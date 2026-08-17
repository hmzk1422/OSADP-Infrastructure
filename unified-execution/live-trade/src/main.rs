mod bybit_ws;
mod db;

use bybit_ws::{BybitTrade, OrderBookWatcher, TradeWatcher};
use db::{ClosedTrade, DbWriter, PnLSnapshot};
use engine::{EngineEvent, MomentumStrategy, StrategyConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use tokio::sync::Mutex;

const SYMBOL: &str = "BTCUSDT";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_path =
        std::env::var("ROCKSDB_PATH").unwrap_or_else(|_| "./data/marketdata.rocksdb".to_string());

    let db = Arc::new(DbWriter::connect(&db_path).await?);
    let strategy = Arc::new(Mutex::new(MomentumStrategy::new(
        SYMBOL,
        StrategyConfig::default(),
    )));

    // Restore dry-run state from the last persisted snapshot.
    if let Ok(snaps) = db.iter_pnl_snapshots(SYMBOL) {
        if let Some(last) = snaps.last() {
            let mut strat = strategy.lock().await;
            strat.restore(last.equity, last.last_exit_ms);
            println!(
                "[restore] equity={:.2} USD, last_exit_ms={}, n_trades={}",
                last.equity, last.last_exit_ms, last.n_trades
            );
        }
    }

    let last_exit_ms = Arc::new(AtomicI64::new(0));

    let trade_strategy = Arc::clone(&strategy);
    let trade_db = Arc::clone(&db);
    let trade_exit = Arc::clone(&last_exit_ms);
    let trades_task = tokio::spawn(async move {
        loop {
            let mut watcher = match TradeWatcher::connect(SYMBOL).await {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("[trades] connect error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }
            };

            loop {
                match watcher.next_batch().await {
                    Ok(trades) => {
                        for trade in trades {
                            trade_db.record_trade(SYMBOL, trade.clone());
                            feed_trade(&trade_strategy, &trade_db, &trade_exit, &trade).await;
                        }
                    }
                    Err(e) => {
                        eprintln!("[trades] stream error: {e}");
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        break; // reconnect
                    }
                }
            }
        }
    });

    let ob_strategy = Arc::clone(&strategy);
    let ob_db = Arc::clone(&db);
    let ob_exit = Arc::clone(&last_exit_ms);
    let ob_task = tokio::spawn(async move {
        loop {
            let mut watcher = match OrderBookWatcher::connect(SYMBOL).await {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("[orderbook] connect error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }
            };

            loop {
                match watcher.next_top().await {
                    Ok(Some(top)) => {
                        ob_db.record_orderbook_top(SYMBOL, top.clone());

                        let mut strat = ob_strategy.lock().await;
                        let events = strat.on_orderbook(
                            top.timestamp,
                            top.bid_price,
                            top.bid_amount,
                            top.ask_price,
                            top.ask_amount,
                        );
                        handle_events(&ob_db, &ob_exit, &mut strat, events, "[LIVE LOB]").await;
                        drop(strat);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("[orderbook] stream error: {e}");
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        break; // reconnect
                    }
                }
            }
        }
    });

    println!(
        "Listening to Bybit Live Trades and LOB for {SYMBOL}, writing to local RocksDB at {db_path}..."
    );

    // Wait for Ctrl+C or for both stream tasks to exit.
    let ctrl_c = tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("ctrl_c monitor");
    });
    let _ = tokio::select! {
        _ = async { let _ = tokio::join!(trades_task, ob_task); } => {},
        _ = ctrl_c => {},
    };

    // Final snapshot + HTML report on shutdown.
    {
        let strat = strategy.lock().await;
        let snap = strat.snapshot();
        let pnl = PnLSnapshot {
            ts_ms: snap.last_features.as_ref().map_or(0, |f| f.ts),
            equity: snap.equity,
            n_trades: snap.n_trades,
            n_wins: snap.n_wins,
            n_losses: snap.n_losses,
            total_r: snap.total_r,
            mid: snap.mid,
            last_exit_ms: last_exit_ms.load(Ordering::SeqCst),
        };
        db.record_pnl_snapshot(SYMBOL, &pnl);
    }

    println!("\n=== Dry-Run Summary ({SYMBOL}) ===");
    let strat = strategy.lock().await;
    let snap = strat.snapshot();
    let wins = snap.n_wins as f64;
    let losses = snap.n_losses as f64;
    let profit_factor = if losses > 0.0 { wins / losses } else { f64::INFINITY };
    println!(
        "equity={:.2} USD | trades={} (W:{}/L:{}) | total_r={:.4} | profit_factor={:.2}",
        snap.equity, snap.n_trades, snap.n_wins, snap.n_losses, snap.total_r, profit_factor
    );
    println!("Generating HTML report -> build/pnl_report_{SYMBOL}.html");
    generate_html_report(&db, SYMBOL).await;

    Ok(())
}


    
/// Feed a single Bybit trade into the engine and persist any emitted events.
async fn feed_trade(
    strategy: &Arc<Mutex<MomentumStrategy>>,
    db: &Arc<DbWriter>,
    exit_ms: &Arc<AtomicI64>,
    trade: &BybitTrade,
) {
    let is_buy = trade.side == "Buy";
    let mut strat = strategy.lock().await;
    let events = strat.on_trade(trade.timestamp, trade.price, trade.amount, is_buy);
    handle_events(db, exit_ms, &mut strat, events, "[LIVE TRADES]").await;
    drop(strat);
}

/// Process engine events: log + persist to RocksDB.
async fn handle_events(
    db: &Arc<DbWriter>,
    exit_ms: &Arc<AtomicI64>,
    strat: &mut MomentumStrategy,
    events: Vec<EngineEvent>,
    tag: &str,
) {
    for ev in events {
        match ev {
            EngineEvent::BarClosed(f) => {
                println!(
                    "{tag} [{}] BAR ts={} close={:.2} rsi2={:.1} win_prob={:.2}",
                    SYMBOL, f.ts, f.close, f.rsi_2, f.win_prob
                );
            }
            EngineEvent::Entered(receipt) => {
                println!(
                    "{tag} [{}] {} ENTRY @ {:.2} qty={:.4} stop={:.2} target={:.2} (rsi2={:.1}, win_prob={:.2})",
                    SYMBOL,
                    if receipt.side == engine::Side::Long { "LONG" } else { "SHORT" },
                    receipt.entry_price,
                    receipt.qty,
                    receipt.stop,
                    receipt.target,
                    receipt.rsi_2,
                    receipt.win_prob,
                );
            }
            EngineEvent::Exited(record) => {
                exit_ms.store(record.exit_ms, Ordering::SeqCst);
                println!(
                    "{tag} [{}] {} EXIT {} @ {:.2} | r={:+.2} | net={:+.2} USD | equity={:.2} ({:.0}s)",
                    SYMBOL,
                    if record.side == engine::Side::Long { "LONG" } else { "SHORT" },
                    record.reason,
                    record.exit_price,
                    record.r_multiple,
                    record.net_usd,
                    record.equity_after,
                    record.duration_s,
                );

                let snap = strat.snapshot();
                let pnl = PnLSnapshot {
                    ts_ms: record.exit_ms,
                    equity: snap.equity,
                    n_trades: snap.n_trades,
                    n_wins: snap.n_wins,
                    n_losses: snap.n_losses,
                    total_r: snap.total_r,
                    mid: snap.mid,
                    last_exit_ms: record.exit_ms,
                };
                db.record_pnl_snapshot(SYMBOL, &pnl);

                let ct = ClosedTrade {
                    entry_ms: record.entry_ms,
                    exit_ms: record.exit_ms,
                    side: if record.side == engine::Side::Long {
                        "long".to_string()
                    } else {
                        "short".to_string()
                    },
                    entry_price: record.entry_price,
                    exit_price: record.exit_price,
                    reason: record.reason.to_string(),
                    qty: record.qty,
                    net_usd: record.net_usd,
                    r_multiple: record.r_multiple,
                    equity_after: record.equity_after,
                    duration_s: record.duration_s,
                };
                db.record_closed_trade(SYMBOL, &ct);
            }
        }
    }
}
/// Generate a standalone HTML PnL report from persisted RocksDB data.
async fn generate_html_report(db: &Arc<DbWriter>, symbol: &str) {
    let snaps = db.iter_pnl_snapshots(symbol).unwrap_or_default();
    let trades = db.iter_closed_trades(symbol).unwrap_or_default();

    let n_trades = trades.len();
    let wins = trades.iter().filter(|t| t.net_usd > 0.0).count();
    let losses = n_trades.saturating_sub(wins);
    let total_net: f64 = trades.iter().map(|t| t.net_usd).sum();
    let total_r: f64 = trades.iter().map(|t| t.r_multiple).sum();
    let profit_factor = if losses > 0 { wins as f64 / losses as f64 } else { f64::INFINITY };
    let final_equity = snaps.last().map(|s| s.equity).unwrap_or(0.0);

    // Build the equity curve as an SVG line chart.
    let curve = if snaps.is_empty() {
        String::new()
    } else {
        let (min_eq, max_eq) = snaps.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(lo, hi), s| (lo.min(s.equity), hi.max(s.equity)),
        );
        let range = (max_eq - min_eq).max(1e-9);
        let w: f64 = 800.0;
        let h: f64 = 200.0;
        let pad: f64 = 30.0;
        let pts: Vec<String> = snaps.iter().enumerate().map(|(i, s)| {
            let span = snaps.len().saturating_sub(1).max(1) as f64;
            let x = pad + (i as f64 / span) * (w - pad * 2.0);
            let y = h - pad - ((s.equity - min_eq) / range) * (h - pad * 2.0);
            format!("{:.1},{:.1}", x, y)
        }).collect();
        format!(
            r##"<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;border-radius:6px">
                <polyline points="{points}" fill="none" stroke="#00d4ff" stroke-width="2"/>
                <text x="{pad}" y="15" fill="#aaa" font-size="12">equity curve (USD)</text>
                <text x="{pad}" y="{h}" fill="#aaa" font-size="10">{min_eq:.2}</text>
                <text x="{pad}" y="{pad}" fill="#aaa" font-size="10">{max_eq:.2}</text>
            </svg>"##,
            w = w, h = h, pad = pad, points = pts.join(" "),
            min_eq = min_eq, max_eq = max_eq,
        )
    };

    let mut rows = String::new();
    for t in &trades {
        let cls = if t.net_usd > 0.0 { "win" } else if t.net_usd < 0.0 { "loss" } else { "breakeven" };
        let side_lbl = if t.side == "long" { "LONG" } else { "SHORT" };
        rows.push_str(&format!(
            r#"<tr class="{cls}">
                <td>{entry_ms}</td><td>{exit_ms}</td><td>{side}</td>
                <td class="num">{entry_price:.4}</td><td class="num">{exit_price:.4}</td>
                <td>{reason}</td><td class="num">{qty:.4}</td>
                <td class="num num-neg">{net:+.2}</td>
                <td class="num">{r_multiple:+.2}</td>
                <td class="num num-pos">{equity_after:+.2}</td>
                <td class="num">{duration:.1}</td>
            </tr>"#,
            cls = cls, entry_ms = t.entry_ms, exit_ms = t.exit_ms, side = side_lbl,
            entry_price = t.entry_price, exit_price = t.exit_price, reason = t.reason,
            qty = t.qty, net = t.net_usd, r_multiple = t.r_multiple,
            equity_after = t.equity_after, duration = t.duration_s,
        ));
    }

    let pf_str = if profit_factor.is_infinite() { "\u{221e}".to_string() } else { format!("{:.2}", profit_factor) };
    let tbody = if rows.is_empty() {
        "<tr><td colspan=\"11\" style=\"text-align:center;color:#888\">No closed trades yet.</td></tr>".to_string()
    } else { rows };

    let html = format!(
        r##"<html><head><meta charset="utf-8"><title>PnL Report — {symbol}</title>
<style>body{{font-family:'Segoe UI',sans-serif;margin:24px;background:#0f0f23;color:#e0e0e0}}
h1{{color:#00d4ff}}.summary{{display:flex;gap:16px;margin:16px 0;flex-wrap:wrap}}
.card{{background:#1a1a2e;border-radius:8px;padding:16px 24px;min-width:140px;text-align:center}}
.card .label{{font-size:12px;color:#888;text-transform:uppercase}}
.card .value{{font-size:24px;font-weight:700;color:#00d4ff}}
table{{border-collapse:collapse;width:100%;margin-top:16px;background:#1a1a2e;border-radius:8px;overflow:hidden}}
th{{background:#162135;padding:8px 12px;text-align:left;font-size:13px;color:#00d4ff;text-transform:uppercase}}
td{{padding:6px 12px;border-bottom:1px solid #0f3460;font-size:13px}}td.num{{text-align:right}}
tr.win{{background:rgba(0,212,255,0.05)}}tr.loss{{background:rgba(255,64,96,0.05)}}
.curve{{margin:16px 0}}footer{{margin-top:24px;color:#555;font-size:12px}}</style>
</head><body><h1>PnL Report — {symbol}</h1>
<div class="summary">
  <div class="card"><div class="label">Equity (USD)</div><div class="value">{equity:.2}</div></div>
  <div class="card"><div class="label">Total Trades</div><div class="value">{n}</div></div>
  <div class="card"><div class="label">Wins / Losses</div><div class="value">{w} / {l}</div></div>
  <div class="card"><div class="label">Total PnL (USD)</div><div class="value">{net:+.2}</div></div>
  <div class="card"><div class="label">Total R</div><div class="value">{r:+.2}</div></div>
  <div class="card"><div class="label">Profit Factor</div><div class="value">{pf}</div></div>
</div>
<div class="curve">{curve}</div>
<table><thead><tr><th>Entry</th><th>Exit</th><th>Side</th><th>Entry Px</th><th>Exit Px</th><th>Reason</th><th>Qty</th><th>Net (USD)</th><th>R-Mult</th><th>Equity After</th><th>Duration (s)</th></tr></thead><tbody>{tbody}</tbody></table>
<footer>Generated by live-trade dry-run engine.</footer></body></html>"##,
        symbol = symbol, equity = final_equity, n = n_trades, w = wins, l = losses,
        net = total_net, r = total_r, pf = pf_str, curve = curve, tbody = tbody,
    );
    let _ = std::fs::create_dir_all("build");
    let out_path = format!("build/pnl_report_{}.html", symbol);
    match std::fs::write(&out_path, &html) {
        Ok(_) => println!("  \u{2192} written to {out_path}"),
        Err(e) => eprintln!("[report] failed to write {out_path}: {e}"),
    }
}
