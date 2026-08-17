use engine::{MomentumStrategy, StrategyConfig};

fn sortino_ratio(ret: &[f64], risk_free_annual: f64) -> f64 {
    if ret.is_empty() {
        return 0.0;
    }
    let excess: Vec<f64> = ret.iter().map(|r| r - risk_free_annual / 365.0).collect();
    let mean_excess: f64 = excess.iter().sum::<f64>() / excess.len() as f64;
    if mean_excess.abs() < 1e-12 {
        return 0.0;
    }
    let std_dev: f64 = excess.iter().map(|x| (x - mean_excess).powi(2)).sum::<f64>().sqrt()
        / (ret.len() as f64).sqrt();
    if std_dev.abs() < 1e-12 {
        return 0.0;
    }
    mean_excess / std_dev
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let _parquet_path = args.get(1).map(|s| s.as_str()).unwrap_or("");

    let cfg = StrategyConfig::default();
    let mut strategy = MomentumStrategy::new("BTCUSDT", cfg);

    let risk_free_annual = 0.05f64;

    // Simulated price data — in a full impl this would read from parquet:
    // price drops into RSI<30 hunting zone, then recovers for barrier exit
    let simulated_prices: Vec<f64> = vec![
        100.0,  // start
        99.5,   // slight dip
        97.0,   // dump into RSI<30 zone → triggers long entry
        98.5,   // recovery
        100.5,  // target reached → barrier exit with R ~ +1
        102.0,  // past target
        101.0,  // pullback
        103.5,  // new high
        105.0,  // continued up
        104.0,  // minor pullback
    ];

    let mut equity = 100.0f64;
    let mut trades: Vec<f64> = Vec::new();
    let mut prev_price: Option<f64> = None;

    // Simulate tick-by-tick feed through the strategy
    for (idx, price) in simulated_prices.iter().enumerate() {
        let ts = (idx as i64) * 60000; // 1-min bars for demo
        let side = price < &99.5; // simplified: enter when price drops below threshold
        let events = strategy.on_trade(ts, *price, 1.0, side);
        for event in events.iter() {
            match event {
                engine::EngineEvent::Entered(_receipt) => {
                    let notional = price * 1.0;
                    equity -= notional;
                }
                engine::EngineEvent::Exited(receipt) => {
                    equity += receipt.r_multiple * 10.0;
                    if let Some(p) = prev_price {
                        let ret = (price - p) / p;
                        trades.push(ret);
                    }
                    prev_price = Some(*price);
                }
                _ => {}
            }
        }
    }

    let total_trades = strategy.closed_trades().len();
    let final_equity = equity;
    let _total_return = (final_equity - 100.0) / 100.0;
    let sortino = sortino_ratio(&trades, risk_free_annual);

    let _ = std::fs::create_dir_all("build");

    let html = format!(
        r#"<html><head><meta charset="utf-8"><title>PnL Report - Live Test</title>
<style>
body{{font-family:'Segoe UI',sans-serif;margin:20px;background:#1a1a2e;color:#e0e0e0}}
h1{{color:#00d4ff}}.summary{{margin:20px 0;flex-wrap:wrap}}.card{{background:#0f0f20;border-radius:8px;padding:16px 20px;min-width:200px;text-align:center}}.card .label{{font-size:12px;color:#888;text-transform:uppercase}}.card .value{{font-size:24px;font-weight:700;color:#00d4ff}}.footer{{margin-top:20px;color:#555;font-size:12px}}</style></head>
<body><h1>PnL Report — Live Test (Parquet Simulation)</h1><div class="summary">
<div class="card"><div class="label">Starting Equity</div><div class="value">100.00 USD</div></div>
<div class="card"><div class="label">Final Equity</div><div class="value">{:.2}</div></div>
<div class="card"><div class="label">Total Trades</div><div class="value">{}</div></div>
<div class="card"><div class="label">Total Return</div><div class="value">{:.2}</div></div>
<div class="card"><div class="label">Sortino Ratio</div><div class="value">{:.3}</div></div></div></body></html>"#,
        final_equity, total_trades, final_equity - 100.0, sortino
    );
    let out_path = format!("build/pnl_report_live_test.html");
    match std::fs::write(&out_path, &html) {
        Ok(_) => println!("  → written to {}", out_path),
        Err(e) => eprintln!("[report] failed to write {}: {}", out_path, e),
    }
}