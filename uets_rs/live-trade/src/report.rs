//! HTML PnL report generation from persisted RocksDB state.
//!
//! Renders an equity curve (inline SVG), a live price chart plus a closed-trade
//! table with win rate, profit factor and cumulative R, written to
//! `build/pnl_report_{symbol}.html`.
//!
//! The report is written BOTH while the session runs (the execution loop
//! regenerates it every [`REPORT_REFRESH_SECS`] seconds) and once more on
//! shutdown. The page carries a `<meta http-equiv="refresh">` so an open
//! browser tab picks up new trades as they happen; scroll position survives
//! the reload via sessionStorage.

use crate::db::DbWriter;

/// Hard cap on plotted price points; longer sessions are strided downsampled
/// (keeps the report a few hundred KB even after days of tape).
const MAX_CHART_POINTS: usize = 4000;

/// How often the execution loop regenerates the report while running. An open
/// browser tab auto-refreshes faster than this (see PAGE_REFRESH_SECS), so new
/// trades appear within ~30 s of closing.
pub const REPORT_REFRESH_SECS: u64 = 30;

/// Browser-side auto-refresh cadence (meta tag in the template). Slightly
/// faster than REPORT_REFRESH_SECS so fresh trades surface promptly; combined
/// with the tmp+rename publish below, a reload can never catch a torn file
/// except across two adjacent syscalls.
const PAGE_REFRESH_SECS: u32 = 15;

/// Which runs persisted in the store the report should show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportScope {
    /// Only the newest/latest run — the numbers the terminal shows for the
    /// current session. This is the default: it keeps the HTML report in sync
    /// with the console status table even when a store holds many runs.
    Latest,
    /// Everything ever persisted to the store (all runs, all time).
    All,
}

impl Default for ReportScope {
    fn default() -> Self {
        ReportScope::Latest
    }
}

/// Generate the standalone HTML report for `symbol` from the open DB handle.
/// `rr_ratio` is the strategy's reward-to-risk (research STEP 5 uses it as the
/// floor of the dual-pass R-clip cap).
pub async fn generate_html_report(
    db: &DbWriter,
    symbol: &str,
    scope: ReportScope,
    rr_ratio: f64,
    out_tag: Option<&str>,
) {
    let all_snaps = db.iter_pnl_snapshots(symbol).unwrap_or_default();
    let all_trades = db.iter_closed_trades(symbol).unwrap_or_default();
    let all_feats = db.iter_features(symbol).unwrap_or_default();
    let all_diags = db.iter_diag(symbol).unwrap_or_default();

    // ---- session scoping ------------------------------------------------------
    // Closed trades / equity snapshots carry a `session_id` (unix-ms at process
    // start). The LATEST session's records become the report's universe; RSI /
    // P(WIN) / latency series have no session id, so they are anchored to the
    // latest run's oldest record timestamp — robust for both live (wall-clock)
    // and replay (historical) timestamps.
    let mut snaps = all_snaps;
    let mut trades = all_trades;
    let mut feats = all_feats;
    let mut diags = all_diags;

    let latest_sid = snaps
        .iter()
        .map(|s| s.session_id)
        .chain(trades.iter().map(|t| t.session_id))
        .max()
        .unwrap_or(0);

    if scope == ReportScope::Latest {
        let mut min_ts: Option<i64> = None;
        let absorb = |ts: i64, m: &mut Option<i64>| {
            *m = Some(m.map_or(ts, |cur| cur.min(ts)));
        };
        for s in snaps.iter().filter(|s| s.session_id == latest_sid) {
            absorb(s.ts_ms, &mut min_ts);
        }
        for t in trades.iter().filter(|t| t.session_id == latest_sid) {
            absorb(t.entry_ms.min(t.exit_ms), &mut min_ts);
        }
        snaps.retain(|s| s.session_id == latest_sid);
        trades.retain(|t| t.session_id == latest_sid);
        match min_ts {
            Some(t0) => {
                feats.retain(|f| f.ts_ms >= t0);
                diags.retain(|d| d.ts_ms >= t0);
            }
            None => {
                // Fresh run with no closed trades yet — don't show stale rows.
                feats.clear();
                diags.clear();
            }
        }
    }

    let scope_lbl = if scope == ReportScope::Latest {
        if latest_sid == 0 {
            "all runs (no sessions recorded)".to_string()
        } else {
            format!("session #{latest_sid}")
        }
    } else {
        "all runs".to_string()
    };

    let n_trades = trades.len();
    let wins = trades.iter().filter(|t| t.net_usd > 0.0).count();
    let losses = n_trades.saturating_sub(wins);
    let total_net: f64 = trades.iter().map(|t| t.net_usd).sum();
    let total_r: f64 = trades.iter().map(|t| t.r_multiple).sum();

    // ---- dual-pass research-parity R clip (STEP 5, ml_state_optimized.py) ----
    // Research computes cap = max(|q(0.99)|, |q(0.01)|, RR_RATIO) over the
    // WHOLE dataset's pre-clip R values and clamps every trade BEFORE
    // expectancy is ever computed. The engine streams trades one at a time, so
    // it records the uncapped value (`r_raw`); here — after the fact — the same
    // cap is derived and applied, making these cards directly comparable with
    // the research walk-forward expectancy tables. Trades written before
    // `r_raw` existed fall back to their (already-final) `r_multiple`.
    let raw_rs: Vec<f64> = trades
        .iter()
        .map(|t| t.r_raw.unwrap_or(t.r_multiple))
        .collect();
    let r_cap = engine::dataset_r_cap(rr_ratio, &raw_rs);
    let total_r_clipped: f64 = match r_cap {
        Some(cap) => raw_rs.iter().map(|r| r.clamp(-cap, cap)).sum(),
        None => total_r,
    };
    let clip_binds = match r_cap {
        Some(cap) => raw_rs.iter().any(|r| r.abs() > cap + 1e-9),
        None => false,
    };
    let expectancy_clipped = if n_trades > 0 {
        total_r_clipped / n_trades as f64
    } else {
        0.0
    };
    let (r_sub, expectancy_sub) = match r_cap {
        Some(cap) if clip_binds => (
            format!("raw · clip ±{cap:.2} → {total_r_clipped:+.2}R (research parity)"),
            format!("R/trade raw · clip ±{cap:.2} → {expectancy_clipped:+.2}"),
        ),
        _ => ("sum of trade R".to_string(), "R per trade".to_string()),
    };

    // Profit factor computed in R-multiples (gross R+ / gross R-) so the card
    // AGREEES with the TOTAL R number. (It used to be on USD net, which is why
    // a +8.76R session could still show a ~1.0 profit factor.)
    let gross_profit_r: f64 = trades.iter().map(|t| t.r_multiple.max(0.0)).sum();
    let gross_loss_r: f64 = trades.iter().map(|t| t.r_multiple.min(0.0)).sum::<f64>().abs();
    let profit_factor = if gross_loss_r > 1e-9 {
        gross_profit_r / gross_loss_r
    } else {
        f64::INFINITY
    };

    // Keep USD gross figures for the Avg Win / Avg Loss card.
    let gross_profit: f64 = trades.iter().map(|t| t.net_usd.max(0.0)).sum();
    let gross_loss: f64 = trades.iter().map(|t| t.net_usd.min(0.0)).sum::<f64>().abs();

    let win_rate = if n_trades > 0 {
        100.0 * wins as f64 / n_trades as f64
    } else {
        0.0
    };

    // Max drawdown over the equity curve (peak-to-trough, USD + %).
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0f64;
    let mut peak_eq = 0.0f64;
    for s in &snaps {
        if s.equity > peak {
            peak = s.equity;
        }
        let dd = peak - s.equity;
        if dd > max_dd {
            max_dd = dd;
            peak_eq = peak;
        }
    }
    let max_dd_pct = if peak_eq > 1e-9 {
        100.0 * max_dd / peak_eq
    } else {
        0.0
    };

    let avg_win = if wins > 0 {
        gross_profit / wins as f64
    } else {
        0.0
    };
    let avg_loss = if losses > 0 {
        gross_loss / losses as f64
    } else {
        0.0
    };
    // Expectancy per trade, in R, so it's consistent with TOTAL R
    // (e.g. 8.76R over 23 trades -> 0.38R/trade).
    let expectancy = if n_trades > 0 { total_r / n_trades as f64 } else { 0.0 };
    let best = trades.iter().map(|t| t.net_usd).fold(f64::NEG_INFINITY, f64::max);
    let worst = trades.iter().map(|t| t.net_usd).fold(f64::INFINITY, f64::min);
    let longs = trades.iter().filter(|t| t.side == "long").count();
    let shorts = n_trades.saturating_sub(longs);
    let final_equity = snaps.last().map(|s| s.equity).unwrap_or(0.0);

    // ---- chart payload -------------------------------------------------------
    // 1-second bucket closes from the raw trade tape (streamed, O(buckets)
    // memory), stride-downsampled to MAX_CHART_POINTS so multi-day sessions
    // stay a few hundred KB of JSON. Trades, equity snapshots AND the full
    // per-second feature history (RSI / P(WIN) â€” every bar, not just the
    // hunting zone) ride along so the canvas chart draws exactly what the
    // algorithm saw, TradingView-style.
    let downsample = |mut v: Vec<(i64, f64)>| -> Vec<(i64, f64)> {
        if v.len() > MAX_CHART_POINTS {
            let stride = (v.len() + MAX_CHART_POINTS - 1) / MAX_CHART_POINTS;
            v = v
                .into_iter()
                .enumerate()
                .filter(|(i, _)| i % stride == 0)
                .map(|(_, p)| p)
                .collect();
        }
        v
    };
    let prices = downsample(db.price_series_1s(symbol).unwrap_or_default());
    let equity: Vec<(i64, f64)> = snaps.iter().map(|s| (s.ts_ms, s.equity)).collect();
    let rsi: Vec<(i64, f64)> = downsample(feats.iter().map(|f| (f.ts_ms, f.rsi_2)).collect());
    let wp: Vec<(i64, f64)> = downsample(feats.iter().map(|f| (f.ts_ms, f.win_prob)).collect());

    // Latency / feed-health diagnostics (one sample per loop tick, keyed with
    // the tape) -> exported so the chart can draw a live diagnostics pane.
    let diags = db.iter_diag(symbol).unwrap_or_default();
    let diag_lat: Vec<(i64, f64)> =
        downsample(diags.iter().map(|d| (d.ts_ms, d.latency_ms)).collect());
    let diag_skew: Vec<(i64, f64)> =
        downsample(diags.iter().map(|d| (d.ts_ms, d.clock_skew_ms)).collect());
    let diag_stal: Vec<(i64, f64)> =
        downsample(diags.iter().map(|d| (d.ts_ms, d.feed_staleness_ms)).collect());

    let start_eq = snaps.first().map(|s| s.equity).unwrap_or(final_equity);
    let payload = serde_json::json!({
        "symbol": symbol,
        "scope": scope_lbl,
        "prices": prices,
        "rsi": rsi,
        "wp": wp,
        "diag": serde_json::json!({
            "latency": diag_lat,
            "skew": diag_skew,
            "stal": diag_stal,
        }),
        "startEq": start_eq,
        "equity": equity,
        "trades": trades
            .iter()
            .map(|t| serde_json::json!({
                "e": t.entry_ms, "x": t.exit_ms, "side": t.side,
                "ep": t.entry_price, "xp": t.exit_price, "qty": t.qty,
                "r": t.reason, "net": t.net_usd, "rm": t.r_multiple,
                "rm_raw": t.r_raw,
                "bp": t.barrier_pct, "tp": t.target_price, "sl": t.stop_price,
            }))
            .collect::<Vec<_>>(),
    });
    let chart_data = serde_json::to_string(&payload)
        .unwrap_or_else(|_| "{}".to_string())
        .replace('<', "\\u003c"); // never break out of the <script> context

    let chart_block = format!(
        r#"<div class="legend">
  <span><i style="background:#26a69a"></i>bull candle</span>
  <span><i style="background:#ef5350"></i>bear candle</span>
  <span><i style="background:#26a69a;border-radius:0"></i>long / exit-win</span>
  <span><i style="background:#ef5350;border-radius:0"></i>short / exit-loss</span>
  <span><i style="background:#7e57c2"></i>RSI(2)</span>
  <span><i style="background:#2962ff"></i>equity</span>
  <span><i style="background:none;border-top:2px dashed #26a69a;height:0;border-radius:0"></i>target</span>
  <span><i style="background:none;border-top:2px dashed #ef5350;height:0;border-radius:0"></i>stop</span>
  <span style="color:#787b86">scroll = zoom Â· drag = pan Â· double-click = reset</span>
</div>
<div id="chartwrap">
  <div id="tvlegend"></div>
  <canvas id="tvchart"></canvas>
  <div id="tip"></div>
</div>
<script>var REPORT_DATA = {data};
{chart_js}</script>"#,
        data = chart_data,
        chart_js = CHART_JS,
    );

    let mut rows = String::new();
    for (idx, t) in trades.iter().enumerate() {
        let cls = if t.net_usd > 0.0 {
            "win"
        } else if t.net_usd < 0.0 {
            "loss"
        } else {
            "breakeven"
        };
        let side_lbl = if t.side == "long" { "LONG" } else { "SHORT" };
        let side_cls = if t.side == "long" { "b-long" } else { "b-short" };
        rows.push_str(&format!(
            r#"<tr class="{cls}" data-e="{entry_ms}" data-x="{exit_ms}">
                <td class="num dim">{i}</td><td>{entry_ms}</td><td>{exit_ms}</td>
                <td><span class="badge {side_cls}">{side}</span></td>
                <td class="num">{entry_price:.2}</td><td class="num">{exit_price:.2}</td>
                <td><span class="badge b-reason">{reason}</span></td><td class="num">{qty:.4}</td>
                <td class="num {pnl_cls}">{net:+.2}</td>
                <td class="num {pnl_cls}">{r_multiple:+.2}</td>
                <td class="num">{equity_after:.2}</td>
                <td class="num">{duration:.1}</td>
            </tr>"#,
            cls = cls,
            i = idx + 1,
            entry_ms = t.entry_ms,
            exit_ms = t.exit_ms,
            side = side_lbl,
            side_cls = side_cls,
            entry_price = t.entry_price,
            exit_price = t.exit_price,
            reason = t.reason,
            qty = t.qty,
            net = t.net_usd,
            pnl_cls = if t.net_usd >= 0.0 { "up" } else { "down" },
            r_multiple = t.r_multiple,
            equity_after = t.equity_after,
            duration = t.duration_s,
        ));
    }

    let pf_str = if profit_factor.is_infinite() {
        "âˆž".to_string()
    } else {
        format!("{:.2}", profit_factor)
    };
    let tbody = if rows.is_empty() {
        "<tr><td colspan=\"12\" style=\"text-align:center;color:#787b86\">No closed trades yet.</td></tr>"
            .to_string()
    } else {
        rows
    };
    let best_str = if trades.is_empty() {
        "â€”".to_string()
    } else {
        format!("{best:+.2}")
    };
    let worst_str = if trades.is_empty() {
        "â€”".to_string()
    } else {
        format!("{worst:+.2}")
    };
    let net_pct = if start_eq > 1e-9 && n_trades > 0 {
        format!(" ({:+.2}%)", 100.0 * total_net / start_eq)
    } else {
        String::new()
    };
    let npnl_cls = if total_net >= 0.0 { "up" } else { "down" };
    let r_cls = if total_r >= 0.0 { "up" } else { "down" };
    let pf_cls = if profit_factor >= 1.0 { "up" } else { "down" };

    let html = format!(
        r##"<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{refresh}"><title>PnL Report â€” {symbol}</title>
<style>
:root{{--bg:#131722;--panel:#1e222d;--border:#2a2e39;--txt:#d1d4dc;--mut:#787b86;
      --up:#26a69a;--dn:#ef5350;--acc:#2962ff}}
*{{box-sizing:border-box}}
body{{font-family:'Trebuchet MS','Segoe UI',sans-serif;margin:0;background:var(--bg);color:var(--txt)}}
.topbar{{display:flex;align-items:center;gap:18px;padding:10px 20px;background:var(--panel);
        border-bottom:1px solid var(--border);position:sticky;top:0;z-index:9}}
.brand{{font-weight:bold;font-size:15px;color:#f0b90b;letter-spacing:.5px}}
.brand span{{color:var(--txt);font-weight:normal;margin-left:6px}}
.sym{{font-size:17px;font-weight:bold}} .sym em{{color:var(--mut);font-style:normal;font-size:12px;margin-left:8px}}
.chips{{margin-left:auto;display:flex;gap:8px;font-size:11px;color:var(--mut)}}
.chips span{{background:#2a2e39;border-radius:4px;padding:3px 9px}}
.pausebb{{background:var(--panel);border:1px solid var(--border);color:var(--mut);border-radius:4px;
        padding:3px 12px;font-size:11px;cursor:pointer;letter-spacing:.4px;text-transform:uppercase}}
.pausebb:hover{{color:var(--txt)}}
.pausebb.on{{color:#fff;border-color:#f0b90b;background:rgba(240,185,11,.15)}}
.wrap{{padding:16px 20px;max-width:1600px;margin:0 auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:10px;margin-bottom:14px}}
.card{{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:10px 12px}}
.card .label{{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.6px;margin-bottom:5px}}
.card .value{{font-size:19px;font-weight:bold;font-variant-numeric:tabular-nums}}
.card .sub{{font-size:10px;color:var(--mut);margin-top:3px}}
.up{{color:var(--up)}} .down{{color:var(--dn)}} .neu{{color:var(--txt)}}
.sect{{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:1px;margin:20px 0 8px}}
.legend{{display:flex;gap:16px;margin:0 0 6px;font-size:11px;color:var(--mut);flex-wrap:wrap;align-items:center}}
.legend i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:-1px}}
#chartwrap{{position:relative;background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:6px}}
#tvlegend{{position:absolute;top:10px;left:12px;z-index:4;font-size:11px;line-height:1.65;pointer-events:none;
          background:rgba(19,23,34,.72);padding:6px 10px;border-radius:4px;font-variant-numeric:tabular-nums}}
#tvlegend b{{font-size:12px}}
#tip{{position:absolute;display:none;pointer-events:none;background:#1e222d;border:1px solid var(--border);
     border-radius:5px;padding:7px 10px;font-size:11px;z-index:6;white-space:nowrap;line-height:1.55;
     box-shadow:0 4px 14px rgba(0,0,0,.45);font-variant-numeric:tabular-nums}}
canvas{{display:block;cursor:crosshair}}
.filters{{display:flex;gap:6px;margin:0 0 8px}}
.filters button{{background:var(--panel);border:1px solid var(--border);color:var(--mut);border-radius:4px;
        padding:4px 14px;font-size:11px;cursor:pointer;text-transform:uppercase;letter-spacing:.5px}}
.filters button.on{{color:#fff;border-color:var(--acc);background:rgba(41,98,255,.15)}}
table{{border-collapse:collapse;width:100%;background:var(--panel);border:1px solid var(--border);
      border-radius:6px;overflow:hidden;font-variant-numeric:tabular-nums}}
th{{background:#2a2e39;padding:8px 10px;text-align:left;font-size:10px;color:var(--mut);
   text-transform:uppercase;letter-spacing:.6px;cursor:pointer;user-select:none}}
th:hover{{color:var(--txt)}} th.sorted::after{{content:" â–¾";color:var(--acc)}}
th.sorted.rev::after{{content:" â–´"}}
td{{padding:6px 10px;border-bottom:1px solid #262a35;font-size:12px}}
td.num{{text-align:right}} td.dim{{color:var(--mut)}}
tbody tr{{cursor:pointer}} tbody tr:hover{{background:#262a35}}
tr.win td:first-child{{box-shadow:inset 2px 0 0 var(--up)}}
tr.loss td:first-child{{box-shadow:inset 2px 0 0 var(--dn)}}
.badge{{display:inline-block;padding:1px 7px;border-radius:3px;font-size:10px;font-weight:bold;letter-spacing:.5px}}
.b-long{{background:rgba(38,166,154,.16);color:var(--up)}}
.b-short{{background:rgba(239,83,80,.16);color:var(--dn)}}
.b-reason{{background:#2a2e39;color:var(--mut);text-transform:uppercase}}
footer{{margin-top:22px;color:#5d606b;font-size:11px;padding-bottom:20px}}
</style>
</head><body>
<div class="topbar">
  <div class="brand">â—† OSADP<span>PnL REPORT</span></div>
  <div class="sym">{symbol}<em>paper Â· feed overlay</em></div>
  <div class="chips"><span id="chipScope">{scope_lbl}</span><span id="chipRange">session â€”</span><span>auto-refresh {refresh}s</span><button id="pauseBtn" type="button" class="pausebb">Pause</button></div>
</div>
<div class="wrap">
<div class="cards">
  <div class="card"><div class="label">Equity</div><div class="value neu">${equity:.2}</div><div class="sub">start ${start_eq:.2}</div></div>
  <div class="card"><div class="label">Net PnL</div><div class="value {npnl_cls}">{net:+.2}</div><div class="sub">{net_pct}</div></div>
  <div class="card"><div class="label">Total R</div><div class="value {r_cls}">{r:+.2}R</div><div class="sub">{r_sub}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value neu">{win_rate:.1}%</div><div class="sub">{w}W / {l}L of {n}</div></div>
  <div class="card"><div class="label">Profit Factor</div><div class="value {pf_cls}">{pf}</div><div class="sub">gross R+ / gross R-</div></div>
  <div class="card"><div class="label">Max Drawdown</div><div class="value down">-{max_dd:.2}</div><div class="sub">{max_dd_pct:.2}% off peak</div></div>
  <div class="card"><div class="label">Expectancy</div><div class="value {r_cls}">{expectancy:+.2}R</div><div class="sub">{expectancy_sub}</div></div>
  <div class="card"><div class="label">Avg Win / Loss</div><div class="value neu"><span class="up">{avg_win:.2}</span>/<span class="down">{avg_loss:.2}</span></div></div>
  <div class="card"><div class="label">Best Trade</div><div class="value up">{best}</div></div>
  <div class="card"><div class="label">Worst Trade</div><div class="value down">{worst}</div></div>
  <div class="card"><div class="label">Long / Short</div><div class="value neu"><span class="up">{longs}</span> / <span class="down">{shorts}</span></div></div>
</div>
<div class="sect">Price Â· RSI(2) Â· Diag (latency/skew/staleness) Â· Equity â€” every second of tape, all indicator values</div>
{chart_block}
<div class="sect">Trade list â€” click a row to locate it on the chart Â· click a header to sort</div>
<div class="filters">
  <button data-f="all" class="on">All ({n})</button>
  <button data-f="win">Wins ({w})</button>
  <button data-f="loss">Losses ({l})</button>
</div>
<table id="trades"><thead><tr>
  <th>#</th><th>Entry</th><th>Exit</th><th>Side</th><th>Entry Px</th><th>Exit Px</th>
  <th>Reason</th><th>Qty</th><th>Net (USD)</th><th>R-Mult</th><th>Equity After</th><th>Dur (s)</th>
</tr></thead><tbody>{tbody}</tbody></table>
<footer>Generated by live-trade execution loop Â· candles aggregated client-side from the 1 s tape.</footer>
</div>
<script>
(function () {{
  var tb = document.querySelector('#trades tbody');
  if (!tb) return;
  // ---- row click -> jump the chart to this trade -------------------------
  tb.addEventListener('click', function (ev) {{
    var tr = ev.target.closest('tr'); if (!tr || !tr.dataset.e) return;
    if (window.locateTrade) window.locateTrade(+tr.dataset.e, +tr.dataset.x || 0);
    var w = document.getElementById('chartwrap');
    if (w) w.scrollIntoView({{behavior:'smooth', block:'center'}});
  }});
  // ---- filter chips -------------------------------------------------------
  function applyFilter(f) {{
    document.querySelectorAll('.filters button').forEach(function (x) {{
      x.classList.toggle('on', x.dataset.f === f);
    }});
    tb.querySelectorAll('tr').forEach(function (tr) {{
      tr.style.display = (f === 'all' || tr.classList.contains(f)) ? '' : 'none';
    }});
  }}
  document.querySelectorAll('.filters button').forEach(function (b) {{
    b.addEventListener('click', function () {{
      var f = b.dataset.f;
      applyFilter(f);
      try {{ sessionStorage.setItem('pnl_report_filter', f); }} catch (e) {{}}
    }});
  }});
  // Keep the reader's win/loss filter across the 15s auto-refresh.
  var savedF = null;
  try {{ savedF = sessionStorage.getItem('pnl_report_filter'); }} catch (e) {{}}
  if (savedF) applyFilter(savedF);
  // ---- column sorting -----------------------------------------------------
  var ths = document.querySelectorAll('#trades th');
  ths.forEach(function (th, ci) {{
    th.addEventListener('click', function () {{
      var rows = Array.prototype.slice.call(tb.querySelectorAll('tr'))
        .filter(function (r) {{ return r.dataset.e; }});
      var rev = th.classList.contains('sorted') && !th.classList.contains('rev');
      ths.forEach(function (x) {{ x.classList.remove('sorted', 'rev'); }});
      th.classList.add('sorted'); if (rev) th.classList.add('rev');
      rows.sort(function (a, b) {{
        var av = a.children[ci].textContent.trim(), bv = b.children[ci].textContent.trim();
        var an = parseFloat(av.replace(/[^0-9.\\-]/g, '')), bn = parseFloat(bv.replace(/[^0-9.\\-]/g, ''));
        var cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : av.localeCompare(bv);
        return rev ? cmp : -cmp;
      }});
      rows.forEach(function (r) {{ tb.appendChild(r); }});
    }});
  }});
}})();
</script>
</body></html>"##,
        symbol = symbol,
        scope_lbl = scope_lbl,
        refresh = PAGE_REFRESH_SECS,
        equity = final_equity,
        start_eq = start_eq,
        net = total_net,
        net_pct = net_pct,
        npnl_cls = npnl_cls,
        r = total_r,
        r_cls = r_cls,
        r_sub = r_sub,
        win_rate = win_rate,
        w = wins,
        l = losses,
        n = n_trades,
        pf = pf_str,
        pf_cls = pf_cls,
        max_dd = max_dd,
        max_dd_pct = max_dd_pct,
        expectancy = expectancy,
        expectancy_sub = expectancy_sub,
        avg_win = avg_win,
        avg_loss = avg_loss,
        best = best_str,
        worst = worst_str,
        longs = longs,
        shorts = shorts,
        chart_block = chart_block,
        tbody = tbody,
    );

    let _ = std::fs::create_dir_all("build");
    // When a per-asset report tag is set (portfolio twins), the output file
    // uses the tag so two processes with the same ticker don't clobber each other.
    let out_path = match out_tag {
        Some(tag) => format!("build/pnl_report_{tag}.html"),
        None => format!("build/pnl_report_{}.html", symbol),
    };
    // Publish via tmp+rename so a browser auto-refresh can essentially never
    // load a half-written file (rename replaces atomically where supported;
    // on Windows we fall back to remove+rename across two adjacent syscalls).
    let tmp_path = format!("{out_path}.tmp");
    match std::fs::write(&tmp_path, &html)
        .ok()
        .and_then(|_| {
            if std::fs::rename(&tmp_path, &out_path).is_ok() {
                Some(())
            } else {
                let _ = std::fs::remove_file(&out_path);
                std::fs::rename(&tmp_path, &out_path).ok()
            }
        }) {
        // Success is silent: the loop republishes every REPORT_REFRESH_SECS and
        // a console line per refresh would drown the [bar]/[entry] telemetry.
        Some(_) => {}
        None => eprintln!("[report] failed to write {out_path}"),
    }
}

/// Inline vanilla-JS renderer for the TradingView-style chart. Kept as a plain
/// string constant (outside `format!`) so the JS needs no brace escaping.
const CHART_JS: &str = include_str!("report_chart.js");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{ClosedTrade, PnLSnapshot};
    use crate::exchange::{MarketTrade, TradeSide};

    /// End-to-end: seed a temp store with a synthetic tape + one winning long,
    /// generate the report, and assert the chart payload carries the price
    /// series, the trade overlay and the equity snapshot.
    #[tokio::test]
    async fn report_embeds_chart_with_trades() {
        let dir = std::env::temp_dir().join(format!("imc_report_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db = DbWriter::connect(dir.to_str().expect("utf8 temp path"))
            .await
            .expect("db open");

        // 300 s tape: one print per second, price drifting 100.0 -> 129.9.
        for i in 0..300u32 {
            db.record_trade(
                "btcusdt",
                MarketTrade {
                    timestamp: 1_700_000_000_000 + i as i64 * 1000,
                    price: 100.0 + f64::from(i) * 0.1,
                    amount: 0.01,
                    side: TradeSide::Buy,
                    id: format!("{i}"),
                },
            );
        }
        db.record_pnl_snapshot(
            "btcusdt",
            &PnLSnapshot {
                ts_ms: 1_700_000_060_000,
                equity: 10_050.0,
                n_trades: 1,
                n_wins: 1,
                n_losses: 0,
                total_r: 2.0,
                mid: 106.0,
                last_exit_ms: 1_700_000_060_000,
                session_id: 0,
            },
        );
        db.record_closed_trade(
            "btcusdt",
            &ClosedTrade {
                entry_ms: 1_700_000_010_000,
                exit_ms: 1_700_000_060_000,
                side: "long".into(),
                entry_price: 101.0,
                exit_price: 106.0,
                reason: "take_profit".into(),
                qty: 0.5,
                net_usd: 50.0,
                r_multiple: 2.0,
                r_raw: Some(2.0),
                equity_after: 10_050.0,
                duration_s: 50.0,
                session_id: 0,
                barrier_pct: 0.05,
                target_price: 106.0,
                stop_price: 96.0,
            },
        );
        // Feature history for the report's RSI pane (full series, all values).
        for i in 0..10u32 {
            db.record_feature(
                "btcusdt",
                &crate::db::FeatureSample {
                    ts_ms: 1_700_000_000_000 + i as i64 * 1000,
                    rsi_2: 20.0 + f64::from(i),
                    win_prob: 0.5,
                },
            );
        }

        generate_html_report(&db, "btcusdt", ReportScope::All, 10.0, None).await;

        let html =
            std::fs::read_to_string("build/pnl_report_btcusdt.html").expect("report written");
        assert!(html.contains("chartwrap"), "chart container present");
        assert!(html.contains("REPORT_DATA"), "chart payload present");
        assert!(html.contains("\"ep\":101"), "trade entry overlay in payload");
        assert!(html.contains("take_profit"), "trade reason in payload");
        assert!(html.contains("\"rsi\":[["), "RSI series embedded in payload");
        assert!(html.contains("tvchart"), "TradingView canvas present");
        // ~300 downsampled price points must be embedded (13-digit ms tuples).
        let pts = html.matches(",1700000").count() + html.matches("[1700000").count();
        assert!(pts >= 100, "expected >=100 price points, found {pts}");

        drop(db);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
