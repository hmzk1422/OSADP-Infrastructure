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

use crate::db::{DbWriter, PnLSnapshot};

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

/// Generate the standalone HTML report for `symbol` from the open DB handle.
pub async fn generate_html_report(db: &DbWriter, symbol: &str) {
    let snaps = db.iter_pnl_snapshots(symbol).unwrap_or_default();
    let trades = db.iter_closed_trades(symbol).unwrap_or_default();

    let n_trades = trades.len();
    let wins = trades.iter().filter(|t| t.net_usd > 0.0).count();
    let losses = n_trades.saturating_sub(wins);
    let total_net: f64 = trades.iter().map(|t| t.net_usd).sum();
    let total_r: f64 = trades.iter().map(|t| t.r_multiple).sum();
    let profit_factor = if losses > 0 {
        wins as f64 / losses as f64
    } else {
        f64::INFINITY
    };
    let final_equity = snaps.last().map(|s| s.equity).unwrap_or(0.0);

    let curve = build_equity_curve(&snaps);

    // ---- price-action chart -------------------------------------------------
    // 1-second bucket closes from the raw trade tape (streamed, O(buckets)
    // memory), stride-downsampled to MAX_CHART_POINTS so multi-day sessions
    // stay a few hundred KB of JSON. Trades + equity snapshots ride along so
    // the canvas chart draws exactly what the algorithm did: shaded holding
    // spans, entry/exit markers, and the equity subplot on a shared clock.
    let mut prices = db.price_series_1s(symbol).unwrap_or_default();
    if prices.len() > MAX_CHART_POINTS {
        let stride = (prices.len() + MAX_CHART_POINTS - 1) / MAX_CHART_POINTS;
        prices = prices
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0)
            .map(|(_, p)| p)
            .collect();
    }
    let payload = serde_json::json!({
        "symbol": symbol,
        "prices": prices,
        "equity": snaps
            .iter()
            .map(|s| (s.ts_ms, s.equity))
            .collect::<Vec<_>>(),
        "trades": trades
            .iter()
            .map(|t| serde_json::json!({
                "e": t.entry_ms, "x": t.exit_ms, "side": t.side,
                "ep": t.entry_price, "xp": t.exit_price, "qty": t.qty,
                "r": t.reason, "net": t.net_usd, "rm": t.r_multiple,
            }))
            .collect::<Vec<_>>(),
    });
    let chart_data = serde_json::to_string(&payload)
        .unwrap_or_else(|_| "{}".to_string())
        .replace('<', "\\u003c"); // never break out of the <script> context

    let chart_block = format!(
        r#"<div class="legend">
  <span><i style="background:#00d4ff"></i>price</span>
  <span><i style="background:#2ecc71"></i>entry (long)</span>
  <span><i style="background:#e67e22"></i>entry (short)</span>
  <span><i style="background:#2ecc71;border-radius:0"></i>exit win</span>
  <span><i style="background:#ff4060;border-radius:0"></i>exit loss</span>
  <span><i style="background:#ffb347"></i>equity</span>
</div>
<div id="chartwrap">
  <canvas id="priceCanvas"></canvas>
  <canvas id="eqCanvas"></canvas>
  <div id="tip"></div>
</div>
<script>var REPORT_DATA = {data};
{chart_js}</script>"#,
        data = chart_data,
        chart_js = CHART_JS,
    );

    let mut rows = String::new();
    for t in &trades {
        let cls = if t.net_usd > 0.0 {
            "win"
        } else if t.net_usd < 0.0 {
            "loss"
        } else {
            "breakeven"
        };
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
            cls = cls,
            entry_ms = t.entry_ms,
            exit_ms = t.exit_ms,
            side = side_lbl,
            entry_price = t.entry_price,
            exit_price = t.exit_price,
            reason = t.reason,
            qty = t.qty,
            net = t.net_usd,
            r_multiple = t.r_multiple,
            equity_after = t.equity_after,
            duration = t.duration_s,
        ));
    }

    let pf_str = if profit_factor.is_infinite() {
        "∞".to_string()
    } else {
        format!("{:.2}", profit_factor)
    };
    let tbody = if rows.is_empty() {
        "<tr><td colspan=\"11\" style=\"text-align:center;color:#888\">No closed trades yet.</td></tr>"
            .to_string()
    } else {
        rows
    };

    let html = format!(
        r##"<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{refresh}"><title>PnL Report — {symbol}</title>
<style>body{{font-family:'Segoe UI',sans-serif;margin:24px;background:#0f0f23;color:#e0e0e0}}
h1{{color:#00d4ff}}.summary{{display:flex;gap:16px;margin:16px 0;flex-wrap:wrap}}
.card{{background:#1a1a2e;border-radius:8px;padding:16px 24px;min-width:140px;text-align:center}}
.card .label{{font-size:12px;color:#888;text-transform:uppercase}}
.card .value{{font-size:24px;font-weight:700;color:#00d4ff}}
table{{border-collapse:collapse;width:100%;margin-top:16px;background:#1a1a2e;border-radius:8px;overflow:hidden}}
th{{background:#162135;padding:8px 12px;text-align:left;font-size:13px;color:#00d4ff;text-transform:uppercase}}
td{{padding:6px 12px;border-bottom:1px solid #0f3460;font-size:13px}}td.num{{text-align:right}}
tr.win{{background:rgba(0,212,255,0.05)}}tr.loss{{background:rgba(255,64,96,0.05)}}
.curve{{margin:16px 0}}
.legend{{display:flex;gap:18px;margin:16px 0 4px;font-size:12px;color:#8899bb;flex-wrap:wrap}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}
#chartwrap{{position:relative;margin:8px 0 16px;background:#1a1a2e;border-radius:8px;padding:8px 8px 2px}}
#tip{{position:absolute;display:none;pointer-events:none;background:#162135;border:1px solid #0f3460;
border-radius:6px;padding:7px 10px;font-size:12px;z-index:5;white-space:nowrap;line-height:1.5}}footer{{margin-top:24px;color:#555;font-size:12px}}</style>
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
<h2 style="color:#00d4ff;font-size:16px">Price action — {symbol} (algorithm overlay)</h2>
{chart_block}
<table><thead><tr><th>Entry</th><th>Exit</th><th>Side</th><th>Entry Px</th><th>Exit Px</th><th>Reason</th><th>Qty</th><th>Net (USD)</th><th>R-Mult</th><th>Equity After</th><th>Duration (s)</th></tr></thead><tbody>{tbody}</tbody></table>
<footer>Generated by live-trade execution loop.</footer></body></html>"##,
        symbol = symbol,
        refresh = PAGE_REFRESH_SECS,
        equity = final_equity,
        n = n_trades,
        w = wins,
        l = losses,
        net = total_net,
        r = total_r,
        pf = pf_str,
        curve = curve,
        tbody = tbody,
    );

    let _ = std::fs::create_dir_all("build");
    let out_path = format!("build/pnl_report_{}.html", symbol);
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

/// Build the equity-curve SVG polyline from persisted snapshots.
fn build_equity_curve(snaps: &[PnLSnapshot]) -> String {
    if snaps.is_empty() {
        return String::new();
    }
    let (min_eq, max_eq) = snaps
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s.equity), hi.max(s.equity))
        });
    let range = (max_eq - min_eq).max(1e-9);
    let w: f64 = 800.0;
    let h: f64 = 200.0;
    let pad: f64 = 30.0;
    let pts: Vec<String> = snaps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let span = snaps.len().saturating_sub(1).max(1) as f64;
            let x = pad + (i as f64 / span) * (w - pad * 2.0);
            let y = h - pad - ((s.equity - min_eq) / range) * (h - pad * 2.0);
            format!("{:.1},{:.1}", x, y)
        })
        .collect();
    format!(
        r##"<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#1a1a2e;border-radius:6px">
            <polyline points="{points}" fill="none" stroke="#00d4ff" stroke-width="2"/>
            <text x="{pad}" y="15" fill="#aaa" font-size="12">equity curve (USD)</text>
            <text x="{pad}" y="{h}" fill="#aaa" font-size="10">{min_eq:.2}</text>
            <text x="{pad}" y="{pad}" fill="#aaa" font-size="10">{max_eq:.2}</text>
        </svg>"##,
        w = w,
        h = h,
        pad = pad,
        points = pts.join(" "),
        min_eq = min_eq,
        max_eq = max_eq,
    )
}
/// Inline vanilla-JS renderer for the price-action chart. Kept as a plain
/// string constant (outside `format!`) so the JS needs no brace escaping.
///
/// Layout: two stacked <canvas> panels sharing one time axis —
///   * top: 1-second downsampled close tape + trade overlays (holding spans
///     shaded green/red by outcome, entry triangles, exit squares) and
///   * bottom: equity snapshots on the same clock.
/// Hovering shows a tooltip with time / price / equity and the active trade.
const CHART_JS: &str = r#"(function () {
  var D = REPORT_DATA || {};
  var P = D.prices || [], EQ = D.equity || [], TR = D.trades || [];
  var wrap = document.getElementById('chartwrap');
  var tip = document.getElementById('tip');
  var pc = document.getElementById('priceCanvas');
  var ec = document.getElementById('eqCanvas');
  if (!wrap || !pc || !ec) return;
  if (!P.length && !TR.length) {
    wrap.innerHTML = '<div style="color:#888;padding:24px;text-align:center">No market data recorded this session.</div>';
    return;
  }

  var PADL = 78, PADR = 104;          // left/right padding (y labels on both sides)
  var PH = 400, EH = 172;             // panel heights (equity includes time axis)
  var AXIS_H = 24;                    // time-label strip inside equity panel
  var view = null;

  function setup(cv, h) {
    var w = Math.max(wrap.clientWidth - 16, 320);
    var dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  function computeView() {
    var t0 = Infinity, t1 = -Infinity, plo = Infinity, phi = -Infinity;
    P.forEach(function (pt) {
      t0 = Math.min(t0, pt[0]); t1 = Math.max(t1, pt[0]);
      plo = Math.min(plo, pt[1]); phi = Math.max(phi, pt[1]);
    });
    TR.forEach(function (tr) {
      t0 = Math.min(t0, tr.e); t1 = Math.max(t1, tr.x);
      plo = Math.min(plo, tr.ep, tr.xp); phi = Math.max(phi, tr.ep, tr.xp);
    });
    if (!isFinite(t0)) return null;
    var pr = (phi - plo) || plo * 0.001 || 1;
    plo -= pr * 0.06; phi += pr * 0.06;
    var elo = Infinity, ehi = -Infinity;
    EQ.forEach(function (pt) { elo = Math.min(elo, pt[1]); ehi = Math.max(ehi, pt[1]); });
    if (!isFinite(elo)) { elo = 0; ehi = 1; }
    var er = (ehi - elo) || Math.abs(elo) * 0.001 || 1;
    elo -= er * 0.12; ehi += er * 0.12;
    return { t0: t0, t1: t1, plo: plo, phi: phi, elo: elo, ehi: ehi };
  }

  function xOf(t, w) {
    return PADL + (t - view.t0) / ((view.t1 - view.t0) || 1) * (w - PADL - PADR);
  }
  function yP(p, h) {
    return h - 14 - (p - view.plo) / ((view.phi - view.plo) || 1) * (h - 28);
  }
  function yE(e, h) {
    return (h - AXIS_H) - 8 - (e - view.elo) / ((view.ehi - view.elo) || 1) * (h - AXIS_H - 24);
  }
  function drawGrid(g, lo, hi, yFn, fmt) {
    var ctx = g.ctx;
    ctx.font = '11px Consolas, monospace'; ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
      var v = lo + (hi - lo) * i / 4;
      var y = yFn(v, g.h);
      ctx.strokeStyle = '#22304f';
      ctx.beginPath(); ctx.moveTo(PADL, y); ctx.lineTo(g.w - PADR + 66, y); ctx.stroke();
      ctx.fillStyle = '#8899bb'; ctx.textAlign = 'left';
      ctx.fillText(fmt(v), g.w - PADR + 10, y + 4);
    }
  }

  function pickStep(span) {
    var steps = [1000, 5000, 15000, 30000, 60000, 300000, 900000, 1800000,
                 3600000, 10800000, 21600000, 43200000, 86400000];
    for (var i = 0; i < steps.length; i++) if (steps[i] >= span / 7) return steps[i];
    return steps[steps.length - 1];
  }

  function hhmm(ms, withSecs) {
    var d = new Date(ms);
    function z(n) { return (n < 10 ? '0' : '') + n; }
    return z(d.getHours()) + ':' + z(d.getMinutes()) +
           (withSecs ? ':' + z(d.getSeconds()) : '');
  }

  function drawTimeAxis(g) {
    var ctx = g.ctx;
    var step = pickStep(view.t1 - view.t0);
    ctx.fillStyle = '#8899bb'; ctx.font = '11px Consolas, monospace';
    for (var t = Math.ceil(view.t0 / step) * step; t <= view.t1; t += step) {
      var x = xOf(t, g.w);
      ctx.strokeStyle = '#22304f';
      ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, g.h - AXIS_H); ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(hhmm(t, step < 60000), x, g.h - 7);
    }
    ctx.textAlign = 'left';
  }

  function line(g, pts, xf, yf, color, width) {
    if (pts.length < 2) return;
    var ctx = g.ctx;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (var i = 0; i < pts.length; i++) {
      var x = xf(pts[i]), y = yf(pts[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function tri(ctx, x, y, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, y - 8); ctx.lineTo(x - 6, y + 5); ctx.lineTo(x + 6, y + 5);
    ctx.closePath(); ctx.fill();
  }
  function sq(ctx, x, y, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x - 5, y - 5, 10, 10);
  }

  function drawTrades(g) {
    var ctx = g.ctx;
    TR.forEach(function (tr) {
      var x1 = xOf(tr.e, g.w), x2 = xOf(tr.x, g.w);
      // holding-span shading by outcome
      ctx.fillStyle = tr.net > 0 ? 'rgba(46,204,113,.09)'
                    : tr.net < 0 ? 'rgba(255,64,96,.11)' : 'rgba(160,160,160,.07)';
      ctx.fillRect(x1, 10, Math.max(x2 - x1, 2), g.h - 24);
      // dashed fill->exit connector
      ctx.strokeStyle = '#3a4a6a'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x1, yP(tr.ep, g.h)); ctx.lineTo(x2, yP(tr.xp, g.h));
      ctx.stroke(); ctx.setLineDash([]);
      // entry triangle (side-colored), exit square (outcome-colored)
      tri(ctx, x1, yP(tr.ep, g.h), tr.side === 'long' ? '#2ecc71' : '#e67e22');
      sq(ctx, x2, yP(tr.xp, g.h),
         tr.net > 0 ? '#2ecc71' : tr.net < 0 ? '#ff4060' : '#aaaaaa');
    });
  }

  function draw() {
    view = computeView(); if (!view) return;
    var g1 = setup(pc, PH), g2 = setup(ec, EH);
    g1.ctx.clearRect(0, 0, g1.w, g1.h);
    g2.ctx.clearRect(0, 0, g2.w, g2.h);
    drawGrid(g1, view.plo, view.phi, yP, function (v) { return v.toFixed(2); });
    drawGrid(g2, view.elo, view.ehi, yE,
             function (v) { return v.toLocaleString(undefined, { maximumFractionDigits: 2 }); });
    drawTrades(g1);
    line(g1, P, function (pt) { return xOf(pt[0], g1.w); },
              function (pt) { return yP(pt[1], PH); }, '#00d4ff', 1.5);
    drawTimeAxis(g1);
    line(g2, EQ, function (pt) { return xOf(pt[0], g2.w); },
              function (pt) { return yE(pt[1], EH); }, '#ffb347', 1.8);
    drawTimeAxis(g2);
  }

  function bisect(t) {           // index of the first price point at/after t
    var lo = 0, hi = P.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (P[mid][0] < t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  wrap.addEventListener('mousemove', function (ev) {
    if (!view || !P.length) return;
    var r = wrap.getBoundingClientRect();
    var px = ev.clientX - r.left - 8;          // minus wrap padding
    var w = pc.clientWidth;
    if (px < PADL || px > w - PADR) { tip.style.display = 'none'; return; }
    var frac = (px - PADL) / ((w - PADL - PADR) || 1);
    var t = view.t0 + frac * (view.t1 - view.t0);
    var pt = P[bisect(t)];
    if (!pt) { tip.style.display = 'none'; return; }
    var html = '<b>' + new Date(pt[0]).toLocaleTimeString() + '</b><br>' +
               'price: ' + pt[1].toFixed(2);
    var bestEq = null;
    EQ.forEach(function (e) {
      if (e[0] <= pt[0] && (!bestEq || e[0] > bestEq[0])) bestEq = e;
    });
    if (bestEq) html += '<br>equity: ' + bestEq[1].toFixed(2);
    for (var k = 0; k < TR.length; k++) {
      if (t >= TR[k].e && t <= TR[k].x) {
        html += '<br><span style="color:#ffb347">IN TRADE</span> ' + TR[k].side +
                ' @ ' + TR[k].ep.toFixed(2) + ' (' + TR[k].r + ')';
        break;
      }
    }
    tip.innerHTML = html;
    tip.style.display = 'block';
    var tx = ev.clientX - r.left + 14, ty = ev.clientY - r.top + 14;
    if (tx + tip.offsetWidth > r.width - 8) tx = ev.clientX - r.left - tip.offsetWidth - 14;
    tip.style.left = tx + 'px';
    tip.style.top = ty + 'px';
  });
  wrap.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
  window.addEventListener('resize', draw);
  draw();
  // The page meta-refreshes while a session runs; keep the reader's scroll
  // position across reloads so the trade table doesn't jump back to the top.
  var prevScroll = sessionStorage.getItem('pnl_report_scroll');
  if (prevScroll) {
    requestAnimationFrame(function () { window.scrollTo(0, +prevScroll); });
  }
  window.addEventListener('beforeunload', function () {
    sessionStorage.setItem('pnl_report_scroll', String(window.scrollY));
  });
})();
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ClosedTrade;
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
                equity_after: 10_050.0,
                duration_s: 50.0,
            },
        );

        generate_html_report(&db, "btcusdt").await;

        let html =
            std::fs::read_to_string("build/pnl_report_btcusdt.html").expect("report written");
        assert!(html.contains("chartwrap"), "chart container present");
        assert!(html.contains("REPORT_DATA"), "chart payload present");
        assert!(html.contains("\"ep\":101"), "trade entry overlay in payload");
        assert!(html.contains("take_profit"), "trade reason in payload");
        // ~300 downsampled price points must be embedded (13-digit ms tuples).
        let pts = html.matches(",1700000").count() + html.matches("[1700000").count();
        assert!(pts >= 100, "expected >=100 price points, found {pts}");

        drop(db);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
