// TradingView-style renderer for the PnL report.
// Three stacked panes on one canvas sharing a crosshair:
//   * price    — candles aggregated client-side from the 1 s tape + trade
//                overlays (holding spans, entry/exit markers)
//   * RSI(2)   — the FULL per-second indicator series with 30/50/70 guides
//                (not just hunting-zone values)
//   * equity   — account curve with start-equity baseline
// Interactions: scroll = zoom (anchored at cursor), drag = pan,
// double-click = reset. The trade table calls window.locateTrade().
(function () {
  var D = window.REPORT_DATA || {};
  var P = D.prices || [], RS = D.rsi || [], WP = D.wp || [];
  var EQ = D.equity || [], TR = D.trades || [];
  var START_EQ = D.startEq || 0;
  var SYM = (D.symbol || '').toUpperCase();

  var wrap = document.getElementById('chartwrap');
  var cv = document.getElementById('tvchart');
  var legendEl = document.getElementById('tvlegend');
  var tip = document.getElementById('tip');
  if (!wrap || !cv) return;
  if (!P.length && !TR.length) {
    wrap.innerHTML = '<div style="color:#787b86;padding:24px;text-align:center">No market data recorded this session.</div>';
    return;
  }

  // ---- session chip --------------------------------------------------------
  function fmtDT(t, withSec) {
    var d = new Date(t);
    function p(n) { return (n < 10 ? '0' : '') + n; }
    var s = (d.getMonth() + 1) + '/' + d.getDate() + ' ' + p(d.getHours()) + ':' + p(d.getMinutes());
    if (withSec) s += ':' + p(d.getSeconds());
    return s;
  }
  var chip = document.getElementById('chipRange');
  if (chip && P.length) {
    chip.textContent = 'session ' + fmtDT(P[0][0], false) + ' → ' + fmtDT(P[P.length - 1][0], true);
  }

  // ---- base 1-second candles ----------------------------------------------
  // The tape stores one close per second; open = previous close so candles
  // chain continuously, high/low span the pair.
  var BASE = [];
  (function buildBase() {
    for (var i = 0; i < P.length; i++) {
      var t = P[i][0], c = P[i][1];
      var o = i > 0 ? P[i - 1][1] : c;
      BASE.push({ t: t, o: o, h: Math.max(o, c), l: Math.min(o, c), c: c });
    }
  })();

  // ---- view state -----------------------------------------------------------
  // Default to the most recent slice of tape (TradingView-style), not the full
  // history — keeps candles readable and the live RSI tail front-and-center.
  var DEF_WINDOW = 1200;                          // seconds shown by default
  var defA = Math.max(0, BASE.length - DEF_WINDOW);
  var view = { a: defA, b: BASE.length };          // half-open candle index range
  var drag = null;                               // {x0, a0, b0}
  var hover = null;                              // {x, y} in css px
  var flashTrade = null;                         // trade being located

  function clampView() {
    var n = view.b - view.a;
    if (n < 5) { var c = (view.a + view.b) / 2; view.a = Math.max(0, c - 2.5); view.b = view.a + 5; }
    if (view.a < 0) { view.b -= view.a; view.a = 0; }
    if (view.b > BASE.length) { view.a -= view.b - BASE.length; view.b = BASE.length; }
    if (view.a < 0) view.a = 0;
    view.a = Math.floor(view.a); view.b = Math.ceil(view.b);
  }

  // ---- visible-candle aggregation (TradingView-style density control) ------
  // Each group records `bi`, its center in ABSOLUTE base-candle index space.
  // Every overlay (markers, ticks, crosshair) uses the same base-index → px
  // mapping, so trade icons stay pinned to their bars at any zoom/pan.
  var MAX_VIS = 480;
  function visibleCandles() {
    clampView();
    var n = view.b - view.a;
    var k = Math.max(1, Math.ceil(n / MAX_VIS));
    var out = [];
    for (var i = view.a; i < view.b; i += k) {
      var j = Math.min(i + k, view.b);
      var o = BASE[i].o, c = BASE[j - 1].c, h = -Infinity, l = Infinity, t = BASE[i].t;
      for (var q = i; q < j; q++) {
        h = Math.max(h, BASE[q].h); l = Math.min(l, BASE[q].l);
      }
      out.push({ t: t, o: o, h: h, l: l, c: c, bi: (i + j) / 2 });
    }
    out.k = k;
    return out;
  }

  // ---- layout ----------------------------------------------------------------
  var AXW = 74;                 // right price-axis width
  var PADL = 2, PADR = AXW;
  var H_PRICE = 430, H_RSI = 120, H_EQ = 140, H_TIME = 26, GAP = 8;

  function paneRects(w) {
    var plotW = w - PADR - PADL;
    var y = GAP;
    var price = { x: PADL, y: y, w: plotW, h: H_PRICE }; y += H_PRICE + GAP;
    var rsi = { x: PADL, y: y, w: plotW, h: H_RSI }; y += H_RSI + GAP;
    var eq = { x: PADL, y: y, w: plotW, h: H_EQ };
    return { price: price, rsi: rsi, eq: eq, timeH: H_TIME };
  }

  var canvasW = 0, canvasH = 0;      // cached — avoid reallocating per frame
  function setupCanvas() {
    var w = Math.max(wrap.clientWidth - 12, 360);
    var rects = paneRects(w);
    var h = GAP + rects.price.h + GAP + rects.rsi.h + GAP + rects.eq.h + rects.timeH + 4;
    var dpr = window.devicePixelRatio || 1;
    if (w !== canvasW || h !== canvasH || cv.width !== Math.round(w * dpr)) {
      canvasW = w; canvasH = h;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      cv.style.width = w + 'px'; cv.style.height = h + 'px';
    }
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h, rects: rects };
  }

  function niceStep(range, target) {
    var raw = range / Math.max(1, target);
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1;
    return step * mag;
  }

  // ---- drawing ----------------------------------------------------------------
  function xForIdx(rects, i, nVis) {
    return rects.price.x + ((i - view.a) + 0.5) * (rects.price.w / nVis);
  }
  // Map an arbitrary timestamp to fractional candle index inside the view.
  function tsToIdx(candles, ts) {
    var t0 = candles[0].t;
    var step = candles.length > 1 ? candles[1].t - candles[0].t : 1000;
    var t1 = candles[candles.length - 1].t + step;
    var f = (ts - t0) / Math.max(1, t1 - t0);
    return view.a + f * (view.b - view.a);
  }

  // rAF-throttled draw: coalesces bursts of mousemove/wheel events into one
  // redraw per frame — the main fix for the glitchy navigation feel.
  var rafPending = false;
  function draw() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(function () { rafPending = false; drawNow(); });
  }

  function drawNow() {
    var S = setupCanvas(), ctx = S.ctx, w = S.w, rects = S.rects;
    var candles = visibleCandles();
    var nVis = candles.length;
    ctx.clearRect(0, 0, w, S.h);

    var plo = Infinity, phi = -Infinity;
    candles.forEach(function (c) { plo = Math.min(plo, c.l); phi = Math.max(phi, c.h); });
    if (!isFinite(plo)) { plo = 0; phi = 1; }
    var padP = (phi - plo) * 0.07 || 1;
    plo -= padP; phi += padP;
    TR.forEach(function (t) {
      if (t.x >= candles[0].t && t.e <= candles[candles.length - 1].t) {
        plo = Math.min(plo, t.ep, t.xp); phi = Math.max(phi, t.ep, t.xp);
      }
    });
    plo -= padP; phi += padP;

    var slot = rects.price.w / nVis;
    function yP(v) { return rects.price.y + (1 - (v - plo) / (phi - plo)) * rects.price.h; }
    function yR(v) { return rects.rsi.y + (1 - v / 100) * rects.rsi.h; }
    var eqLo = Infinity, eqHi = -Infinity;
    EQ.forEach(function (e) {
      if (e[0] < candles[0].t || e[0] > candles[candles.length - 1].t + 36e5) return;
      eqLo = Math.min(eqLo, e[1]); eqHi = Math.max(eqHi, e[1]);
    });
    if (!isFinite(eqLo)) { eqLo = START_EQ - 1; eqHi = START_EQ + 1; }
    var eqPad = (eqHi - eqLo) * 0.12 || 1; eqLo -= eqPad; eqHi += eqPad;
    function yE(v) { return rects.eq.y + (1 - (v - eqLo) / (eqHi - eqLo)) * rects.eq.h; }

    ctx.fillStyle = '#131722';
    [rects.price, rects.rsi, rects.eq].forEach(function (r) { ctx.fillRect(r.x, r.y, r.w, r.h); });
    // Clean right-axis strip + hairline pane separators (TV chrome).
    ctx.fillStyle = '#171b26';
    ctx.fillRect(rects.price.x + rects.price.w, 0, PADR, S.h);
    ctx.strokeStyle = '#2a2e39';
    [rects.rsi.y - GAP / 2, rects.eq.y - GAP / 2,
     rects.eq.y + rects.eq.h + GAP / 2].forEach(function (yy) {
      ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(w, yy); ctx.stroke();
    });

    function hGrid(r, lo, hi, fmt) {
      var step = niceStep(hi - lo, Math.max(2, Math.floor(r.h / 42)));
      ctx.strokeStyle = '#1c2030'; ctx.fillStyle = '#787b86';
      ctx.font = '10px Trebuchet MS'; ctx.textAlign = 'left'; ctx.lineWidth = 1;
      for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
        var y = r.y + (1 - (v - lo) / (hi - lo)) * r.h;
        ctx.beginPath(); ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y); ctx.stroke();
        ctx.fillText(fmt(v), r.x + r.w + 6, y + 3);
      }
    }
    var pxFmt = function (v) { return v >= 1000 ? v.toFixed(0) : v.toFixed(2); };
    hGrid(rects.price, plo, phi, pxFmt);
    hGrid(rects.rsi, 0, 100, String);
    hGrid(rects.eq, eqLo, eqHi, function (v) { return v.toFixed(1); });

    var t0ms = candles[0].t, t1ms = candles[candles.length - 1].t;
    var spanMin = (t1ms - t0ms) / 60000;
    var tickStepMs = niceStep(spanMin, Math.max(2, Math.floor(rects.price.w / 110))) * 60000;
    ctx.strokeStyle = '#1c2030'; ctx.fillStyle = '#787b86';
    ctx.font = '10px Trebuchet MS';
    for (var tt = Math.ceil(t0ms / tickStepMs) * tickStepMs; tt <= t1ms; tt += tickStepMs) {
      var xx = xForIdx(rects, tsToIdx(candles, tt), nVis);
      ctx.beginPath();
      ctx.moveTo(xx, rects.price.y); ctx.lineTo(xx, rects.eq.y + rects.eq.h);
      ctx.stroke();
      ctx.fillText(fmtDT(tt, spanMin < 90), xx - 24, rects.eq.y + rects.eq.h + 16);
    }

    // holding spans + entry/exit markers
    TR.forEach(function (t) {
      if (t.x < t0ms || t.e > t1ms) return;
      var x1 = xForIdx(rects, tsToIdx(candles, t.e), nVis);
      var x2 = xForIdx(rects, tsToIdx(candles, t.x), nVis);
      var winC = t.net >= 0;
      ctx.fillStyle = winC ? 'rgba(38,166,154,.09)' : 'rgba(239,83,80,.09)';
      ctx.fillRect(x1, rects.price.y, Math.max(2, x2 - x1), rects.price.h);
      var long = t.side === 'long';
      var xe = x1, ye = yP(t.ep);
      ctx.fillStyle = long ? '#26a69a' : '#ef5350';
      ctx.beginPath();
      if (long) { ctx.moveTo(xe, ye + 4); ctx.lineTo(xe - 5, ye + 13); ctx.lineTo(xe + 5, ye + 13); }
      else { ctx.moveTo(xe, ye - 4); ctx.lineTo(xe - 5, ye - 13); ctx.lineTo(xe + 5, ye - 13); }
      ctx.closePath(); ctx.fill();
      var xo = x2, yo = yP(t.xp);
      ctx.fillRect(xo - 4, yo - 4, 8, 8);
    });
    if (flashTrade !== null) {
      var fx = xForIdx(rects, tsToIdx(candles, flashTrade), nVis);
      ctx.strokeStyle = 'rgba(240,185,11,.85)'; ctx.lineWidth = 1.5;
      ctx.strokeRect(fx - 16, rects.price.y + 2, 32, rects.price.h - 4);
      ctx.lineWidth = 1;
    }

    // candles — drawn at their base-index centers so markers/ticks align
    var kAgg = candles.k || 1;
    var bw = Math.max(1, Math.min(13, slot * kAgg * 0.72));
    for (var ci = 0; ci < candles.length; ci++) {
      var c = candles[ci];
      var x = xForIdx(rects, c.bi, nVis);
      var up = c.c >= c.o;
      ctx.strokeStyle = up ? '#26a69a' : '#ef5350';
      ctx.fillStyle = up ? '#26a69a' : '#ef5350';
      ctx.beginPath();
      ctx.moveTo(x + 0.5, yP(c.h)); ctx.lineTo(x + 0.5, yP(c.l)); ctx.stroke();
      var yo2 = yP(Math.max(c.o, c.c));
      var hh = Math.max(1, yP(Math.min(c.o, c.c)) - yo2);
      ctx.fillRect(x - bw / 2, yo2, bw, hh);
    }

    // last price line + tag
    var lastC = candles[candles.length - 1];
    var lastY = yP(lastC.c);
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = 'rgba(120,123,134,.55)';
    ctx.beginPath(); ctx.moveTo(rects.price.x, lastY); ctx.lineTo(rects.price.x + rects.price.w, lastY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = lastC.c >= lastC.o ? '#26a69a' : '#ef5350';
    ctx.fillRect(rects.price.x + rects.price.w + 1, lastY - 8, AXW - 3, 16);
    ctx.fillStyle = '#fff'; ctx.font = 'bold 10px Trebuchet MS'; ctx.textAlign = 'left';
    ctx.fillText(pxFmt(lastC.c), rects.price.x + rects.price.w + 6, lastY + 4);

    // timeframe / aggregation chip (top-right of price pane)
    var tfTxt = kAgg > 1 ? kAgg + ' × 1s' : '1s';
    ctx.font = 'bold 10px Trebuchet MS';
    var tw = ctx.measureText(tfTxt).width + 14;
    ctx.fillStyle = '#2a2e39';
    ctx.fillRect(rects.price.x + rects.price.w - tw - 8, rects.price.y + 6, tw, 17);
    ctx.fillStyle = '#d1d4dc'; ctx.textAlign = 'center';
    ctx.fillText(tfTxt, rects.price.x + rects.price.w - tw / 2 - 8, rects.price.y + 18);
    ctx.textAlign = 'left';

    // ---- RSI pane: FULL series, every value, no zone filtering --------------
    ctx.fillStyle = '#787b86'; ctx.font = '10px Trebuchet MS';
    ctx.fillText('RSI(2) · 300 s composite · sampled 1/s', rects.rsi.x + 22, rects.rsi.y + 12);
    // how many samples fall inside the visible window?
    var rsiVis = 0;
    for (var ri = 0; ri < RS.length; ri++) {
      if (RS[ri][0] >= t0ms && RS[ri][0] <= t1ms) rsiVis++;
    }
    if (!rsiVis) {
      ctx.fillStyle = 'rgba(120,123,134,.8)';
      ctx.font = 'italic 11px Trebuchet MS';
      ctx.fillText('no RSI recorded in this window — feature history starts at the first session on this build',
        rects.rsi.x + rects.rsi.w / 2 - 210, rects.rsi.y + rects.rsi.h / 2 + 4);
      ctx.fillStyle = '#787b86'; ctx.font = '10px Trebuchet MS';
    }
    // oversold / overbought zone tints
    ctx.fillStyle = 'rgba(38,166,154,.06)';
    ctx.fillRect(rects.rsi.x, yR(30), rects.rsi.w, rects.rsi.y + rects.rsi.h - yR(30));
    ctx.fillStyle = 'rgba(239,83,80,.06)';
    ctx.fillRect(rects.rsi.x, rects.rsi.y, rects.rsi.w, yR(70) - rects.rsi.y);
    [30, 50, 70].forEach(function (lv) {
      ctx.setLineDash(lv === 50 ? [2, 3] : [5, 4]);
      ctx.strokeStyle = lv === 50 ? 'rgba(120,123,134,.25)' : 'rgba(120,123,134,.45)';
      ctx.beginPath(); ctx.moveTo(rects.rsi.x, yR(lv));
      ctx.lineTo(rects.rsi.x + rects.rsi.w, yR(lv)); ctx.stroke();
    });
    ctx.setLineDash([]);
    function strokeSeries(series, yFn, color, width, dash) {
      if (!series.length) return null;
      ctx.strokeStyle = color; ctx.lineWidth = width || 1.5;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      var started = false, out = null;
      for (var i = 0; i < series.length; i++) {
        var pt = series[i];
        if (pt[0] < t0ms - 60000 || pt[0] > t1ms + 60000) continue;
        var sx = xForIdx(rects, tsToIdx(candles, pt[0]), nVis);
        var sy = yFn(pt[1]);
        if (!started) { ctx.moveTo(sx, sy); started = true; }
        else ctx.lineTo(sx, sy);
        out = { x: sx, y: sy };
      }
      if (started) ctx.stroke();
      ctx.setLineDash([]); ctx.lineWidth = 1;
      return out;
    }
    strokeSeries(RS, yR, '#7e57c2', 1.6);
    // P(WIN) as a thin dotted overlay scaled to 0..100 on the same pane
    strokeSeries(WP.map(function (p2) { return [p2[0], p2[1] * 100]; }),
      yR, 'rgba(41,98,255,.7)', 1, [1, 3]);

    // ---- equity pane ----------------------------------------------------------
    ctx.fillStyle = '#787b86';
    ctx.fillText('EQUITY $', rects.eq.x + 6, rects.eq.y + 12);
    var baseY = yE(START_EQ);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(120,123,134,.4)';
    ctx.beginPath(); ctx.moveTo(rects.eq.x, baseY);
    ctx.lineTo(rects.eq.x + rects.eq.w, baseY); ctx.stroke();
    ctx.setLineDash([]);
    var pts = [];
    for (var ei = 0; ei < EQ.length; ei++) {
      var et = EQ[ei][0];
      if (et < t0ms - 60000 || et > t1ms + 3600000) continue;
      pts.push([xForIdx(rects, tsToIdx(candles, et), nVis), yE(EQ[ei][1])]);
    }
    var tail = null;
    if (pts.length > 1) {
      var grad = ctx.createLinearGradient(0, rects.eq.y, 0, rects.eq.y + rects.eq.h);
      grad.addColorStop(0, 'rgba(41,98,255,.28)');
      grad.addColorStop(1, 'rgba(41,98,255,.02)');
      ctx.beginPath(); ctx.moveTo(pts[0][0], rects.eq.y + rects.eq.h);
      pts.forEach(function (p3) { ctx.lineTo(p3[0], p3[1]); });
      ctx.lineTo(pts[pts.length - 1][0], rects.eq.y + rects.eq.h);
      ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
      ctx.strokeStyle = '#2962ff'; ctx.lineWidth = 1.6;
      ctx.beginPath();
      pts.forEach(function (p3, pi) { pi ? ctx.lineTo(p3[0], p3[1]) : ctx.moveTo(p3[0], p3[1]); });
      ctx.stroke(); ctx.lineWidth = 1;
      tail = pts[pts.length - 1];
    }
    if (tail && EQ.length) {
      ctx.fillStyle = '#2962ff';
      ctx.fillRect(rects.eq.x + rects.eq.w + 1, tail[1] - 8, AXW - 3, 16);
      ctx.fillStyle = '#fff'; ctx.font = 'bold 10px Trebuchet MS';
      ctx.fillText(EQ[EQ.length - 1][1].toFixed(1), rects.eq.x + rects.eq.w + 6, tail[1] + 4);
    }

    // ---- crosshair ------------------------------------------------------------
    if (hover) {
      var hx = Math.max(rects.price.x, Math.min(hover.x, rects.price.x + rects.price.w));
      ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(150,158,175,.55)';
      ctx.beginPath();
      ctx.moveTo(hx, rects.price.y); ctx.lineTo(hx, rects.eq.y + rects.eq.h);
      ctx.moveTo(rects.price.x, hover.y); ctx.lineTo(rects.price.x + rects.price.w, hover.y);
      ctx.stroke(); ctx.setLineDash([]);
      updateLegend(candles, hx);
    } else {
      legendDefault(candles);
    }
    window.__tvState = { candles: candles, rects: rects };
  } // draw()

  // ---- data-window legend ---------------------------------------------------
  function fmtN(v) { return v >= 1000 ? v.toFixed(1) : v.toFixed(2); }
  function rsiAt(ts) {
    var v = null;
    for (var i = 0; i < RS.length; i++) {
      if (RS[i][0] <= ts) v = RS[i][1]; else break;
    }
    return v;
  }
  function wpAt(ts) {
    var v = null;
    for (var i = 0; i < WP.length; i++) {
      if (WP[i][0] <= ts) v = WP[i][1]; else break;
    }
    return v;
  }
  function tradeAt(ts) {
    for (var i = TR.length - 1; i >= 0; i--) {
      if (TR[i].e <= ts && ts <= TR[i].x) return TR[i];
    }
    return null;
  }
  function candleAt(candles, x) {
    var st = window.__tvState;
    if (!st || !candles.length) return null;
    // Invert the same base-index → px mapping used by draw(): xForIdx.
    var k = candles.k || 1;
    var slot = st.rects.price.w / (view.b - view.a);
    var fi = view.a + (x - PADL) / slot - 0.5;          // fractional base idx
    var gi = Math.floor((fi - view.a) / k);             // group index
    if (gi < 0 || gi >= candles.length) return null;
    return candles[gi];
  }
  function legendHTML(c, r, w, eqv, tr, hovered) {
    var chg = ((c.c - c.o) / c.o * 100);
    var cls = chg >= 0 ? 'up' : 'down';
    var h = '<b>' + SYM + '</b> · ' + fmtDT(c.t, true) +
      '<br>O <span class="' + cls + '">' + fmtN(c.o) + '</span>' +
      ' H <span class="' + cls + '">' + fmtN(c.h) + '</span>' +
      ' L <span class="' + cls + '">' + fmtN(c.l) + '</span>' +
      ' C <span class="' + cls + '">' + fmtN(c.c) + '</span>' +
      ' <span class="' + cls + '">(' + (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%)</span>';
    h += '<br><span style="color:#7e57c2">●</span> RSI(2): <b>' +
      (r === null ? '—' : r.toFixed(1)) + '</b>' +
      (r !== null && r < 30 ? ' <span class="up">[in-zone]</span>' : '') +
      ' · <span style="color:#5c7cff">●</span> P(WIN): <b>' +
      (w === null ? '—' : (w * 100).toFixed(0) + '%') + '</b>';
    if (eqv !== null) {
      h += '<br><span style="color:#2962ff">●</span> Equity: <b>$' + eqv.toFixed(2) + '</b>';
    }
    if (tr) {
      h += '<br><span style="color:' + (tr.net >= 0 ? '#26a69a' : '#ef5350') + '">▶ ' +
        tr.side.toUpperCase() + ' open @' + fmtN(tr.ep) + ' → ' + tr.r +
        ' · $' + tr.net.toFixed(2) + ' (' + (tr.rm >= 0 ? '+' : '') + tr.rm.toFixed(2) + 'R)</span>';
    }
    if (!hovered) h += '<br><span style="color:#787b86">hover the chart for values</span>';
    legendEl.innerHTML = h;
  }
  function legendDefault(candles) {
    var last = candles[candles.length - 1];
    if (!last) return;
    var r = rsiAt(last.t), w = wpAt(last.t);
    var eqv = EQ.length ? EQ[EQ.length - 1][1] : null;
    legendHTML(last, r, w, eqv, null, false);
  }
  function updateLegend(candles, x) {
    var c = candleAt(candles, x);
    if (!c) { legendDefault(candles); return; }
    var eqv = null;
    for (var i = EQ.length - 1; i >= 0; i--) {
      if (EQ[i][0] <= c.t) { eqv = EQ[i][1]; break; }
    }
    legendHTML(c, rsiAt(c.t), wpAt(c.t), eqv, tradeAt(c.t), true);
  }

  //__PART4__
  cv.addEventListener('mousemove', function (ev) {
    var rc = cv.getBoundingClientRect();
    hover = { x: ev.clientX - rc.left, y: ev.clientY - rc.top };
    if (drag) {
      // Incremental pan: shift by whole candles since the last event, then
      // re-anchor — no cumulative rounding jitter.
      var S0 = setupCanvas();
      var nVis = view.b - view.a;
      var slotPx = S0.rects.price.w / nVis;
      var shift = Math.round((hover.x - drag.lx) / slotPx);
      if (shift) {
        drag.lx += shift * slotPx;
        view.a -= shift; view.b -= shift;
        clampView();
      }
    }
    draw();
    var st = window.__tvState || {};
    if (st.candles && tip) {
      var c = candleAt(st.candles, hover.x);
      if (c) {
        var tr = tradeAt(c.t);
        if (tr) {
          tip.style.display = 'block';
          tip.style.left = Math.min(hover.x + 14, cv.clientWidth - 200) + 'px';
          tip.style.top = (hover.y + 14) + 'px';
          tip.innerHTML = '<b>' + tr.side.toUpperCase() + '</b> ' + tr.qty.toFixed(4) +
            '<br>@ ' + fmtN(tr.ep) + ' → ' + fmtN(tr.xp) +
            '<br><span style="color:' + (tr.net >= 0 ? '#26a69a' : '#ef5350') + '">$' +
            tr.net.toFixed(2) + ' (' + (tr.rm >= 0 ? '+' : '') + tr.rm.toFixed(2) +
            'R) · ' + tr.r + '</span>';
        } else tip.style.display = 'none';
      }
    }
  });
  cv.addEventListener('mouseleave', function () {
    hover = null; drag = null;
    if (tip) tip.style.display = 'none';
    draw();
  });
  cv.addEventListener('mousedown', function (ev) {
    var rc = cv.getBoundingClientRect();
    drag = { x0: ev.clientX - rc.left, lx: ev.clientX - rc.left, a0: view.a, b0: view.b };
    cv.style.cursor = 'grabbing';
  });
  window.addEventListener('mouseup', function () {
    drag = null; cv.style.cursor = 'grab';
  });
  cv.addEventListener('wheel', function (ev) {
    ev.preventDefault();
    var rc = cv.getBoundingClientRect();
    var mx = ev.clientX - rc.left;
    var S = setupCanvas();
    var frac = view.a + (mx - PADL) / S.rects.price.w * (view.b - view.a);
    var f = ev.deltaY > 0 ? 1.12 : 1 / 1.12;   // gentle steps — less overshoot
    var n = Math.max(8, Math.min(BASE.length, (view.b - view.a) * f));
    view.a = frac - (frac - view.a) * n / (view.b - view.a);
    view.b = view.a + n;
    clampView(); draw();
  }, { passive: false });
  cv.addEventListener('dblclick', function () {
    view.a = Math.max(0, BASE.length - DEF_WINDOW);
    view.b = BASE.length; flashTrade = null; draw();
  });
  // Trade-table rows call this to center the chart on a trade.
  window.locateTrade = function (entryMs) {
    var idx = BASE.length - 1;
    for (var i = 0; i < BASE.length; i++) {
      if (BASE[i].t >= entryMs) { idx = i; break; }
    }
    var half = 45;
    view.a = Math.max(0, idx - half);
    view.b = Math.min(BASE.length, idx + half);
    flashTrade = entryMs;
    clampView(); draw();
  };
  var rsz = null;
  window.addEventListener('resize', function () {
    clearTimeout(rsz); rsz = setTimeout(draw, 120);
  });

  cv.style.cursor = 'grab';
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
