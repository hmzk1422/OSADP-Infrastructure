"""
Live Trade-Count Dashboard for the Rust live-trade path
=======================================================

Serves a no-dependency HTML page that shows, in real time, how many trades the
executor has executed (and equity / wins / losses / total R / feed freshness)
per ticker.

The Rust execution loop already exposes a JSON ``status()`` per strategy (via
the ``engine.Strategy`` PyO3 binding) carrying ``n_trades``, ``n_wins``,
``n_losses``, ``equity``, ``total_r``, ``mid``, ``feed_staleness_ms``, etc. The
runner's control loop pushes the latest snapshot into this object
(``update(ticker, status)``) and a background HTTP thread serves a page whose
JavaScript polls ``/api`` once a second.

It is purely observational — it never touches the exchange or any strategy.

Standalone preview (no bot, synthetic data)::

    python -m main.live_dashboard
    → open http://localhost:8001
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Live Trade Dashboard</title>
<style>
  :root{--bg:#0A0C10;--panel:#12151C;--border:#202632;--text:#E7E9ED;--muted:#7A8493;
        --dim:#4B5568;--long:#34C77B;--short:#F0495B;--accent:#FFB454;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font-family:'Segoe UI',system-ui,sans-serif;padding:24px}
  h1{font-size:15px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-top:12px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
  .card h2{font-size:14px;margin:0 0 10px;font-family:ui-monospace,monospace}
  .stat{display:flex;justify-content:space-between;font-size:13px;padding:4px 0;border-bottom:1px solid var(--border)}
  .stat:last-child{border-bottom:none}
  .stat .k{color:var(--muted)} .stat .v{font-family:ui-monospace,monospace;font-weight:600}
  .v.pos{color:var(--long)} .v.neg{color:var(--short)} .v.big{font-size:22px}
  .muted{color:var(--dim);font-size:12px} #hint{color:var(--dim);font-size:12px;margin-top:8px}
</style></head><body>
<h1>Live trade executor — per-ticker</h1>
<div id="cards"><div class="muted">Waiting for data…</div></div>
<div id="hint">Refresh is automatic (1 s). Ctrl+C in the runner to stop.</div>
<script>
async function refresh(){
  let list;
  try{ list = await (await fetch('/api')).json(); }catch(e){ return; }
  const host = document.getElementById('cards');
  if(!list.length){ host.innerHTML='<div class="muted">No tickers yet — start the runner with --dashboard.</div>'; return; }
  host.innerHTML = list.map(s=>{
    const trades = s.n_trades ?? 0, wins = s.n_wins ?? 0, losses = s.n_losses ?? 0;
    const r = s.total_r ?? 0, eq = s.equity ?? 0, mid = s.mid;
    const stal = (s.feed_staleness_ms != null) ? Number(s.feed_staleness_ms).toFixed(0)+'ms' : '—';
    const eqCls = eq >= 0 ? 'pos' : 'neg', rCls = r >= 0 ? 'pos' : 'neg';
    const midTxt = mid ? Number(mid).toFixed(2) : '—';
    return `<div class="card">
      <h2>${s.symbol || s.ticker || '?'} <span class="muted">${s.exchange || ''}/${s.mode || ''}</span></h2>
      <div class="stat"><span class="k">Equity (USD)</span><span class="v big ${eqCls}">${Number(eq).toFixed(2)}</span></div>
      <div class="stat"><span class="k">Trades executed</span><span class="v" style="color:var(--accent)">${trades}</span></div>
      <div class="stat"><span class="k">Wins / Losses</span><span class="v">${wins} / ${losses}</span></div>
      <div class="stat"><span class="k">Cumulative R</span><span class="v ${rCls}">${Number(r).toFixed(2)}</span></div>
      <div class="stat"><span class="k">Mid</span><span class="v">${midTxt}</span></div>
      <div class="stat"><span class="k">Feed staleness</span><span class="v">${stal}</span></div>
      <div class="stat"><span class="k">Running</span><span class="v">${s.running ? 'yes' : 'no'}</span></div>
    </div>`;
  }).join('');
}
refresh(); setInterval(refresh, 1000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    dashboard: "LiveDashboard"  # set by the server

    def log_message(self, *args):  # silence per-request logging
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/api":
            payload = json.dumps(self.dashboard.snapshots()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()


class LiveDashboard:
    """Thread-safe holder of the latest per-ticker status snapshots plus a
    background HTTP server that exposes them to a browser."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._snapshots: Dict[str, dict] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def update(self, ticker: str, status: dict) -> None:
        """Record the latest status dict for `ticker` (called from the runner
        control loop)."""
        with self._lock:
            self._snapshots[ticker] = dict(status)

    def snapshots(self) -> list:
        """Return the current snapshots as a list (insertion order)."""
        with self._lock:
            return [_with_symbol(v, k) for k, v in self._snapshots.items()]

    def start(self) -> None:
        """Launch the HTTP server on a daemon thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        server = ThreadingHTTPServer((self.host, self.port), _Handler)
        server.daemon_threads = True
        _Handler.dashboard = self
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        import logging

        logging.getLogger("live_dashboard").info(
            "live dashboard → http://localhost:%d", self.port
        )

    def stop(self) -> None:
        """Shut the HTTP server down (best-effort, safe to call multiple times)."""
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._server.server_close()
            self._server = None


def _with_symbol(snapshot: dict, key: str) -> dict:
    """Ensure the API payload always exposes a stable `symbol` name."""
    out = dict(snapshot)
    out.setdefault("symbol", out.get("symbol") or key)
    out.setdefault("ticker", key)
    return out


if __name__ == "__main__":
    # Standalone preview with synthetic data.
    import random
    import time

    dash = LiveDashboard(port=8001)
    dash.start()
    n = 0
    try:
        while True:
            n += 1
            dash.update(
                "btcusdt",
                {
                    "symbol": "btcusdt",
                    "exchange": "bitfinex",
                    "mode": "paper",
                    "running": True,
                    "equity": 10000 + n * 1.5,
                    "n_trades": n,
                    "n_wins": int(n * 0.55),
                    "n_losses": n - int(n * 0.55),
                    "total_r": n * 0.13,
                    "mid": 97000.0 + random.uniform(-50, 50),
                    "feed_staleness_ms": random.uniform(0, 900),
                },
            )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        dash.stop()

