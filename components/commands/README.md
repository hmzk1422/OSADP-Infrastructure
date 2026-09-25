# components/commands

Chain-of-commands layer that wires the Rust binaries to the market data.

Each numbered script is a **command**: it sequences the orchestrators under
`components/rust/<library>/orchestrator/` into one pipeline stage. The
dependency direction is strictly one way — commands import orchestrators, never
the other way round — so every orchestrator stays usable standalone.

```
components/commands/
  README.md          <- you are here
  __init__.py
  _common.py         <- paths, dates, clone, stage logging, parallel runner
  01_ingest.py       <- the L2 ingestion chain
```

---

## Quick start

Run from the **repo root** (so `components.*` imports resolve):

```bash
# one day
python components/commands/01_ingest.py --start-date 2025-06-30 --end-date 2025-06-30

# a week, skipping the optional day-start seed
python components/commands/01_ingest.py --start-date 2025-06-30 --end-date 2025-07-06 --no-seed

# first time only: build the Rust binaries
python components/commands/01_ingest.py --start-date 2025-06-30 --end-date 2025-06-30 --build
```

---

## Where is the output?

**`market_data/matched/<TICKER>/merged_<DATE>_<HH>.parquet`** — 24 files per day,
one per UTC hour, ~8-10 MB each.

```
market_data/matched/BTCUSDT/
  merged_2025-06-30_00.parquet     <- 00:00-00:59 UTC
  merged_2025-06-30_01.parquet     <- 01:00-01:59 UTC
  ...
  merged_2025-06-30_23.parquet     <- 23:00-23:59 UTC
```

Everything else under `matched/<TICKER>/_pipeline/` is **disposable work state**
and is deleted after a successful merge. Canonical `ingest_parquet/` is only
ever read, never modified.

---

## The chain (`01_ingest.py`)

| # | Stage | What runs | Produces |
|---|-------|-----------|----------|
| 1 | Day-start seed *(optional, `--no-seed` to skip)* | `lob_replay ingest-parquet --snapshot-policy day_start` | `matched/<SYM>/snapshots/<DATE>/` |
| 2 | Clone | `matched/parquet.fetch_source_data` | `_pipeline/source/` — 26 files/day |
| 3 | LOB reconstruction | `lob_replay ingest-parquet --snapshot-policy every_event --warm-start` | `_pipeline/orderflow/<DATE>/<HH>_every_event.parquet` |
| 4 | Merge + finalize + cleanup | `event_merger`, then ffill, then delete | **`merged_<DATE>_<HH>.parquet` × 24** |

### Stage 2 — what gets cloned

aggTrades and bookDepth are **one file per day**; only orderbook is hourly.
That gives 26 files/day (not 72):

```
_pipeline/source/aggTrades/<DATE>.parquet       1
_pipeline/source/bookDepth/<DATE>.parquet       1
_pipeline/source/orderbook/<DATE>/<HH>.parquet  24
```

### Stage 4 — the finalize step

The Rust merger writes a *sparse* file where `book_state__*` / `depth_state__*`
are only populated on LOB rows and null on TRADE/BOOKDEPTH rows. The finalize
step forward-fills them so every row carries the most recent book state, seeded
from the previous hour's carry (`_pipeline/state/<DATE>/<HH>.json`). Without
it, the output would ship nulls on every trade.

---

## Options

| Flag | Default | Effect |
|------|---------|--------|
| `--start-date` | *required* | First UTC date, `YYYY-MM-DD` |
| `--end-date` | *required* | Last UTC date, `YYYY-MM-DD` (inclusive) |
| `--root` | `$MARKET_DATA_ROOT` or `<repo>/market_data` | market_data root |
| `--symbol` | `BTCUSDT` | Trading symbol |
| `--tick-size` | `0.1` | Instrument tick size → `lob_replay` |
| `--lob-levels` | `20` | Book levels per side → `event_merger` |
| `--no-seed` | off | Skip stage 1 |
| `--keep-work-inputs` | off | Keep cloned inputs / orderflow / sparse / state for inspection |
| `--no-warm-start` | off | Strict sequence validation (rejects diff-only data) |
| `--workers` | `1` | Parallel dates. Hours within a date always stay sequential |
| `--stop-on-error` | off | Abort the range on the first failure |
| `--build` | off | `cargo build --release` both binaries first |

Only dates having **all three** of aggTrades + bookDepth + orderbook are
processed; incomplete dates in the window are logged and skipped.

---

## Reading the output

128 columns, all namespaced. `timestamp` is UTC epoch **nanoseconds**.

```python
import polars as pl

df = pl.read_parquet("market_data/matched/BTCUSDT/merged_2025-06-30_00.parquet")

df["event_type"].value_counts()          # LOB / TRADE / BOOKDEPTH
df.select("timestamp", "event_type",
          "book_state__best_bid_price", "book_state__best_ask_price",
          "agg__price", "agg__quantity", "agg__is_buyer_maker")
```

| Prefix | Meaning |
|--------|---------|
| `timestamp`, `event_type`, `event_sequence` | merge bookkeeping (chronological order) |
| `lob__*` | LOB event identity (`event_time`, `final_update_id`) |
| `book_state__*` | 20 levels/side: `best_bid_price`, `best_bid_qty`, `best_ask_price`, `best_ask_qty`, then `bid_price_2..20`, `ask_price_2..20` (and matching `_qty`) — 81 cols |
| `agg__*` | aggTrades payload (`price`, `quantity`, `is_buyer_maker`, `agg_trade_id`, ...) |
| `depth__*` | bookDepth event payload |
| `depth_state__*` | cumulative band depth/notional per percentage band (-5..-1, 1..5) — 20 cols |
| `source_agg__*`, `source_depth__*` | verbatim source columns |

`event_type` is one of `LOB`, `TRADE`, `BOOKDEPTH`. `book_state__*` and
`depth_state__*` are carried forward onto every row.

Sample (hour 00, 2025-06-30):

| timestamp (ns) | event_type | seq | best_bid | best_ask | agg price | qty | buyer_maker |
|---|---|---|---|---|---|---|---|
| 1751241599959000000 | LOB | 0 | 108309.8 | — | — | — | — |
| 1751241600006000000 | LOB | 1 | 108309.8 | — | — | — | — |
| 1751241600441900000 | TRADE | 2 | 108665.5 | 108309.8 | 9.338 | true | — |
| 1751241600443900000 | TRADE | 3 | 108665.5 | 108309.9 | 3.354 | false | — |

---

## Health check

Verify no nulls were left behind:

```python
import polars as pl
from pathlib import Path

for p in sorted(Path("market_data/matched/BTCUSDT").glob("merged_*.parquet")):
    df = pl.read_parquet(p)
    state = [c for c in df.columns if c.startswith(("book_state__", "depth_state__"))]
    nulls = sum(df[c].null_count() for c in state)
    print(p.name, df.height, "rows,", nulls, "nulls")
```

---

## Known issue: `WARNING no LOB book state`

If you see this in stage 4, the **LOB builder emitted no book state** for that
hour — the orchestration is fine, the upstream replay is not:

```
[FINALIZE] 2025-07-01 00: WARNING no LOB book state (2162050 null(s) across TRADE/BOOKDEPTH rows)
```

The orderbook parquet for those hours contains 100% updates with no snapshot
frames, and under `--warm-start` the engine rebases (clears the book) on
`first_update_id > last_final_update_id + 1`, which fires on ~98% of frames.
Affected dates in the current sample: `2025-07-01`, `2025-07-02` (0 LOB rows in
all 24 hours). `2025-06-30` and `2025-07-04` are clean (0 nulls).

This is the LOB builder's gap heuristic, not an orchestration bug — it is left
unfixed by design. The chain surfaces it loudly rather than shipping silent
nulls. Grep the run log for `WARNING` to list affected dates:

```bash
python components/commands/01_ingest.py --start-date 2025-07-01 --end-date 2025-07-04 2>&1 \
  | Select-String 'WARNING'
```

---

## Troubleshooting

**`[ERROR] lob_replay binary not found`** — build it:
```bash
cargo build --manifest-path components/Cargo.toml --release
```
(or pass `--build`). `resolve_binary()` looks in `components/target/release`
then `components/target/debug`.

**`no every-event LOB book state for <DATE>`** — stage 3 produced nothing (see
the known issue above), or `--keep-work-inputs` was not used and a previous run
already cleaned the replay output. Re-run the chain for that date.

**`LOB is_bid is null`** (raw Rust error) — the merger was pointed at the raw
orderbook instead of the replay output. `merge_and_finalize` now pre-flight
checks this and fails with a clear message first; you should only see the raw
error when calling `event_merger` directly.

**`unknown flag` from a Rust binary** — the orchestrators validate flags per
mode, so this means a flag was passed straight through to the binary. Check
against the real CLI in `components/rust/lob_builder/btree_ver/main.rs`.

**Output dir inside input dir** — `lob_replay` refuses this. Use a distinct
`--root` or output directory.

---

## Orchestrators (also usable standalone)

| Script | Purpose |
|--------|---------|
| `components/rust/lob_builder/orchestrator/run_lob.py` | `lob_replay` — `ingest-parquet` / `columnar` / `matched` |
| `components/rust/lob_builder/orchestrator/snapshot_builder.py` | day-start seed only |
| `components/rust/event_merger/orchestrator/run_merger.py` | merge + finalize + cleanup over a date range |
| `components/rust/event_merger/orchestrator/smoke_test.py` | one hour/day merge, **nothing** finalized or deleted |
| `components/rust/event_merger/orchestrator/pipeline.py` | standalone LOB+merge driver (assumes sources already cloned) |

```bash
# smoke test a single hour, no finalize, no deletion
python components/rust/event_merger/orchestrator/smoke_test.py \
  --root market_data --symbol BTCUSDT --date 2025-06-30 --hour 0
```

### Sparse output layout

The compiled `event_merger` writes `sparse/<DATE>/<HH>.parquet`; the current
`merge.rs` source would write `sparse/<DATE>_<HH>.parquet`. Because the binary
can lag its source, `find_sparse_hours()` probes **both** (newer mtime wins), so
rebuilding the binary will not break finalize.
