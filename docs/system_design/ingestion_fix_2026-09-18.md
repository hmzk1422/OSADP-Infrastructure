# Ingestion v2 Fixes — 2026-09-18

Root-cause analysis and code changes for the missing BTCUSDT bookDepth parquet
data, the stalled CryptoHFTData conversion, and the matched-data refresh.

---

## 1. Diagnosis (what was actually wrong)

### 1.1 bookDepth parquet never produced — schema bug (root cause)

Downloads were never the problem: the manifest showed `binance bookDepth:
431/431 downloaded`. The failure was in **Stage 2** of
`scripts/ingestion_L2/convert_to_parquet.py`.

Binance's bookDepth daily CSV ships:

- a **header row** (`timestamp,percentage,depth,notional`), and
- `timestamp` as an **ISO datetime string** (e.g. `2025-06-28 00:00:07`),
  NOT an epoch integer.

The converter's `BINANCE_SCHEMAS["bookDepth"]` declared
`"timestamp": "int64"`, so `df.astype(...)` raised on **every** file:

```
ValueError: invalid literal for int() with base 10: '2025-06-28 00:00:07':
Error while type casting for column 'timestamp'
```

All 431 files hit the `except` branch -> `failed=431` -> no parquet written
-> the converter exited `sys.exit(1)` -> `ingestion.py` raised
"STAGE 2 failed". (`aggTrades` converted fine only because its schema happens
to match: numeric ids and integer `transact_time`.)

Consequences of the exit:

- The CHD orderbook conversion never completed either, so all ~10k raw CHD
  files (252 GB) sat in `cryptohftdata/BTCUSDT/orderbook` and the unified
  `parquet/BTCUSDT/orderbook` tree stayed empty.
- The 431 raw bookDepth zips were deleted afterwards (cause outside the
  pipeline code; the converter never deletes a zip without a verified
  parquet), so a re-download was required.

### 1.2 CHD conversion was effectively impossible even if reached

The old CHD path verified each hourly file by materializing it **twice** with
pandas (`len(pd.read_parquet(src))` + re-reading the copy). At 6-7M rows per
hourly file this is on the order of days for ~10k files.

### 1.3 bookDepth dataset semantics (for reference)

Verified empirically against a real zip: rows arrive as 10 percentage bands
per snapshot — `percentage ∈ {-5,-4,-3,-2,-1, +1,+2,+3,+4,+5}` (cumulative
depth/notional within ±1% steps of mid, up to ±5%), one snapshot roughly
every 30-45 s (~28.5k rows/day). `depth` = coin quantity, `notional` = USDT
value. It is *aggregated band depth*, not price-level L2.

---

## 2. Code changes

### `scripts/ingestion_L2/convert_to_parquet.py`

1. **bookDepth timestamp normalization (the core fix).**
   `read_binance_zip_csv` now parses the ISO string via
   `pd.to_datetime(..., utc=True)` and stores it as **UTC epoch milliseconds
   (int64)** — the same convention as `aggTrades.transact_time` and
   CryptoHFTData `event_time` — so joins need no per-reader conversion.
   Note: an intermediate `astype("datetime64[ms]")` is required because pandas
   infers the datetime resolution from the string (second-resolution strings
   yield `datetime64[s]/[us]`, whose raw int64 values are seconds, not ns).
   If a future file ships numeric epoch values, they are kept as-is.
   Verified: `1751068807000 -> 2025-06-28 00:00:07 UTC`.

2. **Fast verification.** `_verify_parquet` now reads Parquet **footer
   metadata only** via a new `_parquet_num_rows` helper (`pyarrow`), instead
   of fully materializing the frame with pandas.

3. **Rename-based CHD move.** When `--delete-raw` is in effect (default), CHD
   hourly files are converted with a **same-volume atomic rename**
   (`Path.replace`) instead of a full read+write copy. This turned the
   ~250 GB conversion into metadata-only renames (minutes instead of days).
   A cross-volume copy fallback is kept for safety; `--keep-raw` still copies.

4. **Canonical CHD filename.** The CHD loop now prefers
   `<SYMBOL>_orderbook.parquet` inside each hour dir instead of blindly using
   `glob("*.parquet")[0]`, which could pick up stray artifacts
   (`*.test.parquet`, leftover `.part` partials) and fail that hour.

5. **`--datasets` filter.** New argument (`aggTrades,bookDepth,orderbook`)
   lets a targeted re-run convert only the requested datasets.

### `scripts/ingestion_L2/ofp_data_downloader.py`

1. **`--datasets` filter** (same values) so a partial re-run does not
   needlessly re-download datasets whose parquet outputs already exist
   (e.g. re-downloading aggTrades zips that were already converted).
2. **Manifest merge-on-write.** `write_manifest` now merges new results into
   the existing manifest keyed by `(provider, dataset, date, hour)` instead
   of clobbering it. A `--datasets bookDepth`-only run no longer erases the
   recorded CHD/aggTrades entries. `start_date`/`end_date` in the manifest
   now reflect the min/max date across all merged entries.

### `scripts/ingestion.py`

Forwards `--datasets` to both pipeline stages.

### Research pipeline re-pointed to the unified parquet tree

- `research/MoE/data/dataconfig.py`: `LOB_PATH` is now
  `research/data_v2/parquet/<TICKER>/orderbook` (previously the raw
  `cryptohftdata/.../orderbook` tree, which is now empty).
- `research/MoE/data/live.py`: `_filter_parquet_files` now understands **both
  layouts** transparently — unified `<day>/<HH>.parquet` hour **files** and
  the legacy `<day>/<HH>/<symbol>_orderbook.parquet` hour **dirs**.
- `scripts/ingestion_L2/zstandard_fix.py`: default `ROOT` moved to the
  unified tree (its repair/compress/decompress modes operate on the same
  files after the move).

### New: `scripts/refresh_matched.py`

Single-command wrapper around `research.MoE.data.live.fetch_source_data`
(defaults from `dataconfig`, `--chunk-size 1` = one hour per pass for low
memory) so matched data can be refreshed reproducibly, including in the
background.

---

## 3. What was run

```bash
# Stage 1 — re-download the 431 bookDepth zips + retry the 22 DNS-errored CHD hours
python scripts/ingestion_L2/ofp_data_downloader.py \
    --symbol BTCUSDT --start-date 2025-06-28 --end-date 2026-09-01 \
    --datasets bookDepth,orderbook --workers 6

# Stage 2 — convert everything (aggTrades no-ops; its parquet already exists)
python scripts/ingestion_L2/convert_to_parquet.py \
    --root research/data_v2 --symbol BTCUSDT \
    --datasets aggTrades,bookDepth,orderbook

# Stage 3 — refresh matched data from the newest sources
python scripts/refresh_matched.py --chunk-size 1
```

## 4. Validation results

| Check | Result |
|---|---|
| bookDepth raw zips | 431/431 downloaded |
| `parquet/BTCUSDT/bookDepth/*.parquet` | **431 files, failed=0** |
| bookDepth schema | `timestamp:int64 (epoch ms)`, `percentage:int64 (−5..+5)`, `depth/notional:float64` |
| bookDepth spot-check (2025-06-28) | 28,570 rows; first ts `1751068807000` = `2025-06-28 00:00:07 UTC`; last `23:59:33` |
| CHD orderbook files | 10,138 moved (10,116 previous + 22 retry successes) |
| Orderbook integrity | byte-identical renames; row counts match pre-move values (e.g. 2025-06-28/05: 6,399,093; 2026-02-01/00: 6,857,730; 2026-02-02/00: 11,326,651) |
| Raw CHD tree after move | 0 files (fully pruned) |
| Converter summary | `converted=10,569 skipped_existing=0 failed=0` |
| `live.py` unified-tree read | 48 matching hours for 2026-02-01 → 2026-02-02, no missing agg/LOB |
| Matched refresh | 24 → **48 files** (now covers 2026-02-02 as well), all rebuilt 2026-09-18 |
| Matched sanity (hour 00, 02-01) | 139,101 trades; 0 negative spreads; 0 null bid/ask; both sides populated on 100% of rows |

Also cleaned up: 4 stale `.part` partial downloads (2026-07-17 17/20/21/22 —
leftovers of the DNS-failure hours) and one 0-byte
`BTCUSDT_orderbook.test.parquet` (2026-08-01/00). All five were re-handled by
the pipeline or were empty artifacts.

## 5. Remaining notes

- 206 CHD hours are `missing` — genuine CryptoHFTData coverage gaps
  (e.g. 2025-06-28 hours before their archive starts mid-day), not errors.
  A later `--datasets orderbook` re-run will retry them automatically.
- The 22 previously DNS-errored CHD hours (2026-07-17/18) downloaded
  successfully on the re-run.
- CryptoHFTData orderbook data is **full-depth L2** (hourly snapshot with
  ~6k rows/side + incremental diff stream incl. quantity-0 level removals).
  `fetch_source_data` currently reduces it to top-of-book before the join;
  multi-level features would need that aggregation extended.
- bookDepth `timestamp` is **epoch ms UTC** now. If you add bookDepth
  consumers, join on `timestamp` (ms) the same way as
  `aggTrades.transact_time`.

---

## 6. Multi-level LOB in matched data + optional bookDepth (2026-09-18, later)

### 6.1 What changed

`research/MoE/data/live.py::fetch_source_data` no longer collapses the
CryptoHFTData stream to top-of-book. It now REPLAYS the depth-diff stream
(`_build_multilevel_book`) and attaches the top-N levels per side to every
trade:

- `fetch_source_data(..., lob_levels=20)` (default 20; 1 = legacy TOB schema):
  columns `best_bid_price/qty`, `bid_price_2..N` / `bid_qty_2..N`,
  `best_ask_price/qty`, `ask_price_2..N` / `ask_qty_2..N`
  (bids best/highest first, asks best/lowest first).
- Replay semantics: `snapshot` rows reset that side's book to exactly those
  rows; `update` rows upsert the level, `quantity == 0` removes it; a price
  lives on exactly one side (bid<->ask transitions handled); update-ID gaps
  (`prev_final_update_id != previous final_update_id`) reset state per CHD's
  docs; rows whose state is crossed (`best_bid >= best_ask`) are dropped and
  reported.
- `fetch_source_data(..., include_bookdepth=True)` additionally asof-joins
  Binance bookDepth as wide columns
  `bd_depth_m1..m5 / bd_notional_m1..m5` (bid side, % below mid) and
  `bd_depth_p1..p5 / bd_notional_p1..p5` (ask side). Optional and OFF by
  default (the CHD L2 levels are strictly richer). bookDepth `percentage == 0`
  rows (duplicated mid markers) are excluded before pivoting.
- CLI: `scripts/refresh_matched.py --lob-levels N --include-bookdepth`.
- `research/MoE/data/dataconfig.py`: new `BOOKDEPTH_PATH`, `LOB_LEVELS = 20`.

### 6.2 Snapshot availability census (empirical, exhaustive scans)

Snapshot rows (`event_type == "snapshot"`) are NOT uniformly present:

| Era | Result |
|---|---|
| 2025-06-28 .. 2025-06-30 | snapshots in (essentially) every hour (~12,000 rows = 6k levels/side) |
| 2025-07-01 .. 2025-10-31 | **0 snapshots in 2,952 files (exhaustive scan)** |
| 2025-11-01 .. 2026-02-02 | **0 snapshots in 2,256 files (exhaustive scan)** |
| 2026-04-18 .. 2026-09+ | sporadic resync snapshots (e.g. 2026-04-18/21, 2026-05-23/21, 2026-05-30/05, 2026-06-06/13, 2026-06-13/21, 2026-08-15/13, 2026-08-22/13+21) |

Consequence: for snapshot-less windows the book is a **warm start** from
diffs alone (state chained across contiguous hours, reseeded on gaps). The
diff stream itself is genuinely full-depth (one 2026 hour: ~945k changed
levels beyond 1% of mid, ~590k beyond 3%), so shallow levels (1-5) are
near-exact quickly; deeper ranks can be stale/missing until the level is
touched. There is no exact alternative source: Binance publishes no
historical full-depth L2, and Binance Vision's futures bookTicker files do
not exist for these dates (verified: NoSuchKey).

### 6.3 Validation (single-hour, 2026-02-01 00:00 UTC)

- replay ~17-20s per hourly file (6-11M rows), ~66-69k book-state rows/hour
- 0 null best bid/ask across 139,101 trades; 0 crossed rows after replay
- spread: min = 1 tick (0.10), no ordering violations in levels 1..20
- state chaining verified: crossed-dropped 2,737 (h00) -> 2,012 (h01)

### 6.4 Tooling

- `main.py scan-snapshots --start <start> --end <end>`: parallel scan for
  snapshot-bearing orderbook files (projection-pushdown, 12 workers).
- `main.py validate-replay [--start <d> --end <d>]`: replay-invariant
  validation over the matched tree.

---

## 7. Layout + executor refactor (2026-09-19)

Everything ingestion now lives in one component and the raw tree is keyed by
**ticker and data type** instead of provider. No ingestion module, helper
tool, or layout constant sits outside the component any more.

### 7.1 Directory changes

- Component: `scripts/` -> `library/python/` -> **`components/python/ingestion_L2/`**
  (`ingestion.py` was folded into `main.py`; `refresh_matched.py`,
  `_scan_snapshots.py` -> `scan_snapshots.py` and `_validate_replay.py` ->
  `validate_replay.py` moved inside).
- New `datapaths.py` owns the whole tree, so a path change is a one-file edit.
  Roots resolve as `--root` > `$MARKET_DATA_ROOT` > `<repo>/market_data`, and
  `--root` always names the `market_data` directory itself.
- New `main.py` is the single executor: `pipeline`, `download`, `convert`,
  `compress` / `decompress` / `repair`, `refresh-matched`, `scan-snapshots`,
  `validate-replay`, `migrate-layout`, `layout`. Stage flags are forwarded
  verbatim, so `main.py <command> --help` prints the stage's own options.
  Stage modules stay individually runnable (each exposes `main(argv)`).

| Before | After |
|---|---|
| `ingest_source/binance/<TICKER>/<dataset>/<DATE>/...` | `ingest_source/<TICKER>/<dataset>/<DATE>/...` |
| `ingest_source/cryptohftdata/<TICKER>/orderbook/<DATE>/<HH>/...` | `ingest_source/<TICKER>/orderbook/<DATE>/<HH>/...` |
| `parquet/<TICKER>/...` | `ingest_parquet/<TICKER>/...` |
| `--root research/data_v2` | `--root market_data` |

The provider is no longer a path segment (the dataset name already implies it:
`aggTrades`/`bookDepth` = Binance, `orderbook` = CryptoHFTData L2). The mapping
lives once in `datapaths.PROVIDER_OF`, used for manifest bookkeeping.

`python main.py migrate-layout --apply` performs the move, re-points every
manifest entry to the canonical path (keeping the original as `legacy_path`),
stamps `layout_version: 2`, and prunes the emptied provider dirs.

### 7.2 Refactor details

- **Centralized paths.** `ofp_data_downloader.py`, `convert_to_parquet.py` and
  `zstandard_fix.py` no longer build paths by hand
  (`root / "binance" / symbol / ...`, a hard-coded
  `research/data_v2/parquet/BTCUSDT/orderbook`, `./research/data_v2` defaults).
  All of them call `datapaths`.
- **`main(argv=None)` on every stage**, so the executor can import and call a
  stage in-process instead of spawning a subprocess; `SystemExit` from argparse
  or `sys.exit(1)` is converted into a return code and stops the pipeline.
- **Raw-file pruning bug fixed.** `_delete_raw_file` walked up removing empty
  parents with no floor, so it could delete the dataset dir and everything
  above it. It now takes `stop=<dataset dir>` and never climbs past it.
- **`migrate-layout` pruning** removes nested empty dirs deepest-first; a
  top-down walk left empty `<dataset>/<DATE>/<HH>` shells behind.
- **`refresh_matched.py`** no longer hard-codes the strategy package path. It
  discovers `<...>/MoE/data/live.py` under the repo root and imports it by its
  relative dotted path (so `research` -> `strategies` -> `strategy_system`
  renames do not break ingestion), treats `dataconfig` as optional, and always
  passes ingestion layout paths explicitly to `fetch_source_data(...)` so a
  stale strategy-side path cannot silently match the wrong tree.
- **`zstandard_fix.py`** gained `--symbol`/optional `--root` and defaults to
  `<market_data>/ingest_parquet/<TICKER>/orderbook`.

### 7.3 Validation (re-run after the refactor)

- `main.py layout`: `aggTrades` 431 days, `bookDepth` 431 days, `orderbook`
  431 days / 10,138 hourly files (2025-06-28 .. 2026-09-01), 48 matched files.
- `main.py migrate-layout --apply`: 11,206 manifest entries re-pointed, 0 stale
  `research/data_v2` paths left, `legacy_path` preserved.
- `main.py convert --symbol BTCUSDT`: idempotent no-op (all outputs present).
- `main.py scan-snapshots --start 2025-06-28 --end 2025-06-28`: reproduces the
  §6.2 census (19/19 hours, 12,000 snapshot rows each).
- `main.py validate-replay --start 2026-02-01 --end 2026-02-02`: 48 files,
  8,133,068 rows, 0 null best bid/ask, 0 crossed, 0 level-order violations,
  spread min 0.1 / max 3,935.3, bookDepth coverage 99.676%. Matches §6.3.
- `main.py refresh-matched`: resolves `strategy_system.strategy.MoE.data`, then
  reports that the strategy layer's own `import research.MoE.data.dataconfig`
  is stale (exit 2). The strategy-side import path is a strategy-layer change,
  outside ingestion; the ingestion trees are intact either way.
