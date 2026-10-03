# L2 Orderbook Replay Engine
```
OSADP updated version: v1.0.0  
Package version      : v1.0.0  
Author               : Himaska Lin
```
A Rust event-driven engine for deterministic reconstruction of full-depth
L2 limit order books from historical exchange data.

Designed for high-volume quantitative research and backtesting, with:

- O(1) expected price-level mutation using HashMap
- strict update-sequence validation
- integer price-tick representation
- full-depth state reconstruction
- daily full-book checkpoints
- deterministic replay
- explicit invalid-state handling

| Commands | Desc     | Speed    |
| -------- | -------- | -------- |
| lob_snapshot_build_source    | Writes directly on PATH to build orderbook snapshot on hour 00 of each day, or optionally, write the snapshots on all rows.     | O(n), where n = number of LOB rows inside parquet but with write all snapshot on, overhead drastically increases    |

## Get Started with L2 Orderbook Replay
Day start checkpoint
```bash
cargo run --release -- ingest-parquet \
    --input <FILE|DIR> \
    --output-dir <DIR> \
    --tick-size <F> \
    [--snapshot-policy <day_start|every_event>] \
    [--warm-start]
```
Per event reconstruction
```bash
cargo run --release -- ingest-parquet \
    --input ./data/raw_depth/BTCUSDT_2026-01-01.parquet \
    --output-dir ./data/event_snapshots/ \
    --tick-size 0.00000001 \
    --snapshot-policy every_event \
    --warm-start
```
Columnar Replay Event
```bash
lob_replay columnar \
    --input <FILE> \
    --output <FILE> \
    --tick-size <F> \
    --levels <N> \
    --snapshot-policy <day_start|every_event> \
    [--snapshot-dir <DIR>] \
    [--snapshot-levels <N>] \
    [--seed-snapshot <FILE>] \
    [--skip-until <MS>] \
    [--warm-start]
```
Single parquet file, full book checkpoint (Legacy)
```
lob_replay matched \
    --input <FILE> \
    --output <FILE> \
    --tick-size <F> \
    [--warm-start]
```


## Why This Exists

Historical L2 data is not directly usable for strategy research.
The book must be reconstructed sequentially from snapshots and updates.

The initial replay is inherently O(n): every relevant event must be processed.
The engineering problem is therefore to minimize the per-event cost and avoid
reconstructing the same history for every backtest.

## lob_snapshot_build
> Replay the market-data stream sequentially once, maintain the canonical book in memory, and materialize a full-book checkpoint only once per UTC day.
This avoids repeatedly reconstructing the same historical order book while keeping the raw event stream as the source of truth.
```
Architecture
Historical L2 Parquet
        │
        ▼
   Event grouping
        │
        ▼
   ReplayEngine
        │
        ├── sequence validation
        ├── snapshot validation
        ├── price normalization
        ├── state reconstruction
        └── book validity tracking
        │
        ▼
     LobBook
        │
        ├── bids: HashMap<PriceTick, Level>
        └── asks: HashMap<PriceTick, Level>
        │
        ▼
   UTC 00:00 boundary
        │
        ▼
 Full-book checkpoint
```
### Design Rationale
The replay engine processes every historical update. There is no attempt to skip events because each event can change the state required to reconstruct the future book. Sequential replay is required, so a full reconstruction is performed (Initial snapshot -> update 1 -> update 2 -> update 3 -> ... -> UTC 00:00 checkpoint, write -> continue replay).
There is no general shortcut around the initial sequential replay. If the historical dataset contains n updates, all relevant updates must be processed at least once to establish the correct state. The optimization is therefore not to eliminate n. The optimization is to make the cost per update as small as possible and avoid repeating the historical reconstruction.
1. Full-Depth Internal State: Engine maintains full book internally (Output levels/top-N depth as output only). Engine cannot retain only top 10 levels safely as if a top level disappears, a deeperlevel may immediately be part of the top 10.
2. Price representation: Prices are represented as integer ticks rather than floating-point keys. For example with tick size 0.10, $42100.10 / 0.10 = 421001. This avoids floating point values as map keys and gives exact discrete price-level identity.
3. Book data structure: The book uses `HashMap<PriceTick, Level>` for both sides (`bids: HashMap<PriceTick, Level>` and `asks: HashMap<PriceTick, Level>`). A level contains its current absolute quantity with `pub struct Level { pub quantity: f64, }`. Hashmap is used for the following.
    - Locating existing price level
    - Inserting new price level
    - Updating existing level
    - Removing a level
    
The operations are O(1) expected with a hash table. This removes the
O(log r) tree operation from the dominant mutation path. The trade-off is
that ordered traversal is no longer maintained continuously and must be
performed only when required.

### Alternative Design Choices
BTreeMap naturally maintains a sorted price `BTreeMap<PriceTick, Level>` which makes best-price lookup very straightforward (best bid = largest key and best ask = smallest key). However, updates becomes O(log r) where r is the number of active price levels. With approximately 6000 active levels per side:
$$log_2(6000) = 12.55$$
A hashmap instead provides O(1) expected for the common mutation path. The trade-off is sorted traversal must be performed when it is actually needed. This engine therefore optimizes for the dominant operation of large numbers of sequential state mutations rather than continuously maintaining a globally sorted representation.

# Performance Benchmark
placeholder
Benchmarks are measured on the full historical L2 replay pipeline.

| Metric | Result |
|---|---:|
| Input L2 rows | XX,XXX,XXX |
| Replay throughput | X.XX M rows/sec |
| Avg replay time / row | X.XX μs |
| Peak process memory | X.XX GB |
| Checkpoint generation time | X.XX ms/day |
| Number of daily checkpoints | XXX |
| Build | `cargo build --release` |
| Threads | 1 |
| Dataset | BTCUSDT L2 |
| Measurement | XX runs / median |

The benchmark includes event grouping, sequence validation, book-state mutation,
and checkpoint generation. Parquet read/write cost is reported separately.

Charts and tables showing actual performance, failure case, validation, measure, etc