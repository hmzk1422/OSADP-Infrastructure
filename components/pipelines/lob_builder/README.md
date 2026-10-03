# L2 Orderbook Replay Engine
```
OSADP updated version: v1.0.0  
Package version      : v1.0.0  
Author               : Himaska Lin
```
| Version| Changes|
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v1.0.0**<br>2026-09-20 | **Initial stable L2 replay engine.** Implemented Rust limit-order-book reconstruction with integer price ticks, bid/ask price-level management via `BTreeMap`, full snapshot reseeding, incremental absolute-quantity updates, update-ID continuity validation, crossed-book detection, malformed-event validation, and daily UTC checkpointing. Added strict and warm-start replay modes to support both snapshot-seeded and diff-only streams. Current implementation prioritizes **state correctness, deterministic replay, and data integrity** over maximum update-path performance. `BTreeMap` is intentionally retained as the initial ordered-book implementation; a lower-latency structure may be evaluated in a future version after correctness and benchmark baselines are established. |
