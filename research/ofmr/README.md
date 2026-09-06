# Orderflow Mean Reversion
Orderflow mean reversion is the variation of Connor's strategy that trades RSI2 and d^2 price relationship. It produces 2 model txt files used in the executor.

## Main Executor
| File Name | Component | Description |
| --- | --- | --- |
| `ml_state_optimized_regime.py` | Output | Variation of ml_state_optimized specifically to export features based on  |
| `ml_state_optimized.py` | Output | (1) Feature combiner via historical state space by chunking of state space. (2) Zone filtering of RSI2, d^2 price. Reformatting to Triple Barrier Labelling. (3) Deduplication and downsampling of macro/micro with second layer of dataset cleaning. (4) Purged chronological walkforwards and reporting. |
| `ml_state_builder.py` | Abstraction | (1) Memory management with disk formatting and ensuring ingestion of CSV or parquet. (2) Ensures time-series alignment by handling empty time intervals and most importantly, micro/macro timeline alignment. (3) Anti-lookahead shield with state persistence for chunking (recursive/non-recursive features) |
| `feature_builder` | Strategy Logic | Pure feature calculation based on orderflow and price features |