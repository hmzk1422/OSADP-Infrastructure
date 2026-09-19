# Ingestion
The ingestion layer is responsible for acquiring and preserving the raw market data required by research/. Its primary design objective is data integrity and reproducibility, not aggressive normalization or abstraction.
The ingestion system therefore preserves each provider's native data representation and separates raw archival data from downstream analytical formats.
## Data Source Choice
Binance Data Vision and CryptoHFTData provide different types of market data.
- Binance aggTrades: Executed aggregated trades
- Binance BookDepth: Cumulative depth summarized across percentage bands (Rough, fast to process data).
- CryptoHFTData L2: Price-level orderbook snapshots and incremental updates, true L2 data.

Initially, only Binance aggTrades and bookDepth are used to account for data limitations (Both span 3 years, large quantity of data means model can be trained without dangers of overfitting). aggTrades provides the incremental accuracy while bookDepth provides a shallow, rough look at L2 data.
## Design choices
```bash
scripts
|_ ingestion.py
|_ ingestion_L2
   |_ ofp_data_downloader.py # Downloads binance and cryptohftdata files.
   |_ parquet_converter.py # Converts binance data to parquet files
   |_ spec_data_downloader.py # Orderflow and price are universal but certain features (liquidation, etc) are considered exclusive data.

Sequence
ingestion.py --> download binance aggTrades + bookDepth --> download CryptoHFTData --> conversion to parquet files
```
Everything is executed via unified runner ingestion.py. This is useful for single-command, reproducible research execution. This allows for possible future implementation (e.g. Self-aware trading strategy for autonomous alpha discovery, or complex web of trading systems).
- **Controlled Concurrent workers/threading**: Run concurrent workers for aggTrades and BookDepth where each workers work on different dates/trade sequences. Threading works very well here as it speeds up download speed massively and there is no risk of interleaving by having workers work on each feed. Binance and CryptoHFTData is explicitly split to prevent disk overloading where running too many workers might cause overhead bottleneck and diminishing performance (CryptoHFTData being true L2 data makes it heavy)
- **Provider-Specific Ingestion, Abstraction on Executor**: Data stream are chosen exclusively to be easily constructable as the following.
    - AggTrades: Directly obtained from websocket.
    - BookDepth: Reconstructed from the live L2 order-book WebSocket. The primary limitation is the available book-depth range; for example, ±3% may be used as the standard live representation even when the historical dataset extends to ±5%. An alternative is to preserve the same overall range while using finer percentage increments (e.g., 0–1% in 0.2% increments). Before deployment, differences between representations should be empirically evaluated using measures such as cosine similarity and mutual information to determine whether the finer-scale representation preserves the relevant predictive information.
    - CryptoHFTData: Directly obtained from websocket.
- **Immutable Market Data, Mutable Research**: Raw market data remains immutable. Research transformations are performed downstream (immutable -> reproducible conversion -> features -> models -> result). This makes it possible to distinguish a change caused by the data source from a change caused by feature engineering or model methodology. This means raw data still remains the source of truth.
- **Preserving Native Source Semantics**: Datasets are not converted or recalculated into any format during ingestion. This will be done optionally during research and the ingested data should remain lossless.