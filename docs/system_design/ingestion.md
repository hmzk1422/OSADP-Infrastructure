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
components
|_ python
   |_ ingestion_L2
      |_ main.py                # unified executor: python main.py <command> [flags]
      |_ datapaths.py           # single source of truth for the on-disk layout
      |_ ofp_data_downloader.py # Stage 1: downloads binance and cryptohftdata files.
      |_ convert_to_parquet.py  # Stage 2: converts raw files into the parquet tree
      |_ zstandard_fix.py       # orderbook parquet: compress / decompress / repair
      |_ refresh_matched.py     # rebuilds matched/ (wraps the strategy data layer)
      |_ scan_snapshots.py      # census of snapshot-bearing orderbook hours
      |_ validate_replay.py     # replay-invariant validation over matched/
      |_ spec_data_downloader.py # Orderflow and price are universal but certain features (liquidation, etc) are considered exclusive data.

Sequence
main.py pipeline --> download binance aggTrades + bookDepth --> download CryptoHFTData --> conversion to parquet files --> (optional) compress / refresh matched
```
Everything is executed via the unified executor `main.py`. Every stage is also runnable on its own (`python main.py <command> --help` prints that stage's own flags), which keeps single-command reproducible research execution without losing the ability to run one step. This allows for possible future implementation (e.g. Self-aware trading strategy for autonomous alpha discovery, or complex web of trading systems).
- **Everything ingestion lives in `ingestion_L2`**: no ingestion module, helper tool, or layout constant sits outside the component. `datapaths.py` owns the tree so a layout change is a one-file edit, and `main.py` is the only entry point needed to reach any of it.
## Data layout
Raw archival data and downstream analytical formats are separated, and each tree is keyed by **ticker and data type** — never by provider:
```bash
market_data
|_ ingest_source                     # raw, provider-native, immutable
|  |_ manifests/BTCUSDT_manifest.json  # atomic write log: status, paths, bytes, errors
|  |_ BTCUSDT
|     |_ aggTrades/<DATE>/BTCUSDT-aggTrades-<DATE>.zip     # Binance (daily)
|     |_ bookDepth/<DATE>/BTCUSDT-bookDepth-<DATE>.zip     # Binance (daily)
|     |_ orderbook/<DATE>/<HH>/BTCUSDT_orderbook.parquet   # CryptoHFTData L2 (hourly)
|_ ingest_parquet                    # normalized, uniform parquet
|  |_ BTCUSDT
|     |_ aggTrades/<DATE>.parquet
|     |_ bookDepth/<DATE>.parquet
|     |_ orderbook/<DATE>/<HH>.parquet
|_ matched                           # aggTrades x L2 join output, research read/write
   |_ BTCUSDT/BTCUSDT_<start>_<end>_<start-HH>-<end-HH>.parquet
```
The provider is deliberately not a path segment: the dataset name already identifies it (`aggTrades` / `bookDepth` are Binance, `orderbook` is CryptoHFTData L2), and every consumer selects by data type. The mapping is recorded once, in code (`datapaths.PROVIDER_OF`), for manifest bookkeeping. `--root` (or `$MARKET_DATA_ROOT`) always names the `market_data` directory itself.
- A tree predating this layout (`ingest_source/<provider>/<TICKER>/...`) is migrated with `python main.py migrate-layout --apply`, which re-homes the files and re-points the manifest (the old path is kept per entry as `legacy_path`).
- **Controlled Concurrent workers/threading**: Run concurrent workers for aggTrades and BookDepth where each workers work on different dates/trade sequences. Threading works very well here as it speeds up download speed massively and there is no risk of interleaving by having workers work on each feed. Binance and CryptoHFTData is explicitly split to prevent disk overloading where running too many workers might cause overhead bottleneck and diminishing performance (CryptoHFTData being true L2 data makes it heavy)
- **Provider-Specific Ingestion, Abstraction on Executor**: Data stream are chosen exclusively to be easily constructable as the following.
    - AggTrades: Directly obtained from websocket.
    - BookDepth: Reconstructed from the live L2 order-book WebSocket. The primary limitation is the available book-depth range; for example, ±3% may be used as the standard live representation even when the historical dataset extends to ±5%. An alternative is to preserve the same overall range while using finer percentage increments (e.g., 0–1% in 0.2% increments). Before deployment, differences between representations should be empirically evaluated using measures such as cosine similarity and mutual information to determine whether the finer-scale representation preserves the relevant predictive information.
    - CryptoHFTData: Directly obtained from websocket.
- **Immutable Market Data, Mutable Research**: Raw market data remains immutable. Research transformations are performed downstream (immutable -> reproducible conversion -> features -> models -> result). This makes it possible to distinguish a change caused by the data source from a change caused by feature engineering or model methodology. This means raw data still remains the source of truth.
- **Preserving Native Source Semantics**: Datasets are not converted or recalculated into any format during ingestion. This will be done optionally during research and the ingested data should remain lossless.