from binance_downloader import download_and_convert

trades_file, orderbook_file = download_and_convert(
    symbol="BTCUSDT",
    start_date="2026-05-01",
    end_date="2026-05-05",
    outdir="./research/csv",
    cache_dir="./binance_cache",
    out_format="csv",  # <--- Change this to "csv"
)