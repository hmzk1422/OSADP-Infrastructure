import pandas as pd

path = r"C:\Users\Himaska\Documents\Projects\Business_Projects\intraday-momentum-composite\market_data\ingest_parquet\BTCUSDT\orderbook\2025-07-15\00.parquet"

df = pd.read_parquet(path)

if False:
    print("shape:", df.shape)
    print("\ndtypes:")
    print(df.dtypes)

    print("\nhead:")
    print(df.head())

    print("\nTimestamp:")
    print(df["timestamp"].head(20))

    print("\nTimestamp dtype:")
    print(df["timestamp"].dtype)

    print("\nUnique timestamps:", df["timestamp"].nunique())
    print("Unique agg_trade_id:", df["agg_trade_id"].nunique())

    print("\nRows per timestamp:")
    print(
        df.groupby("timestamp")
        .size()
        .describe()
    )
else:
    print("shape:", df.shape)
    print("\ndtypes:")
    print(df.dtypes)

    print("\nhead:")
    print(df.head())

    print("\nTimestamp:")
    print(df["received_time"].head(20))

    print("\nTimestamp dtype:")
    print(df["received_time"].dtype)

    print("\nUnique timestamps:", df["received_time"].nunique())
    print("Unique agg_trade_id:", df["received_time"].nunique())

    print("\nRows per timestamp:")
    print(
        df.groupby("received_time")
        .size()
        .describe()
    )
    ts = pd.to_datetime(df["received_time"], errors="coerce").dropna()

    min_time = ts.min()
    max_time = ts.max()
    time_span = max_time - min_time

    print("Min time:", min_time)
    print("Max time:", max_time)
    print("Total span:", time_span)

    if len(ts) > 1:
        avg_interval = time_span / (len(ts) - 1)
        print("Average interval between rows:", avg_interval)
        print(
            "Average interval (ms):",
            avg_interval.total_seconds() * 1000
        )