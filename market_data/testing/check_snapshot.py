import pandas as pd

if False:
    df = pd.read_parquet("08_every_event.parquet")

    # Print event times
    print(df["event_time"])

    # Calculate gaps between consecutive events
    gaps = df["event_time"].sort_values().diff().dropna()

    print("\nAverage gap:")
    print(gaps.mean(), "ms")

    print("\nAverage gap in seconds:")
    print(gaps.mean() / 1000, "seconds")
else:
    df = pd.read_parquet("00.parquet")

    unique_event_times = df["event_time"].nunique()

    print("Unique event times:", unique_event_times)