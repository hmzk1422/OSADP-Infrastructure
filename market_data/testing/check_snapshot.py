import pandas as pd

df = pd.read_parquet("00.parquet")

# Print event times
print(df["event_time"])

# Calculate gaps between consecutive events
gaps = df["event_time"].sort_values().diff().dropna()

print("\nAverage gap:")
print(gaps.mean(), "ms")

print("\nAverage gap in seconds:")
print(gaps.mean() / 1000, "seconds")