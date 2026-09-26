import pandas as pd

path = r"C:\Users\Himaska\Documents\Projects\Business_Projects\intraday-momentum-composite\market_data\matched\BTCUSDT\merged_2025-06-30_01.parquet"

df = pd.read_parquet(path).to_csv("sample.csv")

# Ensure the timestamp column is datetime format and sorted chronologically
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Compute interval between consecutive rows
avg_interval = df['timestamp'].diff().mean()

print(f"Average time interval: {avg_interval}")

# Optional: Format into specific units
avg_seconds = avg_interval.total_seconds()
print(f"Average interval in seconds: {avg_seconds:.6f} s")
print(f"Average interval in milliseconds: {avg_seconds * 1000:.3f} ms")