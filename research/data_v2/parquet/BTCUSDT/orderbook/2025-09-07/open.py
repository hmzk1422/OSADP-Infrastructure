import pandas as pd

# Pandas handles Zstandard automatically via pyarrow
df = pd.read_parquet("00.parquet", engine="pyarrow")

print(df.columns)
print(df)