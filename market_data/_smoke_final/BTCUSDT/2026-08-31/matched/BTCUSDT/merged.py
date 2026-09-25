import pandas as pd

df = pd.read_parquet("merged_2026-08-31_23.parquet")
for i in df.columns:
    print(i)