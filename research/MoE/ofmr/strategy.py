from research.MoE.data import live
from research.MoE.ofmr.feature_builder import macro_features
from pathlib import Path

#live.fetch_source_data(start="2026-02-01", end="2026-02-02", chunk_size=1)
df = live.read_source_data(chunk_size=1)
print(df.columns)