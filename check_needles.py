"""Needle check: tick vs 100ms-bar jumps before/after loader repair (read-only proof)."""
import pyarrow.parquet as pq
from strategy_system.exchange.loader import load_L2
from components.pipelines.event_merger.matched.reconstruct import reconstruct

# --- raw file 07-09_09 (worst toggling hour) ---
t = pq.read_table(
    "market_data/matched/BTCUSDT/merged_2025-07-09_09.parquet",
    columns=["timestamp", "event_type",
             "book_state__best_bid_price", "book_state__best_ask_price"],
).to_pandas().sort_values("timestamp").reset_index(drop=True)
raw_mid = (t["book_state__best_bid_price"] + t["book_state__best_ask_price"]) / 2.0
lob_mid = raw_mid.where(t["event_type"] == "LOB")
print("RAW 07-09_09 tick jumps>10 :", int((raw_mid.diff().abs() > 10).sum()))
print("RAW 07-09_09 tick jumps>100:", int((raw_mid.diff().abs() > 100).sum()))
print("RAW 07-09_09 LOB-only jumps>100:",
      int(lob_mid.dropna().diff().abs().gt(100).sum()))

# --- repaired loader output (1 file default = 06-30_00, then bars) ---
df = load_L2(file_limit=1)
m = df["mid_price"]
print("FIXED tick jumps>10 :", int((m.diff().abs() > 10).sum()))
print("FIXED tick jumps>100:", int((m.diff().abs() > 100).sum()))
print("FIXED dust rows:", int(
    m.apply(lambda x: len(repr(float(x)).split(".")[-1]) > 4 if x == x else False).sum()))
print("FIXED null mids:", int(m.isna().sum()))

r = reconstruct(df, "100ms")
mc = r["mid_price_close"].dropna()
print("FIXED 100ms-bar mid_close jumps>10 :", int((mc.diff().abs() > 10).sum()))
print("FIXED 100ms-bar mid_close jumps>100:", int((mc.diff().abs() > 100).sum()))
print("bars:", len(r))
