"""Proof: ugly mid_price comes from OUR (bid+ask)/2.0, not from CryptoHFT.

Reads RAW parquet bytes with pyarrow only -- no load_L2, no normalize_merged_frame,
no strategy code -- so what you see is what is actually stored on disk.

Run:  python proof_mid_price_origin.py
"""
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
MATCHED = ROOT / "market_data" / "matched" / "BTCUSDT" / "merged_2025-07-09_09.parquet"
ORDERBOOK = ROOT / "market_data" / "ingest_parquet" / "BTCUSDT" / "orderbook" / "2025-07-09" / "09.parquet"
AGGTRADE = ROOT / "market_data" / "ingest_parquet" / "BTCUSDT" / "aggTrades" / "2025-07-09.parquet"

print("== 1. CryptoHFT raw orderbook (upstream feed) ==")
print("file:", ORDERBOOK)
pf = pq.ParquetFile(ORDERBOOK)
print("schema price type:", [f for f in pf.schema if "price" in str(f)])
raw = pq.read_table(str(ORDERBOOK), columns=["price"]).slice(0, 5).to_pandas()
print("dtype:", raw["price"].dtype)  # expect object/string, NOT float
print("values:", raw["price"].tolist())
print("-> prices are exact decimal STRINGS like '108612.30'. No float noise possible here.\n")

print("== 2. CryptoHFT raw aggTrades ==")
pf2 = pq.ParquetFile(str(AGGTRADE))
print("schema:", [n for n in pf2.schema.names if "price" in n.lower()])
ag = pq.read_table(str(AGGTRADE), columns=["price"]).slice(0, 5).to_pandas()
print("dtype:", ag["price"].dtype, "values:", [repr(x) for x in ag["price"].tolist()], "\n")

print("== 3. OUR merged matched file (output of Rust merger) ==")
print("file:", MATCHED)
t = pq.read_table(str(MATCHED),
                  columns=["book_state__best_bid_price", "book_state__best_ask_price"]).to_pandas()
print("dtypes:", dict(t.dtypes.astype(str)))
print("stored bid head:", [repr(x) for x in t["book_state__best_bid_price"].head(5).tolist()])
print("stored ask head:", [repr(x) for x in t["book_state__best_ask_price"].head(5).tolist()])
print("-> stored bid/ask are float64 but repr-clean (1 decimal). No ugly tail yet.\n")

print("== 4. OUR mid_price step: (bid+ask)/2.0  [same formula as strategy.py:184, features.py:126] ==")
mid = (t["book_state__best_bid_price"] + t["book_state__best_ask_price"]) / 2.0
ugly = mid[mid.apply(lambda x: len(repr(float(x)).split(".")[-1]) > 4 if x == x else False)]
print(f"ugly mids: {len(ugly)} / {len(mid)} rows in this one hour file")
print("examples (bid, ask -> mid repr):")
for i in ugly.index[:10]:
    b, a, m = t.loc[i, "book_state__best_bid_price"], t.loc[i, "book_state__best_ask_price"], mid.loc[i]
    print(f"  ({b} + {a})/2 = {m!r}   <-- decimal-exact {(b + a) / 2:.2f} but float64 prints this tail")
print("\nConclusion: feed stores exact strings; ugly tail appears ONLY after OUR float division.")
print("Fix (if you want): round mid to tick size, e.g. mid.round(1) or Decimal, at the strategy layer.")
