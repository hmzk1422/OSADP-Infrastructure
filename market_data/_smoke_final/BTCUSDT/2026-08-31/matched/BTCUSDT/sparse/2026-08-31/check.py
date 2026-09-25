import pandas as pd
import numpy as np

INPUT = "23.parquet"
TICK_SIZE = 0.01

df = pd.read_parquet(INPUT)

lob = df[df["event_type"] == "LOB"].copy()

print(f"LOB rows: {len(lob):,}")

# Best prices
lob["spread"] = (
    lob["book_state__best_ask_price"]
    - lob["book_state__best_bid_price"]
)

lob["spread_ticks"] = (
    lob["spread"] / TICK_SIZE
)

print("\nSpread statistics:")
print(
    lob["spread"].describe(
        percentiles=[
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    )
)

print("\nSpread in ticks:")
print(
    lob["spread_ticks"].describe(
        percentiles=[
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    )
)

# Suspicious spreads
for threshold in [1, 2, 5, 10, 50, 100]:
    count = (
        lob["spread_ticks"] > threshold
    ).sum()

    pct = count / len(lob) * 100

    print(
        f"> {threshold:>3} ticks: "
        f"{count:>8,} "
        f"({pct:.3f}%)"
    )

# Number of populated levels
bid_levels = [
    f"book_state__bid_price_{i}"
    for i in range(2, 21)
]

ask_levels = [
    f"book_state__ask_price_{i}"
    for i in range(2, 21)
]

lob["bid_levels"] = lob[
    ["book_state__best_bid_price"] + bid_levels
].notna().sum(axis=1)

lob["ask_levels"] = lob[
    ["book_state__best_ask_price"] + ask_levels
].notna().sum(axis=1)

print("\nBid levels:")
print(lob["bid_levels"].describe())

print("\nAsk levels:")
print(lob["ask_levels"].describe())

print("\nRows with fewer than 20 bid levels:")
print(
    (lob["bid_levels"] < 20).sum()
)

print("\nRows with fewer than 20 ask levels:")
print(
    (lob["ask_levels"] < 20).sum()
)

# Crossed book
crossed = (
    lob["book_state__best_bid_price"]
    >= lob["book_state__best_ask_price"]
)

print("\nCrossed books:")
print(crossed.sum())

# Show worst examples
print("\nWorst 20 spreads:")
print(
    lob[
        [
            "timestamp",
            "lob__event_time",
            "lob__final_update_id",
            "book_state__best_bid_price",
            "book_state__best_ask_price",
            "spread",
            "spread_ticks",
            "bid_levels",
            "ask_levels",
        ]
    ]
    .sort_values("spread", ascending=False)
    .head(20)
    .to_string(index=False)
)