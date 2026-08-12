"""Regression coverage for chunk-safe historical feature construction."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import polars as pl

from research.ml_state_optimized import (
    TIMEFRAME_BARRIER,
    build_session_feature_context,
    load_and_build_state_space,
    partition_state_space,
)


class ChunkInvariantRsiTest(unittest.TestCase):
    def test_csv_chunks_preserve_continuous_feature_state(self) -> None:
        """Chunking must not change the state space, including data gaps."""
        bars_per_chunk = 40
        chunk_count = 3
        base_time = datetime(2024, 1, 1)
        total_bars = bars_per_chunk * chunk_count
        gap_start, gap_end = 45, 85
        raw_indices = [index for index in range(total_bars) if not gap_start <= index < gap_end]
        bar_times = [
            base_time + timedelta(seconds=(index * TIMEFRAME_BARRIER) + 1)
            for index in raw_indices
        ]
        epoch = datetime(1970, 1, 1)
        timestamps = [int((value - epoch).total_seconds() * 1000) for value in bar_times]
        all_prices = 100 + np.cumsum(np.sin(np.arange(total_bars) / 2.7) + 0.15)
        prices = all_prices[raw_indices]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            orderbook_path = temp_path / "orderbook.csv"
            trades_path = temp_path / "trades.csv"
            pl.DataFrame({
                "timestamp": timestamps,
                "bid_price": prices - 0.01,
                "ask_price": prices + 0.01,
                "bid_amount": np.full(len(prices), 2.0),
                "ask_amount": np.full(len(prices), 2.0),
            }).write_csv(orderbook_path)
            pl.DataFrame({
                "timestamp": timestamps,
                "price": prices,
                "amount": np.full(len(prices), 0.25),
                "side": np.where(np.arange(len(prices)) % 2, "buy", "sell"),
            }).write_csv(trades_path)

            context = build_session_feature_context(str(orderbook_path), str(trades_path))
            whole_session = load_and_build_state_space(
                str(orderbook_path), str(trades_path), session_id=0, session_features=context
            )

            # A raw-event gap is the case that broke parity in the real CSVs:
            # full-session resampling carries the continuous state through it.
            # Partitioning the completed state space must preserve every row.
            partitions = partition_state_space(
                whole_session,
                base_time,
                base_time + timedelta(seconds=total_bars * TIMEFRAME_BARRIER),
                chunk_count,
            )
            reassembled = pl.concat(partitions).sort("datetime")
            self.assertEqual(reassembled.height, whole_session.height)
            self.assertEqual(reassembled["datetime"].to_list(), whole_session["datetime"].to_list())
            gap_timestamp = base_time + timedelta(seconds=(gap_start * TIMEFRAME_BARRIER) + 60)
            self.assertIn(gap_timestamp, reassembled["datetime"].to_list())


if __name__ == "__main__":
    unittest.main()
