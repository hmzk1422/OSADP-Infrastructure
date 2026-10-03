import time
from typing import Optional, Union
import ccxt
import pandas as pd


def get_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    since: Optional[Union[str, int]] = None,
    until: Optional[Union[str, int]] = None,
    exchange_id: str = "binance",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetches historical OHLCV data, auto-paginating through CCXT limits.

    :param symbol: Trading pair (e.g., 'BTC/USDT')
    :param timeframe: Candle size ('1m', '5m', '1h', '1d')
    :param since: Start date as ISO string ("2024-01-01T00:00:00Z") or epoch ms
    :param until: End date as ISO string ("2024-02-01T00:00:00Z") or epoch ms
    :param exchange_id: CCXT exchange ID ('binance', 'bybit', etc.)
    :param limit: Max candles per request (default 1000)
    :return: DataFrame indexed by UTC datetime
    """
    # Initialize exchange
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT.")

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    # Parse timestamps to epoch milliseconds
    current_since = (
        exchange.parse8601(since) if isinstance(since, str) else since
    )
    until_ms = exchange.parse8601(until) if isinstance(until, str) else until

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    all_candles = []

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit,
            )
        except ccxt.BaseError as e:
            print(f"[cryptoL1/price] API error: {e}. Retrying...")
            time.sleep(exchange.rateLimit / 1000)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]

        # Stop conditions
        if until_ms and last_ts >= until_ms:
            break

        # Move timeframe cursor past the last retrieved bar
        next_since = last_ts + timeframe_ms
        if current_since is not None and next_since <= current_since:
            break
        current_since = next_since

        # Reached current live bar if exchange returned fewer than requested
        if len(candles) < limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Format into DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    df.drop_duplicates(subset=["timestamp"], inplace=True)

    if until_ms:
        df = df[df["timestamp"] <= until_ms]

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)

    return df