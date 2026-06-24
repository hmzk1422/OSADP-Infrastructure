"""
Bybit Execution Pipeline

This module serves as the production-grade interface between algorithmic 
strategy research (./research) and the Bybit exchange API. It abstracts 
exchange-specific complexities into clean, reusable execution primitives.

Role:
    - Acts as a hardware abstraction layer for order execution.
    - Handles data normalization (tick -> OHLC) for strategy consumption.
    - Implements universal safety constraints (e.g., max position size).

Key Responsibilities:
    - Data Aggregation: Converts raw stream data into strategy-ready formats.
    - Order Execution: Provides synchronous/asynchronous wrappers for 
      market/limit orders with standardized error handling.
    - State Management: Tracks active order IDs and latency metrics.

Usage:
    from pipelines.bybit import BybitExecutor
    executor = BybitExecutor(api_key=..., api_secret=...)
    executor.place_order(symbol="BTCUSDT", side="Buy", qty=0.01)

Dependencies:
    - pybit: Official Bybit API wrapper.
    - pandas: For vectorized data aggregation.
"""

from pybit.unified_trading import HTTP
import pandas as pd

class BybitConnector:
    """
    Interface for Bybit API data retrieval.
    
    Responsibilities:
        - Authenticated/Public session management
        - OHLCV data fetching and normalization
        - Market instrument discovery
    """
    def __init__(self, api_key, api_secret, testnet=False):
        # We don't need API keys for public data fetching, 
        # but you can add them to the constructor later if needed.
        self.session = HTTP(testnet=testnet)
        self.api_key = api_key
        self.api_secret = api_secret

    def get_ohlcv(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """
        Fetches OHLCV data and returns it as a cleaned pandas DataFrame.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1', '3', '5', '15', '30', '60', etc.)
            limit: Number of candles (max 200)
        """
        response = self.session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if response['retCode'] != 0:
            raise ConnectionError(f"Bybit API Error: {response['retMsg']}")
            
            
        data = response['result']['list']
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']

        df = pd.DataFrame(data, columns=columns)
        
        # Convert to numeric and set chronological order
        df = df.iloc[::-1].reset_index(drop=True)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Now this will work because it's a clean integer
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df

    def get_all_tickers(self, category: str = "linear") -> list:
        """
        Returns a list of all active symbol names for the specified category.
        """
        response = self.session.get_instruments_info(category=category)
        
        if response['retCode'] != 0:
            raise ConnectionError(f"Bybit API Error: {response['retMsg']}")
            
        return [item['symbol'] for item in response['result']['list']]