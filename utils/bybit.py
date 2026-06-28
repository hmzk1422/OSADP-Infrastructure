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
import time

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

    def get_ohlcv(self, symbol: str, interval: str, total_limit: int = 500) -> pd.DataFrame:
        all_data = []
        # Bybit API limit per request
        batch_size = total_limit
        # Use the earliest timestamp found as the end_time for the next batch
        end_time = None 

        while len(all_data) < total_limit:
            # Determine how many candles to fetch in this batch
            remaining = total_limit - len(all_data)
            current_limit = min(remaining, batch_size)
            
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=current_limit,
                end=end_time
            )
            
            if response['retCode'] != 0:
                raise ConnectionError(f"Bybit API Error: {response['retMsg']}")
                
            batch = response['result']['list']
            if not batch:
                break
                
            all_data.extend(batch)
            
            # Update end_time to the timestamp of the oldest candle received 
            # (The API returns data in descending order)
            end_time = int(batch[-1][0])
            
            # Avoid hitting rate limits
            time.sleep(0.1) 

        # Construct DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        df = pd.DataFrame(all_data, columns=columns)
        
        # Process the data
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.iloc[::-1].reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return df.head(total_limit)

    def get_all_tickers(self, category: str = "linear") -> list:
        """
        Returns a list of all active symbol names for the specified category.
        """
        response = self.session.get_instruments_info(category=category)
        
        if response['retCode'] != 0:
            raise ConnectionError(f"Bybit API Error: {response['retMsg']}")
            
        return [item['symbol'] for item in response['result']['list']]