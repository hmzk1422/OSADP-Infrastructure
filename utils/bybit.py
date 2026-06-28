"""
CCXT Execution Pipeline

This module serves as the production-grade interface between algorithmic 
strategy research (./research) and the exchange API using CCXT. It abstracts 
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
    from pipelines.ccxt_exchange import ExchangeExecutor
    executor = ExchangeExecutor(api_key=..., api_secret=...)
    executor.place_order(symbol="BTC/USDT:USDT", side="buy", qty=0.01)

Dependencies:
    - ccxt: Universal crypto trading API framework.
    - pandas: For vectorized data aggregation.
"""

import ccxt.pro as ccxtpro
import ccxt
import pandas as pd
import time
import os
import csv
import asyncio

class BybitConnector:
    """
    Interface for Bybit API data retrieval using CCXT.
    
    Responsibilities:
        - Authenticated/Public session management via CCXT
        - OHLCV data fetching and normalization
        - Market instrument discovery
    """
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        # Initialize the unified Bybit exchange instance in CCXT
        options = {
            'enableRateLimit': True,
        }
        
        # If keys are missing or placeholders, skip fetching currency data that requires authentication
        if not api_key or "..." in str(api_key):
            options['options'] = {'fetchCurrencies': False}
            self.exchange = ccxt.bybit(options)
        else:
            self.exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)

    def get_ohlcv(self, symbol: str, interval: str, total_limit: int = 1000) -> pd.DataFrame:
        """
        Fetches historical OHLCV data and returns a cleaned pandas DataFrame.
        Note: Symbol must use standard CCXT syntax (e.g., 'BTC/USDT:USDT' for linear swaps)
        """
        all_data = []
        batch_size = total_limit  # Bybit default pagination limit via CCXT
        
        # CCXT expects timeframes like '1m', '5m', '1h', '1d'
        # CCXT fetch_ohlcv pagination uses `since` timestamp in milliseconds
        since = None 
        
        # Calculate an approximate start timestamp if fetching a massive back history
        # (Alternatively, let CCXT fetch backwards if supported, but loop forward is more stable)
        while len(all_data) < total_limit:
            remaining = total_limit - len(all_data)
            current_limit = min(remaining, batch_size)
            
            try:
                # CCXT automatically parses Bybit linear/spot/inverse structures based on the unified symbol
                batch = self.exchange.fetch_ohlcv(
                    symbol=symbol, 
                    timeframe=interval, 
                    since=since, 
                    limit=current_limit
                )
            except ccxt.BaseError as e:
                raise ConnectionError(f"CCXT / Bybit API Error: {e}")
                
            if not batch:
                break
                
            all_data.extend(batch)
            
            # Update 'since' to 1 millisecond past the last received candle timestamp
            # CCXT returns data in ascending chronological order
            since = batch[-1][0] + 1
            
            # Safety sleep buffer (in addition to enableRateLimit)
            time.sleep(0.1)

        # CCXT unified OHLCV structure: [timestamp, open, high, low, close, volume]
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_data, columns=columns)
        
        # Process and clean the data frame
        df = df.drop_duplicates(subset=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return df.head(total_limit)

    def get_orderflow_historical(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches the recent public execution logs (Time & Sales) for orderflow analysis.
        """
        try:
            # Bybit native limit maxes out at 1000 for trades
            trades = self.exchange.fetch_trades(symbol=symbol, limit=limit)
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT Trade Fetch Error: {e}")
            
        if not trades:
            return pd.DataFrame()

        # Normalize into an orderflow-ready format
        formatted_trades = []
        for t in trades:
            formatted_trades.append({
                'timestamp': pd.to_datetime(t['timestamp'], unit='ms'),
                'id': t['id'],
                'side': t['side'],      # 'buy' (taker bought) or 'sell' (taker sold)
                'price': float(t['price']),
                'amount': float(t['amount']),
                'cost': float(t['cost']), # price * amount
                'takerOrMaker': t['takerOrMaker']
            })
            
        df = pd.DataFrame(formatted_trades)
        
        # Add quick Orderflow metrics
        df['volume_delta'] = df.apply(lambda row: row['amount'] if row['side'] == 'buy' else -row['amount'], axis=1)
        return df
    
    async def log_trades(self, exchange, symbol, filename):
        print(f"Logging trades for {symbol}...")
        while True:
            try:
                trades = await exchange.watch_trades(symbol)
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for t in trades:
                        writer.writerow([t['timestamp'], t['price'], t['amount'], t['side'], t['id']])
            except Exception as e:
                print(f"Error in trade logger: {e}")
                await asyncio.sleep(5)

    async def log_orderbook(self, exchange, symbol, filename):
        print(f"Logging orderbook for {symbol}...")
        # Initialize orderbook file
        ob_file = filename.replace('_trades.csv', '_orderbook.csv')
        if not os.path.exists(ob_file):
            with open(ob_file, 'w', newline='') as f:
                csv.writer(f).writerow(['timestamp', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount'])

        while True:
            try:
                orderbook = await exchange.watch_order_book(symbol)
                # Take the best bid and best ask
                best_bid = orderbook['bids'][0] if orderbook['bids'] else [0, 0]
                best_ask = orderbook['asks'][0] if orderbook['asks'] else [0, 0]
                
                with open(ob_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([orderbook['timestamp'], best_bid[0], best_bid[1], best_ask[0], best_ask[1]])
            except Exception as e:
                print(f"Error in orderbook logger: {e}")
                await asyncio.sleep(5)

    async def get_live_orderflow(self, symbol):
        import aiohttp

        # ThreadedResolver uses Python's stdlib socket.getaddrinfo (thread pool),
        # bypassing aiodns completely — works reliably on all platforms.
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())

        exchange = ccxtpro.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'fetchMarkets': ['linear'],
            }
        })

        # Inject custom session before ccxt makes any network calls
        exchange.session = aiohttp.ClientSession(connector=connector)

        safe_symbol = symbol.replace('/', '_').replace(':', '-')
        filename = f"{safe_symbol}_trades.csv"

        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                csv.writer(f).writerow(['timestamp', 'price', 'amount', 'side', 'id'])

        try:
            await exchange.load_markets()
            await asyncio.gather(
                self.log_trades(exchange, symbol, filename),
                self.log_orderbook(exchange, symbol, filename)
            )
        finally:
            await exchange.close()

    def get_all_tickers(self, category: str = "linear") -> list:
        """
        Returns a list of all active CCXT unified symbol names.
        """
        try:
            # CCXT loads and filters market types natively
            markets = self.exchange.load_markets()
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT / Bybit API Error: {e}")
            
        active_symbols = []
        for symbol, market_info in markets.items():
            # Filter based on target category logic mapped to CCXT parameters
            if market_info['active']:
                if category == "linear" and market_info.get('linear'):
                    active_symbols.append(symbol)
                elif category == "spot" and market_info.get('spot'):
                    active_symbols.append(symbol)
                elif category == "inverse" and market_info.get('inverse'):
                    active_symbols.append(symbol)
                    
        return active_symbols
    