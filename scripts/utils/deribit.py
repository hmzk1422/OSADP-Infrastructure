"""
Deribit Orderflow Connector

Mirrors the structure of `BybitConnector` in pipelines/ccxt_exchange.py, scoped
down to what a market-making research pipeline actually needs: trades,
order book snapshots, live streaming of both, and funding rate (for perps,
useful as an MM signal/bias input).

Symbol conventions (CCXT unified format on Deribit):
    - Inverse perpetual:  'BTC/USD:BTC', 'ETH/USD:ETH'
    - Linear (USDC) perp: 'BTC/USDC:USDC', 'ETH/USDC:USDC'
    - Options:             'BTC/USD:BTC-250926-100000-C'  (unified option symbol)
Use `get_all_tickers()` to confirm exact live symbols rather than hardcoding.

Note on scope: for prototyping a market-making method, CCXT's snapshot-based
watch_order_book is sufficient. If you later need tick-perfect L2 book
reconstruction (sequence-numbered deltas), Deribit's native WS `book` channel
gives you that directly -- CCXT abstracts it away. Swap that one piece out
if/when you need it; everything else here can stay as-is.

Dependencies:
    - ccxt / ccxt.pro
    - pandas
"""

import ccxt.pro as ccxtpro
import ccxt
import pandas as pd
import time
import os
import csv
import asyncio


class DeribitConnector:
    """
    Interface for Deribit API data retrieval using CCXT, scoped to orderflow
    data for market-making research (trades, order book, funding).
    """

    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        if not api_key or "..." in str(api_key):
            self.exchange = ccxt.deribit({'enableRateLimit': True})
        else:
            self.exchange = ccxt.deribit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })

        if testnet:
            self.exchange.set_sandbox_mode(True)

        self.exchange.options['adjustForTimeDifference'] = True
        try:
            self.exchange.load_time_difference()
        except ccxt.BaseError as e:
            print(f"Warning: could not sync time with exchange: {e}")

    def get_ohlcv(self, symbol: str, interval: str, total_limit: int = 10000) -> pd.DataFrame:
        """
        Fetches historical OHLCV data and returns a cleaned pandas DataFrame.
        Symbol must use CCXT unified syntax, e.g. 'BTC/USD:BTC' for the
        inverse perpetual.
        """
        all_data = []
        max_per_request = 1000  # Deribit's practical per-request cap via CCXT

        try:
            timeframe_ms = self.exchange.parse_timeframe(interval) * 1000
        except Exception:
            raise ValueError(f"Invalid CCXT interval format: {interval}")

        buffer_limit = int(total_limit * 1.05)
        since = self.exchange.milliseconds() - (buffer_limit * timeframe_ms)

        while len(all_data) < buffer_limit:
            remaining = buffer_limit - len(all_data)
            current_limit = min(remaining, max_per_request)

            try:
                batch = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=interval,
                    since=since,
                    limit=current_limit
                )
            except ccxt.BaseError as e:
                raise ConnectionError(f"CCXT / Deribit API Error: {e}")

            if not batch:
                break

            all_data.extend(batch)
            since = batch[-1][0] + 1
            time.sleep(self.exchange.rateLimit / 1000)

        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_data, columns=columns)

        df = df.drop_duplicates(subset=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df.sort_values('timestamp')

        return df.tail(total_limit).reset_index(drop=True)

    def get_orderflow_historical(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetches recent public execution logs (Time & Sales) for orderflow analysis.
        """
        try:
            # Deribit's native cap via CCXT is 1000 trades per request
            trades = self.exchange.fetch_trades(symbol=symbol, limit=limit)
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT Trade Fetch Error: {e}")

        if not trades:
            return pd.DataFrame()

        formatted_trades = []
        for t in trades:
            formatted_trades.append({
                'timestamp': pd.to_datetime(t['timestamp'], unit='ms'),
                'id': t['id'],
                'side': t['side'],      # 'buy' (taker bought) or 'sell' (taker sold)
                'price': float(t['price']),
                'amount': float(t['amount']),
                'cost': float(t['cost']) if t.get('cost') is not None else float(t['price']) * float(t['amount']),
                'takerOrMaker': t.get('takerOrMaker'),
            })

        df = pd.DataFrame(formatted_trades)
        df['volume_delta'] = df.apply(lambda row: row['amount'] if row['side'] == 'buy' else -row['amount'], axis=1)
        return df

    def get_orderbook_snapshot(self, symbol: str, depth: int = 50) -> dict:
        """
        Fetches a single L2 order book snapshot. Useful for spot-checking
        spread/depth or seeding a local book before switching to streaming.
        """
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=depth)
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT Order Book Fetch Error: {e}")
        return ob

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
        ob_file = filename.replace('_trades.csv', '_orderbook.csv')
        if not os.path.exists(ob_file):
            with open(ob_file, 'w', newline='') as f:
                csv.writer(f).writerow(['timestamp', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount'])

        while True:
            try:
                orderbook = await exchange.watch_order_book(symbol)
                best_bid = orderbook['bids'][0] if orderbook['bids'] else [0, 0]
                best_ask = orderbook['asks'][0] if orderbook['asks'] else [0, 0]

                with open(ob_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([orderbook['timestamp'], best_bid[0], best_bid[1], best_ask[0], best_ask[1]])
            except Exception as e:
                print(f"Error in orderbook logger: {e}")
                await asyncio.sleep(5)

    async def get_live_orderflow(self, symbol):
        """
        Streams live trades + top-of-book to CSV concurrently, same pattern
        as BybitConnector.get_live_orderflow. Deribit's WS gateway doesn't
        have the aiodns quirk Bybit's does, but the ThreadedResolver is kept
        here too for consistency and to sidestep any DNS resolver issues in
        containerized environments.
        """
        import aiohttp

        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())

        exchange = ccxtpro.deribit({
            'enableRateLimit': True,
        })

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

    def get_funding_rate(self, symbol: str, since: int = None, limit: int = 1000) -> pd.DataFrame:
        """
        Funding rate history. Only applies to perpetual swap symbols
        (e.g. 'BTC/USD:BTC') -- will raise/return empty for options.
        Useful as a directional bias input for a MM method.
        """
        try:
            data = self.exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT Funding Rate History Error: {e}")
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_all_tickers(self, category: str = "swap") -> list:
        """
        Returns a list of all active CCXT unified symbol names.
        category: 'swap' (perpetuals), 'option', or 'future' (dated futures).
        """
        try:
            markets = self.exchange.load_markets()
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT / Deribit API Error: {e}")

        active_symbols = []
        for symbol, market_info in markets.items():
            if market_info['active'] and market_info.get('type') == category:
                active_symbols.append(symbol)

        return active_symbols


if __name__ == "__main__":
    # Quick smoke test -- adjust symbol as needed once you confirm it via
    # get_all_tickers(). No credentials required for public orderflow data.
    connector = DeribitConnector()
    print(connector.get_all_tickers(category="swap")[:10])