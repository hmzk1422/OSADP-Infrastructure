import ccxt.pro as ccxtpro
import ccxt
import pandas as pd
import numpy as np
import time
import os
import csv
import asyncio
import aiohttp


class BybitConnector:
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        options = {
            'enableRateLimit': True,
            'options': {'fetchCurrencies': False},
        }

        if not api_key or "..." in str(api_key):
            self.exchange = ccxt.bybit(options)
        else:
            self.exchange = ccxt.bybit({
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
        all_data = []
        batch_size = total_limit
        since = None

        while len(all_data) < total_limit:
            remaining = total_limit - len(all_data)
            current_limit = min(remaining, batch_size)

            try:
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
            since = batch[-1][0] + 1
            time.sleep(0.1)

        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_data, columns=columns)

        df = df.drop_duplicates(subset=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        return df.head(total_limit)

    def get_orderflow_historical(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        try:
            trades = self.exchange.fetch_trades(symbol=symbol, limit=limit)
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT Trade Fetch Error: {e}")

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['price'] = df['price'].astype(float)
        df['amount'] = df['amount'].astype(float)
        df['cost'] = df['cost'].astype(float)
        df['volume_delta'] = np.where(df['side'] == 'buy', df['amount'], -df['amount'])

        return df[['timestamp', 'id', 'side', 'price', 'amount', 'cost', 'takerOrMaker', 'volume_delta']]

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
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())

        exchange = ccxtpro.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'fetchMarkets': ['linear'],
            }
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
        try:
            data = self.exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT Funding Rate History Error: {e}")
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_open_interest(self, symbol: str, timeframe: str = '1h', since: int = None, limit: int = 1000) -> pd.DataFrame:
        try:
            data = self.exchange.fetch_open_interest_history(symbol, timeframe=timeframe, since=since, limit=limit)
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT Open Interest History Error: {e}")
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_all_tickers(self, category: str = "linear") -> list:
        try:
            markets = self.exchange.load_markets()
        except ccxt.BaseError as e:
            raise ConnectionError(f"CCXT / Bybit API Error: {e}")

        active_symbols = []
        for symbol, market_info in markets.items():
            if market_info['active']:
                if category == "linear" and market_info.get('linear'):
                    active_symbols.append(symbol)
                elif category == "spot" and market_info.get('spot'):
                    active_symbols.append(symbol)
                elif category == "inverse" and market_info.get('inverse'):
                    active_symbols.append(symbol)

        return active_symbols