from utils.bybit import BybitConnector
from utils.engine import *
import asyncio
import sys

def run():
    # aiodns (used by aiohttp/ccxt.pro) is incompatible with Windows'
    # default ProactorEventLoop. SelectorEventLoop resolves this.
    client = BybitConnector(api_key="...", api_secret="...", testnet=False)
    ff = client.get_funding_rate("BTCUSDT")
    oi = client.get_open_interest("BTCUSDT")

    ff.to_csv("ff.csv")
    oi.to_csv("oi.csv")
    

if __name__ == "__main__":
    run()