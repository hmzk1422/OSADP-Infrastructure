from utils.bybit import BybitConnector
from utils.engine import *
import asyncio
import sys

def run():
    # aiodns (used by aiohttp/ccxt.pro) is incompatible with Windows'
    # default ProactorEventLoop. SelectorEventLoop resolves this.
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    client = BybitConnector(api_key="...", api_secret="...", testnet=False)
    asyncio.run(client.get_live_orderflow("BEATUSDT"))

if __name__ == "__main__":
    run()