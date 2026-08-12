from utils.bybit import BybitConnector
from utils.engine import *
import asyncio
import sys

async def main(tickers):
    client = BybitConnector(api_key="...", api_secret="...", testnet=False)
    
    # Create concurrent coroutines for every ticker in the list
    tasks = [client.get_live_orderflow(ticker) for ticker in tickers]
    
    # Gather and run all streams simultaneously
    await asyncio.gather(*tasks)

def run():
    # Fix for Windows selector event loop (required for aiodns / ccxt.pro)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Define your target tickers here
    #tickers = ["ETHUSDT", "SOLUSDT", "LINKUSDT", "ARBUSDT", "OPUSDT", "XAUTUSDT", "BEATUSDT"]
    tickers = ["BTCUSDT"]

    asyncio.run(main(tickers))

if __name__ == "__main__":
    run()