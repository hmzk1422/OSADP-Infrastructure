import sys
import asyncio
from utils.bybit import BybitConnector


def run():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    client = BybitConnector(api_key="...", api_secret="...", testnet=False)
    asyncio.run(client.get_live_orderflow("XAUTUSDT"))


if __name__ == "__main__":
    run()