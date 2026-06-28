from utils.bybit import BybitConnector
from utils.engine import *

def run():
    # Now you can use the class
    client = BybitConnector(api_key="abc", api_secret="abc", testnet=False)
    price_a = client.get_ohlcv("BTCUSDT", "1")
    price_b = client.get_ohlcv("ETHUSDT", "1")

    plot_pnl_history(price_a)

if __name__ == "__main__":
    run()