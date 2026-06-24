from utils.bybit import BybitConnector
from utils.engine import *

def run():
    # Now you can use the class
    client = BybitConnector(api_key="abc", api_secret="abc", testnet=False)
    price_a = client.get_ohlcv("BTCUSDT", "1")
    price_b = client.get_ohlcv("ETHUSDT", "1")
    spread, beta = get_kalman_spread(price_a['close'], price_b['close'])
    print(check_stationarity(beta))

    historical_performance = run_historical_strategy(beta, price_a['close'], price_b['close'])

    plot_pnl_history(historical_performance)

if __name__ == "__main__":
    run()