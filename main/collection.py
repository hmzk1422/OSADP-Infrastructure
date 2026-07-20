from utils.bybit import BybitConnector


def run():
    client = BybitConnector(api_key="...", api_secret="...", testnet=False)
    tickers = client.get_all_tickers()
    for t in tickers:
        print(t)


if __name__ == "__main__":
    run()