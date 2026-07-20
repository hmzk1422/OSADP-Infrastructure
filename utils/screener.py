import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from .indicators import get_wilders_rsi, sma
from .bybit import BybitConnector


def trend_persistency(df):
    close = df["close"].values
    delta = np.diff(close, prepend=close[0])

    rsi_14 = get_wilders_rsi(delta, 14)
    sma_14 = sma(rsi_14, 14)

    rsi_sma_dist = rsi_14 - sma_14
    bias = sma(rsi_sma_dist, 200)

    return bias, rsi_sma_dist


def fetch_single_ticker(connector, ticker):
    try:
        df = connector.get_ohlcv(ticker, "1")
        if df is None or len(df) < 300:
            print(f"Skipping {ticker}: Insufficient data ({len(df) if df is not None else 0})")
            return ticker, None, None, None

        persistency, rsi_sma_dist = trend_persistency(df)
        latest_bias = persistency[-1]

        if np.isnan(latest_bias):
            print(f"Skipping {ticker}: Bias is NaN")
            return ticker, None, None, None

        bias_std2 = 2 * np.nanstd(rsi_sma_dist[~np.isnan(rsi_sma_dist)])

        if np.isnan(bias_std2):
            print(f"Skipping {ticker}: bias_std2 is NaN")
            return ticker, None, None, None

        return ticker, df, latest_bias, bias_std2
    except Exception as e:
        print(f"Error on {ticker}: {e}")
        return ticker, None, None, None


def rank_bias(screened_data):
    data_list = [
        {"ticker": ticker, "bias": data["bias"], "bias_std2": data["bias_std2"]}
        for ticker, data in screened_data.items()
    ]
    df_rank = pd.DataFrame(data_list)
    df_rank["abs_bias"] = df_rank["bias"].abs()
    return df_rank.sort_values(by="abs_bias", ascending=False)


def _worker_factory(connector_class, api_key, api_secret, testnet):
    def worker(ticker):
        local_conn = connector_class(api_key=api_key, api_secret=api_secret, testnet=testnet)
        return fetch_single_ticker(local_conn, ticker)
    return worker


def screener(connector_input):
    if isinstance(connector_input, type):
        connector_class = connector_input
        temp_conn = connector_class()
    else:
        connector_class = connector_input.__class__
        temp_conn = connector_input

    all_tickers = temp_conn.get_all_tickers()
    linear_tickers = [t for t in all_tickers if "/USDT:USDT" in t and "-" not in t.split("/USDT:USDT")[1]]

    print(f"Total tickers: {len(all_tickers)} | After filtering: {len(linear_tickers)}")

    api_key = getattr(temp_conn, "api_key", "")
    api_secret = getattr(temp_conn, "api_secret", "")
    testnet = getattr(temp_conn, "testnet", False)

    worker = _worker_factory(connector_class, api_key, api_secret, testnet)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(worker, linear_tickers))

    screened_data = {
        ticker: {"df": df, "bias": bias, "bias_std2": bias_std2}
        for ticker, df, bias, bias_std2 in results
        if df is not None
        and not np.isnan(bias)
        and bias_std2 is not None
        and not np.isnan(bias_std2)
    }

    return screened_data