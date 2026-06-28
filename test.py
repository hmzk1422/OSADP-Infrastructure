# Libraries needed to run the code
from utils.bybit import BybitConnector
from utils.engine import *
from utils.indicators import *

client = BybitConnector(api_key="", api_secret="", testnet=False)
#price = client.get_ohlcv("BTC/USDT:USDT", "1", 200)
price = client.get_ohlcv("BTC/USDT:USDT", "1", 10000)

import pandas as pd

# Fetch the data
ohlcv_data = client.get_ohlcv("BTC/USDT:USDT", "1", 10000)

# Define column names
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# Convert to DataFrame
df = pd.DataFrame(ohlcv_data, columns=columns)

# Convert timestamp to readable format (optional)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Write to CSV
df.to_csv("btc_ohlcv_data.csv", index=False)