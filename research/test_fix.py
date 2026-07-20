import pandas as pd
import numpy as np

# Test: Load all 3 sessions fully with low_memory=False
print('=== Full pipeline test ===')
def load_and_build_state_space(orderbook_path, trades_path, session_id):
    df_orderbook = pd.read_csv(orderbook_path, low_memory=False)
    df_trades = pd.read_csv(trades_path, low_memory=False)
    
    # Convert ALL numeric columns to numeric, coercing errors (handles malformed rows)
    num_cols_ob = ['timestamp', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount']
    num_cols_tr = ['timestamp', 'price', 'amount']
    for c in num_cols_ob:
        df_orderbook[c] = pd.to_numeric(df_orderbook[c], errors='coerce')
    for c in num_cols_tr:
        df_trades[c] = pd.to_numeric(df_trades[c], errors='coerce')
    
    df_orderbook['datetime'] = pd.to_datetime(df_orderbook['timestamp'] // 1000, unit='s')
    df_trades['datetime'] = pd.to_datetime(df_trades['timestamp'] // 1000, unit='s')
    
    df_orderbook = df_orderbook.set_index('datetime').sort_index()
    df_trades = df_trades.set_index('datetime').sort_index()
    
    # Drop any rows with NaN timestamps (malformed rows)
    df_orderbook = df_orderbook.dropna(subset=['timestamp'])
    df_trades = df_trades.dropna(subset=['timestamp'])
    
    df_orderbook['mid_price'] = (df_orderbook['bid_price'] + df_orderbook['ask_price']) / 2
    
    df_ob_1m = df_orderbook.resample('1min').agg({
        'mid_price': ['first', 'max', 'min', 'last'],
        'bid_price': 'last',
        'ask_price': 'last',
        'bid_amount': 'sum',
        'ask_amount': 'sum',
    })
    df_ob_1m.columns = ['open', 'high', 'low', 'close', 'bid_price', 'ask_price', 'bid_amount', 'ask_amount']
    df_ob_1m['spread'] = df_ob_1m['ask_price'] - df_ob_1m['bid_price']
    
    df_trades['buy_vol'] = df_trades['amount'].where(df_trades['side'] == 'buy', 0)
    df_trades['sell_vol'] = df_trades['amount'].where(df_trades['side'] == 'sell', 0)
    
    df_micro = df_trades.resample('1s').agg({
        'price': 'last',
        'buy_vol': 'sum',
        'sell_vol': 'sum'
    }).ffill()
    
    df_trades_1m = df_trades.resample('1min').agg({
        'price': ['first', 'max', 'min', 'last', 'count'],
        'amount': 'sum',
        'buy_vol': ['sum', 'max'],
        'sell_vol': ['sum', 'max']
    })
    df_trades_1m.columns = ['open', 'high', 'low', 'close', 'num_trades', 'volume',
                             'agg_buyer', 'largest_buyer', 'agg_seller', 'largest_seller']
    df_trades_1m = df_trades_1m.dropna(subset=['close'])
    
    df_price_1m = pd.DataFrame({
        'open': df_ob_1m['open'],
        'high': df_ob_1m['high'],
        'low': df_ob_1m['low'],
        'close': df_ob_1m['close'],
        'largest_buyer': df_trades_1m['largest_buyer'] * 100 / (df_trades_1m['volume'] + 1e-9),
        'largest_seller': df_trades_1m['largest_seller'] * 100 / (df_trades_1m['volume'] + 1e-9),
    })
    
    df_state_space = df_micro.shift(1).join(df_price_1m.shift(1), how='left').ffill()
    df_state_space = df_state_space.dropna()
    df_state_space['session_id'] = session_id
    return df_state_space

s0 = load_and_build_state_space('./csv/BTCUSDT_orderbook.csv', './csv/BTCUSDT_trades.csv', 0)
print(f'Session 0 rows: {len(s0)}')
s1 = load_and_build_state_space('./csv/BTCUSDT_orderbook_2.csv', './csv/BTCUSDT_trades_2.csv', 1)
print(f'Session 1 rows: {len(s1)}')
s2 = load_and_build_state_space('./csv/BTCUSDT_orderbook_3.csv', './csv/BTCUSDT_trades_3.csv', 1)
print(f'Session 2 rows: {len(s2)}')
df_all = pd.concat([s0, s1, s2]).sort_index()
print(f'Total combined rows: {len(df_all)}')
print('PASS: Full pipeline completed without errors!')