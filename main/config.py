"""
Strategy Configuration — Live Execution
=========================================
Central config matching the research parameters in corrected_state.py.
"""

# --- Exchange ---
SYMBOL = "BTC/USDT:USDT"         # CCXT symbol format for linear perpetual
EXCHANGE = "bybit"
TESTNET = False                    # False = real API for data, True = testnet (testnet sucks)
API_KEY = "..."                   # Your real Bybit API key (read-only is fine for data)
API_SECRET = "..."                # Your real Bybit API secret

# --- Strategy Parameters (mirrors corrected_state.py) ---
LOOKAHEAD_SECONDS    = 240       # 4-min barrier horizon
BARRIER_MULTIPLIER   = 1.0       # Volatility-scaled barrier width
RSI_FILTER_COLUMN    = 'rsi_14'
RSI_FILTER_THRESHOLD = 30        # Only trade when RSI(14) < oversold
CONFIDENCE_THRESHOLD = 0.85      # P(LONG) must be >= this to fire a SHORT fade

# --- Barrier & Position ---
STOP_LOSS_BPS        = 30        # Fallback stop if barrier logic fails
TAKE_PROFIT_BPS      = 30
MAX_POSITION_USD     = 100       # Max notional per position
MIN_ORDER_SIZE       = 0.001     # Minimum contract size

# --- Feature Windows ---
RSI_14_WINDOW   = 14
RSI_2_WINDOW    = 2
SMA_14_WINDOW   = 14
OFI_LOOKBACK    = 60
CVD_ROLLING     = 10
BARRIER_VOL_WINDOW = 100         # Rolling window for vol estimation

# --- Label encoding (INVERTED FOR SHORT STRATEGY) ---
LABEL_SHORT = 0                   # Price goes DOWN = profit for short
LABEL_LONG  = 1                   # Price goes UP = profit for long

# --- Training ---
TRAIN_ON_STARTUP = True           # Retrain model on initial historical fetch?
N_HISTORICAL_BARS = 5000          # Fetch this many 1-min bars for initial training
MODEL_PATH = "main/models/lgb_model.txt"

# --- Execution ---
DRY_RUN = True                    # Set False to place real orders!
MAX_TRADES_PER_HOUR = 2
FEE_BPS = 0.055                   # Binance/Bybit taker fee ~0.055%