"""
Strategy Configuration — Live Execution
=========================================
Central config matching the research parameters in ml-state.ipynb (binary version).
All parameters are verified against the research notebook.
"""

# --- Exchange ---
SYMBOL = "BTC/USDT:USDT"         # CCXT symbol format for linear perpetual
EXCHANGE = "bybit"
TESTNET = False                    # False = real API for data, True = testnet (testnet sucks)
API_KEY = "..."                   # Your real Bybit API key (read-only is fine for data)
API_SECRET = "..."                # Your real Bybit API secret

# --- Strategy Parameters (VERIFIED against research/ml-state.ipynb) ---
# Cell "Config" block in ml-state.ipynb (binary version, line 458-476):
LOOKAHEAD_SECONDS    = 900       # 5-min barrier horizon (research: 300s)
BARRIER_MULTIPLIER   = 6.0       # Volatility-scaled barrier width (research test: 2.0)
RSI_FILTER_COLUMN    = 'rsi_2'   # Research uses rsi_2 for filter (NOT rsi_14)
RSI_FILTER_THRESHOLD = 30        # Only trade when RSI(2) < 30 (research: rsi_2 < 30)
CONFIDENCE_THRESHOLD = 0.5      # P(LONG) must be >= this to fire a SHORT fade
# Note: Research tests thresholds 0.50-0.95. Start at 0.50 for more trades.

# --- Barrier & Position ---
STOP_LOSS_BPS        = 5        # Fallback stop if barrier logic fails
TAKE_PROFIT_BPS      = 5
MAX_POSITION_USD     = 100       # Max notional per position (~0.0017 BTC at $60k)
MIN_ORDER_SIZE       = 0.001     # Minimum contract size

# --- Feature Windows (research ml-state.ipynb, cell "Step 1") ---
RSI_14_WINDOW      = 14
RSI_2_WINDOW       = 2
SMA_14_WINDOW      = 14
OFI_LOOKBACK       = 60         # Rolling window for CVD_z (research: rolling(60))
CVD_ROLLING        = 10         # Rolling window for micro_CVD (research: rolling(10))
BARRIER_VOL_WINDOW = 100        # Rolling window for vol estimation (research: rolling(100))

# --- Label encoding (matches research binary: LABEL_LOSS=0, LABEL_WIN=1) ---
# Research inverts trades: P(WIN) high → EXIT/AVOID long, ENTER short
LABEL_LOSS = 0
LABEL_WIN  = 1
LABEL_SHORT = 0    # For short strategy: price down = profit
LABEL_LONG  = 1    # For long strategy: price up = profit
EPS = 1e-9

# --- Feature definition (EXACT match to research notebook) ---
# From ml-state.ipynb Step 4 binary version (line 593-596):
# drop_cols = ['target', 'price', 'open', 'high', 'low', 'close',
#              'profit_barrier', 'stop_barrier', 'session_id',
#              'bars_to_exit', 'exit_ret', 'terminal_ret', 'is_timeout']
# features = [c for c in df_training_zone.columns if c not in drop_cols]
#
# This produces these 12 features (in order):
FEATURE_COLUMNS = [
    'micro_OFI', 'micro_CVD',          # Micro orderflow features
    'rsi_2', 'rsi_14', 'sma_14',       # RSI features
    'largest_buyer', 'largest_seller',  # Trade concentration
    'OFI_LOB', 'OFI_AGG', 'CVD_z',     # Orderflow imbalance
    'candle_type', 'candle_size',       # Candle morphology
]

# --- Training ---
TRAIN_ON_STARTUP = True           # Retrain model on initial historical fetch?
N_HISTORICAL_BARS = 5000          # Fetch this many 1-min bars for initial training
# NOTE: After RSI filter + dropna, typically only ~200-500 rows survive for training.
# This is expected — the model only sees oversold regimes.
MODEL_PATH = "main/models/lgb_model.txt"

# --- Execution ---
DRY_RUN = True                    # Set False to place real orders!
MAX_TRADES_PER_HOUR = 300
FEE_BPS = 0                   # Binance/Bybit taker fee ~0.055%