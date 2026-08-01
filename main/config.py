"""
Strategy Configuration — Live Execution
=========================================
Central config matching the research parameters in ml_state_optimized.py.
All parameters verified against the optimized research.
"""

# --- Exchange ---
SYMBOL = "BTC/USDT:USDT"         # CCXT symbol format for linear perpetual
EXCHANGE = "bybit"
TESTNET = False                    # False = real API for data, True = testnet (testnet sucks)
API_KEY = "..."                   # Your real Bybit API key (read-only is fine for data)
API_SECRET = "..."                # Your real Bybit API secret

# --- Strategy Parameters (VERIFIED against research/ml_state_optimized.py) ---
LOOKAHEAD_SECONDS    = 3000      # 10-min barrier horizon (research: 600s)
BARRIER_MULTIPLIER   = 0.5       # Volatility-scaled barrier width (research: 3.0)
RSI_FILTER_COLUMN    = 'rsi_2'   # Research uses rsi_2 for filter
RSI_FILTER_THRESHOLD = 30        # Only trade when RSI(2) < 30
CONFIDENCE_THRESHOLD = 0.5       # P(WIN) must be >= this to fire a trade
# Research tests thresholds 0.50-0.95. Start at 0.50 for more trades.

# --- Trade Direction ---
# 'short' = fade (current behavior) — bet AGAINST the model's P(WIN) prediction
# 'long'  = snapback — bet WITH the model's P(WIN) prediction (price bounces up)
TRADE_DIRECTION = 'short'        # 'short' or 'long'

# note for myself: based on result, inversing signal for macro asset dont work. only for small asset.

# --- Barrier & Position ---
STOP_LOSS_BPS        = 5        # Fallback stop if barrier logic fails
TAKE_PROFIT_BPS      = 5
MAX_POSITION_USD     = 100       # Max notional per position (~0.0017 BTC at $60k)
MIN_ORDER_SIZE       = 0.001     # Minimum contract size
RR_RATIO             = 1.5       # Risk:Reward ratio (2.0 = 1:2, profit target = 2x stop distance)

# --- Volatility / Barrier-Size Filter ---
# Reject entries when the volatility-scaled barrier is outside this range.
# barrier_pct = ret_std * sqrt(LOOKAHEAD_SECONDS) * BARRIER_MULTIPLIER
# MIN: reject if market is too dead (barrier too tight → no profit potential)
# MAX: reject if market is too wild (barrier too wide → uncontrollable risk)
# Set MIN=0.0 and MAX=1.0 to disable the filter entirely.
MIN_BARRIER_PCT = 0.0001        # 0.01% — reject if barrier smaller than this
MAX_BARRIER_PCT = 0.05          # 5%    — reject if barrier larger than this
TIMEFRAME_BARRIER = 300         # In unit seconds (5 minutes)

# --- Feature Windows (research ml_state_optimized.py) ---
RSI_14_WINDOW      = 14
RSI_2_WINDOW       = 2
SMA_14_WINDOW      = 14
OFI_LOOKBACK       = 60         # Rolling window for CVD_z (research: rolling(60))
CVD_ROLLING        = 10         # Rolling window for micro_CVD (research: rolling(10))
BARRIER_VOL_WINDOW = 100        # Rolling window for vol estimation (research: rolling(100))

# --- DeepLOB feature windows ---
DEEPLOB_Z_WINDOW    = 10        # Z-score window for spread_bps_z (research: window=10)
RSI_Z_WINDOW        = 20        # Z-score window for rsi_z (research: window=20)
MA_SLOPE_WINDOW     = 5         # MA slope window (research: 5)
PRICE_ROC_WINDOW_3  = 3         # Price rate of change window 3
PRICE_ROC_WINDOW_5  = 5         # Price rate of change window 5 (excluded in optimized)
RSI_ROC_WINDOW_3    = 3         # RSI rate of change window 3 (excluded in optimized)
RSI_ROC_WINDOW_5    = 5         # RSI rate of change window 5 (excluded in optimized)

# --- Label encoding (matches research binary: LABEL_LOSS=0, LABEL_WIN=1) ---
LABEL_LOSS = 0
LABEL_WIN  = 1
LABEL_SHORT = 0
LABEL_LONG  = 1
EPS = 1e-9

# --- Feature definition (EXACT match to ml_state_optimized.py Step 4) ---
# From ml_state_optimized.py lines 492-499:
# drop_cols = ['target', 'price', 'open', 'high', 'low', 'close',
#              'profit_barrier', 'stop_barrier', 'session_id',
#              'bars_to_exit', 'exit_ret', 'terminal_ret', 'is_timeout',
#              'execution_price',
#              'buy_vol', 'sell_vol', 'micro_OFI', 'micro_CVD', 'volume',
#              'spread_pct', 'rsi_14', 'rsi_roc_3', 'rsi_roc_5',
#              'ma_accel_regime', 'price_roc_5', 'depth_imbalance_z']
#
# The 7 remaining features (by PFI ranking):
#   spread_bps_z (0.081), ma_slope_5 (0.024), rsi_x_ma (0.006),
#   ma_accel (0.005), price_roc_3 (0.005), depth_imbalance_z (0.002)
FEATURE_COLUMNS = [
    'spread_bps_z',              # DeepLOB spread in bps, z-scored (PFI=0.081)
    'ma_slope_5',                # MA velocity (PFI=0.024)
    'rsi_x_ma',                  # RSI × MA slope interaction (PFI=0.006)
    'ma_accel',                  # MA acceleration (PFI=0.005)
    'price_roc_3',               # Price rate of change (PFI=0.005)
    'rsi_2',                     # RSI(2) filter/feature
    'sma_14',                    # SMA of RSI(14)
]

# Note on depth_imbalance_z (PFI=0.002): excluded because it contributes
# negligibly and adds complexity. The research also excludes it from model
# features since it's in the drop_cols.

# --- Training ---
TRAIN_ON_STARTUP = True           # Retrain model on initial historical fetch?
N_HISTORICAL_BARS = 5000          # Fetch this many 1-min bars for initial training
MODEL_PATH = "main/models/lgb_model.txt"

# --- Execution ---
DRY_RUN = True                    # Set False to place real orders!
MAX_TRADES_PER_HOUR = 300
FEE_BPS = 0                   # Binance/Bybit taker fee ~0.055%