"""Legacy compatibility shim for the strategy configuration.

The canonical, single source of truth is ``config/portfolio.json``: its
``_strategy_defaults`` block carries every strategy parameter (snake_case,
mirroring ``engine::StrategyConfig``) and is read DIRECTLY by the Rust engine.
This shim maps those JSON keys onto the old UPPER_CASE module constants so
older modules that still do ``from uets_legacy.config import *`` keep working
unchanged. Nothing here duplicates a parameter value — edit
``config/portfolio.json`` instead.
"""

import json as _json
import os as _os

with open(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "..", "..", "..", "config", "portfolio.json",
    ),
    "r", encoding="utf-8-sig",
) as _fh:
    _RAW = _json.load(_fh)

_S = _RAW.get("_strategy_defaults", {})


def _get(key, default=None):
    return _S.get(key, default)


# --- Exchange ---
# `ticker` (in the assets array) is the ONLY instrument identifier in the
# config. The CCXT symbol format needed by the legacy executor is DERIVED:
# btcusdt -> BTC/USDT:USDT (USDT-margined linear perpetual).
def _ccxt_symbol(ticker: str) -> str:
    t = str(ticker).lower()
    if t.endswith("usdt"):
        return f"{t[:-4].upper()}/USDT:USDT"
    return str(ticker).upper()


_TICKER0 = (_RAW.get("assets") or [{}])[0].get("ticker", "btcusdt")
TICKER = str(_TICKER0).lower()
SYMBOL = _ccxt_symbol(_TICKER0)
EXCHANGE = _RAW.get("exchange", "bybit")
TESTNET = False                    # False = real API for data, True = testnet (testnet sucks)
API_KEY = "..."                    # Your real Bybit API key (read-only is fine for data)
API_SECRET = "..."                 # Your real Bybit API secret

# --- Strategy Parameters (VERIFIED against research/ofmr/ml_state_optimized.py) ---
LOOKAHEAD_SECONDS = _get("lookahead_seconds", 6000)
BARRIER_MULTIPLIER = _get("barrier_multiplier", 0.3)
RSI_FILTER_COLUMN = _get("rsi_filter_column", "rsi_2")
RSI_FILTER_THRESHOLD = _get("rsi_filter_threshold", 30)
CONFIDENCE_THRESHOLD = _get("confidence_threshold", 0.2)

# --- RSI + acceleration (momentum) entry filter toggle ---
ENABLE_RSI_ACCEL_FILTER = _get("enable_rsi_accel_filter", False)

# --- Trade Direction ---
TRADE_DIRECTION = "long" if _get("trade_direction_is_long", True) else "short"

# --- Barrier & Position ---
STOP_LOSS_BPS = _get("stop_loss_bps", 5)
TAKE_PROFIT_BPS = _get("take_profit_bps", 5)
MAX_POSITION_USD = _get("max_position_usd", 100)
MIN_ORDER_SIZE = _get("min_order_size", 0.001)
RR_RATIO = _get("rr_ratio", 10)
R_CLIP_CAP = _get("r_clip_cap", 0.0)

# --- Volatility / Barrier-Size Filter ---
MIN_BARRIER_PCT = _get("min_barrier_pct", 0.0001)
MAX_BARRIER_PCT = _get("max_barrier_pct", 0.05)
TIMEFRAME_BARRIER = _get("timeframe_barrier_secs", 300)

# --- Volatility-normalized position sizing (risk per trade) ---
RISK_PCT_PER_TRADE = _get("risk_pct_per_trade", 0.002)
MAX_LEVERAGE = _get("max_leverage", 100.0)

# --- Feature Windows (research ofmr/ml_state_optimized.py) ---
RSI_14_WINDOW = _get("rsi_14_window", 14)
RSI_2_WINDOW = _get("rsi_2_window", 2)
SMA_14_WINDOW = _get("sma_14_window", 14)
OFI_LOOKBACK = _get("ofi_lookback", 60)
CVD_ROLLING = _get("cvd_rolling", 10)
BARRIER_VOL_WINDOW = _get("barrier_vol_window", 100)
BARRIER_VOL_BAR_WINDOW = _get("barrier_vol_bar_window", 20)

# --- DeepLOB feature windows ---
DEEPLOB_Z_WINDOW = _get("deeplob_z_window", 10)
RSI_Z_WINDOW = _get("rsi_z_window", 20)
MA_SLOPE_WINDOW = _get("ma_slope_window", 5)
PRICE_ROC_WINDOW_3 = _get("price_roc_window_3", 3)
PRICE_ROC_WINDOW_5 = _get("price_roc_window_5", 5)
RSI_ROC_WINDOW_3 = _get("rsi_roc_window_3", 3)
RSI_ROC_WINDOW_5 = _get("rsi_roc_window_5", 5)

# --- Label encoding (matches research binary: LABEL_LOSS=0, LABEL_WIN=1) ---
LABEL_LOSS = _get("label_loss", 0)
LABEL_WIN = _get("label_win", 1)
LABEL_SHORT = LABEL_LOSS
LABEL_LONG = LABEL_WIN
EPS = _get("eps", 1e-9)

# --- Feature definition (EXACT match to ofmr/ml_state_optimized.py Step 4) ---
FEATURE_COLUMNS = _get("feature_columns", [
    "spread_bps_z", "ma_slope_5", "rsi_x_ma", "ma_accel",
    "price_roc_3", "rsi_2", "sma_14",
])

# --- Orderflow Features (optional 9-feature block) ---
USE_ORDERFLOW_FEATURES = _get("use_orderflow_features", False)
ORDERFLOW_FEATURE_COLUMNS = _get("orderflow_feature_columns", [
    "ofi_z", "cvd_delta_z", "cvd_slope_z", "absorption_z",
    "tick_autocorr", "inst_ratio", "delta_divergence_z", "depth_imbalance_z",
])
ORDERFLOW_Z_WINDOW = _get("orderflow_z_window", 10)
RETAIL_THRESHOLD = _get("retail_threshold", 0.05)

# --- VPIN (volume-clock informed-trading probability) ---
VPIN_TARGET_BUCKETS_PER_SESSION = _get("vpin_target_buckets_per_session", 50)
VPIN_WINDOW_BUCKETS = _get("vpin_window_buckets", 10)
VPIN_MIN_PERIODS = _get("vpin_min_periods", 5)
VPIN_SESSION_LOOKBACK_SECONDS = _get("vpin_session_lookback_seconds", 86400)

# --- Research replay / validation — feed override (read by research pipeline) ---
REPLAY_FEED = _get("replay_feed", "")

# --- Training ---
TRAIN_ON_STARTUP = _get("train_on_startup", True)
N_HISTORICAL_BARS = _get("n_historical_bars", 5000)
MODEL_PATH = _get("model_path", "uets_rs/models/lgb_model.txt")

# --- Execution ---
DRY_RUN = _get("dry_run", True)
MAX_TRADES_PER_HOUR = _get("max_trades_per_hour", 300)
FEE_BPS = _get("fee_bps", 0)

# --- Signal Debounce & Re-entry Lockout (matches research walk-forward) ---
HYSTERESIS_BUFFER = _get("hysteresis_buffer", 0)
LOCKOUT_SECONDS = _get("lockout_seconds", 1000)

# --- Slippage simulation ---
ENABLE_SLIPPAGE = _get("enable_slippage", False)
SLIPPAGE_BPS_FALLBACK = _get("slippage_bps_fallback", 0)
