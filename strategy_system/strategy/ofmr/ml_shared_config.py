"""OFMR strategy research config (+ compatibility re-exports).

Platform-level constants — paths, data-source selection, session bounds, the
universal walk-forward fold scheme, shared sampling defaults — live in
`strategy_system/config/platform.json` (plain JSON, previously `config/platform.py`), the
single source of truth for EVERY strategy. New code should read that file.

This module loads the platform JSON once and re-exports its values as module
constants so all existing importers keep working unchanged, and keeps the
OFMR-strategy-specific constants (feature lists, model windows, VPIN
diagnostics, legacy engine knobs).
"""
import json
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

with open(os.path.join(PROJECT_ROOT, "strategy_system", "config", "platform.json"),
          "r", encoding="utf-8-sig") as _pf:
    _PLATFORM = json.load(_pf)

BUILD_DIR = os.path.join(PROJECT_ROOT, _PLATFORM["build_dir"])
STATE_SPACE_DIR = os.path.join(PROJECT_ROOT, _PLATFORM["state_space_dir"])

DATA_FORMAT = _PLATFORM["data_format"]
DATA_DIR = _PLATFORM["data_dir"]
CHUNK_SESSIONS = _PLATFORM["chunk_sessions"]
N_CHUNKS_PER_SESSION = _PLATFORM["n_chunks_per_session"]
AUTO_CONVERT_CSV_TO_PARQUET = _PLATFORM["auto_convert_csv_to_parquet"]

SESSION_BOUNDS = [tuple(b) for b in _PLATFORM["session_bounds"]]
SESSION_ORDER = [b[0] for b in SESSION_BOUNDS]

# --- Matched-database ingestion (strategy_system/strategy/ofmr/abstraction.py) ---
# When True, `ml_state_optimized.py __main__` skips the legacy orderbook/trades
# parquet sessions and builds the state space from
# `market_data/matched/<SYMBOL>/merged_*.parquet` via `abstraction.py`, then
# continues through the IDENTICAL downstream (labels -> walk-forward ->
# production model under ofmr/models/). Off by default: existing csv/parquet
# and REPLAY_FEED paths are untouched.
USE_MATCHED_DB = os.environ.get("OFMR_USE_MATCHED_DB", "1").strip().lower() not in (
    "0", "false", "no", "off", "")
MATCHED_TAG = None            # e.g. "2025-06-30" to ingest one day; None = all
MATCHED_SESSION_ID = 0
# NOTE: MATCHED_SYMBOL is defined after the strategy config import block
# below — TICKER only exists from that point on.

REGIME_FOLD_MODE = _PLATFORM["regime_fold_mode"]
REGIME_FOLD_N_SAMPLES = _PLATFORM["regime_fold_n_samples"]
REGIME_FOLD_BLOCK_SIZE = _PLATFORM["regime_fold_block_size"]
REGIME_FOLD_HORIZON_SECONDS = _PLATFORM["regime_fold_horizon_seconds"]
REGIME_FOLD_EMBARGO_SECONDS = _PLATFORM["regime_fold_embargo_seconds"]

BARRIER_CHUNK_SIZE = _PLATFORM["barrier_chunk_size"]
N_RANDOM_SAMPLES = _PLATFORM["n_random_samples"]
TEST_BLOCK_SIZE = _PLATFORM["test_block_size"]


def ensure_dir(path: str) -> str:
    """Create `path` (and parents) if needed, then return it."""
    os.makedirs(path, exist_ok=True)
    return path

from strategy_system.strategy.ofmr.config import (
    LOOKAHEAD_SECONDS, BARRIER_MULTIPLIER, RSI_FILTER_COLUMN, RSI_FILTER_THRESHOLD,
    ENABLE_RSI_ACCEL_FILTER,
    RR_RATIO, TRADE_DIRECTION, TIMEFRAME_BARRIER, SYMBOL, TICKER,
    RSI_14_WINDOW, RSI_2_WINDOW, SMA_14_WINDOW, DEEPLOB_Z_WINDOW, RSI_Z_WINDOW,
    MA_SLOPE_WINDOW, PRICE_ROC_WINDOW_3, LABEL_LOSS, LABEL_WIN, FEATURE_COLUMNS,
    FEE_BPS, ENABLE_SLIPPAGE, MIN_BARRIER_PCT, MAX_BARRIER_PCT, EPS,
    HYSTERESIS_BUFFER, LOCKOUT_SECONDS, MODEL_PATH,
    REPLAY_FEED,
)

try:
    from strategy_system.strategy.ofmr.config import BARRIER_VOL_BAR_WINDOW
except ImportError:
    BARRIER_VOL_BAR_WINDOW = 20

MATCHED_SYMBOL = TICKER.upper()  # e.g. "BTCUSDT" dir under market_data/matched

# ============================================================================
# OFMR-SPECIFIC CONSTANTS (platform paths/sessions/folds live in strategy_system/config/platform)
# ============================================================================
SYMBOL_HIST = TICKER.upper()  # btcusdt -> BTCUSDT (historical data-file prefix)
BAR_FREQ = f"{TIMEFRAME_BARRIER}s"
USE_ORDERFLOW_FEATURES = True
EXPORT_PRODUCTION_MODEL = True

KNOWN_BAR_WINDOWS = [
    RSI_14_WINDOW, RSI_2_WINDOW, SMA_14_WINDOW, DEEPLOB_Z_WINDOW, RSI_Z_WINDOW,
    MA_SLOPE_WINDOW, PRICE_ROC_WINDOW_3, BARRIER_VOL_BAR_WINDOW,
    10,  # hardcoded window=10 rolling z-scores
    5,   # hardcoded window=5 rolling features (price_roc_5 / rsi_roc_5)
]
_CHUNK_WARMUP_SAFETY_BARS = 10  # extra margin on top of the largest window found
CHUNK_WARMUP_BARS = max(KNOWN_BAR_WINDOWS) + _CHUNK_WARMUP_SAFETY_BARS
CHUNK_BUFFER_SECONDS = TIMEFRAME_BARRIER * CHUNK_WARMUP_BARS

BARRIER_CHUNK_SIZE = 50_000  # legacy re-export; canonical home: config/platform
EMBARGO_SECONDS = LOOKAHEAD_SECONDS
CLASS_NAMES = {LABEL_LOSS: 'LOSS', LABEL_WIN: 'WIN'}

VPIN_TARGET_BUCKETS_PER_SESSION = 50
VPIN_WINDOW_BUCKETS = 10
VPIN_MIN_PERIODS = 5
RETAIL_THRESHOLD = 0.05
ROLLING_WINDOW = 30

ORDERFLOW_FEATURE_COLUMNS = [
        'ofi_z', 'cvd_delta_z', 'cvd_slope_z', 'absorption_z',
        'tick_autocorr', 'inst_ratio', 'delta_divergence_z', 'depth_imbalance_z',
    ]

# ============================================================================
# TRADE LOG FEATURE WINDOWS (research walk-forward CSV enrichment)
# ----------------------------------------------------------------------------
# When True, the walk-forward trade log CSV (`build/walk_forward_trade_log_v3.csv`)
# gains per-feature OPEN / HIGH / LOW / CLOSE columns measured over each trade's
# holding window:
#   - `{feature}_open`  : feature value on the 1-second row the trade entered
#   - `{feature}_high`  : highest value reached while the trade was held
#   - `{feature}_low`   : lowest value reached while the trade was held
#   - `{feature}_close` : value on the row the trade exited (barrier/timeout)
# This is observational diagnostics ONLY — never used for prediction or
# labeling — so it cannot cause lookahead bias. The values are read from the
# already-shifted, point-in-time-safe micro grid. Set False to save memory on
# very long datasets (it materializes one numpy column per feature).
# ============================================================================
TRADE_FEATURE_WINDOWS = True
 