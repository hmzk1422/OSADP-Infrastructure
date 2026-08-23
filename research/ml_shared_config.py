import os
from main.config import (
    LOOKAHEAD_SECONDS, BARRIER_MULTIPLIER, RSI_FILTER_COLUMN, RSI_FILTER_THRESHOLD,
    ENABLE_RSI_ACCEL_FILTER,
    RR_RATIO, TRADE_DIRECTION, TIMEFRAME_BARRIER,
    RSI_14_WINDOW, RSI_2_WINDOW, SMA_14_WINDOW, DEEPLOB_Z_WINDOW, RSI_Z_WINDOW,
    MA_SLOPE_WINDOW, PRICE_ROC_WINDOW_3, LABEL_LOSS, LABEL_WIN, FEATURE_COLUMNS,
    FEE_BPS, ENABLE_SLIPPAGE, MIN_BARRIER_PCT, MAX_BARRIER_PCT, EPS,
    HYSTERESIS_BUFFER, LOCKOUT_SECONDS, MODEL_PATH,
)

try:
    from main.config import BARRIER_VOL_BAR_WINDOW
except ImportError:
    BARRIER_VOL_BAR_WINDOW = 20

# ============================================================================
# OUTPUT DIRECTORY CONFIGURATION
# ----------------------------------------------------------------------------
# Every generated artifact (backtest reports, feature-importance CSVs, trade
# logs, HTML charts) is written under the project's `build/` folder so the
# working directory stays clean. Intermediate state-space parquet chunks are
# written inside the research package (`research/state_space/`) instead of the
# repository root.
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
STATE_SPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_space")


def ensure_dir(path: str) -> str:
    """Create `path` (and parents) if needed, then return it."""
    os.makedirs(path, exist_ok=True)
    return path

SYMBOL_HIST = "BTCUSDT"
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

BARRIER_CHUNK_SIZE = 50_000
N_RANDOM_SAMPLES = 400
TEST_BLOCK_SIZE = 150
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
# DATA SOURCE CONFIGURATION
# ----------------------------------------------------------------------------
# Flip DATA_FORMAT to switch the entire pipeline (both the time-bounds scan
# and the per-chunk load) between parquet and csv input files, instead of
# editing hardcoded paths in __main__.
# ============================================================================
DATA_FORMAT = "parquet"  # "parquet" or "csv"
DATA_DIR = "./research/csv"
CHUNK_SESSIONS = True
N_CHUNKS_PER_SESSION = 24

AUTO_CONVERT_CSV_TO_PARQUET = False  # see _ensure_parquet() below

SESSION_BOUNDS = [
    ("Asia (00:00-08:00)", 0, 480),
    ("EU-Asia (08:00-09:00)", 480, 540),
    ("EU (09:00-13:30)", 540, 810),
    ("EU-US (13:30-16:00)", 810, 960),
    ("US (16:00-20:00)", 960, 1200),
    ("Pacific (20:00-00:00)", 1200, 1440),
]
SESSION_ORDER = [label for label, _, _ in SESSION_BOUNDS]
 