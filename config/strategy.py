"""
Strategy Configuration — Live Execution
=========================================
CANONICAL HOME: this file now lives at `config/strategy.py` (repo-root level)
so the toolchain does NOT depend on the legacy `main/` package. Edit parameters
HERE — `main/config.py` is just a compatibility shim that re-exports these.

Central config matching the research parameters in ofmr/ml_state_optimized.py.
All parameters verified against the optimized research.
"""

# --- Exchange ---
SYMBOL = "BTC/USDT:USDT"         # CCXT symbol format for linear perpetual
EXCHANGE = "bybit"
TESTNET = False                    # False = real API for data, True = testnet (testnet sucks)
API_KEY = "..."                   # Your real Bybit API key (read-only is fine for data)
API_SECRET = "..."                # Your real Bybit API secret

# --- Strategy Parameters (VERIFIED against research/ofmr/ml_state_optimized.py) ---
LOOKAHEAD_SECONDS    = 6000      # Barrier horizon / timeout in seconds (research
                                 # walk-forward used 600 s; live runs 10x that)
BARRIER_MULTIPLIER   = 0.3       # Volatility-scaled barrier width (matches research 0.3)
RSI_FILTER_COLUMN    = 'rsi_2'   # Research uses rsi_2 for filter
RSI_FILTER_THRESHOLD = 30        # Only trade when RSI(2) < 30
CONFIDENCE_THRESHOLD = 0.2       # P(WIN) must be >= this to fire a trade
# Research tests thresholds 0.50-0.95. Start at 0.50 for more trades.

# --- RSI + acceleration (momentum) entry filter toggle ---
# True  = NEW two-sided directional zone:
#         (RSI(2)<30 & ma_accel_EWA3>0) -> long candidate
#         OR (RSI(2)>70 & ma_accel_EWA3<0) -> short candidate
#         direction is inverted by TRADE_DIRECTION.
# False = ORIGINAL single-zone filter (the very first condition): RSI(2)<30 only,
#         no ma_accel filter, original P(WIN)>=threshold gate + global invert.
# The ma_accel_ewma3 column is always computed in feature_builder; it is only
# USED to filter entries when this flag is True.
ENABLE_RSI_ACCEL_FILTER = False

# --- Trade Direction ---
# 'short' = fade (current behavior) — bet AGAINST the model's P(WIN) prediction
# 'long'  = snapback — bet WITH the model's P(WIN) prediction (price bounces up)
TRADE_DIRECTION = 'long'        # 'short' or 'long'

# note for myself: based on result, inversing signal for macro asset dont work. only for small asset.

# --- Barrier & Position ---
STOP_LOSS_BPS        = 5        # Fallback stop if barrier logic fails
TAKE_PROFIT_BPS      = 5
MAX_POSITION_USD     = 100       # Max notional per position (~0.0017 BTC at $60k)
MIN_ORDER_SIZE       = 0.001     # Minimum contract size
RR_RATIO             = 10       # Risk:Reward ratio (2.0 = 1:2, profit target = 2x stop distance)

# R-multiple clip (research STEP 5 parity, ofmr/ml_state_optimized.py): research
# clamps normalized_return dataset-wide with
#     cap = max(|q(0.99)|, |q(0.01)|, RR_RATIO)
# BEFORE any expectancy is computed. A streaming engine cannot know that cap in
# advance, so the live engine records the UNCLIPPED R per trade (`r_raw`) and
# live-test parity applies the cap post-hoc (dual pass, see
# research/parity_diagnostics.py). This fixed ceiling is only a fallback for
# long-running live sessions that never get a second pass. 0.0 = disabled
# (default — barrier trades are self-limiting at +RR/-1 anyway, so this only
# ever matters for timeout exits on tiny barriers).
R_CLIP_CAP           = 0.0

# --- Volatility / Barrier-Size Filter ---
# Reject entries when the volatility-scaled barrier is outside this range.
# barrier_pct = ret_std * sqrt(LOOKAHEAD_SECONDS) * BARRIER_MULTIPLIER
# MIN: reject if market is too dead (barrier too tight → no profit potential)
# MAX: reject if market is too wild (barrier too wide → uncontrollable risk)
# Set MIN=0.0 and MAX=1.0 to disable the filter entirely.
MIN_BARRIER_PCT = 0.0001   # 0.01% — reject if barrier smaller than this (was 0 = disabled)
MAX_BARRIER_PCT = 0.05          # 5%    — reject if barrier larger than this
TIMEFRAME_BARRIER = 300         # In unit seconds (5 minutes)

# --- Volatility-normalized position sizing (risk per trade) ---
# Size each position so that hitting the stop barrier loses exactly
# RISK_PCT_PER_TRADE of CURRENT equity, regardless of symbol or volatility:
#
#     notional = (equity * RISK_PCT_PER_TRADE) / barrier_pct
#
# capped at MAX_LEVERAGE * equity (a tight/low-vol barrier needs a large
# notional to hit the risk target — this caps the implied leverage; when the
# cap binds, effective risk < target). Set RISK_PCT_PER_TRADE to 0.0 to fall
# back to legacy fixed-notional sizing (MAX_POSITION_USD / full slot capital).
# 0.2% per trade: a -1R stop-out costs 0.2% equity while a +10R winner returns
# ~2% equity, and every symbol in the portfolio now contributes comparable
# dollar risk — R-multiples and $-PnL stay consistent across BTC/ETH/SOL.
RISK_PCT_PER_TRADE = 0.002   # 0.2% of equity risked per trade
MAX_LEVERAGE = 100.0           # notional ceiling = 3x slot equity

# --- Feature Windows (research ofmr/ml_state_optimized.py) ---
RSI_14_WINDOW      = 14
RSI_2_WINDOW       = 2
SMA_14_WINDOW      = 14
OFI_LOOKBACK       = 60         # Rolling window for CVD_z (research: rolling(60))
CVD_ROLLING        = 10         # Rolling window for micro_CVD (research: rolling(10))
BARRIER_VOL_WINDOW = 100        # Rolling window for vol estimation (research: rolling(100))

# --- Aggregated-bar volatility window (barrier sizing) ---
# Number of TIMEFRAME_BARRIER bars used to estimate bar-to-bar return std for
# barrier sizing. This is the window both LiveFeatureEngine (features.py) and
# research/ofmr/ml_state_optimized.py fall back to via globals().get(..., 20) when
# it's absent. Defining it here explicitly closes that silent-drift fallback so
# the live engine, the research pipeline, and the training-data builder all use
# the SAME window. Matches the research add_barrier_labels() bootstrap floor
# (bar_vol_std.where(bar_vol_std > 0, 0.001)) and PositionTracker.get_barrier_pct().
BARRIER_VOL_BAR_WINDOW = 20

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

# --- Feature definition (EXACT match to ofmr/ml_state_optimized.py Step 4) ---
# From ofmr/ml_state_optimized.py lines 492-499:
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

# --- Orderflow Features (optional 9-feature block, ofmr/ml_state_optimized.py Step 4) ---
# These mirror the constants ofmr/ml_state_optimized.py hardcodes locally near the
# top of that file (USE_ORDERFLOW_FEATURES, ORDERFLOW_FEATURE_COLUMNS,
# RETAIL_THRESHOLD, VPIN_TARGET_BUCKETS_PER_SESSION, VPIN_WINDOW_BUCKETS,
# VPIN_MIN_PERIODS). The research script is read-only and imports its copies
# from itself, not from here — so if you ever change RETAIL_THRESHOLD or the
# VPIN bucket/window constants in ofmr/ml_state_optimized.py, update the matching
# values below too, or the live orderflow features will silently be computed
# on different definitions than whatever produced the model's manifest.
#
# USE_ORDERFLOW_FEATURES here controls what LiveFeatureEngine.get_feature_columns()
# (features.py) actually emits. It must match the USE_ORDERFLOW_FEATURES the
# model was exported with (see <model>.features.json "includes_orderflow"),
# or executor.py's feature-count guard will refuse to trade every tick.
USE_ORDERFLOW_FEATURES = False

# Exact same 9 names, same order, as ORDERFLOW_FEATURE_COLUMNS in
# ofmr/ml_state_optimized.py — order matters, it's the column order the model was
# trained on.
ORDERFLOW_FEATURE_COLUMNS = [
    'ofi_z', 'cvd_delta_z', 'cvd_slope_z', 'absorption_z',
    'tick_autocorr', 'inst_ratio', 'delta_divergence_z', 'depth_imbalance_z',
]

# Rolling z-score window (in bars) for ofi_z, cvd_delta_z, cvd_slope_z,
# absorption_z, delta_divergence_z, depth_imbalance_z — mirrors the
# hardcoded window=10 in every calculate_zscore(...) call for these features
# in ofmr/ml_state_optimized.py.
ORDERFLOW_Z_WINDOW = 10

# Trade-size cutoff (base-asset units) separating "retail" from
# "institutional" for inst_ratio / delta_divergence_z. Matches
# RETAIL_THRESHOLD in ofmr/ml_state_optimized.py.
RETAIL_THRESHOLD = 0.05

# --- VPIN (volume-clock informed-trading probability) ---
# Research derives its volume-per-bucket size from each CSV session's total
# traded volume (total_session_volume / VPIN_TARGET_BUCKETS_PER_SESSION),
# because a research "session" is a fixed, already-complete file. Live
# trading has no such fixed endpoint — the bucket size instead self-
# calibrates from a trailing VPIN_SESSION_LOOKBACK_SECONDS window of realized
# volume (see LiveFeatureEngine._compute_macro_features), recomputed on every
# bar rollover so it tracks changing liquidity. This is the live analogue of
# the research definition, not a byte-for-byte replica of it — flagging this
# because VPIN is the one orderflow feature that can't be made identical
# between an offline CSV session and an open-ended live stream.
VPIN_TARGET_BUCKETS_PER_SESSION = 50
VPIN_WINDOW_BUCKETS = 10   # smoothing window in closed buckets
VPIN_MIN_PERIODS    = 5    # buckets needed before vpin leaves its 0.0 default
VPIN_SESSION_LOOKBACK_SECONDS = 86400   # trailing window used to size buckets (1 day)

# ---------------------------------------------------------------------------
# Research replay / validation — feed override (read by research pipeline)
# ---------------------------------------------------------------------------
# Point `research/ofmr/ml_state_optimized` (and `feature_builder`) at a replay feed
# TEXT FILE instead of the `research/csv/*.parquet` store. The replay format is
# exactly what the Rust live-test replay exchange consumes — one event per line
# (`#` header lines ignored):
#
#     #TICKER btcusdt
#     T <ts_ms> <price> <amount> <buy|sell> <id>          (trade print)
#     B <ts_ms> <bid_px> <bid_amount> <ask_px> <ask_amount>   (L1 orderbook)
#
# e.g. `REPLAY_FEED = "build/live_test_feed_btcusdt.replay"`.
# Leave "" (the default) to run the normal parquet pipeline. When a feed file
# has no orderbook (`B`) lines, a degenerate bid==ask==trade-price book is
# synthesized so feature construction still runs (spread/depth features then
# collapse toward zero — fine for replay validation, not for training).
REPLAY_FEED = ""

# --- Training ---
TRAIN_ON_STARTUP = True           # Retrain model on initial historical fetch?
N_HISTORICAL_BARS = 5000          # Fetch this many 1-min bars for initial training
MODEL_PATH = "uets_rs/models/lgb_model.txt"   # exported by research STEP 12 (moved out of uets_legacy)

# --- Execution ---
DRY_RUN = True                    # Set False to place real orders!
MAX_TRADES_PER_HOUR = 300
FEE_BPS = 0                   # Binance/Bybit taker fee ~0.055%

# --- Signal Debounce & Re-entry Lockout (matches research walk-forward) ---
# Research (ofmr/ml_state_optimized.py) uses a hysteresis buffer to collapse
# probability flicker into one trade per elevated-probability period, and a
# non-overlapping position filter that blocks re-entry until the previous
# trade's barrier horizon has elapsed. Live previously used only a loose
# MAX_TRADES_PER_HOUR rate limit, which let it fire far more trades than
# research. These options make live reproduce research's trade cadence.
#
# HYSTERESIS_BUFFER: once P(WIN) >= CONFIDENCE_THRESHOLD fires a signal, the
# detector disarms and only re-arms once P(WIN) drops below
# (CONFIDENCE_THRESHOLD - HYSTERESIS_BUFFER). Set 0.0 to disable (fire on
# every crossing, like the old behavior).
HYSTERESIS_BUFFER = 0

# LOCKOUT_SECONDS: after a trade closes, block new entries for this many
# seconds (research uses the full LOOKAHEAD_SECONDS barrier horizon).
# Set 0 to disable the lockout entirely (re-enter as soon as position closes).
#
# CURRENT SETTING: 0 = DISABLED. Hysteresis stays ON (0.08) so flicker is
# still collapsed, but there is NO re-entry lockout — the executor re-enters
# as soon as a position closes, producing many trades (executor's original
# high-cadence behavior). Set to LOOKAHEAD_SECONDS to reproduce the research
# walk-forward's low-cadence one-position-per-horizon behavior.
LOCKOUT_SECONDS = 1000

# Set False to disable all simulated slippage: entry fills exactly at
# mid_price, exits fill exactly at the barrier price, 0.00 bps both ways.
# Useful for isolating whether slippage vs. barrier sizing (BARRIER_MULTIPLIER,
# MIN_BARRIER_PCT) is driving the R-curve decay.
ENABLE_SLIPPAGE = False

# Fallback slippage (bps) applied when live order-book depth lacks a quote on
# the far side at fill time. Mirrors engine SLIPPAGE_BPS_FALLBACK.
SLIPPAGE_BPS_FALLBACK = 0


# ---------------------------------------------------------------------------
# Strategy-config export — single source of truth for the Rust engine /
# live-test replay. `unified_runner.py` calls `export_strategy_config(...)` to
# materialize `build/live_test_config.json`, then passes `--config` to the
# `live-test`/`live-trade` binaries so the live engine is provably running the
# same parameters as `main/config.py` (validated by the engine's
# `consts_match_main_config` unit test). Only keys present in StrategyConfig
# are emitted.
# ---------------------------------------------------------------------------
import json as _json

#: Mirrors `engine::StrategyConfig` field names (snake_case).
_STRATEGY_CONFIG_KEYS = (
    "micro_barrier_secs",
    "timeframe_barrier_secs",
    "lookahead_seconds",
    "barrier_multiplier",
    "rr_ratio",
    "rsi_filter_threshold",
    "enable_rsi_accel_filter",
    "trade_direction_is_long",
    "confidence_threshold",
    "lockout_seconds",
    "max_position_usd",
    "min_order_size",
    "fee_bps",
    "enable_slippage",
    "slippage_bps_fallback",
    "min_barrier_pct",
    "max_barrier_pct",
    "risk_pct_per_trade",
    "max_leverage",
    "r_clip_cap",
    "use_orderflow_features",
    "hysteresis_buffer",
    "require_fold_coverage",
    "one_candidate_per_bar",
)


def strategy_config_dict() -> dict:
    """Build a JSON-ready mapping of StrategyConfig fields from this module."""
    d = {
        "micro_barrier_secs": 1,  # research micro bar = 1 s
        "timeframe_barrier_secs": TIMEFRAME_BARRIER,
        "lookahead_seconds": LOOKAHEAD_SECONDS,
        "barrier_multiplier": BARRIER_MULTIPLIER,
        "rr_ratio": RR_RATIO,
        "rsi_filter_threshold": RSI_FILTER_THRESHOLD,
        "enable_rsi_accel_filter": ENABLE_RSI_ACCEL_FILTER,
        "trade_direction_is_long": TRADE_DIRECTION == "long",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "lockout_seconds": LOCKOUT_SECONDS,
        "max_position_usd": MAX_POSITION_USD,
        "min_order_size": MIN_ORDER_SIZE,
        "fee_bps": FEE_BPS,
        "enable_slippage": ENABLE_SLIPPAGE,
        "slippage_bps_fallback": SLIPPAGE_BPS_FALLBACK,
        "min_barrier_pct": MIN_BARRIER_PCT,
        "max_barrier_pct": MAX_BARRIER_PCT,
        "risk_pct_per_trade": RISK_PCT_PER_TRADE,
        "max_leverage": MAX_LEVERAGE,
        "r_clip_cap": R_CLIP_CAP,
        "hysteresis_buffer": HYSTERESIS_BUFFER,
    }
    # Orderflow block is always computed/used with the research model.
    d["use_orderflow_features"] = True
    # Walk-forward parity gates: off for normal live trading; unified_runner's
    # `live-test --wf-dir` flips them on in the exported JSON copy.
    d["require_fold_coverage"] = False
    d["one_candidate_per_bar"] = False
    return d


def export_strategy_config(path: str) -> str:
    """Write StrategyConfig JSON (single source of truth) and return its path."""
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(strategy_config_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path
