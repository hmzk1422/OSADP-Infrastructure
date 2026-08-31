//! Unified Execution Engine — research-faithful dry-run trading strategy.
//!
//! A Rust port of the exact entry/exit mechanics from
//! [`research/ofmr/ml_state_optimized.py`](../../research/ofmr/ml_state_optimized.py) and
//! the live [`main/execution.py`](../../main/execution.py) `PositionTracker`,
//! using the SAME numbers as [`main/config.py`](../../main/config.py):
//!
//! * **1-second micro bars** — the tick stream is aggregated into 1 s bars and
//!   the full 15-feature vector is assembled + evaluated on EVERY 1 s bar close
//!   (the research state space is `df_trades.resample('1s')` → one row/second);
//! * **`TIMEFRAME_BARRIER` (300 s) regime filter** — on each 300 s rollover the
//!   composite-bar features are recomputed once (Wilder RSI(2), RSI(14), SMA(14)
//!   of RSI(14), MA(5) slope + acceleration / d² derivative, spread z-score
//!   w=10, `rsi_x_ma`, and bar-to-bar return vol over `BARRIER_VOL_BAR_WINDOW`
//!   bars (ddof=0) for barrier sizing) and forward-filled to every 1 s bar —
//!   exactly `df_micro.join(df_macro, how='left').ffill()` in research;
//! * hunting zone entry — `rsi_2 < 30` (single-zone, `TRADE_DIRECTION='long'`
//!   → buy the snapback), gated by a *model-free confidence proxy* calibrated
//!   to reproduce the LightGBM P(WIN) ≥ 0.50 gate of the research pipeline
//!   (Rust loads the exported `lgb_model.txt`; when it is unavailable, it uses a
//!   transparent deterministic proxy over the base features and keeps every
//!   other mechanic byte-for-byte identical);
//! * vol-scaled barriers — `barrier_pct = bar_vol_std × BARRIER_MULTIPLIER`
//!   (bootstrap floor `0.001 × BARRIER_MULTIPLIER`), stop at `entry × (1 ∓ bp)`,
//!   target at `entry × (1 ± bp × RR_RATIO)`, timeout at `LOOKAHEAD_SECONDS`;
//! * dry-run accounting matching `PositionTracker._exit` exactly — taker fees
//!   on both sides, R-multiple (clipped to `[-1, +RR_RATIO]` for barrier exits),
//!   side-aware slippage, and a running equity curve.
//!
//! The engine is dependency-free so any workspace crate (live-trade,
//! live-test) can link it and it can be unit-tested in isolation.

#![allow(clippy::too_many_arguments)]

use std::collections::VecDeque;

pub mod lgbm;

/// Manifest of which feature slots the model expects, matched to
/// `uets_rs/models/lgb_model.features.json` base + orderflow blocks.
#[derive(Clone, Debug)]
pub struct FeatureManifest {
    pub base: [&'static str; 7],
    pub orderflow: [&'static str; 8],
    pub all: [&'static str; 15],
}

// ============================================================================
// Research configuration — exact mirror of main/config.py (+ ml_shared_config)
// ============================================================================

/// Composite / regime bar size (seconds). Macro features (RSI(2), MA slope +
/// acceleration, spread z, `bar_vol_std`) are recomputed once per boundary and
/// forward-filled to every micro bar (matches `TIMEFRAME_BARRIER` = 300).
pub const TIMEFRAME_BARRIER_SECS: i64 = 300; // 5-minute composite bars

/// Micro bar size (seconds). Ticks are aggregated into 1 s bars and the full
/// feature vector (micro `price_roc_3` + forward-filled macro) is evaluated on
/// every micro bar close — the research model runs on EVERY 1-second row.
pub const MICRO_BARRIER_SECS: i64 = 1;
pub const LOOKAHEAD_SECONDS: i64 = 6000; // barrier horizon / timeout
pub const BARRIER_MULTIPLIER: f64 = 0.3;
/// If true, the engine computes the 8 orderflow features (OFI z-score, CVD delta/slope,
/// absorption, tick autocorrelation, institutional ratio, delta divergence, depth imbalance)
/// and forwards all 15 features to the LightGBM model. Set to true when running with the
/// research model (`uets_rs/models/lgb_model.txt`), false for the model-free proxy mode.
pub const USE_ORDERFLOW_FEATURES: bool = true;
pub const RR_RATIO: f64 = 10.0;
pub const RSI_FILTER_THRESHOLD: f64 = 30.0;
pub const ENABLE_RSI_ACCEL_FILTER: bool = false; // config ENABLE_RSI_ACCEL_FILTER = False -> single-zone long-only
pub const TRADE_DIRECTION_IS_LONG: bool = true; // config TRADE_DIRECTION = "long"
pub const CONFIDENCE_THRESHOLD: f64 = 0.5; // config CONFIDENCE_THRESHOLD = 0.5 (was 0.2 before research-parity fix)
pub const LOCKOUT_SECONDS: i64 = 1000;
pub const MAX_POSITION_USD: f64 = 100.0;
pub const MIN_ORDER_SIZE: f64 = 0.001;
pub const FEE_BPS: f64 = 0.1;
pub const ENABLE_SLIPPAGE: bool = false; // research config disables it
pub const SLIPPAGE_BPS_FALLBACK: f64 = 0.75;
/// Matches `config/strategy.py` `MIN_BARRIER_PCT` (0.01%): reject entries whose
/// volatility-scaled barrier is tighter than this (research applies the same
/// filter in its training/live universe).
pub const MIN_BARRIER_PCT: f64 = 0.0001;
pub const MAX_BARRIER_PCT: f64 = 0.05;

/// Volatility-normalized position sizing: fraction of CURRENT equity risked
/// per trade (1R = equity × this, in USD — identical on every symbol and every
/// entry). 0.0 = disabled → legacy fixed-notional sizing (`max_position_usd`),
/// which is what research parity and existing backtests use.
pub const RISK_PCT_PER_TRADE: f64 = 0.0;
/// Notional ceiling for risk-normalized sizing, in multiples of current
/// equity (an implied-leverage cap). Hitting a fixed % risk on a tight
/// (low-volatility) barrier requires proportionally larger notional; this
/// caps how far that scaling is allowed to go.
pub const MAX_LEVERAGE: f64 = 1.0;
/// Fixed R-multiple clip applied in `close_position` when > 0: each trade's
/// r_multiple is clamped to [-cap, cap], mirroring research STEP 5's dataset-wide
/// `np.clip(normalized_return, -cap, cap)` (ml_state_optimized.py). Research
/// derives `cap = max(|q(0.99)|, |q(0.01)|, RR_RATIO)` from the WHOLE dataset,
/// which a streaming engine cannot know in advance — so this fixed ceiling is
/// OFF (0.0) by default and every trade also records its UNCLIPPED `r_raw`.
/// For live-test parity diagnostics use the dual-pass data-driven cap in
/// `research/parity_diagnostics.py` (computed from the recorded raw R values
/// after the replay completes) — that reproduces research's cap exactly.
pub const R_CLIP_CAP: f64 = 0.0;

pub const HYSTERESIS_BUFFER: f64 = 0.0; // config HYSTERESIS_BUFFER = 0

pub const RSI_14_WINDOW: usize = 14;
pub const RSI_2_WINDOW: usize = 2;
pub const SMA_14_WINDOW: usize = 14;
pub const MA_SLOPE_WINDOW: usize = 5;
pub const PRICE_ROC_WINDOW_3: usize = 3;
pub const DEEPLOB_Z_WINDOW: usize = 10;
pub const RSI_Z_WINDOW: usize = 20;
    pub const BARRIER_VOL_WINDOW: usize = 20; // BARRIER_VOL_BAR_WINDOW (main/config.py: 20) — research feature_builder.py rolling_std(20, ddof=0)
pub const ORDERFLOW_Z_WINDOW: usize = 10; // rolling z window for orderflow features
pub const RETAIL_THRESHOLD: f64 = 0.05; // trade-size cutoff retail vs institutional

const EPS: f64 = 1e-9;
const MAX_MICRO_CLOSES: usize = 300; // features.py `_price_buffer` — 1s closes → price_roc_3
const MAX_MACRO_CLOSES: usize = 200; // features.py `_close_prices` — composite closes → RSI/MA/vol
const MAX_SPREADS: usize = 1024;
const MAX_RSI_HIST: usize = 64;
const MAX_Z_HIST: usize = 64; // orderflow rolling z-score history (features.py RingBuffer(30))

/// Tunable strategy configuration. `Default` reproduces `main/config.py`.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Micro bar size (seconds) — 1 s in research/live.
    pub micro_barrier_secs: i64,
    /// Composite / regime bar size (seconds) — 300 s in research/live.
    pub timeframe_barrier_secs: i64,
    pub lookahead_seconds: i64,
    pub barrier_multiplier: f64,
    pub rr_ratio: f64,
    pub rsi_filter_threshold: f64,
    pub enable_rsi_accel_filter: bool,
    pub trade_direction_is_long: bool,
    pub confidence_threshold: f64,
    pub lockout_seconds: i64,
    pub max_position_usd: f64,
    pub min_order_size: f64,
    pub fee_bps: f64,
    pub enable_slippage: bool,
    pub slippage_bps_fallback: f64,
    pub min_barrier_pct: f64,
    pub max_barrier_pct: f64,
    /// Fraction of current equity risked per trade (volatility-normalized
    /// sizing: notional = equity × risk_pct / barrier_pct, capped by
    /// `max_leverage × equity`). 0.0 disables it → legacy fixed-notional
    /// (`max_position_usd`) sizing.
    pub risk_pct_per_trade: f64,
    /// Implied-leverage cap for risk-normalized sizing: max notional is
    /// `max_leverage × current equity`.
    pub max_leverage: f64,
    /// Fixed R-multiple clip (see [`R_CLIP_CAP`]). 0.0 = disabled; the raw
    /// (unclipped) R is always recorded as `TradeRecord::r_raw`, and the
    /// dual-pass data-driven cap lives in `research/parity_diagnostics.py`.
    pub r_clip_cap: f64,
    /// Whether the model was exported with the orderflow block (see
    /// `lgb_model.features.json` `includes_orderflow`). When off, the 8
    /// orderflow accumulators are skipped (fallback/proxy mode).
    pub use_orderflow_features: bool,
    /// Hysteresis dead-band (research `HYSTERESIS_BUFFER`). The de-bounced
    /// signal detector re-arms only when P(WIN) drops below
    /// `confidence_threshold - hysteresis_buffer`.
    pub hysteresis_buffer: f64,
    /// Walk-forward parity gates (live-test replay). When
    /// `require_fold_coverage` is set, rows outside every exported fold's OOS
    /// test window score P(WIN) = -1 (never tradable), matching research
    /// where such rows are simply never evaluated. When
    /// `one_candidate_per_bar` is set, entries come only from the LAST
    /// volume>0 1 s row of each composite bar -- the research universe
    /// (`volume > 0` filter runs BEFORE `deduplicate_to_bar`, keep-last).
    pub require_fold_coverage: bool,
    pub one_candidate_per_bar: bool,
    pub starting_equity: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            micro_barrier_secs: MICRO_BARRIER_SECS,
            timeframe_barrier_secs: TIMEFRAME_BARRIER_SECS,
            lookahead_seconds: LOOKAHEAD_SECONDS,
            barrier_multiplier: BARRIER_MULTIPLIER,
            rr_ratio: RR_RATIO,
            rsi_filter_threshold: RSI_FILTER_THRESHOLD,
            enable_rsi_accel_filter: ENABLE_RSI_ACCEL_FILTER,
            trade_direction_is_long: TRADE_DIRECTION_IS_LONG,
            confidence_threshold: CONFIDENCE_THRESHOLD,
            lockout_seconds: LOCKOUT_SECONDS,
            max_position_usd: MAX_POSITION_USD,
            min_order_size: MIN_ORDER_SIZE,
            fee_bps: FEE_BPS,
            enable_slippage: ENABLE_SLIPPAGE,
            slippage_bps_fallback: SLIPPAGE_BPS_FALLBACK,
            min_barrier_pct: MIN_BARRIER_PCT,
            max_barrier_pct: MAX_BARRIER_PCT,
            risk_pct_per_trade: RISK_PCT_PER_TRADE,
            max_leverage: MAX_LEVERAGE,
            r_clip_cap: R_CLIP_CAP,
            use_orderflow_features: USE_ORDERFLOW_FEATURES,
            hysteresis_buffer: HYSTERESIS_BUFFER,
            require_fold_coverage: false,
            one_candidate_per_bar: false,
            starting_equity: 10_000.0,
        }
    }
}

/// Composite (TIMEFRAME_BARRIER = 300 s) macro features — the regime filter.
/// Recomputed once per 300 s boundary and forward-filled to every 1 s micro bar
/// (matches `LiveFeatureEngine._compute_macro_features` + `.ffill()`).
/// One exported walk-forward fold model plus its out-of-sample test window
/// (`build/walkforward_models/catalog.json`, written by
/// `research/ofmr/ml_state_optimized.py` STEP 6a). Scoring time t with the fold
/// covering t reproduces walk-forward evaluation in a live-legal way: in real
/// trading you periodically retrain and deploy the freshly fitted model --
/// exactly what each fold represents.
#[derive(Debug, Clone)]
pub struct FoldModel {
    pub fold: u32,
    /// Inclusive OOS test-window bounds (ms epoch), from the catalog.
    pub test_start_ms: i64,
    pub test_end_ms: i64,
    pub model: crate::lgbm::LightGbm,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MacroFeatures {
    /// RSI(2) on composite-bar closes — the hunting-zone regime filter.
    pub rsi_2: f64,
    /// SMA(14) of RSI(14) on composite-bar closes.
    pub sma_14: f64,
    /// MA(5) velocity on composite-bar closes.
    pub ma_slope_5: f64,
    /// MA acceleration — d² derivative of composite-bar closes.
    pub ma_accel: f64,
    /// DeepLOB spread in bps, z-scored (window = DEEPLOB_Z_WINDOW).
    pub spread_bps_z: f64,
    /// RSI(2) z-score × MA slope interaction.
    pub rsi_x_ma: f64,
    /// Std of composite-bar-to-bar returns (barrier sizing only, not an ML feature).
    pub bar_vol_std: f64,
    /// Latest MA(5) value — used only by the model-free P(WIN) stand-in.
    pub ma5_last: f64,
    /// price_roc_3 on the 300 s COMPOSITE close series (`pct_change(3)`),
    /// matching research `compute_price_features` (feature_builder.py:336).
    /// The model was trained against this macro ROC, NOT a 1 s micro ROC, so
    /// the live engine must feed the same quantity (see "research/live drift").
    pub price_roc_3: f64,

    // --- orderflow features (ml_state_optimized.py feature block) ---
    pub ofi_z: f64,
    pub cvd_delta_z: f64,
    pub cvd_slope_z: f64,
    pub absorption_z: f64,
    pub tick_autocorr: f64,
    pub inst_ratio: f64,
    pub delta_divergence_z: f64,
    pub depth_imbalance_z: f64,
}

/// Feature snapshot assembled at every closed 1-second (micro) bar — the full
/// 15-feature vector = micro `price_roc_3` + forward-filled macro features
/// plus the orderflow block (mirrors `LiveFeatureEngine.compute_1s_feature_vector`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Feature {
    pub ts: i64, // bar close timestamp (ms)
    pub close: f64,
    pub n_bars: usize,
    pub spread_bps: f64,
    pub spread_bps_z: f64,
    pub ma_slope_5: f64,
    pub rsi_x_ma: f64,
    pub ma_accel: f64,
    pub price_roc_3: f64,
    pub rsi_2: f64,
    pub sma_14: f64,
    pub bar_vol_std: f64,
    /// Model-free P(WIN) ≥ 0.50 gate (LightGBM stand-in). See module docs.
    pub win_prob: f64,
    /// Traded base-asset volume of this 1 s bar (research `volume` column;
    /// zero-volume rows sit outside the research trade universe).
    pub volume: f64,
}

/// Trade direction of an open/closed position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Long,
    Short,
}

impl Side {
    pub fn as_str(&self) -> &'static str {
        match self {
            Side::Long => "long",
            Side::Short => "short",
        }
    }
}

/// Live position state (mirrors `PositionTracker`).
#[derive(Debug, Clone)]
pub struct Position {
    pub side: Side,
    pub entry_ms: i64,
    pub entry_mid: f64,
    pub entry_price: f64,
    pub qty: f64,
    pub notional: f64,
    pub barrier_pct: f64,
    pub stop: f64,
    pub target: f64,
    pub hi: f64,
    pub lo: f64,
}

/// Event emitted at order entry (mirrors `PositionTracker.enter_*` receipt).
#[derive(Debug, Clone)]
pub struct EntryReceipt {
    pub ts: i64,
    pub side: Side,
    pub entry_mid: f64,
    pub entry_price: f64,
    pub qty: f64,
    pub notional: f64,
    pub barrier_pct: f64,
    pub stop: f64,
    pub target: f64,
    pub rsi_2: f64,
    pub win_prob: f64,
}

/// Closed-trade record; `reason` ∊ {stop_loss, take_profit, timeout}.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub entry_ms: i64,
    pub exit_ms: i64,
    pub side: Side,
    pub entry_price: f64,
    pub exit_price: f64,
    pub reason: &'static str,
    pub is_timeout: bool,
    pub qty: f64,
    pub notional: f64,
    pub barrier_pct: f64,
    pub gross_usd: f64,
    pub net_usd: f64,
    pub fees: f64,
    pub r_multiple: f64,
    /// R-multiple BEFORE any clip (`r_multiple` == `r_raw` unless
    /// `r_clip_cap`/the dual-pass parity clip bound). Research scores timeouts
    /// as `raw_move / barrier_pct` — with a tiny barrier that explodes, and
    /// research clamps it dataset-wide; this field keeps the raw value so the
    /// same clamp can be applied post-hoc.
    pub r_raw: f64,
    pub equity_after: f64,
    pub duration_s: f64,
}

/// Events the live-trade/main.rs layer reacts to.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// A TIMEFRAME_BARRIER bar closed with its computed feature set.
    BarClosed(Feature),
    /// A dry-run order was filled.
    Entered(EntryReceipt),
    /// A position was closed (barrier / timeout).
    Exited(TradeRecord),
}

/// Cheap read-only snapshot for consoles / report writers.
#[derive(Debug, Clone, Default)]
pub struct StrategySnapshot {
    pub symbol: String,
    pub equity: f64,
    pub n_trades: u64,
    pub n_wins: u64,
    pub n_losses: u64,
    pub total_r: f64,
    pub mid: f64,
    pub open_side: Option<Side>,
    pub last_features: Option<Feature>,
}

/// Streaming feature + dry-run execution engine (research-faithful).
pub struct MomentumStrategy {
    pub cfg: StrategyConfig,
    pub symbol: String,

    // --- micro (1 s) bar aggregation state ---
    micro_idx: Option<i64>,
    micro_high: f64,
    micro_low: f64,
    micro_close: f64,
    micro_closes: VecDeque<f64>,
    micro_volume: f64,           // traded volume of the in-progress 1 s bar // completed 1 s closes → price_roc_3 (micro)

    // --- macro (TIMEFRAME_BARRIER / regime) aggregation state ---
    macro_idx: Option<i64>,
    macro_high: f64,
    macro_low: f64,
    macro_close: f64,
    macro_closes: VecDeque<f64>, // composite-bar closes → RSI/MA slope+accel/vol
    spreads: VecDeque<f64>,      // composite-bar spread_bps → spread_bps_z
    rsi2_hist: VecDeque<f64>,    // composite-bar RSI(2) → rsi_x_ma
    macro_features: Option<MacroFeatures>, // forward-filled to every 1 s bar

    // --- research LightGBM model → real P(WIN); None = fallback logit proxy ---
    model: Option<crate::lgbm::LightGbm>,
    /// Walk-forward fold models (live-test replay parity). When non-empty,
    /// every 1 s row is scored by the fold whose OOS test window covers it;
    /// rows outside every window score -1.0 (untradable) -- mirroring
    /// research's fold-local out-of-sample evaluation.
    fold_models: Vec<FoldModel>,

    // --- orderflow accumulation (mirrors features.py, reset per composite bar) ---
    ofi_prev_bid_price: Option<f64>,
    ofi_prev_bid_amount: f64,
    ofi_prev_ask_price: Option<f64>,
    ofi_prev_ask_amount: f64,
    bar_ofi_sum: f64,
    prev_bar_ofi: f64,
    ofi_diff_buffer: VecDeque<f64>,
    bar_cvd_net: f64,
    cvd_delta_buffer: VecDeque<f64>,
    cvd_slope_buffer: VecDeque<f64>,
    bar_ask_high: Option<f64>,
    bar_bid_low: Option<f64>,
    absorption_buffer: VecDeque<f64>,
    last_trade_price: Option<f64>,
    second_last_tick_sign: i32,
    bar_tick_signs: Vec<i32>,
    bar_retail_buy: f64,
    bar_retail_sell: f64,
    bar_inst_buy: f64,
    bar_inst_sell: f64,
    delta_divergence_buffer: VecDeque<f64>,
    depth_imbalance_buffer: VecDeque<f64>,
    last_bid_amount: f64,
    last_ask_amount: f64,

    // --- market state ---
    mid: f64,
    last_spread_bps: f64,

    // --- feature / accounting state ---
    last_features: Option<Feature>,
    last_bar_close_ms: i64,
    /// Newest volume>0 feature row of the current composite bar (used by the
    /// one-candidate-per-bar research-universe gate; entered at rollover).
    candidate: Option<Feature>,

    position: Option<Position>,
    last_exit_ms: i64,
    /// Which entry gate rejected the most recent entry evaluation ("" = none /
    /// passed). Purely observational live telemetry — see `entry_gate()`.
    last_entry_block: &'static str,
    /// Next re-entry allowed timestamp (ms). Research lockout: a trade taken
    /// at entry time T with actual duration D blocks re-entry until
    /// T + D + 1 s — i.e. exactly `exit_ms + 1000`. Mirrors research
    /// `next_available_time = actual_global_time + (bars_to_exit + 1)`.
    next_available_entry_ms: i64,
    equity: f64,
    n_trades: u64,
    n_wins: u64,
    n_losses: u64,
    total_r: f64,
    closed: Vec<TradeRecord>,
}

// ============================================================================
// R-clip (research STEP 5 parity)
// ============================================================================

/// Research STEP 5's dataset-wide R clip cap:
/// `cap = max(|q(0.99)|, |q(0.01)|, rr_ratio)` over the R-multiple series
/// (NaN-skipping, linear interpolation — matches `np.nanquantile`). Returns
/// `None` for an empty/all-NaN series.
///
/// This is the DUAL PASS of the live-test parity flow: the replay streams each
/// trade's uncapped `r_raw`, then this cap is computed post-hoc over the full
/// session and applied — reproducing exactly what research does to
/// `normalized_return` before expectancy is ever computed.
pub fn dataset_r_cap(rr_ratio: f64, r_multiples: &[f64]) -> Option<f64> {
    let mut rs: Vec<f64> = r_multiples
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();
    if rs.is_empty() {
        return None;
    }
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = |p: f64| -> f64 {
        let n = rs.len();
        if n == 1 {
            return rs[0];
        }
        let pos = p * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        rs[lo] + (pos - lo as f64) * (rs[hi] - rs[lo])
    };
    Some(quantile(0.99).abs().max(quantile(0.01).abs()).max(rr_ratio))
}

impl MomentumStrategy {
    pub fn new(symbol: &str, cfg: StrategyConfig) -> Self {
        let starting_equity = cfg.starting_equity;
        Self {
            cfg,
            symbol: symbol.to_string(),
            micro_idx: None,
            micro_high: 0.0,
            micro_low: f64::INFINITY,
            micro_close: 0.0,
            micro_closes: VecDeque::with_capacity(MAX_MICRO_CLOSES),
            micro_volume: 0.0,
            macro_idx: None,
            macro_high: 0.0,
            macro_low: f64::INFINITY,
            macro_close: 0.0,
            macro_closes: VecDeque::with_capacity(MAX_MACRO_CLOSES),
            spreads: VecDeque::with_capacity(MAX_SPREADS),
            rsi2_hist: VecDeque::with_capacity(MAX_RSI_HIST),
            macro_features: Some(MacroFeatures {
                rsi_2: 100.0,
                sma_14: 50.0,
                ..Default::default()
            }),
            model: None,
            fold_models: Vec::new(),
            ofi_prev_bid_price: None,
            ofi_prev_bid_amount: 0.0,
            ofi_prev_ask_price: None,
            ofi_prev_ask_amount: 0.0,
            bar_ofi_sum: 0.0,
            prev_bar_ofi: 0.0,
            ofi_diff_buffer: VecDeque::with_capacity(MAX_Z_HIST),
            bar_cvd_net: 0.0,
            cvd_delta_buffer: VecDeque::with_capacity(MAX_Z_HIST),
            cvd_slope_buffer: VecDeque::with_capacity(MAX_Z_HIST),
            bar_ask_high: None,
            bar_bid_low: None,
            absorption_buffer: VecDeque::with_capacity(MAX_Z_HIST),
            last_trade_price: None,
            second_last_tick_sign: 0,
            bar_tick_signs: Vec::new(),
            bar_retail_buy: 0.0,
            bar_retail_sell: 0.0,
            bar_inst_buy: 0.0,
            bar_inst_sell: 0.0,
            delta_divergence_buffer: VecDeque::with_capacity(MAX_Z_HIST),
            depth_imbalance_buffer: VecDeque::with_capacity(MAX_Z_HIST),
            last_bid_amount: 0.0,
            last_ask_amount: 0.0,
            mid: 0.0,
            last_spread_bps: 0.0,
            last_features: None,
            last_bar_close_ms: 0,
            candidate: None,
            position: None,
            last_exit_ms: i64::MIN,
            last_entry_block: "",
            next_available_entry_ms: i64::MIN,
            equity: starting_equity,
            n_trades: 0,
            n_wins: 0,
            n_losses: 0,
            total_r: 0.0,
            closed: Vec::new(),
        }
    }

    /// Restore persisted dry-run state across process restarts.
    pub fn restore(&mut self, equity: f64, last_exit_ms: i64) {
        if equity > 0.0 {
            self.equity = equity;
        }
        self.last_exit_ms = last_exit_ms;
    }

    // ---- accessors --------------------------------------------------------

    pub fn equity(&self) -> f64 {
        self.equity
    }

    pub fn mid(&self) -> f64 {
        self.mid
    }

    pub fn closed_trades(&self) -> &[TradeRecord] {
        &self.closed
    }

    pub fn open_position(&self) -> Option<&Position> {
        self.position.as_ref()
    }

    /// Which entry gate rejected the most recent bar-close evaluation:
    /// `"in_position"`, `"lockout"`, `"vol_filter"`, `"hunting_zone"`,
    /// `"confidence"`, or `""` when nothing blocked (an entry fired or no
    /// candidate was evaluated). Observational only — never gates behavior.
    pub fn entry_gate(&self) -> &'static str {
        self.last_entry_block
    }

    pub fn snapshot(&self) -> StrategySnapshot {
        StrategySnapshot {
            symbol: self.symbol.clone(),
            equity: self.equity,
            n_trades: self.n_trades,
            n_wins: self.n_wins,
            n_losses: self.n_losses,
            total_r: self.total_r,
            mid: self.mid,
            open_side: self.position.as_ref().map(|p| p.side),
            last_features: self.last_features,
        }
    }

    // ---- market-data entry points ----------------------------------------

    /// Feed a single trade print (`is_buy` = aggressor bought).
    pub fn on_trade(&mut self, ts: i64, price: f64, amount: f64, is_buy: bool) -> Vec<EngineEvent> {
        if self.mid == 0.0 {
            self.mid = price; // bootstrap before first orderbook snapshot
        }
        if self.orderflow_enabled() {
            // CVD: signed volume per composite bar (features.py `_bar_cvd_net`).
            self.bar_cvd_net += if is_buy { amount } else { -amount };

            // Retail vs. institutional cohorts (ml_state_optimized Step 5).
            if amount < RETAIL_THRESHOLD {
                if is_buy {
                    self.bar_retail_buy += amount;
                } else {
                    self.bar_retail_sell += amount;
                }
            } else if is_buy {
                self.bar_inst_buy += amount;
            } else {
                self.bar_inst_sell += amount;
            }

            // Tick sign (tick rule) vs the previous TRADE price. `resample('1s')
            // .last()` means only the final tick sign of each second survives —
            // captured at micro-bar rollover in `finish_micro`, so here we just
            // keep the running "last seen" sign.
            if let Some(prev) = self.last_trade_price {
                self.second_last_tick_sign = if price > prev {
                    1
                } else if price < prev {
                    -1
                } else {
                    0
                };
            }
            self.last_trade_price = Some(price);
        }
        if !self.orderflow_enabled() {
            // Keep the fill-price reference (research `execution_price`) even
            // when orderflow accumulators are off — `evaluate_entry` uses it.
            self.last_trade_price = Some(price);
        }
        self.push_price(ts, price, Some((amount, is_buy)))
    }

    /// Feed a level-1 orderbook snapshot. Quotes update midpoint/spread and
    /// orderflow state, and the composite (macro) bar's OHLC — research
    /// `_build_price_bars` is built from the orderbook MID, so RSI/MA/vol and
    /// price_roc_3 track mid, not last-trade price. Does not create
    /// trade-derived micro bars or exits.
    pub fn on_orderbook(
        &mut self,
        ts: i64,
        bid: f64,
        _bid_amt: f64,
        ask: f64,
        _ask_amt: f64,
    ) -> Vec<EngineEvent> {
        let mut events: Vec<EngineEvent> = Vec::new();
        let mid = 0.5 * (bid + ask);
        if mid > 0.0 {
            self.mid = mid;
            self.last_spread_bps = (ask - bid) / mid * 10_000.0;
            // Composite (macro) bars are built from the orderbook MID, exactly
            // like research `_build_price_bars(df_orderbook)`. A rollover here
            // can emit a deferred candidate entry (one-candidate-per-bar mode).
            events.extend(self.push_macro_price(ts, mid));
        }
        if self.orderflow_enabled() {
            // OFI (Cont et al.) — needs the PREVIOUS tick, computed BEFORE
            // overwriting the prev snapshot (features.py on_orderbook Step 1).
            if let (Some(prev_bid_p), Some(prev_ask_p)) = (
                self.ofi_prev_bid_price.as_ref(),
                self.ofi_prev_ask_price.as_ref(),
            ) {
                let delta_bid = if bid > *prev_bid_p {
                    _bid_amt
                } else if bid < *prev_bid_p {
                    -self.ofi_prev_bid_amount
                } else {
                    _bid_amt - self.ofi_prev_bid_amount
                };
                let delta_ask = if ask < *prev_ask_p {
                    _ask_amt
                } else if ask > *prev_ask_p {
                    -self.ofi_prev_ask_amount
                } else {
                    _ask_amt - self.ofi_prev_ask_amount
                };
                self.bar_ofi_sum += delta_bid - delta_ask;
            }
            self.ofi_prev_bid_price = Some(bid);
            self.ofi_prev_bid_amount = _bid_amt;
            self.ofi_prev_ask_price = Some(ask);
            self.ofi_prev_ask_amount = _ask_amt;

            // Absorption ratio price range: bar's ask-high / bid-low
            // (features.py `_bar_ask_high` / `_bar_bid_low`).
            self.bar_ask_high = Some(self.bar_ask_high.map_or(ask, |h| h.max(ask)));
            self.bar_bid_low = Some(self.bar_bid_low.map_or(bid, |l| l.min(bid)));

            // Latest depth snapshot for depth_imbalance_z.
            self.last_bid_amount = _bid_amt;
            self.last_ask_amount = _ask_amt;
        }
        events
    }

    // ---- core bar/price pipeline -----------------------------------------

    /// Advance the composite (TIMEFRAME_BARRIER) bar with a price tick from
    /// either a trade print or an orderbook mid. On the first event of a new
    /// composite bar the macro/regime features are recomputed once and then
    /// forward-filled to every subsequent micro bar until the next rollover
    /// (matches research `df_micro.join(df_macro).ffill()`).
    fn push_macro_price(&mut self, ts: i64, price: f64) -> Vec<EngineEvent> {
        let macro_ms = self.cfg.timeframe_barrier_secs * 1000;
        let macro_idx = ts / macro_ms;
        match self.macro_idx {
            None => {
                self.start_macro(macro_idx, price);
                Vec::new()
            }
            Some(cur) if macro_idx != cur => {
                self.compute_macro_features();
                // Research trade universe: ONE entry candidate per composite
                // bar -- its LAST volume>0 1-second row (the `volume > 0`
                // filter runs BEFORE `deduplicate_to_bar`, which keeps the
                // last row per bucket). The candidate is stashed in
                // `finish_micro`; enter from it now, at its own bar-close
                // time and price, exactly like the research loop.
                let mut events = Vec::new();
                if self.cfg.one_candidate_per_bar {
                    if let Some(cand) = self.candidate.take() {
                        let fill = cand.close;
                        self.try_enter(cand, fill, &mut events);
                    }
                }
                self.start_macro(macro_idx, price);
                events
            }
            Some(_) => {
                if self.macro_idx == Some(macro_idx) {
                    self.macro_high = self.macro_high.max(price);
                    self.macro_low = self.macro_low.min(price);
                    self.macro_close = price;
                }
                Vec::new()
            }
        }
    }

    fn push_price(&mut self, ts: i64, price: f64, trade: Option<(f64, bool)>) -> Vec<EngineEvent> {
        let micro_ms = self.cfg.micro_barrier_secs * 1000;
        let micro_idx = ts / micro_ms;
        let mut events = Vec::new();

        // 1. Micro (1 s) bar rollover → finalize the completed 1 s bar with the
        //    full feature vector (micro grid + forward-filled macro) and
        //    evaluate a fresh entry from it. The model runs on every 1 s row.
        match self.micro_idx {
            None => self.start_micro(micro_idx, price),
            Some(cur) if micro_idx > cur => {
                // The research state space is dense at one-second resolution.
                // Close every missing interval at the last trade price so a
                // sparse feed cannot change model cadence or timeout timing.
                let mut idx = cur;
                while idx < micro_idx {
                    let bar_close = self.micro_close;
                    let bar_ms = idx * micro_ms;
                    let feat = self.finish_micro(bar_close, bar_ms);
                    events.push(EngineEvent::BarClosed(feat));

                    // Research `_sequential_triple_barrier` checks barrier
                    // crossings against the 1-SECOND BAR CLOSE series only
                    // (sliding_window_view of 1s closes) — never on an
                    // intra-second wick. Evaluate exits on the same grid.
                    if let Some(rec) = self.check_exit(bar_ms, bar_close) {
                        events.push(EngineEvent::Exited(rec));
                    }

                    self.evaluate_entry(&mut events);
                    let next_idx = idx + 1;
                    self.start_micro(next_idx, bar_close);
                    idx = next_idx;
                }
            }
            Some(cur) if micro_idx < cur => return events,
            Some(_) => {}
        }

        // 2. Macro (regime) rollover + OHLC — driven by price events from
        //    EITHER source (trade print or orderbook mid). The first event of
        //    a new 300 s bar recomputes the composite features ONCE from the
        //    just-closed bar and forward-fills them to subsequent 1 s bars.
        events.extend(self.push_macro_price(ts, price));

        // 3. Update the in-progress micro bar OHLC (macro handled above).
        if self.micro_idx == Some(micro_idx) {
            self.micro_high = self.micro_high.max(price);
            self.micro_low = self.micro_low.min(price);
            self.micro_close = price;
        }

        if let Some((amount, _)) = trade {
            self.mid = price; // last-trade fallback mirrors live last price feed
            if self.micro_idx == Some(micro_idx) {
                self.micro_volume += amount; // research `volume` per 1 s row
            }
        }

        // Exits are evaluated on 1-second bar closes only (see the rollover
        // loop above) to match research's triple-barrier-on-1s-close semantics.
        events
    }

    fn start_micro(&mut self, idx: i64, price: f64) {
        self.micro_idx = Some(idx);
        self.micro_high = price;
        self.micro_low = price;
        self.micro_close = price;
        self.micro_volume = 0.0;
    }

    fn start_macro(&mut self, idx: i64, price: f64) {
        self.macro_idx = Some(idx);
        self.macro_high = price;
        self.macro_low = price;
        self.macro_close = price;
    }

    /// Finalize the completed 1-second (micro) bar: assemble the 15-feature
    /// vector — micro `price_roc_3` (fresh every second) + macro/regime and
    /// orderflow features forward-filled from the last TIMEFRAME_BARRIER
    /// rollover (mirrors `LiveFeatureEngine.compute_1s_feature_vector`).
    fn finish_micro(&mut self, close: f64, close_ms: i64) -> Feature {
        self.last_bar_close_ms = close_ms;

        // Micro price rate-of-change is deliberately NOT computed here on 1 s
        // closes: the research state space computes `price_roc_3` on the 300 s
        // COMPOSITE close series (`feature_builder.py: close.pct_change(3)`)
        // and broadcasts it to every 1 s row. `m.price_roc_3` carries that
        // macro ROC (computed in `compute_macro_features`). The model was
        // trained on the macro ROC — feeding a 1 s ROC is a research/live
        // divergence that shifts the single most important model input.
        self.micro_closes.push_back(close);
        if self.micro_closes.len() > MAX_MICRO_CLOSES {
            self.micro_closes.pop_front();
        }

        // Capture this (just-closed) 1-second bar's surviving tick sign into the
        // composite-bar autocorrelation series, then reset (features.py
        // compute_1s_feature_vector appends `_second_last_tick_sign` and resets).
        if self.orderflow_enabled() {
            self.bar_tick_signs.push(self.second_last_tick_sign);
            self.second_last_tick_sign = 0;
        }

        let m = self.macro_features.unwrap_or_default();

        // --- Real model P(WIN) over the full 15-feature vector, in the exact
        // order produced by ml_state_optimized.py / recorded in
        // lgb_model.features.json: 7 base + 8 orderflow. Falls back to the
        // deterministic logit proxy only when no model is loaded (so the
        // engine is still usable offline / without an export). ---
        let features: [f64; 15] = [
            m.spread_bps_z,
            m.ma_slope_5,
            m.rsi_x_ma,
            m.ma_accel,
            m.price_roc_3,
            m.rsi_2,
            m.sma_14,
            m.ofi_z,
            m.cvd_delta_z,
            m.cvd_slope_z,
            m.absorption_z,
            m.tick_autocorr,
            m.inst_ratio,
            m.delta_divergence_z,
            m.depth_imbalance_z,
        ];
        // Walk-forward parity: when fold models are loaded, the row MUST be
        // scored by the fold whose OOS test window covers it (research never
        // evaluates a row with any other fold's model). Rows outside every
        // window score -1.0, which can never pass the entry threshold --
        // matching research, where such rows are simply never evaluated.
        let win_prob = if !self.fold_models.is_empty() {
            match self
                .fold_models
                .iter()
                .find(|f| close_ms >= f.test_start_ms && close_ms <= f.test_end_ms)
            {
                Some(f) => f.model.predict_proba(&features),
                None => -1.0,
            }
        } else {
            match self.model.as_ref() {
                Some(mdl) => mdl.predict_proba(&features),
                None => self.proxy_win_prob(&m, m.price_roc_3, close),
            }
        };

        let feat = Feature {
            ts: close_ms,
            close,
            n_bars: self.micro_closes.len(),
            spread_bps: self.last_spread_bps,
            spread_bps_z: m.spread_bps_z,
            ma_slope_5: m.ma_slope_5,
            rsi_x_ma: m.rsi_x_ma,
            ma_accel: m.ma_accel,
            price_roc_3: m.price_roc_3,
            rsi_2: m.rsi_2,
            sma_14: m.sma_14,
            bar_vol_std: m.bar_vol_std,
            win_prob,
            volume: self.micro_volume,
        };
        self.last_features = Some(feat);
        // One-entry-candidate-per-composite-bar mode: remember the newest
        // volume>0 row of this composite bar; the macro rollover enters from
        // the last one (research `volume>0` filter -> `deduplicate_to_bar`
        // keep-last).
        if self.cfg.one_candidate_per_bar && self.micro_volume > 0.0 {
            self.candidate = Some(feat);
        }
        feat
    }

    fn evaluate_entry(&mut self, events: &mut Vec<EngineEvent>) {
        if self.position.is_some() {
            return;
        }
        let Some(feat) = self.last_features else {
            return;
        };

        // Research fills at `execution_price` = the LAST TRADE PRICE (the
        // state-space grid's unshifted micro-bar price). The engine's `mid`
        // mixes book mid with last-trade price; prefer the pure trade-print
        // fill to match the trained entry basis.
        let fill_price = self
            .last_trade_price
            .filter(|p| *p > 0.0)
            .or_else(|| if self.mid > 0.0 { Some(self.mid) } else { None })
            .unwrap_or(feat.close);
        self.try_enter(feat, fill_price, events);
    }

    /// Shared entry pipeline: lockout -> barrier-size filter -> hunting zone
    /// -> pure P(WIN) threshold. Research `find_debounced_signals` is a PURE
    /// threshold (`np.where(y_proba >= t)`) with NO hysteresis state; the
    /// per-trade-duration lockout alone governs trade spacing. A negative
    /// `win_prob` means "no walk-forward fold covers this time" (research
    /// never evaluates such rows) and can never pass the threshold.
    fn try_enter(&mut self, feat: Feature, fill_price: f64, events: &mut Vec<EngineEvent>) {
        if self.position.is_some() {
            self.last_entry_block = "in_position";
            return;
        }
        // Research-style lockout: re-entry blocked until
        // entry_time + actual_duration + 1 s (== exit_ms + 1000 for barrier
        // exits; entry + LOOKAHEAD for timeouts). LOCKOUT_SECONDS is used as
        // an on/off flag only -- exactly like the research loop.
        if self.cfg.lockout_seconds > 0 && feat.ts < self.next_available_entry_ms {
            self.last_entry_block = "lockout";
            return;
        }

        // Vol filter mirrors `_passes_vol_filter` (MIN/MAX_BARRIER_PCT).
        let bp = self.barrier_pct();
        if bp < self.cfg.min_barrier_pct || bp > self.cfg.max_barrier_pct {
            self.last_entry_block = "vol_filter";
            return;
        }

        let (in_zone, side) = self.resolve_zone_side(&feat);
        if !in_zone {
            self.last_entry_block = "hunting_zone";
            return;
        }

        if feat.win_prob + 1e-12 < self.cfg.confidence_threshold {
            self.last_entry_block = "confidence";
            return; // LOW_CONFIDENCE (also rejects the -1 no-fold sentinel)
        }

        self.last_entry_block = "";
        if !self.enter_position(feat.ts, side, fill_price, bp) {
            self.last_entry_block = "min_order_size";
            return;
        }

        let p = self.position.as_ref().expect("position just opened");
        events.push(EngineEvent::Entered(EntryReceipt {
            ts: p.entry_ms,
            side: p.side,
            entry_mid: p.entry_mid,
            entry_price: p.entry_price,
            qty: p.qty,
            notional: p.notional,
            barrier_pct: p.barrier_pct,
            stop: p.stop,
            target: p.target,
            rsi_2: feat.rsi_2,
            win_prob: feat.win_prob,
        }));
    }

    /// Resolve the hunting zone + trade side exactly like research
    /// `_infer_trade_side` (+ executor's zone rules).
    fn resolve_zone_side(&self, feat: &Feature) -> (bool, Side) {
        let thr = self.cfg.rsi_filter_threshold;
        let high_thr = 100.0 - thr;
        let dir_side = if self.cfg.trade_direction_is_long {
            Side::Long
        } else {
            Side::Short
        };

        if self.cfg.enable_rsi_accel_filter {
            // Two-sided directional zone: (rsi<thr & acc>0)→long, (rsi>100-thr
            // & acc<0)→short, then flipped by TRADE_DIRECTION.
            let zone_side = if feat.rsi_2 < thr && feat.ma_accel > 0.0 {
                1
            } else if feat.rsi_2 > high_thr && feat.ma_accel < 0.0 {
                -1
            } else {
                return (false, dir_side);
            };
            let side = if (zone_side == 1) == self.cfg.trade_direction_is_long {
                Side::Long
            } else {
                Side::Short
            };
            (true, side)
        } else {
            // Single zone: rsi_2 < threshold; direction = global invert.
            (feat.rsi_2 < thr, dir_side)
        }
    }

    /// Open a position. Returns `false` (nothing opened) when risk-normalized
    /// sizing cannot honor the exchange's `min_order_size` without exceeding
    /// the per-trade risk target.
    fn enter_position(&mut self, ts: i64, side: Side, mid: f64, bp: f64) -> bool {
        let slip = if self.cfg.enable_slippage {
            self.cfg.slippage_bps_fallback
        } else {
            0.0
        };
        let entry_price = match side {
            Side::Long => mid * (1.0 + slip / 10_000.0),
            Side::Short => mid * (1.0 - slip / 10_000.0),
        };
        // Position sizing — two modes:
        //
        // - `risk_pct_per_trade == 0` (default / research parity): fixed
        //   notional = `max_position_usd`. Dollar risk then floats with each
        //   entry's volatility-scaled barrier (notional × barrier_pct), so 1R
        //   is worth a DIFFERENT dollar amount per symbol and per trade — this
        //   is why $-PnL and R-multiples diverge across the portfolio.
        //
        // - `risk_pct_per_trade > 0` (volatility-normalized): risk
        //   `equity × risk_pct_per_trade` USD per trade by sizing
        //   notional = risk / barrier_pct, capped at `max_leverage × equity`.
        //   A full stop-out now always loses the same fraction of equity,
        //   regardless of symbol or current volatility.
        let qty = if self.cfg.risk_pct_per_trade > 0.0 && bp > EPS && self.equity > 0.0 {
            let risk_usd = self.equity * self.cfg.risk_pct_per_trade;
            let notional = (risk_usd / bp).min(self.equity * self.cfg.max_leverage);
            let qty = notional / entry_price;
            if qty < self.cfg.min_order_size {
                // The venue minimum would force oversizing beyond the risk
                // target — decline the entry instead of silently taking more
                // risk than configured. (The legacy path below keeps its floor
                // for backwards compatibility.)
                return false;
            }
            qty
        } else {
            (self.cfg.max_position_usd / entry_price).max(self.cfg.min_order_size)
        };
        let notional = qty * entry_price;
        let (stop, target) = match side {
            Side::Long => (
                entry_price * (1.0 - bp),
                entry_price * (1.0 + bp * self.cfg.rr_ratio),
            ),
            Side::Short => (
                entry_price * (1.0 + bp),
                entry_price * (1.0 - bp * self.cfg.rr_ratio),
            ),
        };
        self.position = Some(Position {
            side,
            entry_ms: ts,
            entry_mid: mid,
            entry_price,
            qty,
            notional,
            barrier_pct: bp,
            stop,
            target,
            hi: mid,
            lo: mid,
        });
        true
    }

    /// Vol-scaled stop distance: `bar_vol_std × BARRIER_MULTIPLIER`, with the
    /// bootstrap floor `0.001 × BARRIER_MULTIPLIER` (research + PositionTracker).
    fn barrier_pct(&self) -> f64 {
        let vol = self.last_features.map(|f| f.bar_vol_std).unwrap_or(0.0);
        let bp = vol * self.cfg.barrier_multiplier;
        if bp > EPS && bp.is_finite() {
            bp
        } else {
            0.001 * self.cfg.barrier_multiplier
        }
    }

    // ---- exit monitoring (mirrors PositionTracker.check_exit) --------------

    /// Exit detection mirrors research `_sequential_triple_barrier`: it is
    /// called once per 1-SECOND bar close with the bar's CLOSE price, and a
    /// barrier only counts as hit when that close reaches it — never on an
    /// intra-second wick (the research state space has no sub-second data).
    /// Target-first → win, stop-first → loss, else the lookahead timeout at
    /// `entry + LOOKAHEAD_SECONDS`.
    fn check_exit(&mut self, ts: i64, close: f64) -> Option<TradeRecord> {
        let decision = match &mut self.position {
            Some(p) => {
                let timed_out = ts - p.entry_ms >= self.cfg.lookahead_seconds * 1000;
                let hit_stop = match p.side {
                    Side::Long => close <= p.stop,
                    Side::Short => close >= p.stop,
                };
                let hit_tp = match p.side {
                    Side::Long => close >= p.target,
                    Side::Short => close <= p.target,
                };

                if timed_out {
                    Some(("timeout", close))
                } else if hit_tp {
                    // First barrier the 1s close reaches wins (a single close
                    // cannot cross both opposite-side barriers, so stop-vs-target
                    // ordering is unambiguous; target-first = research win_mask).
                    Some(("take_profit", p.target))
                } else if hit_stop {
                    Some(("stop_loss", p.stop))
                } else {
                    None
                }
            }
            None => return None,
        };

        let (reason, theoretical_exit) = decision?;
        Some(self.close_position(ts, reason, theoretical_exit))
    }

    /// PnL accounting — byte-for-byte port of `PositionTracker._exit`.
    fn close_position(
        &mut self,
        ts: i64,
        reason: &'static str,
        theoretical_exit: f64,
    ) -> TradeRecord {
        let p = self
            .position
            .take()
            .expect("close_position with no open position");
        let is_timeout = reason == "timeout";

        let slip_bps = if self.cfg.enable_slippage {
            self.cfg.slippage_bps_fallback
        } else {
            0.0
        };
        let exit_price = match p.side {
            Side::Long => theoretical_exit * (1.0 - slip_bps / 10_000.0),
            Side::Short => theoretical_exit * (1.0 + slip_bps / 10_000.0),
        };

        let gross_pct = match p.side {
            Side::Long => (exit_price - p.entry_price) / p.entry_price,
            Side::Short => (p.entry_price - exit_price) / p.entry_price,
        };
        let gross_usd = gross_pct * p.notional;

        // Taker fee on both legs (FEE_BPS).
        let entry_fee = p.notional * self.cfg.fee_bps / 10_000.0;
        let exit_fee = p.qty * exit_price * self.cfg.fee_bps / 10_000.0;
        let fees = entry_fee + exit_fee;
        let net_usd = gross_usd - fees;

        let bp = if p.barrier_pct > 1e-9 {
            p.barrier_pct
        } else {
            0.001 * self.cfg.barrier_multiplier
        };

        // Research (ml_state_optimized STEP 5) scores barrier exits as a
        // hardcoded +RR_RATIO win / -1.0 loss (the actual exit overshoot is
        // ignored) and timeouts as the real side-adjusted return / bp. This is
        // exactly what the walk-forward reports as R — replicate it here.
        let side_sign = match p.side {
            Side::Long => 1.0,
            Side::Short => -1.0,
        };
        let base_r = if is_timeout {
            (theoretical_exit - p.entry_price) / p.entry_price * side_sign / bp
        } else {
            let favourable = (theoretical_exit - p.entry_price) * side_sign > 0.0;
            if favourable {
                self.cfg.rr_ratio
            } else {
                -1.0
            }
        };
        let slip_penalty = (slip_bps / 10_000.0) / bp;
        let fee_in_r = (fees / p.notional) / bp;
        // Uncapped R — exactly research's pre-clip `normalized_return`. Kept on
        // every record so the dual-pass dataset cap (research parity) or a
        // fixed `r_clip_cap` can be applied without losing information.
        let r_raw = base_r - slip_penalty - fee_in_r;
        let r_multiple = if self.cfg.r_clip_cap > 0.0 {
            r_raw.clamp(-self.cfg.r_clip_cap, self.cfg.r_clip_cap)
        } else {
            r_raw
        };

        self.equity += net_usd;
        self.n_trades += 1;
        if net_usd > 0.0 {
            self.n_wins += 1;
        } else {
            self.n_losses += 1;
        }
        self.total_r += r_multiple;
        self.last_exit_ms = ts;
        // Research lockout: `next_available_time = entry + (bars_to_exit + 1)s`.
        // Barrier exits: bars_to_exit = seconds held → next = exit_ms + 1s.
        // Timeouts: bars_to_exit = LOOKAHEAD - 1 → next = entry + LOOKAHEAD
        // (research sets the timeout offset to `lookahead - 1`).
        self.next_available_entry_ms = if is_timeout {
            p.entry_ms + self.cfg.lookahead_seconds * 1000
        } else {
            ts + 1000
        };

        let rec = TradeRecord {
            entry_ms: p.entry_ms,
            exit_ms: ts,
            side: p.side,
            entry_price: p.entry_price,
            exit_price,
            reason,
            is_timeout,
            qty: p.qty,
            notional: p.notional,
            barrier_pct: p.barrier_pct,
            gross_usd,
            net_usd,
            fees,
            r_multiple,
            r_raw,
            equity_after: self.equity,
            duration_s: (ts - p.entry_ms) as f64 / 1000.0,
        };
        self.closed.push(rec.clone());
        rec
    }

    // ---- model + orderflow lifecycle ------------------------------------------

    /// Install the research LightGBM model (loaded at startup). Once present,
    /// every 1-second feature vector is scored with the real 15-feature model
    /// instead of `proxy_win_prob`. Mirrors `ModelManager.load` + the
    /// feature-manifest check in `executor.py` (`_one_sec_tick`).
    pub fn set_model(&mut self, model: crate::lgbm::LightGbm) {
        self.model = Some(model);
    }

    /// Install walk-forward fold models (see [`FoldModel`]). When non-empty
    /// they take precedence over any single production model: each 1 s row is
    /// scored by the fold covering its timestamp, and rows outside every
    /// fold's test window score -1.0 (research never evaluates them).
    pub fn set_fold_models(&mut self, models: Vec<FoldModel>) {
        self.fold_models = models;
    }

    /// True if a real LightGBM model is installed and will be scored on the
    /// full 15-feature vector. `live-trade` uses this to decide whether to
    /// seed orderflow feeds.
    pub fn model_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Orderflow accumulators are only updated when the active model was
    /// exported with the orderflow block (`includes_orderflow`). This is the
    /// Rust mirror of `features.py`'s `USE_ORDERFLOW_FEATURES` gate.
    pub fn orderflow_enabled(&self) -> bool {
        self.cfg.use_orderflow_features
    }

    /// Seed the composite (macro) ring buffers from historical closes to avoid
    /// the cold-start gap (executor.py `seed_historical_data`). `closes` are
    /// TIMEFRAME_BARRIER-aggregated closes (5-min), same timeframe RSI/MA are
    /// computed on. Replicates: push each close through `_close_prices`/
    /// `_price_diffs`, then one `_compute_macro_features()` to warm rsi_2/sma_14
    /// /ma_slope/ma_accel immediately.
    pub fn seed_macro_closes(&mut self, closes: &[f64]) {
        if closes.is_empty() {
            return;
        }
        self.macro_closes.clear();
        for &c in closes {
            if self.macro_closes.len() >= MAX_MACRO_CLOSES {
                self.macro_closes.pop_front();
            }
            self.macro_closes.push_back(c);
        }
        // Warm the OHLC trackers for the in-progress composite bar.
        self.macro_close = *closes.last().expect("non-empty");
        self.macro_high = self
            .macro_closes
            .iter()
            .copied()
            .fold(self.macro_close, f64::max);
        self.macro_low = self
            .macro_closes
            .iter()
            .copied()
            .fold(self.macro_close, f64::min);
        // Force one macro feature computation so rsi_2/sma_14 are warm on the
        // first live bar (compute_macro_features will push `macro_close` once
        // more — identical to features.py seed path at lines 731-749).
        self.compute_macro_features();
    }

    /// Deterministic logit proxy for P(WIN) — the model-free stand-in used only
    /// when no research model is loaded (so the engine is still runnable
    /// offline / in unit tests). Preserved verbatim from the original engine
    /// logic; `finish_micro` calls the real LightGBM whenever `self.model`
    /// is `Some`.
    fn proxy_win_prob(&self, m: &MacroFeatures, price_roc_3: f64, close: f64) -> f64 {
        let dir = if self.cfg.trade_direction_is_long {
            1.0
        } else {
            -1.0
        };
        let vol2 = m.bar_vol_std.max(EPS);
        let thr = self.cfg.rsi_filter_threshold;
        let oversold = if dir > 0.0 {
            (thr - m.rsi_2) / 10.0
        } else {
            (m.rsi_2 - (100.0 - thr)) / 10.0
        };
        let score = 0.40 * tanh(oversold)
            + 0.20 * tanh(dir * m.ma_slope_5 / vol2 * 0.5)
            + 0.20 * tanh(dir * m.ma_accel / vol2 * 0.5)
            + 0.10 * tanh(dir * (close - m.ma5_last) / vol2 * 0.5)
            + 0.10 * tanh(dir * price_roc_3 / vol2 * 0.5);
        1.0 / (1.0 + (-score).exp())
    }

    // ---- composite (macro / regime) feature computation (mirrors main/features.py) ----

    /// Recompute the composite (TIMEFRAME_BARRIER = 300 s) features once at each
    /// macro-bar boundary. These ARE the regime filter — RSI(2) + MA
    /// acceleration (d² derivative) — and are forward-filled to every 1 s bar
    /// until the next 300 s boundary (matches
    /// `LiveFeatureEngine._compute_macro_features` + `.ffill()`).
    fn compute_macro_features(&mut self) {
        let close = self.macro_close;
        self.macro_closes.push_back(close);
        if self.macro_closes.len() > MAX_MACRO_CLOSES {
            self.macro_closes.pop_front();
        }

        let closes: Vec<f64> = self.macro_closes.iter().copied().collect();
        let n = closes.len();
        let diffs: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();

        // price_roc_3 — research `compute_price_features` (feature_builder.py):
        // `close.pct_change(3)` on the COMPOSITE (BAR_FREQ) close series. The
        // model was trained against this macro ROC; the live 1 s micro bars
        // inherit it via the forward-filled `MacroFeatures`, exactly like the
        // state space's asof broadcast.
        let mut price_roc_3 = 0.0;
        if closes.len() >= PRICE_ROC_WINDOW_3 + 1 {
            let prev = closes[closes.len() - 1 - PRICE_ROC_WINDOW_3];
            if prev.abs() > EPS {
                price_roc_3 = (close - prev) / prev;
            }
        }

        // Wilder RSI(2) / RSI(14) on composite-bar close diffs (get_wilders_rsi).
        let rsi2 = if diffs.len() >= RSI_2_WINDOW + 1 && !diffs.is_empty() {
            wilders_rsi(&diffs, RSI_2_WINDOW).unwrap_or(50.0)
        } else {
            50.0
        };
        let rsi14_series = if diffs.len() >= RSI_14_WINDOW + 1 {
            Some(wilders_rsi_series(&diffs, RSI_14_WINDOW))
        } else {
            None
        };
        let sma_14 = match rsi14_series.as_deref() {
            Some(s) => {
                let start = s.len().saturating_sub(SMA_14_WINDOW);
                let w = &s[start..];
                if w.is_empty() {
                    50.0
                } else {
                    w.iter().sum::<f64>() / w.len() as f64
                }
            }
            None => 50.0,
        };

        // bar_vol_std: population std (ddof=0) of composite-bar-to-bar returns
        // over the trailing BARRIER_VOL_WINDOW bars — feeds barrier sizing only.
        let mut bar_vol_std = 0.0;
        if n >= 3 {
            let rets: Vec<f64> = closes
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0].abs().max(EPS))
                .collect();
            let win = rets.len().min(BARRIER_VOL_WINDOW);
            if win >= 2 {
                bar_vol_std = pop_std(&rets[rets.len() - win..]);
            }
        }

        // MA(5) slope + acceleration — the d² derivative (features.py use of
        // np.convolve 'valid', then np.diff twice).
        let ma5 = sma_series(&closes, MA_SLOPE_WINDOW);
        let ma5_last = ma5.last().copied().unwrap_or(close);
        let ma_slope_5 = if ma5.len() >= 2 {
            ma5[ma5.len() - 1] - ma5[ma5.len() - 2]
        } else {
            0.0
        };
        let ma_accel = if ma5.len() >= 3 {
            (ma5[ma5.len() - 1] - ma5[ma5.len() - 2]) - (ma5[ma5.len() - 2] - ma5[ma5.len() - 3])
        } else {
            0.0
        };

        // DeepLOB spread in bps, sampled once per composite bar, z-scored
        // (features.py lines 443-461).
        self.spreads.push_back(self.last_spread_bps);
        if self.spreads.len() > MAX_SPREADS {
            self.spreads.pop_front();
        }
        let spread_vals: Vec<f64> = self.spreads.iter().copied().collect();
        let spread_bps_z = roll_z(&spread_vals, DEEPLOB_Z_WINDOW);

        // RSI(2) rolling z (w=20) then cross with MA slope (features.py 463-476).
        self.rsi2_hist.push_back(rsi2);
        if self.rsi2_hist.len() > MAX_RSI_HIST {
            self.rsi2_hist.pop_front();
        }
        let rsi_vals: Vec<f64> = self.rsi2_hist.iter().copied().collect();
        let rsi_z = roll_z(&rsi_vals, RSI_Z_WINDOW);
        let rsi_x_ma = rsi_z * ma_slope_5;

        // ================================================================
        // Orderflow features (ml_state_optimized.py Step 4 + z-scoring).
        // Mirrors features.py's per-bar orderflow block: raw accumulators
        // are summed over the composite bar, z-scored vs the trailing
        // ORDERFLOW_Z_WINDOW, then reset for the next bar.
        // ================================================================
        let (
            ofi_z,
            cvd_delta_z,
            cvd_slope_z,
            absorption_z,
            tick_autocorr,
            inst_ratio,
            delta_divergence_z,
            depth_imbalance_z,
        ) = if self.cfg.use_orderflow_features {
            let zw = ORDERFLOW_Z_WINDOW;

            // --- OFI: this bar's total minus last bar's total, z-scored ---
            let ofi_diff = self.bar_ofi_sum - self.prev_bar_ofi;
            let ofi_z = push_roll_z(&mut self.ofi_diff_buffer, ofi_diff, zw);
            self.prev_bar_ofi = self.bar_ofi_sum;

            // --- CVD: per-bar net delta + 5-bar slope, each z-scored ---
            let cvd_delta_z = push_roll_z(&mut self.cvd_delta_buffer, self.bar_cvd_net, zw);
            let cvd_hist: Vec<f64> = self.cvd_delta_buffer.iter().copied().collect();
            let cvd_slope_5 = if cvd_hist.len() >= 5 {
                cvd_hist[cvd_hist.len() - 5..].iter().sum()
            } else {
                0.0
            };
            let cvd_slope_z = push_roll_z(&mut self.cvd_slope_buffer, cvd_slope_5, zw);

            // --- Absorption ratio: CVD net / orderbook price range ---
            let price_range = match (self.bar_ask_high, self.bar_bid_low) {
                (Some(h), Some(l)) => h - l,
                _ => 0.0,
            };
            let abs_raw = self.bar_cvd_net / (price_range + EPS);
            let absorption_z = push_roll_z(&mut self.absorption_buffer, abs_raw, zw);

            // --- Trade-sign lag-1 autocorrelation over this bar's
            //     per-second tick-sign series ---
            let tick_autocorr = {
                let s: Vec<f64> = self.bar_tick_signs.iter().map(|&v| v as f64).collect();
                if s.len() > 5 && pop_std(&s) > 0.0 {
                    pearson_lag1(&s)
                } else {
                    0.0
                }
            };

            // --- Retail vs institutional flow divergence ---
            let bar_volume =
                self.bar_retail_buy + self.bar_retail_sell + self.bar_inst_buy + self.bar_inst_sell;
            let inst_delta = self.bar_inst_buy - self.bar_inst_sell;
            let retail_delta = self.bar_retail_buy - self.bar_retail_sell;
            let delta_divergence = inst_delta - retail_delta;
            let inst_ratio = (self.bar_inst_buy + self.bar_inst_sell) / (bar_volume + EPS);
            let delta_divergence_z =
                push_roll_z(&mut self.delta_divergence_buffer, delta_divergence, zw);

            // --- Depth imbalance: latest snapshot (bid_amt - ask_amt)/sum ---
            let depth_imb = (self.last_bid_amount - self.last_ask_amount)
                / (self.last_bid_amount + self.last_ask_amount + EPS);
            let depth_imbalance_z = push_roll_z(&mut self.depth_imbalance_buffer, depth_imb, zw);

            // --- Reset per-bar accumulators (features.py block at 562-571) ---
            self.bar_ofi_sum = 0.0;
            self.bar_cvd_net = 0.0;
            self.bar_ask_high = None;
            self.bar_bid_low = None;
            self.bar_tick_signs.clear();
            self.bar_retail_buy = 0.0;
            self.bar_retail_sell = 0.0;
            self.bar_inst_buy = 0.0;
            self.bar_inst_sell = 0.0;

            (
                ofi_z,
                cvd_delta_z,
                cvd_slope_z,
                absorption_z,
                tick_autocorr,
                inst_ratio,
                delta_divergence_z,
                depth_imbalance_z,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };

        self.macro_features = Some(MacroFeatures {
            rsi_2: rsi2,
            sma_14,
            ma_slope_5,
            ma_accel,
            spread_bps_z,
            rsi_x_ma,
            bar_vol_std,
            ma5_last,
            price_roc_3,
            ofi_z,
            cvd_delta_z,
            cvd_slope_z,
            absorption_z,
            tick_autocorr,
            inst_ratio,
            delta_divergence_z,
            depth_imbalance_z,
        });
    }
}

// ============================================================================
// Indicator helpers
// ============================================================================

fn tanh(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_sign_positive() { 1.0 } else { -1.0 }
    } else {
        x.tanh()
    }
}

/// Wilder's RSI — full series matching `utils/indicators.py::get_wilders_rsi`
/// (first `period` outputs are 0, seed mean at index `period`, then Wilder
/// smoothing to the end).
fn wilders_rsi_series(delta: &[f64], period: usize) -> Vec<f64> {
    let n = delta.len();
    let mut out = vec![0.0; n];
    if n < period + 1 {
        return out;
    }
    let mut avg_gain: f64 = delta[..period].iter().map(|d| d.max(0.0)).sum::<f64>() / period as f64;
    let mut avg_loss: f64 =
        delta[..period].iter().map(|d| (-d).max(0.0)).sum::<f64>() / period as f64;
    out[period] = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-10));
    for i in (period + 1)..n {
        avg_gain = (avg_gain * (period as f64 - 1.0) + delta[i].max(0.0)) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + (-delta[i]).max(0.0)) / period as f64;
        out[i] = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-10));
    }
    out
}

/// Last RSI value of the Wilder series, if computable (needs ≥ period+1 diffs).
fn wilders_rsi(delta: &[f64], period: usize) -> Option<f64> {
    if delta.len() < period + 1 {
        return None;
    }
    wilders_rsi_series(delta, period).last().copied()
}

/// Simple moving average over `window` trailing values.
fn sma_series(values: &[f64], window: usize) -> Vec<f64> {
    values
        .windows(window)
        .map(|w| w.iter().sum::<f64>() / window as f64)
        .collect()
}

/// Population standard deviation (ddof=0), like `np.nanstd` (0.0 on empty).
fn pop_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    var.sqrt()
}

/// Rolling z-score of the last element vs the trailing `window` samples
/// (population std, matches `calculate_zscore(..., window=…)`).
fn roll_z(series: &[f64], window: usize) -> f64 {
    let n = series.len();
    if n < window || window == 0 {
        return 0.0;
    }
    let w = &series[n - window..];
    let mean = w.iter().sum::<f64>() / w.len() as f64;
    let sd = pop_std(w);
    if sd <= EPS {
        return 0.0;
    }
    (*w.last().expect("non-empty window") - mean) / sd
}

/// Lag-1 autocorrelation (Pearson r of the series against itself shifted by 1),
/// matching `np.corrcoef(tick_arr[:-1], tick_arr[1:])[0, 1]` from features.py.
/// Returns 0.0 when there's not enough variation to define it.
fn pearson_lag1(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 2 {
        return 0.0;
    }
    let a = &series[..n - 1];
    let b = &series[1..];
    let len = a.len() as f64;
    let ma = a.iter().sum::<f64>() / len;
    let mb = b.iter().sum::<f64>() / len;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..a.len() {
        let da_i = a[i] - ma;
        let db_i = b[i] - mb;
        num += da_i * db_i;
        da += da_i * da_i;
        db += db_i * db_i;
    }
    let den = (da * db).sqrt();
    if den <= EPS { 0.0 } else { num / den }
}

/// Append `value` to `buf`, trim to `MAX_Z_HIST`, then return its rolling
/// z-score vs the trailing `window` samples (population std, matches
/// features.py `_rolling_zscore` → `calculate_zscore(window=…)`).
fn push_roll_z(buf: &mut VecDeque<f64>, value: f64, window: usize) -> f64 {
    buf.push_back(value);
    if buf.len() > MAX_Z_HIST {
        buf.pop_front();
    }
    let vals: Vec<f64> = buf.iter().copied().collect();
    roll_z(&vals, window)
}

// ============================================================================
// Tests — a small synthetic walk that exercises bars → features → entry → exit
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine consts must mirror `main/config.py` (single source of
    /// truth). If this fails, live-test/backtest parity is broken by design.
    #[test]
    fn consts_match_main_config() {
        assert!(!ENABLE_RSI_ACCEL_FILTER, "config ENABLE_RSI_ACCEL_FILTER = False");
        assert!(TRADE_DIRECTION_IS_LONG, "config TRADE_DIRECTION = 'long'");
        assert_eq!(CONFIDENCE_THRESHOLD, 0.5, "config CONFIDENCE_THRESHOLD");
        assert_eq!(LOOKAHEAD_SECONDS, 6000, "config LOOKAHEAD_SECONDS");
        assert_eq!(TIMEFRAME_BARRIER_SECS, 300, "config TIMEFRAME_BARRIER");
        assert_eq!(BARRIER_MULTIPLIER, 0.3, "config BARRIER_MULTIPLIER");
        assert_eq!(RR_RATIO, 10.0, "config RR_RATIO");
        assert_eq!(RSI_FILTER_THRESHOLD, 30.0, "config RSI_FILTER_THRESHOLD");
        assert_eq!(LOCKOUT_SECONDS, 1000, "config LOCKOUT_SECONDS (on/off only)");
        assert_eq!(MIN_BARRIER_PCT, 0.0001, "config MIN_BARRIER_PCT");
        assert_eq!(MAX_BARRIER_PCT, 0.05, "config MAX_BARRIER_PCT");
        assert_eq!(FEE_BPS, 0.1, "config FEE_BPS");
        assert_eq!(BARRIER_VOL_WINDOW, 20, "config BARRIER_VOL_BAR_WINDOW = 20 (research feature_builder rolling_std)");
        assert_eq!(R_CLIP_CAP, 0.0, "config R_CLIP_CAP = off (dual-pass cap used for parity instead)");
    }

    /// The dataset R-cap matches research STEP 5: max(|q99|, |q1|, RR_RATIO)
    /// with numpy's linear-interpolation quantile. A handful of huge timeout
    /// outliers must NOT inflate the cap past their own magnitude, and the
    /// RR_RATIO floor must hold for well-behaved series.
    #[test]
    fn dataset_r_cap_matches_research_step5() {
        // 88 values: 41 losses (~-1), 9.9, 40x 9.93, 9.95, 10.0, 10.1.
        // numpy q(0.99) = 10.0 + 0.13*(10.1-10.0) = 10.013 > RR floor.
        let mut rs = vec![9.9f64, 10.1, -1.05, -1.0, 9.95, -1.1, 10.0, -1.02];
        rs.extend(std::iter::repeat_n(-1.04f64, 40));
        rs.extend(std::iter::repeat_n(9.93f64, 40));
        let cap = dataset_r_cap(10.0, &rs).unwrap();
        assert!((cap - 10.013).abs() < 1e-9, "cap={cap}");

        // One exploding timeout (+120R on a tiny barrier): n=89, numpy
        // q(0.99) = 10.1 + 0.12*(120-10.1) = 23.288 → cap = 23.288, i.e. the
        // outlier inflates the cap but the clip still binds on it.
        let mut rs2 = rs.clone();
        rs2.push(120.0);
        let cap2 = dataset_r_cap(10.0, &rs2).unwrap();
        assert!((cap2 - 23.288).abs() < 1e-9, "cap2={cap2}");
        assert!(120.0 > cap2);

                // Well-behaved series: n=4, q(0.99) = 9.9 + 0.97*0.2 = 10.094 (RR floor 10).
        let calm = [-1.02f64, -1.05, 9.9, 10.1];
        assert!((dataset_r_cap(10.0, &calm).unwrap() - 10.094).abs() < 1e-9);
        assert_eq!(dataset_r_cap(10.0, &[]), None);
        assert_eq!(dataset_r_cap(10.0, &[f64::NAN]), None);
    }

    /// A trade whose raw R exceeds `r_clip_cap` gets clamped while `r_raw`
    /// keeps the uncapped value.
    #[test]
    fn close_position_applies_fixed_r_clip() {
        // `enter_position`/`close_position` math is exercised end-to-end via a
        // synthetic timeout: base_r = move/bp, fees ~0 (FEE_BPS=0.1 on tiny
        // notional), so pick numbers where raw R is far above the cap.
        let cfg = StrategyConfig {
            r_clip_cap: 10.0,
            fee_bps: 0.0,
            enable_slippage: false,
            ..Default::default()
        };
        let mut s = MomentumStrategy::new("CLIP", cfg.clone());
        // Hand-rolled position: long 100 @ barrier 1% → stop 99, target 110.
        s.enter_position(1_000, Side::Long, 100.0, 0.01);
        // Timeout at 105 → raw R = +5 (under cap, untouched).
        let rec = s.close_position(2_000, "timeout", 105.0);
        assert!((rec.r_raw - 5.0).abs() < 1e-9, "r_raw={}", rec.r_raw);
        assert!((rec.r_multiple - 5.0).abs() < 1e-9);

        // Now a move far past the target: raw R = +20 → clipped to 10.
        let mut s = MomentumStrategy::new("CLIP2", cfg.clone());
        s.enter_position(1_000, Side::Long, 100.0, 0.01);
        let rec = s.close_position(2_000, "timeout", 120.0);
        assert!((rec.r_raw - 20.0).abs() < 1e-9, "r_raw={}", rec.r_raw);
        assert!((rec.r_multiple - 10.0).abs() < 1e-9, "r_multiple={}", rec.r_multiple);
    }


    /// Volatility-normalized sizing: the dollar value of 1R (notional ×
    /// barrier_pct) tracks `equity × risk_pct_per_trade` regardless of barrier
    /// width, and the leverage cap binds for tight barriers. The legacy path
    /// (`risk_pct_per_trade = 0`) is unchanged fixed-notional sizing.
    #[test]
    fn risk_normalized_sizing_targets_fixed_dollar_risk() {
        let mk_cfg = || StrategyConfig {
            risk_pct_per_trade: 0.002,
            max_leverage: 3.0,
            starting_equity: 10_000.0,
            min_order_size: 0.0, // keep the math exact in this test
            ..Default::default()
        };

        // Wide barrier (1% stop): notional = 20 / 0.01 = $2,000 — under the
        // $30,000 leverage cap, so full risk target is honored.
        let mut s = MomentumStrategy::new("WIDE", mk_cfg());
        s.enter_position(1_000, Side::Long, 100.0, 0.01);
        let p = s.position.as_ref().unwrap();
        assert!((p.notional - 2_000.0).abs() < 1e-6, "notional={}", p.notional);
        assert!((p.qty * p.entry_price * 0.01 - 20.0).abs() < 1e-6); // $ risk = 0.2% of equity

        // Tight barrier (0.05% stop, like low-vol BTC): uncapped notional
        // would be $40,000, but the 3x leverage cap binds at $30,000.
        let mut s = MomentumStrategy::new("TIGHT", mk_cfg());
        s.enter_position(1_000, Side::Long, 100.0, 0.0005);
        let p = s.position.as_ref().unwrap();
        assert!((p.notional - 30_000.0).abs() < 1e-6, "notional={}", p.notional);
        assert!(p.notional * 0.0005 < 20.0); // capped: effective risk < target

        // Legacy path unchanged: fixed notional regardless of barrier width.
        let cfg = StrategyConfig {
            starting_equity: 10_000.0,
            min_order_size: 0.0,
            ..Default::default() // risk_pct_per_trade = 0.0
        };
        for bp in [0.0005, 0.01] {
            let mut s = MomentumStrategy::new("LEGACY", cfg.clone());
            s.enter_position(1_000, Side::Long, 100.0, bp);
            let p = s.position.as_ref().unwrap();
            assert!((p.notional - cfg.max_position_usd).abs() < 1e-6);
        }
    }

    /// RSI sanity: constant uptrend → RSI(14) near 100, downtrend → 0.
    #[test]
    fn wilders_rsi_trend_direction() {
        let up = vec![1.0; 20];
        let down: Vec<f64> = up.iter().map(|v| -v).collect();
        let rsi_up = wilders_rsi(&up, 14).expect("computable");
        let rsi_down = wilders_rsi(&down, 14).expect("computable");
        assert!(rsi_up > 99.0, "rsi_up={rsi_up}");
        assert!(rsi_down < 1.0, "rsi_down={rsi_down}");
    }

    /// Population std matches the classic sigma² = 4 reference.
    #[test]
    fn pop_std_reference() {
        let v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((pop_std(&v) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn sparse_trades_finalize_every_missing_second() {
        let mut s = MomentumStrategy::new("GAP", StrategyConfig::default());
        let _ = s.on_trade(0, 100.0, 1.0, true);
        let events = s.on_trade(5_000, 101.0, 1.0, true);
        let bars: Vec<i64> = events
            .iter()
            .filter_map(|event| match event {
                EngineEvent::BarClosed(feature) => Some(feature.ts),
                _ => None,
            })
            .collect();
        assert_eq!(bars, vec![0, 1_000, 2_000, 3_000, 4_000]);
    }

    /// End-to-end: warm up the 300 s composite regime features on a steady
    /// uptrend (macro RSI(2) ≈ 100, no warm-up entries) → one full sharp-down
    /// composite bar collapses macro RSI(2) into the hunting zone → a 1-second
    /// micro bar close in the next window enters a LONG (snapback) → recovery
    /// reaches the RR-scaled target and the position exits with R clipped to
    /// ≤ +RR_RATIO (10.0 — see `RR_RATIO` / `close_position`).
    #[test]
    fn hunting_zone_entry_then_barrier_exit() {
        let cfg = StrategyConfig {
            confidence_threshold: 0.0, // bypass the model-free gate for this unit test
            min_barrier_pct: 0.0,
            max_barrier_pct: 0.5,
            lockout_seconds: 0,
            enable_rsi_accel_filter: false,
            ..Default::default()
        };
        let micro_ms = cfg.micro_barrier_secs * 1000;
        let macro_ms = cfg.timeframe_barrier_secs * 1000;
        let macro_ticks = (macro_ms / micro_ms) as usize;
        let mut s = MomentumStrategy::new("TEST", cfg);
        // Start exactly on a composite boundary so the dump spans one window.
        let mut ts = ((1_700_000_000_000 + macro_ms - 1) / macro_ms) * macro_ms;
        let mut price = 100.0f64;
        let mut entered: Option<EntryReceipt> = None;
        let mut exited = false;

        // Phase 1 — 90 composite (300 s) bars of steady uptrend: warms the
        // RSI / MA / vol rings while macro RSI(2) stays ≈ 100, so no warm-up
        // entry can fire.
        for b in 0..90usize {
            let base = 100.0 * (1.0 + b as f64 * 0.0004);
            for j in 0..macro_ticks {
                price = base * (1.0 + (j as f64).sin() * 0.00005);
                let mut ev = s.on_trade(ts, price, 1.0, j % 2 == 0);
                ev.extend(s.on_orderbook(ts, price * 0.9999, 1.0, price * 1.0001, 1.0));
                let _ = ev;
                ts += micro_ms;
            }
        }

        // Phase 2 — one full sharp-down composite bar (~ -20%): macro RSI(2)
        // collapses into the hunting zone at the next 300 s boundary.
        for _ in 0..macro_ticks {
            price *= 0.99926;
            let mut ev = s.on_trade(ts, price, 1.0, false);
            ev.extend(s.on_orderbook(ts, price * 0.9999, 1.0, price * 1.0001, 1.0));
            for e in ev {
                if let EngineEvent::Entered(r) = e {
                    entered = Some(r);
                }
            }
            ts += micro_ms;
        }

        // Phase 3 — a few recovery seconds in the next composite window: the
        // freshly-armed macro RSI(2) < 30 lets the first 1 s bar close enter.
        for _ in 0..8 {
            price *= 1.001;
            let mut ev = s.on_trade(ts, price, 1.0, true);
            ev.extend(s.on_orderbook(ts, price * 0.9999, 1.0, price * 1.0001, 1.0));
            for e in ev {
                match e {
                    EngineEvent::Entered(r) => entered = Some(r),
                    EngineEvent::Exited(_) => exited = true,
                    _ => {}
                }
            }
            ts += micro_ms;
        }
        assert!(
            entered.is_some(),
            "expected a hunting-zone entry after the dump"
        );
        let entry = entered.expect("entry present");
        assert!(
            entry.rsi_2 < RSI_FILTER_THRESHOLD,
            "entry must be inside the rsi_2 < 30 zone, got {}",
            entry.rsi_2
        );

        // Phase 4 — recovery pushes price up to the RR-scaled target.
        let mut guard = 0;
        while !exited && guard < 500 {
            guard += 1;
            price *= 1.01;
            let mut ev = s.on_trade(ts, price, 1.0, true);
            ev.extend(s.on_orderbook(ts, price * 0.9999, 1.0, price * 1.0001, 1.0));
            for e in ev {
                if let EngineEvent::Exited(r) = e {
                    exited = true;
                    assert!(
                        r.r_multiple <= RR_RATIO + 1e-9,
                        "barrier R clipped to +RR_RATIO, got {}",
                        r.r_multiple
                    );
                    assert_eq!(r.reason, "take_profit");
                    assert_eq!(r.side, Side::Long);
                }
            }
            ts += micro_ms;
        }
        assert!(exited, "position never hit its target");
        assert_eq!(s.closed_trades().len(), 1);
        assert!(s.snapshot().n_trades == 1);
    }
}
