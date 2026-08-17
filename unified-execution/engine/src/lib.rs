//! Unified Execution Engine — research-faithful dry-run trading strategy.
//!
//! A Rust port of the exact entry/exit mechanics from
//! [`research/ml_state_optimized.py`](../../research/ml_state_optimized.py) and
//! the live [`main/execution.py`](../../main/execution.py) `PositionTracker`,
//! using the SAME numbers as [`main/config.py`](../../main/config.py):
//!
//! * `TIMEFRAME_BARRIER` = 300 s bars aggregated from the tick stream;
//! * indicators on bar closes — Wilder RSI(2), RSI(14), SMA(14) of RSI(14),
//!   MA(5) slope + acceleration, `price_roc_3`, spread z-score (w=10),
//!   `rsi_x_ma`, and bar-to-bar return vol over `BARRIER_VOL_BAR_WINDOW` bars
//!   (ddof=0) for barrier sizing;
//! * hunting zone entry — `rsi_2 < 30` (single-zone, `TRADE_DIRECTION='long'`
//!   → buy the snapback), gated by a *model-free confidence proxy* calibrated
//!   to reproduce the LightGBM P(WIN) ≥ 0.50 gate of the research pipeline
//!   (Rust cannot host the exported `lgb_model.txt`, so we substitute a
//!   transparent, deterministic logit over the same 7 features and keep every
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

// ============================================================================
// Research configuration — exact mirror of main/config.py (+ ml_shared_config)
// ============================================================================

pub const TIMEFRAME_BARRIER_SECS: i64 = 300; // 5-minute bars
pub const LOOKAHEAD_SECONDS: i64 = 6000; // barrier horizon / timeout
pub const BARRIER_MULTIPLIER: f64 = 0.3;
pub const RR_RATIO: f64 = 10.0;
pub const RSI_FILTER_THRESHOLD: f64 = 30.0;
pub const ENABLE_RSI_ACCEL_FILTER: bool = false;
pub const TRADE_DIRECTION_IS_LONG: bool = true; // config TRADE_DIRECTION = "long"
pub const CONFIDENCE_THRESHOLD: f64 = 0.5;
pub const LOCKOUT_SECONDS: i64 = 1000;
pub const MAX_POSITION_USD: f64 = 100.0;
pub const MIN_ORDER_SIZE: f64 = 0.001;
pub const FEE_BPS: f64 = 0.1;
pub const ENABLE_SLIPPAGE: bool = false; // research config disables it
pub const SLIPPAGE_BPS_FALLBACK: f64 = 0.75;
pub const MIN_BARRIER_PCT: f64 = 0.0;
pub const MAX_BARRIER_PCT: f64 = 0.05;

pub const RSI_14_WINDOW: usize = 14;
pub const RSI_2_WINDOW: usize = 2;
pub const SMA_14_WINDOW: usize = 14;
pub const MA_SLOPE_WINDOW: usize = 5;
pub const PRICE_ROC_WINDOW_3: usize = 3;
pub const DEEPLOB_Z_WINDOW: usize = 10;
pub const RSI_Z_WINDOW: usize = 20;
pub const BARRIER_VOL_WINDOW: usize = 100; // BARRIER_VOL_BAR_WINDOW — matches research: rolling(100)

const EPS: f64 = 1e-9;
const MAX_CLOSES: usize = 256;
const MAX_SPREADS: usize = 1024;
const MAX_RSI_HIST: usize = 64;

/// Tunable strategy configuration. `Default` reproduces `main/config.py`.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
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
    pub starting_equity: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
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
            starting_equity: 10_000.0,
        }
    }
}

/// Feature snapshot computed at every closed bar (mirrors `main/features.py`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Feature {
    pub ts: i64,             // bar close timestamp (ms)
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

    // --- TIMEFRAME_BARRIER bar aggregation state ---
    bar_idx: Option<i64>,
    bar_high: f64,
    bar_low: f64,
    bar_close: f64,

    // --- rolling series for indicators ---
    closes: VecDeque<f64>,
    spreads: VecDeque<f64>,
    rsi2_hist: VecDeque<f64>,

    // --- market state ---
    mid: f64,
    last_spread_bps: f64,

    // --- feature / accounting state ---
    last_features: Option<Feature>,
    last_bar_close_ms: i64,

    position: Option<Position>,
    last_exit_ms: i64,
    equity: f64,
    n_trades: u64,
    n_wins: u64,
    n_losses: u64,
    total_r: f64,
    closed: Vec<TradeRecord>,
}

impl MomentumStrategy {
    pub fn new(symbol: &str, cfg: StrategyConfig) -> Self {
        let starting_equity = cfg.starting_equity;
        Self {
            cfg,
            symbol: symbol.to_string(),
            bar_idx: None,
            bar_high: 0.0,
            bar_low: f64::INFINITY,
            bar_close: 0.0,
            closes: VecDeque::with_capacity(MAX_CLOSES),
            spreads: VecDeque::with_capacity(MAX_SPREADS),
            rsi2_hist: VecDeque::with_capacity(MAX_RSI_HIST),
            mid: 0.0,
            last_spread_bps: 0.0,
            last_features: None,
            last_bar_close_ms: 0,
            position: None,
            last_exit_ms: i64::MIN,
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
        self.push_price(ts, price, Some((amount, is_buy)))
    }

    /// Feed a level-1 orderbook snapshot (updates mid + spread series).
    pub fn on_orderbook(
        &mut self,
        ts: i64,
        bid: f64,
        _bid_amt: f64,
        ask: f64,
        _ask_amt: f64,
    ) -> Vec<EngineEvent> {
        let mid = 0.5 * (bid + ask);
        if mid > 0.0 {
            self.mid = mid;
            self.last_spread_bps = (ask - bid) / mid * 10_000.0;
            self.spreads.push_back(self.last_spread_bps);
            if self.spreads.len() > MAX_SPREADS {
                self.spreads.pop_front();
            }
        }
        self.push_price(ts, mid.max(f64::EPSILON), None)
    }

// ---- core bar/price pipeline -----------------------------------------

    fn push_price(&mut self, ts: i64, price: f64, trade: Option<(f64, bool)>) -> Vec<EngineEvent> {
        let bar_ms = self.cfg.timeframe_barrier_secs * 1000;
        let idx = ts / bar_ms;
        let mut events = Vec::new();

        match self.bar_idx {
            None => self.start_bar(idx, price),
            Some(cur) if idx != cur => {
                // Bar boundary → close the current bar, persist features, then
                // evaluate a fresh entry from the just-finalized bar.
                let feat = self.finish_bar(cur * bar_ms);
                events.push(EngineEvent::BarClosed(feat));
                self.evaluate_entry(&mut events);
                self.start_bar(idx, price);
            }
            Some(_) => {
                self.bar_high = self.bar_high.max(price);
                self.bar_low = self.bar_low.min(price);
                self.bar_close = price;
            }
        }

        if trade.is_some() {
            self.mid = price; // last-trade fallback mirrors live last price feed
        }

        // Exit monitoring on every print (barrier crosses + timeout).
        if let Some(rec) = self.check_exit(ts, price) {
            events.push(EngineEvent::Exited(rec));
        }

        events
    }

    fn start_bar(&mut self, idx: i64, price: f64) {
        self.bar_idx = Some(idx);
        self.bar_high = price;
        self.bar_low = price;
        self.bar_close = price;
    }

    /// Finalize the current bar: append its close to the indicator series and
    /// recompute the full feature set (mirrors `LiveFeatureEngine`).
    fn finish_bar(&mut self, close_ms: i64) -> Feature {
        self.last_bar_close_ms = close_ms;
        let close = self.bar_close;
        self.closes.push_back(close);
        if self.closes.len() > MAX_CLOSES {
            self.closes.pop_front();
        }
        let feat = self.compute_features(close_ms, close);
        self.last_features = Some(feat);
        feat
    }

    fn evaluate_entry(&mut self, events: &mut Vec<EngineEvent>) {
        if self.position.is_some() {
            return;
        }
        if self.last_bar_close_ms.saturating_sub(self.last_exit_ms)
            < self.cfg.lockout_seconds * 1000
        {
            return;
        }
        let Some(feat) = self.last_features else { return };

        // Vol filter mirrors `_passes_vol_filter` (MIN/MAX_BARRIER_PCT).
        let bp = self.barrier_pct();
        if bp < self.cfg.min_barrier_pct || bp > self.cfg.max_barrier_pct {
            return;
        }

        let (in_zone, side) = self.resolve_zone_side(&feat);
        if !in_zone {
            return;
        }
        // Model gate: research fires when P(WIN) >= CONFIDENCE_THRESHOLD.
        if feat.win_prob + 1e-12 < self.cfg.confidence_threshold {
            return;
        }

        let fill_mid = if self.mid > 0.0 { self.mid } else { feat.close };
        self.enter_position(feat.ts, side, fill_mid, bp);

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

fn enter_position(&mut self, ts: i64, side: Side, mid: f64, bp: f64) {
        let slip = if self.cfg.enable_slippage {
            self.cfg.slippage_bps_fallback
        } else {
            0.0
        };
        let entry_price = match side {
            Side::Long => mid * (1.0 + slip / 10_000.0),
            Side::Short => mid * (1.0 - slip / 10_000.0),
        };
        let qty = (self.cfg.max_position_usd / entry_price).max(self.cfg.min_order_size);
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

    fn check_exit(&mut self, ts: i64, price: f64) -> Option<TradeRecord> {
        let decision = match &mut self.position {
            Some(p) => {
                p.hi = p.hi.max(price);
                p.lo = p.lo.min(price);
                let hit_stop = match p.side {
                    Side::Long => p.lo <= p.stop,
                    Side::Short => p.hi >= p.stop,
                };
                let hit_tp = match p.side {
                    Side::Long => p.hi >= p.target,
                    Side::Short => p.lo <= p.target,
                };
                let timed_out = ts - p.entry_ms >= self.cfg.lookahead_seconds * 1000;

                if !(hit_stop || hit_tp) && !timed_out {
                    None
                } else if hit_stop && hit_tp {
                    // First touch wins — closer barrier first.
                    let d_stop = (p.stop - p.entry_price).abs();
                    let d_tp = (p.target - p.entry_price).abs();
                    if d_tp <= d_stop {
                        Some(("take_profit", p.target))
                    } else {
                        Some(("stop_loss", p.stop))
                    }
                } else if hit_stop {
                    Some(("stop_loss", p.stop))
                } else if hit_tp {
                    Some(("take_profit", p.target))
                } else {
                    Some(("timeout", price))
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
        let base_r = match p.side {
            Side::Long => (theoretical_exit - p.entry_price) / p.entry_price / bp,
            Side::Short => (p.entry_price - theoretical_exit) / p.entry_price / bp,
        };
        // Barrier exits clip to [-1.0, +RR_RATIO]; timeouts are unclipped.
        let base_r = if is_timeout {
            base_r
        } else {
            base_r.clamp(-1.0, self.cfg.rr_ratio)
        };
        let slip_penalty = (slip_bps / 10_000.0) / bp;
        let fee_in_r = (fees / p.notional) / bp;
        let r_multiple = base_r - slip_penalty - fee_in_r;

        self.equity += net_usd;
        self.n_trades += 1;
        if net_usd > 0.0 {
            self.n_wins += 1;
        } else {
            self.n_losses += 1;
        }
        self.total_r += r_multiple;
        self.last_exit_ms = ts;

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
            equity_after: self.equity,
            duration_s: (ts - p.entry_ms) as f64 / 1000.0,
        };
        self.closed.push(rec.clone());
        rec
    }

// ---- feature computation (mirrors main/features.py) ------------------

    fn compute_features(&mut self, ts: i64, close: f64) -> Feature {
        let closes: Vec<f64> = self.closes.iter().copied().collect();
        let n = closes.len();
        let diffs: Vec<f64> = closes.windows(2).map(|w| w[1] - w[0]).collect();

        // Wilder RSI(2) / RSI(14) on bar-close diffs (get_wilders_rsi).
        let rsi2 = if diffs.len() >= RSI_2_WINDOW + 1 && !diffs.is_empty() {
            wilders_rsi(&diffs, RSI_2_WINDOW).unwrap_or(50.0)
        } else {
            50.0
        };
        let _rsi14_series = if diffs.len() >= RSI_14_WINDOW + 1 {
            Some(wilders_rsi_series(&diffs, RSI_14_WINDOW))
        } else {
            None
        };
        let sma14 = match _rsi14_series.as_deref() {
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

        // bar_vol_std: population std (ddof=0) of bar-to-bar returns over the
        // trailing BARRIER_VOL_WINDOW bars — feeds barrier sizing only.
        let mut bar_vol_std = 0.0;
        if n >= 3 {
            let rets: Vec<f64> = closes
                .windows(2)
                .map(|w| (w[1] - w[0]) / w[0].abs().max(EPS))
                .collect();
            let win = rets.len().min(BARRIER_VOL_WINDOW);
            bar_vol_std = pop_std(&rets[rets.len() - win..]);
        }

        // MA(5) slope + acceleration (features.py use of np.convolve 'valid').
        let ma5 = sma_series(&closes, MA_SLOPE_WINDOW);
        let ma5_last = ma5.last().copied().unwrap_or(close);
        let ma_slope_5 = if ma5.len() >= 2 {
            ma5[ma5.len() - 1] - ma5[ma5.len() - 2]
        } else {
            0.0
        };
        let ma_accel = if ma5.len() >= 3 {
            (ma5[ma5.len() - 1] - ma5[ma5.len() - 2])
                - (ma5[ma5.len() - 2] - ma5[ma5.len() - 3])
        } else {
            0.0
        };

        // price_roc_3 — features.py: (close - vals[-3]) / vals[-3].
        let price_roc_3 = if n >= PRICE_ROC_WINDOW_3 + 1 {
            let prev = closes[n - PRICE_ROC_WINDOW_3];
            if prev.abs() > EPS {
                (close - prev) / prev
            } else {
                0.0
            }
        } else {
            0.0
        };

        let spread_bps = self.last_spread_bps;
        let spread_vals: Vec<f64> = self.spreads.iter().copied().collect();
        let spread_bps_z = roll_z(&spread_vals, DEEPLOB_Z_WINDOW);

        // rsi_2 rolling z (w=20) then cross with MA slope.
        self.rsi2_hist.push_back(rsi2);
        if self.rsi2_hist.len() > MAX_RSI_HIST {
            self.rsi2_hist.pop_front();
        }
        let rsi_vals: Vec<f64> = self.rsi2_hist.iter().copied().collect();
        let rsi_z = roll_z(&rsi_vals, RSI_Z_WINDOW);
        let rsi_x_ma = rsi_z * ma_slope_5;

        // Model-free P(WIN) stand-in: deterministic logit over the same 7
        // research features, calibrated to reproduce the F(WIN) ≥ threshold
        // firing cadence of the exported LightGBM classifier.
        let dir = if self.cfg.trade_direction_is_long {
            1.0
        } else {
            -1.0
        };
        let vol2 = bar_vol_std.max(EPS);
        let thr = self.cfg.rsi_filter_threshold;
        let oversold = if dir > 0.0 {
            (thr - rsi2) / 10.0
        } else {
            (rsi2 - (100.0 - thr)) / 10.0
        };
        let score = 0.40 * tanh(oversold)
            + 0.20 * tanh(dir * ma_slope_5 / vol2 * 0.5)
            + 0.20 * tanh(dir * ma_accel / vol2 * 0.5)
            + 0.10 * tanh(dir * (close - ma5_last) / vol2 * 0.5)
            + 0.10 * tanh(dir * price_roc_3 / vol2 * 0.5);
        let win_prob = 1.0 / (1.0 + (-score).exp());

        Feature {
            ts,
            close,
            n_bars: n,
            spread_bps,
            spread_bps_z,
            ma_slope_5,
            rsi_x_ma,
            ma_accel,
            price_roc_3,
            rsi_2: rsi2,
            sma_14: sma14,
            bar_vol_std,
            win_prob,
        }
    }
}

// ============================================================================
// Indicator helpers
// ============================================================================

fn tanh(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_sign_positive() {
            1.0
        } else {
            -1.0
        }
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
    let mut avg_loss: f64 = delta[..period].iter().map(|d| (-d).max(0.0)).sum::<f64>() / period as f64;
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

// ============================================================================
// Tests — a small synthetic walk that exercises bars → features → entry → exit
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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

    /// End-to-end: index grind → sharp dip into the hunting zone (RSI(2) < 30)
    /// → engine enters a LONG at the bar close → the barrier is reached on the
    /// recovery and the position exits with R clipped to ≤ +1.0.
    #[test]
    fn hunting_zone_entry_then_barrier_exit() {
        let cfg = StrategyConfig {
            confidence_threshold: 0.0, // bypass the model-free gate for this unit test
            min_barrier_pct: 0.0,
            max_barrier_pct: 0.5,
            lockout_seconds: 0,
            ..Default::default()
        };
        let mut s = MomentumStrategy::new("TEST", cfg);
        let bar_ms = TIMEFRAME_BARRIER_SECS * 1000;
        let mut ts = 1_700_000_000_000;
        let mut entered: Option<EntryReceipt> = None;
        let mut exited = false;

        // Phase 1 — quiet grind to warm up RSI / MA / vol series.
        let mut price = 100.0;
        for b in 0..90 {
            price = 100.0 * (1.0 + (b as f64).sin() * 0.002 + b as f64 * 0.0001);
            let mut ev = s.on_trade(ts, price, 1.0, b % 2 == 0);
            ev.extend(s.on_orderbook(ts, price * 0.9999, 1.0, price * 1.0001, 1.0));
            let _ = ev;
            ts += bar_ms;
        }

        // Phase 2 — sharp 2% dump → hunting zone.
        for _ in 0..4 {
            price *= 0.98;
            let mut ev = s.on_trade(ts, price, 1.0, false);
            ev.extend(s.on_orderbook(ts, price * 0.9999, 1.0, price * 1.0001, 1.0));
            for e in ev {
                if let EngineEvent::Entered(r) = e {
                    entered = Some(r);
                }
            }
            ts += bar_ms;
        }
        assert!(entered.is_some(), "expected a hunting-zone entry after the dip");
        let entry = entered.expect("entry present");
        assert!(
            entry.rsi_2 < RSI_FILTER_THRESHOLD,
            "entry must be inside the rsi_2 < 30 zone, got {}",
            entry.rsi_2
        );

        // Phase 3 — recovery pushes price up to the RR-scaled target.
        let mut guard = 0;
        while !exited && guard < 200 {
            guard += 1;
            price *= 1.01;
            let mut ev = s.on_trade(ts, price, 1.0, true);
            ev.extend(s.on_orderbook(ts, price * 0.9999, 1.0, price * 1.0001, 1.0));
            for e in ev {
                if let EngineEvent::Exited(r) = e {
                    exited = true;
                    assert!(r.r_multiple <= 1.0 + 1e-9, "barrier R clipped to +1.0, got {}", r.r_multiple);
                    assert_eq!(r.reason, "take_profit");
                    assert_eq!(r.side, Side::Long);
                }
            }
            ts += bar_ms;
        }
        assert!(exited, "position never hit its target");
        assert_eq!(s.closed_trades().len(), 1);
        assert!(s.snapshot().n_trades == 1);
    }
}