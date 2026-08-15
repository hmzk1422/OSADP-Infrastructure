/// Strategy execution layer — port of `main/executor.py` signal gates, the
/// `QuantSimulator` vol-filter/rate-limit/quantity sizing, and the
/// `PositionTracker` re-entry lockout.
///
/// This does NOT do exchange I/O. It consumes a `FeatureEngine` (fed by the
/// sidecar) and an `OnnxModel`, and emits `Action` decisions. Position PnL and
/// barrier math live in `crate::positions::Position`.
use crate::config::Config;
use crate::features::FeatureEngine;
use crate::model::OnnxModel;
use crate::positions::{ExitResult, Position};
use chrono::{DateTime, Utc};

/// Result of attempting to enter a position — captured for logging/analytics.
#[derive(Debug, Clone)]
pub struct EnterReceipt {
    pub side: String,
    pub mid_price: f32,
    pub fill_price: f32,
    pub quantity: f32,
    pub notional_usd: f32,
    pub barrier_pct: f32,
    pub stop_barrier: f32,
    pub profit_barrier: f32,
    pub slippage_bps: f32,
}

/// What the strategy decided to do on a given clock tick.
#[derive(Debug, Clone)]
pub enum Action {
    /// No action taken this tick (warming up / waiting for data).
    Wait,
    RsiFiltered,
    InPosition,
    Lockout,
    Debounced,
    LowConfidence,
    QtyTooSmall,
    EntryRejected,
    EnterShort(EnterReceipt),
    EnterLong(EnterReceipt),
    /// An open position was closed by a barrier or the lookahead timeout.
    Exit(ExitResult),
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Wait => write!(f, "WAIT"),
            Action::RsiFiltered => write!(f, "RSI_FILTERED"),
            Action::InPosition => write!(f, "IN_POSITION"),
            Action::Lockout => write!(f, "LOCKOUT"),
            Action::Debounced => write!(f, "DEBOUNCED"),
            Action::LowConfidence => write!(f, "LOW_CONFIDENCE"),
            Action::QtyTooSmall => write!(f, "QTY_TOO_SMALL"),
            Action::EntryRejected => write!(f, "ENTRY_REJECTED"),
            Action::EnterShort(r) => write!(f, "ENTER_SHORT fill={:.2} qty={:.4}", r.fill_price, r.quantity),
            Action::EnterLong(r) => write!(f, "ENTER_LONG fill={:.2} qty={:.4}", r.fill_price, r.quantity),
            Action::Exit(e) => write!(f, "EXIT {} R={:.3}", e.reason, e.r_multiple),
        }
    }
}

/// The hysteresis + lockout + gate state owned by the `QuantExecutor`.
pub struct StrategyExecutor {
    cfg: Config,
    model: OnnxModel,
    position: Position,
    // Signal debounce state (mirrors executor.py `_update_signal_state`).
    signal_armed: bool,
    debounce_active: bool,
    // Re-entry lockout (mirrors `_in_lockout` + `_lockout_until`).
    lockout_until: Option<DateTime<Utc>>,
    last_exit_time: Option<DateTime<Utc>>,
    // Per-hour rate limit (mirrors `MAX_TRADES_PER_HOUR`).
    recent_entries: Vec<DateTime<Utc>>,
    // Bookkeeping for the 1-second decision clock.
        last_decision_second: u64,
}

impl StrategyExecutor {
    pub fn new(cfg: Config, model: OnnxModel) -> Self {
        Self {
            cfg,
            model,
            position: Position::new(),
            // Research starts armed (True): the first qualifying tick fires.
            signal_armed: true,
            debounce_active: false,
            lockout_until: None,
            last_exit_time: None,
            recent_entries: Vec::new(),
                        last_decision_second: 0,
        }
    }

    pub fn position(&self) -> &Position { &self.position }
    pub fn position_mut(&mut self) -> &mut Position { &mut self.position }

    /// Latest tradable mid price known to the position tracker.
    pub fn mid_price(&self) -> f32 {
        self.position.entry_mid.max(self.position.entry_price)
    }

    // ------------------------------------------------------------------
    // Hysteresis signal state — exact port of `_update_signal_state`.
    // ------------------------------------------------------------------
    fn update_signal_state(&mut self, p_win: f32) -> bool {
        let threshold = self.cfg.confidence_threshold;
        let lower_bound = (threshold - self.cfg.hysteresis_buffer).max(0.0);
        if !self.signal_armed && p_win < lower_bound {
            self.signal_armed = true;
            self.debounce_active = false;
        }
        self.signal_armed
    }

    /// Disarm the detector after a trade actually fires (mirrors research).
    fn disarm_signal(&mut self) {
        self.signal_armed = false;
        self.debounce_active = true;
    }

        // ------------------------------------------------------------------
    // QuantSimulator-like sizing + vol filter
    // ------------------------------------------------------------------
    /// Contract quantity for `MAX_POSITION_USD` notional at `price`.
    fn calculate_quantity(&self, price: f32) -> f32 {
        if price <= 0.0 {
            return self.cfg.min_order_size;
        }
        let qty = self.cfg.max_position_usd / price;
        // Match research rounding (default precision 4).
        let prec = 4usize;
        let factor = 10f32.powi(prec as i32);
        let qty = (qty * factor).round() / factor;
        qty.max(self.cfg.min_order_size)
    }

    /// Barrier range filter — mirrors `_passes_vol_filter`.
    fn passes_vol_filter(&self) -> bool {
        let barrier_pct = self.position.barrier_pct;
        if barrier_pct < self.cfg.min_barrier_pct {
            log::info!(
                "Vol filter: barrier_pct {} < MIN {} - market too dead",
                barrier_pct,
                self.cfg.min_barrier_pct
            );
            return false;
        }
        if barrier_pct > self.cfg.max_barrier_pct {
            log::info!(
                "Vol filter: barrier_pct {} > MAX {} - market too wild",
                barrier_pct,
                self.cfg.max_barrier_pct
            );
            return false;
        }
        true
    }

    /// Hourly trade-count rate limit — mirrors `can_trade`.
    fn can_trade(&self, now: DateTime<Utc>) -> bool {
        let cutoff = now - chrono::Duration::hours(1);
        let recent: usize = self.recent_entries.iter().filter(|t| **t > cutoff).count();
        recent < self.cfg.max_trades_per_hour as usize && !self.position.active
    }

    fn prune_rate_window(&mut self, now: DateTime<Utc>) {
        let cutoff = now - chrono::Duration::hours(1);
        self.recent_entries.retain(|t| *t > cutoff);
    }

    // ------------------------------------------------------------------
    // Public decision surfaces
    // ------------------------------------------------------------------
    /// Feed a trade tick to an open position: push intra-bar extremes and
    /// check for an exit. Mirrors executor `on_trade`'s exit-check branch.
    pub fn on_trade(&mut self, price: f32, now: DateTime<Utc>) -> Option<Action> {
                if !self.position.active {
            self.position.update_tick(price, now, &self.cfg);
            return None;
        }
        self.position.update_tick(price, now, &self.cfg);
        match self.position.check_exit(price, now, &self.cfg) {
            Some(exit_result) => {
                let cloned = exit_result.clone();
                self.on_position_exit(exit_result, now);
                Some(Action::Exit(cloned))
            }
            None => None,
        }
    }

    fn on_position_exit(&mut self, _exit: ExitResult, now: DateTime<Utc>) {
        self.last_exit_time = Some(now);
        if self.cfg.lockout_seconds > 0 {
            self.lockout_until =
                Some(now + chrono::Duration::seconds(self.cfg.lockout_seconds as i64));
            log::info!(
                "[LOCKOUT] re-entry blocked for {}s until {}",
                self.cfg.lockout_seconds,
                self.lockout_until.unwrap()
            );
        }
        // Re-arm the hysteresis detector once the trade closes (research).
        self.signal_armed = true;
        self.debounce_active = false;
    }

    fn build_receipt(&self, side: &str, mid_price: f32) -> EnterReceipt {
        let slip_bps = if self.cfg.enable_slippage { 0.75 } else { 0.0 };
        EnterReceipt {
            side: side.to_string(),
            mid_price,
            fill_price: self.position.entry_price,
            quantity: self.position.quantity,
            notional_usd: self.position.notional_usd,
            barrier_pct: self.position.barrier_pct,
                        stop_barrier: self.position.stop_barrier,
            profit_barrier: self.position.profit_barrier,
            slippage_bps: slip_bps,
        }
    }

        /// Re-entry lockout window (mirrors executor `_in_lockout`). Returns true
    /// while `now` is inside the post-exit lockout.
    fn in_lockout(&self, now: DateTime<Utc>) -> bool {
        self.lockout_until.map_or(false, |until| now < until)
    }

    /// One iteration of the 1-second decision clock.
    /// Mirrors `executor.py` `_one_sec_tick` + the entry-gate priority block
    /// (lines ~689-755) + the lockout/disarm lifecycle.
    pub fn on_clock_tick(&mut self, features: &FeatureEngine, now: DateTime<Utc>) -> Action {
        // De-duplicate: only act when the engine has advanced to a new second.
        let seconds = features.second_count();
        if seconds <= self.last_decision_second {
            return Action::Wait;
        }
        self.last_decision_second = seconds;

        // Warmup gate: need enough 1s closes for price_roc_3 + RSI(2).
        if features.price_buffer_len() < (self.cfg.price_roc_window_3 + 1) {
            return Action::Wait;
        }

        let current_price = features.last_mid();
        let rsi_2_val = features.rsi_2();
        let bar_vol_std = features.bar_vol_std();

        // Primary barrier-sizing input: aggregated-bar return std.
        self.position.set_barrier_pct(bar_vol_std);
        // Keep intra-bar extremes current between trades.
                self.position.update_tick(current_price, now, &self.cfg);

        let feature_vector = features.compute_1s_feature_vector();
        let p_win = match self.model.predict_pwin(&feature_vector) {
            Ok(p) => p,
            Err(e) => {
                log::error!("model inference failed: {e}");
                return Action::Wait;
            }
        };

        log::debug!(
            "clock sec={} mid={:.2} rsi2={:.1} p_win={:.3}",
            seconds, current_price, rsi_2_val, p_win
        );

        // ---- Clock-driven position exit (timeout + barrier progress) ----
        if self.position.active {
                        if let Some(exit) = self.position.check_exit(current_price, now, &self.cfg) {
                let exit_clone = exit.clone();
                self.on_position_exit(exit, now);
                return Action::Exit(exit_clone);
            }
        }

        // ---- Entry gates (priority order, ~exact mirror of executor.py) ----
        let mut fresh_signal = false;
        if rsi_2_val < self.cfg.rsi_filter_threshold {
            fresh_signal = self.update_signal_state(p_win);
        }

        if rsi_2_val >= self.cfg.rsi_filter_threshold {
            return Action::RsiFiltered;
        }
        if self.position.active {
            return Action::InPosition;
        }
        if self.in_lockout(now) {
            return Action::Lockout;
        }
        if !fresh_signal {
            return Action::Debounced;
        }
        if p_win < self.cfg.confidence_threshold {
            return Action::LowConfidence;
        }

        // Confidence gate passed — size the trade.
        if !self.can_trade(now) {
            return Action::EntryRejected;
        }
        let qty = self.calculate_quantity(current_price);
        if qty < self.cfg.min_order_size {
            return Action::QtyTooSmall;
        }
        if !self.passes_vol_filter() {
            return Action::EntryRejected;
        }

        self.prune_rate_window(now);
        self.recent_entries.push(now);

        let side = self.cfg.trade_direction.clone();
        if side == "long" {
            self.position.enter_long(current_price, qty, now, &self.cfg);
        } else {
            self.position.enter_short(current_price, qty, now, &self.cfg);
        }

        if !self.position.active {
            return Action::EntryRejected;
        }
        let receipt = self.build_receipt(&side, current_price);
        self.disarm_signal();
        log::info!(
            "[{}] fill={:.2} qty={:.4} npv={:.2} stop={:.2} tgt={:.2}",
            if side == "long" { "ENTER LONG" } else { "ENTER SHORT" },
            receipt.fill_price,
            receipt.quantity,
            receipt.notional_usd,
            receipt.stop_barrier,
            receipt.profit_barrier
        );

        if side == "long" {
            Action::EnterLong(receipt)
        } else {
            Action::EnterShort(receipt)
        }
    }
}

