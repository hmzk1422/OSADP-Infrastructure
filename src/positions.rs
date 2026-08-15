/// Position tracking with volatility-scaled barriers and R-multiple PnL.
/// Mirrors `main/execution.py` `PositionTracker`.
use crate::config::Config;

const EPS: f32 = 1e-9;

#[derive(Debug, Clone)]
pub struct Position {
    pub active: bool,
    pub side: String,                    // "short" | "long"
    pub entry_price: f32,
    pub entry_mid: f32,
    pub entry_time: chrono::DateTime<chrono::Utc>,
    pub quantity: f32,
    pub notional_usd: f32,
    pub barrier_pct: f32,
    pub stop_barrier: f32,
    pub profit_barrier: f32,
    pub reason: String,
    pub is_timeout: bool,
    pub r_multiple: f32,
    pub gross_pnl_pct: f32,
    pub net_pnl_pct: f32,
    pub fees_paid: f32,
    bar_high: f32,
    bar_low: f32,
}

#[derive(Debug, Clone)]
pub struct ExitResult {
    pub reason: String,
    pub exit_price: f32,
    pub r_multiple: f32,
    pub gross_pnl_pct: f32,
    pub net_pnl_pct: f32,
    pub fees_paid: f32,
    pub side: String,
    pub entry_price: f32,
    pub barrier_pct: f32,
    pub quantity: f32,
    pub notional_usd: f32,
}

impl Position {
    pub fn new() -> Self {
        Self {
            active: false, side: String::new(),
            entry_price: 0.0, entry_mid: 0.0, entry_time: chrono::Utc::now(),
            quantity: 0.0, notional_usd: 0.0, barrier_pct: 0.0,
            stop_barrier: 0.0, profit_barrier: 0.0, reason: String::new(),
            is_timeout: false, r_multiple: 0.0, gross_pnl_pct: 0.0,
            net_pnl_pct: 0.0, fees_paid: 0.0,
            bar_high: 0.0, bar_low: f32::INFINITY,
        }
    }

    /// Update the volatility-derived barrier distance (engine sets this on
    /// every TIMEFRAME_BARRIER bar close).
    pub fn set_barrier_pct(&mut self, pct: f32) {
        if pct > 0.0 { self.barrier_pct = pct; }
    }

    fn effective_bp(&self, cfg: &Config) -> f32 {
        if self.barrier_pct > 1e-9 { self.barrier_pct } else { 0.001 * cfg.barrier_multiplier }
    }

    pub fn enter_short(&mut self, mid: f32, qty: f32, now: chrono::DateTime<chrono::Utc>, cfg: &Config) {
        let slip = if cfg.enable_slippage { 0.75 } else { 0.0 };
        self.entry_price = mid * (1.0 - slip / 10000.0);
        self.entry_mid = mid; self.entry_time = now;
        self.quantity = qty; self.notional_usd = qty * self.entry_price;
        self.side = "short".into(); self.active = true;
        let bp = self.effective_bp(cfg);
        self.barrier_pct = bp;
        // Short: stop ABOVE entry, profit BELOW entry.
        self.stop_barrier = self.entry_price * (1.0 + bp);
        self.profit_barrier = self.entry_price * (1.0 - bp * cfg.rr_ratio);
        self.bar_high = mid; self.bar_low = mid;
    }

    pub fn enter_long(&mut self, mid: f32, qty: f32, now: chrono::DateTime<chrono::Utc>, cfg: &Config) {
        let slip = if cfg.enable_slippage { 0.75 } else { 0.0 };
        self.entry_price = mid * (1.0 + slip / 10000.0);
        self.entry_mid = mid; self.entry_time = now;
        self.quantity = qty; self.notional_usd = qty * self.entry_price;
        self.side = "long".into(); self.active = true;
        let bp = self.effective_bp(cfg);
        self.barrier_pct = bp;
        // Long: stop BELOW entry, profit ABOVE entry.
        self.stop_barrier = self.entry_price * (1.0 - bp);
        self.profit_barrier = self.entry_price * (1.0 + bp * cfg.rr_ratio);
        self.bar_high = mid; self.bar_low = mid;
    }

    /// Update intra-bar high/low. Returns Some(ExitResult) if the position exits.
    pub fn update_tick(&mut self, mid: f32, now: chrono::DateTime<chrono::Utc>, cfg: &Config) -> Option<ExitResult> {
        if !self.active { return None; }
        self.bar_high = self.bar_high.max(mid);
        self.bar_low = self.bar_low.min(mid);
                self.check_exit(mid, now, cfg)
    }

    pub fn check_exit(&mut self, current_mid: f32, now: chrono::DateTime<chrono::Utc>, cfg: &Config) -> Option<ExitResult> {
        if !self.active { return None; }

        let hit_stop = if self.side == "short" {
            self.bar_high >= self.stop_barrier
        } else {
            self.bar_low <= self.stop_barrier
        };
        let hit_profit = if self.side == "short" {
            self.bar_low <= self.profit_barrier
        } else {
            self.bar_high >= self.profit_barrier
        };

        let elapsed = (now - self.entry_time).num_seconds() as u64;
        let timeout = elapsed >= cfg.lookahead_seconds;

        if !hit_stop && !hit_profit {
            if timeout {
                return Some(self.exit_internal("timeout", current_mid, cfg));
            }
            return None;
        }

        if hit_stop && hit_profit {
            // Both touched within the same tick — closer barrier first.
            let stop_dist = (self.stop_barrier - self.entry_price).abs();
            let profit_dist = (self.profit_barrier - self.entry_price).abs();
            if profit_dist <= stop_dist {
                Some(self.exit_internal("take_profit", self.profit_barrier, cfg))
            } else {
                Some(self.exit_internal("stop_loss", self.stop_barrier, cfg))
            }
        } else if hit_stop {
            Some(self.exit_internal("stop_loss", self.stop_barrier, cfg))
        } else {
            Some(self.exit_internal("take_profit", self.profit_barrier, cfg))
        }
    }

    fn exit_internal(&mut self, reason: &str, theoretical_exit: f32, cfg: &Config) -> ExitResult {
        self.active = false;
        self.reason = reason.into();
        self.is_timeout = reason == "timeout";

        let slip_bps = if cfg.enable_slippage { 0.75 } else { 0.0 };
        // Exit slippage: shorts cover at ask (up), longs sell at bid (down).
        let exit_price = if self.side == "short" {
            theoretical_exit * (1.0 + slip_bps / 10000.0)
        } else {
            theoretical_exit * (1.0 - slip_bps / 10000.0)
        };

        let gross_pnl_pct = if self.side == "short" {
            (self.entry_price - exit_price) / self.entry_price
        } else {
            (exit_price - self.entry_price) / self.entry_price
        };
        let entry_fee = self.notional_usd * cfg.fee_bps / 10000.0;
        let exit_fee = self.quantity * exit_price * cfg.fee_bps / 10000.0;
        let fees_paid = entry_fee + exit_fee;
        let total_fee_pct = fees_paid / self.notional_usd.max(EPS);
        let net_pnl_pct = gross_pnl_pct - total_fee_pct;

        // R-multiple (matches execution.py _exit + research clipping).
        let bp = self.barrier_pct.max(0.001 * cfg.barrier_multiplier);
        let base_gross_r = if self.side == "short" {
            (self.entry_price - theoretical_exit) / self.entry_price / bp
        } else {
            (theoretical_exit - self.entry_price) / self.entry_price / bp
        };
        let base_gross_r = if self.is_timeout {
            base_gross_r
        } else {
                        base_gross_r.clamp(-1.0_f32, 1.0_f32 * cfg.rr_ratio)
        };
        let slippage_r = (slip_bps / 10000.0) / bp;
        let fee_in_r = total_fee_pct / bp;
        let r_multiple = base_gross_r - slippage_r - fee_in_r;

        self.gross_pnl_pct = gross_pnl_pct;
        self.net_pnl_pct = net_pnl_pct;
        self.fees_paid = fees_paid;
        self.r_multiple = r_multiple;

        ExitResult {
            reason: reason.into(), exit_price, r_multiple,
            gross_pnl_pct, net_pnl_pct, fees_paid,
            side: self.side.clone(), entry_price: self.entry_price,
            barrier_pct: bp, quantity: self.quantity,
            notional_usd: self.notional_usd,
        }
    }
}