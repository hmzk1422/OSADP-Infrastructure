//! Realized-pnl ledger: closed trades and the equity curve.
//!
//! The ledger is the only place a win is decided. Keeping that judgement here
//! means the dashboard's win rate, the Sortino ratio and the drawdown limit all
//! read the same number instead of three approximations of it.

use super::order::Side;

/// One completed round trip.
#[derive(Clone, Debug, PartialEq)]
pub struct Trade {
    pub symbol: String,
    /// Which position was opened: `Buy` = long, `Sell` = short.
    pub side: Side,
    pub entry_price: f64,
    pub exit_price: f64,
    pub qty: f64,
    /// Commission on both legs, already summed.
    pub fee: f64,
    pub entry_sequence: u64,
    pub exit_sequence: u64,
}

impl Trade {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: impl Into<String>,
        side: Side,
        entry_price: f64,
        exit_price: f64,
        qty: f64,
        fee: f64,
        entry_sequence: u64,
        exit_sequence: u64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            entry_price,
            exit_price,
            qty,
            fee,
            entry_sequence,
            exit_sequence,
        }
    }

    /// Pnl before fees.
    pub fn gross_pnl(&self) -> f64 {
        let direction = if self.side.is_buy() { 1.0 } else { -1.0 };
        (self.exit_price - self.entry_price) * self.qty * direction
    }

    /// Pnl after fees: the number a win/loss judgement must use. A trade that
    /// cleared 5 and paid 6 in commission was not a win.
    pub fn net_pnl(&self) -> f64 {
        self.gross_pnl() - self.fee
    }

    pub fn is_win(&self) -> bool {
        self.net_pnl() > 0.0
    }

    /// Return in units of the initial risk, when the stop distance is known.
    ///
    /// `risk_per_unit` is the price distance from entry to stop. `None` for a
    /// non-positive distance: an R-multiple against no risk is meaningless.
    pub fn r_multiple(&self, risk_per_unit: f64) -> Option<f64> {
        if !risk_per_unit.is_finite() || risk_per_unit <= 0.0 {
            return None;
        }
        Some(self.net_pnl() / (risk_per_unit * self.qty))
    }
}

/// Append-only record of what the account earned and how deep it drew down.
#[derive(Clone, Debug)]
pub struct PnlLedger {
    starting_equity: f64,
    trades: Vec<Trade>,
    equity: Vec<f64>,
    peak: f64,
    max_drawdown: f64,
}

impl PnlLedger {
    pub fn new(starting_equity: f64) -> Self {
        Self {
            starting_equity,
            trades: Vec::new(),
            equity: vec![starting_equity],
            peak: starting_equity,
            max_drawdown: 0.0,
        }
    }

    pub fn record_trade(&mut self, trade: Trade) {
        self.trades.push(trade);
    }

    /// Append a mark-to-market equity sample and fold it into the drawdown.
    pub fn mark(&mut self, equity: f64) {
        self.equity.push(equity);
        if equity > self.peak {
            self.peak = equity;
        }
        if self.peak > 0.0 {
            let drawdown = (self.peak - equity) / self.peak;
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }
        }
    }

    pub fn starting_equity(&self) -> f64 {
        self.starting_equity
    }

    pub fn current_equity(&self) -> f64 {
        self.equity.last().copied().unwrap_or(self.starting_equity)
    }

    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    pub fn equity_curve(&self) -> &[f64] {
        &self.equity
    }

    pub fn realized_pnl(&self) -> f64 {
        self.trades.iter().map(Trade::net_pnl).sum()
    }

    pub fn gross_pnl(&self) -> f64 {
        self.trades.iter().map(Trade::gross_pnl).sum()
    }

    pub fn fees_paid(&self) -> f64 {
        self.trades.iter().map(|t| t.fee).sum()
    }

    /// Largest peak-to-trough fall of the equity curve, as a fraction.
    pub fn max_drawdown(&self) -> f64 {
        self.max_drawdown
    }

    pub fn wins(&self) -> usize {
        self.trades.iter().filter(|t| t.is_win()).count()
    }

    /// Strictly losing trades; a breakeven is counted as neither.
    pub fn losses(&self) -> usize {
        self.trades.iter().filter(|t| t.net_pnl() < 0.0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long(entry: f64, exit: f64, qty: f64, fee: f64) -> Trade {
        Trade::new("BTCUSDT", Side::Buy, entry, exit, qty, fee, 1, 2)
    }

    #[test]
    fn a_long_profits_when_price_rises_and_a_short_when_it_falls() {
        assert_eq!(long(100.0, 110.0, 2.0, 0.0).gross_pnl(), 20.0);
        let short = Trade::new("BTCUSDT", Side::Sell, 100.0, 90.0, 2.0, 0.0, 1, 2);
        assert_eq!(short.gross_pnl(), 20.0);
    }

    #[test]
    fn fees_turn_a_small_gross_win_into_a_loss() {
        let t = long(100.0, 101.0, 1.0, 2.0);
        assert_eq!(t.gross_pnl(), 1.0);
        assert_eq!(t.net_pnl(), -1.0);
        assert!(!t.is_win(), "cleared 1, paid 2: not a win");
    }

    #[test]
    fn r_multiple_is_measured_against_the_stop_distance() {
        let t = long(100.0, 110.0, 1.0, 0.0);
        assert_eq!(t.r_multiple(5.0), Some(2.0));
        assert_eq!(t.r_multiple(0.0), None);
        assert_eq!(t.r_multiple(f64::NAN), None);
    }

    #[test]
    fn a_fresh_ledger_starts_at_its_starting_equity() {
        let ledger = PnlLedger::new(10_000.0);
        assert_eq!(ledger.current_equity(), 10_000.0);
        assert_eq!(ledger.equity_curve(), &[10_000.0]);
        assert_eq!(ledger.max_drawdown(), 0.0);
        assert!(ledger.trades().is_empty());
    }

    #[test]
    fn marks_track_the_peak_and_the_worst_drawdown() {
        let mut ledger = PnlLedger::new(100.0);
        ledger.mark(120.0);
        ledger.mark(90.0); // 25% below the 120 peak
        ledger.mark(110.0); // recovering does not erase the drawdown
        assert_eq!(ledger.max_drawdown(), 0.25);
        assert_eq!(ledger.current_equity(), 110.0);
    }

    #[test]
    fn trades_are_summed_net_of_fees() {
        let mut ledger = PnlLedger::new(1_000.0);
        ledger.record_trade(long(100.0, 110.0, 1.0, 1.0));
        ledger.record_trade(long(100.0, 90.0, 1.0, 1.0));
        assert_eq!(ledger.gross_pnl(), 0.0, "+10 then -10");
        assert_eq!(ledger.fees_paid(), 2.0);
        assert_eq!(ledger.realized_pnl(), -2.0);
        assert_eq!(ledger.wins(), 1);
        assert_eq!(ledger.losses(), 1);
    }
}
