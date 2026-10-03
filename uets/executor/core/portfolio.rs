//! Cash, margin and equity for one account.
//!
//! This is deliberately a *risk budget*, not a venue statement:
//! `connectors/<venue>` supplies the venue's own figures through
//! `AccountConnector::snapshot`, and the two are reconciled rather than
//! conflated. What is modelled here is the realized-cash convention —
//!
//! ```text
//!   cash   = starting_cash + realized pnl - fees
//!   equity = cash + unrealized pnl at the current marks
//!   margin = gross exposure / leverage
//! ```
//!
//! — which needs no cross/isolated distinction and therefore behaves the same
//! on every venue.

use std::collections::BTreeMap;

use super::order::Fill;
use super::pnl::{PnlLedger, Trade};
use super::position::PositionBook;

#[derive(Clone, Debug)]
pub struct Portfolio {
    starting_cash: f64,
    leverage: f64,
    holdings: PositionBook,
    ledger: PnlLedger,
}

impl Portfolio {
    /// `starting_cash` must be finite and positive, `leverage` finite and
    /// positive. Both are refused rather than defaulted: a zero leverage would
    /// divide by zero on the first margin calculation.
    pub fn new(starting_cash: f64, leverage: f64) -> Result<Self, String> {
        if !starting_cash.is_finite() || starting_cash <= 0.0 {
            return Err(format!(
                "starting cash must be positive, got {starting_cash}"
            ));
        }
        if !leverage.is_finite() || leverage <= 0.0 {
            return Err(format!("leverage must be positive, got {leverage}"));
        }
        Ok(Self {
            starting_cash,
            leverage,
            holdings: PositionBook::new(),
            ledger: PnlLedger::new(starting_cash),
        })
    }

    pub fn starting_cash(&self) -> f64 {
        self.starting_cash
    }

    pub fn leverage(&self) -> f64 {
        self.leverage
    }

    pub fn holdings(&self) -> &PositionBook {
        &self.holdings
    }

    pub fn ledger(&self) -> &PnlLedger {
        &self.ledger
    }

    /// Realized cash balance. Unrealized pnl never touches it.
    pub fn cash(&self) -> f64 {
        self.starting_cash + self.holdings.realized_pnl() - self.holdings.fees_paid()
    }

    /// Fold a fill into the book, returning the realized pnl it produced.
    pub fn apply_fill(&mut self, fill: &Fill) -> f64 {
        self.holdings.apply(fill)
    }

    /// Bank a completed trade so win rate and Sortino see it.
    pub fn record_trade(&mut self, trade: Trade) {
        self.ledger.record_trade(trade);
    }

    pub fn unrealized_pnl(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.holdings.unrealized_pnl(marks)
    }

    pub fn gross_exposure(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.holdings.gross_exposure(marks)
    }

    pub fn net_exposure(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.holdings.net_exposure(marks)
    }

    /// Margin the book currently requires.
    pub fn margin_used(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.gross_exposure(marks) / self.leverage
    }

    pub fn equity(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.cash() + self.unrealized_pnl(marks)
    }

    /// Equity not already committed to margin.
    pub fn available_margin(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.equity(marks) - self.margin_used(marks)
    }

    /// Gross exposure as a multiple of equity.
    ///
    /// `INFINITY` when there is exposure but no equity left: a wiped account is
    /// maximally levered, not unlimitedly safe. A flat book is `0.0`.
    pub fn effective_leverage(&self, marks: &BTreeMap<String, f64>) -> f64 {
        let gross = self.gross_exposure(marks);
        if gross == 0.0 {
            return 0.0;
        }
        let equity = self.equity(marks);
        if equity <= 0.0 {
            f64::INFINITY
        } else {
            gross / equity
        }
    }

    /// Mark the account, record the equity sample, return the equity.
    pub fn mark(&mut self, marks: &BTreeMap<String, f64>) -> f64 {
        let equity = self.equity(marks);
        self.ledger.mark(equity);
        equity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::order::{Fill, OrderId, Side};

    fn marks(price: f64) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        m.insert("BTCUSDT".to_string(), price);
        m
    }

    fn fill(side: Side, qty: f64, price: f64) -> Fill {
        Fill::new(OrderId::new("t"), "BTCUSDT", side, qty, price, 0.0)
    }

    #[test]
    fn a_portfolio_needs_positive_cash_and_leverage() {
        assert!(Portfolio::new(0.0, 10.0).is_err());
        assert!(Portfolio::new(1_000.0, 0.0).is_err());
        assert!(Portfolio::new(f64::NAN, 10.0).is_err());
        assert!(Portfolio::new(1_000.0, 10.0).is_ok());
    }

    #[test]
    fn cash_only_moves_on_a_realized_close() {
        let mut p = Portfolio::new(1_000.0, 10.0).unwrap();
        p.apply_fill(&Fill::new(
            OrderId::new("a"),
            "BTCUSDT",
            Side::Buy,
            2.0,
            100.0,
            1.0,
        ));
        assert_eq!(p.cash(), 999.0, "starting cash less the entry fee");
        p.apply_fill(&Fill::new(
            OrderId::new("b"),
            "BTCUSDT",
            Side::Sell,
            2.0,
            120.0,
            1.0,
        ));
        assert_eq!(p.cash(), 1_038.0, "+40 realized, -2 fees");
    }

    #[test]
    fn equity_adds_unrealized_pnl_to_cash() {
        let mut p = Portfolio::new(1_000.0, 10.0).unwrap();
        p.apply_fill(&fill(Side::Buy, 1.0, 100.0));
        assert_eq!(p.equity(&marks(150.0)), 1_050.0);
        assert_eq!(p.unrealized_pnl(&marks(150.0)), 50.0);
    }

    #[test]
    fn margin_and_available_margin_follow_gross_exposure() {
        let mut p = Portfolio::new(1_000.0, 10.0).unwrap();
        p.apply_fill(&fill(Side::Buy, 1.0, 100.0));
        let m = marks(100.0);
        assert_eq!(p.gross_exposure(&m), 100.0);
        assert_eq!(p.margin_used(&m), 10.0, "100 of exposure at 10x");
        assert_eq!(p.available_margin(&m), 990.0);
        assert_eq!(p.effective_leverage(&m), 0.1);
    }

    #[test]
    fn a_flat_book_has_no_leverage_and_a_wiped_book_is_infinite() {
        let mut p = Portfolio::new(1_000.0, 10.0).unwrap();
        assert_eq!(p.effective_leverage(&marks(100.0)), 0.0, "nothing at risk");
        p.apply_fill(&fill(Side::Buy, 20.0, 100.0));
        // Mark collapses to 1: equity is -980 while 20 units are still exposed.
        assert_eq!(p.effective_leverage(&marks(1.0)), f64::INFINITY);
    }

    #[test]
    fn mark_records_the_equity_curve() {
        let mut p = Portfolio::new(1_000.0, 10.0).unwrap();
        p.apply_fill(&fill(Side::Buy, 1.0, 100.0));
        assert_eq!(p.mark(&marks(200.0)), 1_100.0);
        assert_eq!(p.ledger().current_equity(), 1_100.0);
        p.mark(&marks(50.0));
        assert!((p.ledger().max_drawdown() - 150.0 / 1_100.0).abs() < 1e-12);
    }
}
