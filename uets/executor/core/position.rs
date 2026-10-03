//! Net exposure per symbol, marked to market.
//!
//! `core::strategy::Position` is what the strategy *wants* (flat / long /
//! short). This module is what the account *has*: a signed size, an average
//! entry, and the realized pnl that got it there. Keeping the two apart is what
//! lets a decision stay a pure function while the money stays stateful.

use std::collections::BTreeMap;

use super::order::{Fill, Side};

/// Sizes below this are float dust from an exact close, not a position.
const FLAT_EPS: f64 = 1e-12;

/// Net position in one symbol.
///
/// Named `Holding` rather than `Position` on purpose: with
/// `core::strategy::Position` in scope too, one word for two different things
/// is how a sign error gets written.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Holding {
    /// Signed size: positive long, negative short, zero flat.
    pub size: f64,
    /// Average entry price of the open size. Zero while flat.
    pub avg_entry: f64,
    /// Pnl banked by closes. Fees are tracked separately in `fees_paid`.
    pub realized_pnl: f64,
    /// Commission paid on every fill that touched this symbol.
    pub fees_paid: f64,
}

impl Holding {
    pub fn is_flat(&self) -> bool {
        self.size.abs() < FLAT_EPS
    }

    /// The side the position is held on, or `None` while flat.
    pub fn side(&self) -> Option<Side> {
        if self.is_flat() {
            None
        } else if self.size > 0.0 {
            Some(Side::Buy)
        } else {
            Some(Side::Sell)
        }
    }

    /// Paper value of the open size at `mark`.
    pub fn notional(&self, mark: f64) -> f64 {
        self.size * mark
    }

    /// Pnl if the open size were closed at `mark`.
    pub fn unrealized_pnl(&self, mark: f64) -> f64 {
        self.size * (mark - self.avg_entry)
    }

    /// Realized plus unrealized.
    pub fn equity_pnl(&self, mark: f64) -> f64 {
        self.realized_pnl + self.unrealized_pnl(mark)
    }
}

/// Fold a fill into a holding, returning the realized pnl it produced.
///
/// Three cases, and only three:
///
/// * **opening** from flat — the entry price is the fill price;
/// * **adding** to the same side — the entry becomes the weighted average, so
///   the holding keeps a single meaningful entry;
/// * **reducing or flipping** — the closed part realizes pnl against the old
///   entry; any remainder opens at the fill price.
pub fn apply_fill(holding: &mut Holding, fill: &Fill) -> f64 {
    let signed = fill.side.signed(fill.qty);
    let previous = holding.size;

    let realized = if previous.abs() < FLAT_EPS {
        holding.avg_entry = fill.price;
        0.0
    } else if previous.signum() == signed.signum() {
        let weight = previous.abs() + signed.abs();
        holding.avg_entry =
            (holding.avg_entry * previous.abs() + fill.price * signed.abs()) / weight;
        0.0
    } else {
        let closed = signed.abs().min(previous.abs());
        // `previous.signum()` collapses the long and short cases into one line.
        let pnl = closed * (fill.price - holding.avg_entry) * previous.signum();
        if signed.abs() > previous.abs() {
            // Flipped through flat: the leftover is a new position at this price.
            holding.avg_entry = fill.price;
        }
        pnl
    };

    holding.size = previous + signed;
    if holding.size.abs() < FLAT_EPS {
        // Snap the dust away so `is_flat` is exact and a later open starts clean.
        holding.size = 0.0;
        holding.avg_entry = 0.0;
    }
    holding.realized_pnl += realized;
    holding.fees_paid += fill.fee;
    realized
}

/// Every symbol's holding, keyed by the venue's symbol spelling.
#[derive(Clone, Debug, Default)]
pub struct PositionBook {
    holdings: BTreeMap<String, Holding>,
}

impl PositionBook {
    pub fn new() -> Self {
        Self {
            holdings: BTreeMap::new(),
        }
    }

    /// A copy of the holding, flat when the symbol has never traded.
    ///
    /// Returns a value rather than an `Option` so callers that only need a size
    /// never have to special-case "no row yet".
    pub fn holding(&self, symbol: &str) -> Holding {
        self.holdings.get(symbol).copied().unwrap_or_default()
    }

    pub fn get(&self, symbol: &str) -> Option<&Holding> {
        self.holdings.get(symbol)
    }

    /// Apply a fill under its own symbol, returning the realized pnl.
    pub fn apply(&mut self, fill: &Fill) -> f64 {
        let entry = self.holdings.entry(fill.symbol.clone()).or_default();
        apply_fill(entry, fill)
    }

    /// Symbols with a non-flat holding, in sorted order.
    pub fn open_holdings(&self) -> impl Iterator<Item = (&String, &Holding)> {
        self.holdings.iter().filter(|(_, h)| !h.is_flat())
    }

    pub fn symbols(&self) -> impl Iterator<Item = &String> {
        self.holdings.keys()
    }

    pub fn realized_pnl(&self) -> f64 {
        self.holdings.values().map(|h| h.realized_pnl).sum()
    }

    pub fn fees_paid(&self) -> f64 {
        self.holdings.values().map(|h| h.fees_paid).sum()
    }

    /// The mark for `symbol`, falling back to its entry price.
    ///
    /// An unmarked symbol therefore contributes zero unrealized pnl instead of
    /// a fabricated one: a stale mark is worse than an admitted gap.
    fn mark(marks: &BTreeMap<String, f64>, symbol: &str, holding: &Holding) -> f64 {
        marks.get(symbol).copied().unwrap_or(holding.avg_entry)
    }

    pub fn unrealized_pnl(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.holdings
            .iter()
            .map(|(symbol, holding)| holding.unrealized_pnl(Self::mark(marks, symbol, holding)))
            .sum()
    }

    /// Sum of absolute notionals: what a margin model actually sizes against.
    pub fn gross_exposure(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.holdings
            .iter()
            .map(|(symbol, holding)| holding.notional(Self::mark(marks, symbol, holding)).abs())
            .sum()
    }

    /// Signed notionals: net directional bias, which can sit near zero while
    /// gross exposure is large.
    pub fn net_exposure(&self, marks: &BTreeMap<String, f64>) -> f64 {
        self.holdings
            .iter()
            .map(|(symbol, holding)| holding.notional(Self::mark(marks, symbol, holding)))
            .sum()
    }

    pub fn is_flat(&self) -> bool {
        self.holdings.values().all(Holding::is_flat)
    }

    pub fn len(&self) -> usize {
        self.holdings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.holdings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::order::OrderId;

    fn fill(side: Side, qty: f64, price: f64, fee: f64) -> Fill {
        Fill::new(OrderId::new("t"), "BTCUSDT", side, qty, price, fee)
    }

    fn marks(symbol: &str, price: f64) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        m.insert(symbol.to_string(), price);
        m
    }

    #[test]
    fn opening_from_flat_sets_the_entry_price() {
        let mut book = PositionBook::new();
        assert_eq!(book.apply(&fill(Side::Buy, 1.0, 100.0, 0.5)), 0.0);
        let h = book.holding("BTCUSDT");
        assert_eq!(h.size, 1.0);
        assert_eq!(h.avg_entry, 100.0);
        assert_eq!(h.realized_pnl, 0.0);
        assert_eq!(h.fees_paid, 0.5);
        assert_eq!(h.side(), Some(Side::Buy));
    }

    #[test]
    fn adding_to_the_same_side_weight_averages_the_entry() {
        let mut book = PositionBook::new();
        book.apply(&fill(Side::Buy, 1.0, 100.0, 0.0));
        book.apply(&fill(Side::Buy, 1.0, 110.0, 0.0));
        assert_eq!(book.holding("BTCUSDT").avg_entry, 105.0);
        assert_eq!(book.holding("BTCUSDT").realized_pnl, 0.0);
    }

    #[test]
    fn closing_a_long_realizes_against_the_entry() {
        let mut book = PositionBook::new();
        book.apply(&fill(Side::Buy, 1.0, 100.0, 0.0));
        let realized = book.apply(&fill(Side::Sell, 1.0, 120.0, 0.0));
        assert_eq!(realized, 20.0);
        let h = book.holding("BTCUSDT");
        assert!(h.is_flat());
        assert_eq!(h.avg_entry, 0.0, "a flat holding carries no entry");
        assert_eq!(h.realized_pnl, 20.0);
        assert!(book.is_flat());
    }

    #[test]
    fn closing_a_short_profits_when_price_falls() {
        let mut book = PositionBook::new();
        book.apply(&fill(Side::Sell, 1.0, 100.0, 0.0));
        assert_eq!(book.holding("BTCUSDT").size, -1.0);
        let realized = book.apply(&fill(Side::Buy, 1.0, 90.0, 0.0));
        assert_eq!(realized, 10.0);
        assert!(book.holding("BTCUSDT").is_flat());
    }

    #[test]
    fn flipping_through_flat_reopens_at_the_fill_price() {
        let mut book = PositionBook::new();
        book.apply(&fill(Side::Buy, 1.0, 100.0, 0.0));
        let realized = book.apply(&fill(Side::Sell, 3.0, 110.0, 0.0));
        assert_eq!(realized, 10.0, "only the 1.0 that existed was closed");
        let h = book.holding("BTCUSDT");
        assert_eq!(h.size, -2.0);
        assert_eq!(h.avg_entry, 110.0, "the leftover is a fresh short");
    }

    #[test]
    fn unrealized_and_exposure_use_the_mark() {
        let mut book = PositionBook::new();
        book.apply(&fill(Side::Buy, 2.0, 100.0, 0.0));
        let m = marks("BTCUSDT", 130.0);
        assert_eq!(book.unrealized_pnl(&m), 60.0);
        assert_eq!(book.gross_exposure(&m), 260.0);
        assert_eq!(book.net_exposure(&m), 260.0);
        // Without a mark the position contributes nothing rather than a guess.
        assert_eq!(book.unrealized_pnl(&BTreeMap::new()), 0.0);
    }

    #[test]
    fn open_holdings_excludes_flat_symbols() {
        let mut book = PositionBook::new();
        book.apply(&fill(Side::Buy, 1.0, 100.0, 0.0));
        let mut eth = fill(Side::Buy, 1.0, 10.0, 0.0);
        eth.symbol = "ETHUSDT".to_string();
        book.apply(&eth);
        let mut close_eth = fill(Side::Sell, 1.0, 10.0, 0.0);
        close_eth.symbol = "ETHUSDT".to_string();
        book.apply(&close_eth);
        let open: Vec<&String> = book.open_holdings().map(|(s, _)| s).collect();
        assert_eq!(open, vec!["BTCUSDT"]);
        assert_eq!(book.len(), 2, "the flat row is still remembered");
    }
}
