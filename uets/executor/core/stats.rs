//! Performance statistics: what the dashboard renders at the front.
//!
//! Pure functions over the equity curve and the trade list, so every number on
//! screen is recomputable from the log — and therefore assertable in a test
//! instead of taken on faith.

use super::pnl::{PnlLedger, Trade};

/// The headline numbers.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PerformanceStats {
    /// Annualised Sortino ratio (downside deviation only).
    pub sortino: f64,
    /// Winning trades / all trades, in `[0, 1]`. Zero when there are none.
    pub win_rate: f64,
    /// `(ending - starting) / starting`.
    pub cumulative_return: f64,
    /// Largest peak-to-trough fall, as a fraction.
    pub max_drawdown: f64,
    pub trades: usize,
    pub wins: usize,
    pub losses: usize,
}

impl PerformanceStats {
    /// Everything the GUI shows, computed from one ledger.
    ///
    /// `periods_per_year` annualises the ratio; pass `1.0` for an
    /// un-annualised figure, which is what a short intraday session wants.
    pub fn from_ledger(ledger: &PnlLedger, periods_per_year: f64) -> Self {
        let equity = ledger.equity_curve();
        let returns = returns_from_equity(equity);
        let ending = equity
            .last()
            .copied()
            .unwrap_or_else(|| ledger.starting_equity());
        Self {
            sortino: sortino(&returns, 0.0, periods_per_year),
            win_rate: win_rate(ledger.trades()),
            cumulative_return: cumulative_return(ledger.starting_equity(), ending),
            max_drawdown: ledger.max_drawdown(),
            trades: ledger.trades().len(),
            wins: ledger.wins(),
            losses: ledger.losses(),
        }
    }
}

/// Period-over-period simple returns from an equity curve.
///
/// A non-positive previous equity yields no return: division by a wiped account
/// is undefined, not infinite.
pub fn returns_from_equity(equity: &[f64]) -> Vec<f64> {
    equity
        .windows(2)
        .filter_map(|w| {
            if w[0] > 0.0 {
                Some((w[1] - w[0]) / w[0])
            } else {
                None
            }
        })
        .collect()
}

/// Annualised Sortino ratio.
///
/// The denominator is the *downside* deviation over **all** periods, not only
/// the losing ones. That is the standard definition, and the reason a strategy
/// with one very large loss scores worse here than on a Sharpe ratio.
pub fn sortino(returns: &[f64], target: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f64;
    let excess_sum: f64 = returns.iter().map(|r| r - target).sum();
    let mean = excess_sum / n;
    let downside_sq: f64 = returns
        .iter()
        .map(|r| {
            let e = r - target;
            if e < 0.0 { e * e } else { 0.0 }
        })
        .sum();
    let downside = (downside_sq / n).sqrt();
    if downside > 0.0 {
        let annualise = if periods_per_year > 0.0 {
            periods_per_year.sqrt()
        } else {
            1.0
        };
        (mean / downside) * annualise
    } else if mean > 0.0 {
        // No downside at all: infinite ratio, which the JSON writer renders as
        // `null` so the dashboard shows "--" rather than a fake number.
        f64::INFINITY
    } else {
        0.0
    }
}

/// Winning trades / all trades. A breakeven counts as neither.
pub fn win_rate(trades: &[Trade]) -> f64 {
    if trades.is_empty() {
        return 0.0;
    }
    trades.iter().filter(|t| t.is_win()).count() as f64 / trades.len() as f64
}

pub fn cumulative_return(starting_equity: f64, ending_equity: f64) -> f64 {
    if starting_equity <= 0.0 {
        return 0.0;
    }
    (ending_equity - starting_equity) / starting_equity
}

/// Largest peak-to-trough fall as a fraction, from an equity curve alone.
///
/// Exposed separately from the ledger so a curve pulled out of a file can be
/// scored without replaying it through the ledger.
pub fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut worst: f64 = 0.0;
    for &value in equity {
        if value > peak {
            peak = value;
        }
        if peak > 0.0 {
            let drawdown = (peak - value) / peak;
            if drawdown > worst {
                worst = drawdown;
            }
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::order::Side;

    fn trade(entry: f64, exit: f64) -> Trade {
        Trade::new("BTCUSDT", Side::Buy, entry, exit, 1.0, 0.0, 1, 2)
    }

    #[test]
    fn equity_returns_skip_a_wiped_step() {
        assert_eq!(returns_from_equity(&[100.0, 110.0, 99.0]), vec![0.1, -0.1]);
        assert_eq!(returns_from_equity(&[0.0, 5.0]), Vec::<f64>::new());
        assert!(returns_from_equity(&[]).is_empty());
    }

    #[test]
    fn sortino_penalises_only_downside_volatility() {
        // 10% up then 10% down: mean excess is zero, so the ratio is zero.
        assert_eq!(sortino(&[0.1, -0.1], 0.0, 1.0), 0.0);
        // Up only: there is no downside deviation at all.
        assert_eq!(sortino(&[0.1, 0.1], 0.0, 1.0), f64::INFINITY);
        assert!(sortino(&[-0.1, -0.1], 0.0, 1.0) < 0.0);
        assert_eq!(sortino(&[], 0.0, 1.0), 0.0, "no data is zero, not infinite");
    }

    #[test]
    fn sortino_annualises_by_the_square_root_of_the_period_count() {
        let unannualised = sortino(&[0.02, -0.01], 0.0, 1.0);
        let annualised = sortino(&[0.02, -0.01], 0.0, 252.0);
        assert!((annualised - unannualised * 252.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn win_rate_ignores_breakevens_in_the_numerator_only() {
        assert_eq!(win_rate(&[]), 0.0);
        let trades = vec![trade(100.0, 110.0), trade(100.0, 90.0), trade(100.0, 100.0)];
        assert!((win_rate(&trades) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn cumulative_return_is_a_fraction_of_the_starting_equity() {
        assert_eq!(cumulative_return(1_000.0, 1_250.0), 0.25);
        assert_eq!(cumulative_return(1_000.0, 750.0), -0.25);
        assert_eq!(
            cumulative_return(0.0, 100.0),
            0.0,
            "no starting equity, no percentage"
        );
    }

    #[test]
    fn max_drawdown_finds_the_worst_peak_to_trough_fall() {
        assert_eq!(max_drawdown(&[100.0, 120.0, 90.0, 110.0]), 0.25);
        assert_eq!(max_drawdown(&[100.0, 110.0, 120.0]), 0.0, "monotonic rise");
        assert_eq!(max_drawdown(&[]), 0.0);
    }

    #[test]
    fn stats_are_recomputable_from_the_ledger() {
        let mut ledger = PnlLedger::new(1_000.0);
        ledger.mark(1_100.0);
        ledger.mark(990.0);
        ledger.record_trade(trade(100.0, 110.0));
        let stats = PerformanceStats::from_ledger(&ledger, 1.0);
        assert_eq!(stats.trades, 1);
        assert_eq!(stats.wins, 1);
        assert_eq!(stats.losses, 0);
        assert_eq!(stats.win_rate, 1.0);
        assert!((stats.cumulative_return - (-0.01)).abs() < 1e-12);
        assert!((stats.max_drawdown - 0.1).abs() < 1e-12, "110 of 1100");
        assert!(stats.sortino.is_finite());
    }
}
