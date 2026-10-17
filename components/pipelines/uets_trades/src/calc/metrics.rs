//! Performance metrics — identical formulas across all execution modes.
//!
//! These are pure functions over an equity curve and trade list, so every
//! number on screen (or in a report) is recomputable from the log.

use crate::types::Capital;

/// One complete reporting snapshot, computed incrementally.
#[derive(Clone, Debug, PartialEq)]
pub struct PerformanceSnapshot {
    /// Strategy instance identity, e.g. `ofmr@btcusdt`.
    pub strategy_id: String,
    /// Asset ticker, e.g. `btcusdt`.
    pub asset_id: String,
    /// Snapshot timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Starting capital allocated to this instance.
    pub starting_capital: f64,
    /// Current cash balance.
    pub cash: f64,
    /// Current equity (cash + unrealized PnL).
    pub equity: f64,
    /// Unrealized PnL on open positions.
    pub unrealized_pnl: f64,
    /// Realized PnL from closed trades.
    pub realized_pnl: f64,
    /// Total fees paid.
    pub fees_paid: f64,
    /// `(equity - starting) / starting`.
    pub cumulative_return: f64,
    /// Absolute return in quote currency.
    pub absolute_return: f64,
    /// Largest peak-to-trough fall, as a fraction.
    pub max_drawdown: f64,
    /// Current drawdown from peak.
    pub current_drawdown: f64,
    /// Annualised Sortino ratio (downside deviation only).
    pub sortino: f64,
    /// Winning trades / all trades, in [0, 1].
    pub win_rate: f64,
    /// Mean realized R multiple over trades whose initial risk was recorded.
    pub expectancy_rr: f64,
    /// Total trade count.
    pub trades: usize,
    /// Winning trade count.
    pub wins: usize,
    /// Losing trade count.
    pub losses: f64,
    /// Gross exposure (sum of |position| * mark).
    pub gross_exposure: f64,
    /// Net exposure (sum of signed position * mark).
    pub net_exposure: f64,
    /// Margin used at current leverage.
    pub margin_used: f64,
    /// Available margin.
    pub available_margin: f64,
    /// Effective leverage.
    pub effective_leverage: f64,
}

/// Combine strategy-instance accounting for portfolio-level reporting.
/// Ratios that depend on a time series (Sortino and drawdown) are calculated
/// from the caller's aggregated equity curve, not averaged from instances.
pub fn aggregate_capital(capitals: &[Capital]) -> Option<Capital> {
    if capitals.is_empty() {
        return None;
    }
    let mut aggregate = Capital::new(0.0);
    for capital in capitals {
        aggregate.starting += capital.starting;
        aggregate.cash += capital.cash;
        aggregate.equity += capital.equity;
        aggregate.unrealized_pnl += capital.unrealized_pnl;
        aggregate.realized_pnl += capital.realized_pnl;
        aggregate.fees_paid += capital.fees_paid;
        aggregate.gross_exposure += capital.gross_exposure;
        aggregate.net_exposure += capital.net_exposure;
        aggregate.margin_used += capital.margin_used;
        aggregate.available_margin += capital.available_margin;
        aggregate.trade_count += capital.trade_count;
        aggregate.win_count += capital.win_count;
        aggregate.loss_count += capital.loss_count;
        aggregate.r_multiple_sum += capital.r_multiple_sum;
        aggregate.r_trades += capital.r_trades;
    }
    aggregate.effective_leverage = if aggregate.equity > 0.0 {
        aggregate.gross_exposure / aggregate.equity
    } else {
        0.0
    };
    Some(aggregate)
}

impl PerformanceSnapshot {
    pub fn from_capital(
        strategy_id: &str,
        asset_id: &str,
        timestamp: chrono::DateTime<chrono::Utc>,
        capital: &Capital,
        equity_curve: &[f64],
        periods_per_year: f64,
    ) -> Self {
        let starting = capital.starting;
        let equity = capital.equity;
        let cumulative_return = if starting > 0.0 {
            (equity - starting) / starting
        } else {
            0.0
        };
        let absolute_return = equity - starting;

        // Drawdown from equity curve
        let (max_dd, current_dd) = drawdown_stats(equity_curve);

        // Sortino from returns
        let returns = returns_from_equity(equity_curve);
        let sortino = sortino(&returns, 0.0, periods_per_year);

        let win_rate = if capital.trades() > 0 {
            capital.wins() as f64 / capital.trades() as f64
        } else {
            0.0
        };

        Self {
            strategy_id: strategy_id.to_string(),
            asset_id: asset_id.to_string(),
            timestamp,
            starting_capital: starting,
            cash: capital.cash,
            equity,
            unrealized_pnl: capital.unrealized_pnl,
            realized_pnl: capital.realized_pnl,
            fees_paid: capital.fees_paid,
            cumulative_return,
            absolute_return,
            max_drawdown: max_dd,
            current_drawdown: current_dd,
            sortino,
            win_rate,
            expectancy_rr: capital.expectancy_rr(),
            trades: capital.trades(),
            wins: capital.wins(),
            losses: capital.losses() as f64,
            gross_exposure: capital.gross_exposure,
            net_exposure: capital.net_exposure,
            margin_used: capital.margin_used,
            available_margin: capital.available_margin,
            effective_leverage: capital.effective_leverage,
        }
    }
}

/// Largest peak-to-trough fall and current drawdown from an equity curve.
pub fn drawdown_stats(equity: &[f64]) -> (f64, f64) {
    let mut peak = f64::NEG_INFINITY;
    let mut worst: f64 = 0.0;
    for &value in equity {
        if value > peak {
            peak = value;
        }
        if peak > 0.0 {
            let dd = (peak - value) / peak;
            if dd > worst {
                worst = dd;
            }
        }
    }
    let current = equity.last().copied().unwrap_or(0.0);
    let current_dd = if peak > 0.0 && current < peak {
        (peak - current) / peak
    } else {
        0.0
    };
    (worst, current_dd)
}

/// Period-over-period simple returns from an equity curve.
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
        f64::INFINITY
    } else {
        0.0
    }
}

/// Win rate: wins / total trades.
pub fn win_rate(wins: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        wins as f64 / total as f64
    }
}

/// Cumulative return: (ending - starting) / starting.
pub fn cumulative_return(starting: f64, ending: f64) -> f64 {
    if starting > 0.0 {
        (ending - starting) / starting
    } else {
        0.0
    }
}

/// Max drawdown from an equity curve alone.
pub fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut worst: f64 = 0.0;
    for &value in equity {
        if value > peak {
            peak = value;
        }
        if peak > 0.0 {
            let dd = (peak - value) / peak;
            if dd > worst {
                worst = dd;
            }
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawdown_finds_worst_peak_to_trough() {
        let (mdd, _) = drawdown_stats(&[100.0, 120.0, 90.0, 110.0]);
        assert!((mdd - 0.25).abs() < 1e-12);
    }

    #[test]
    fn drawdown_monotonic_rise_is_zero() {
        let (mdd, _) = drawdown_stats(&[100.0, 110.0, 120.0]);
        assert_eq!(mdd, 0.0);
    }

    #[test]
    fn sortino_penalises_only_downside() {
        assert_eq!(sortino(&[0.1, -0.1], 0.0, 1.0), 0.0);
        assert_eq!(sortino(&[0.1, 0.1], 0.0, 1.0), f64::INFINITY);
        assert_eq!(sortino(&[], 0.0, 1.0), 0.0);
    }

    #[test]
    fn cumulative_return_is_fraction() {
        assert_eq!(cumulative_return(1_000.0, 1_250.0), 0.25);
        assert_eq!(cumulative_return(1_000.0, 750.0), -0.25);
    }

    #[test]
    fn win_rate_zero_when_no_trades() {
        assert_eq!(win_rate(0, 0), 0.0);
        assert!((win_rate(1, 3) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn aggregate_capital_sums_portfolio_values_and_rr_denominators() {
        let mut first = Capital::new(1_000.0);
        first.cash = 950.0;
        first.equity = 1_050.0;
        first.r_multiple_sum = 2.0;
        first.r_trades = 2;
        let mut second = Capital::new(2_000.0);
        second.cash = 1_900.0;
        second.equity = 2_100.0;
        second.r_multiple_sum = 3.0;
        second.r_trades = 1;
        let total = aggregate_capital(&[first, second]).unwrap();
        assert_eq!(total.starting, 3_000.0);
        assert_eq!(total.cash, 2_850.0);
        assert_eq!(total.equity, 3_150.0);
        assert!((total.expectancy_rr() - 5.0 / 3.0).abs() < 1e-12);
    }
}
