//! 100 ms reporting cadence — a report update must not block strategy
//! execution or order reporting.

use crate::calc::metrics::PerformanceSnapshot;
use crate::types::Capital;
use chrono::{DateTime, Utc};

/// Reporting configuration.
#[derive(Clone, Debug)]
pub struct ReportConfig {
    /// Reporting cadence (nominal 100 ms).
    pub interval_ms: u64,
    /// Periods per year for Sortino annualisation.
    pub periods_per_year: f64,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            interval_ms: 100,
            periods_per_year: 1.0,
        }
    }
}

/// Incremental reporter — maintains live state and produces snapshots.
pub struct Reporter {
    config: ReportConfig,
    /// Last report timestamp.
    last_report: Option<DateTime<Utc>>,
    /// Equity curve samples (bounded).
    equity_curve: Vec<f64>,
}

impl Reporter {
    pub fn new(config: ReportConfig) -> Self {
        Self {
            config,
            last_report: None,
            equity_curve: Vec::new(),
        }
    }

    /// Check whether it's time to produce a report based on `now`.
    pub fn should_report(&self, now: DateTime<Utc>) -> bool {
        match self.last_report {
            None => true,
            Some(last) => {
                (now - last).num_milliseconds() >= self.config.interval_ms as i64
            }
        }
    }

    /// Produce a report snapshot if the cadence has elapsed.
    /// Returns `None` if not yet time to report.
    pub fn maybe_report(
        &mut self,
        now: DateTime<Utc>,
        strategy_id: &str,
        asset_id: &str,
        capital: &Capital,
    ) -> Option<PerformanceSnapshot> {
        if !self.should_report(now) {
            return None;
        }

        self.equity_curve.push(capital.equity);
        // Keep the curve bounded (last 10_000 samples ~ 16 minutes at 100ms)
        if self.equity_curve.len() > 10_000 {
            self.equity_curve.drain(..self.equity_curve.len() - 10_000);
        }

        self.last_report = Some(now);

        Some(PerformanceSnapshot::from_capital(
            strategy_id,
            asset_id,
            now,
            capital,
            &self.equity_curve,
            self.config.periods_per_year,
        ))
    }

    /// Force a report immediately (e.g. on session end).
    pub fn force_report(
        &mut self,
        now: DateTime<Utc>,
        strategy_id: &str,
        asset_id: &str,
        capital: &Capital,
    ) -> PerformanceSnapshot {
        self.equity_curve.push(capital.equity);
        if self.equity_curve.len() > 10_000 {
            self.equity_curve.drain(..self.equity_curve.len() - 10_000);
        }
        self.last_report = Some(now);

        PerformanceSnapshot::from_capital(
            strategy_id,
            asset_id,
            now,
            capital,
            &self.equity_curve,
            self.config.periods_per_year,
        )
    }

    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }
}

impl Default for Reporter {
    fn default() -> Self {
        Self::new(ReportConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Capital;
    use chrono::{TimeZone, Utc};

    fn sample_capital() -> Capital {
        Capital {
            starting: 10_000.0,
            cash: 10_000.0,
            equity: 10_500.0,
            unrealized_pnl: 500.0,
            realized_pnl: 0.0,
            fees_paid: 0.0,
            gross_exposure: 0.0,
            net_exposure: 0.0,
            margin_used: 0.0,
            available_margin: 10_500.0,
            effective_leverage: 0.0,
            trade_count: 0,
            win_count: 0,
            loss_count: 0,
            r_multiple_sum: 0.0,
            r_trades: 0,
        }
    }

    #[test]
    fn report_respects_100ms_cadence() {
        let mut reporter = Reporter::new(ReportConfig {
            interval_ms: 100,
            periods_per_year: 1.0,
        });
        let base = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();

        // First call should report
        let r1 = reporter.maybe_report(base, "ofmr@btcusdt", "btcusdt", &sample_capital());
        assert!(r1.is_some());

        // Immediate second call should not report (same timestamp)
        let r2 = reporter.maybe_report(base, "ofmr@btcusdt", "btcusdt", &sample_capital());
        assert!(r2.is_none());

        // After 100ms should report again
        let later = base + chrono::Duration::milliseconds(100);
        let r3 = reporter.maybe_report(later, "ofmr@btcusdt", "btcusdt", &sample_capital());
        assert!(r3.is_some());
    }

    #[test]
    fn force_report_ignores_cadence() {
        let mut reporter = Reporter::new(ReportConfig {
            interval_ms: 1000,
            periods_per_year: 1.0,
        });
        let base = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();

        let r1 = reporter.maybe_report(base, "ofmr@btcusdt", "btcusdt", &sample_capital());
        assert!(r1.is_some());

        // Force report immediately after
        let r2 = reporter.force_report(base, "ofmr@btcusdt", "btcusdt", &sample_capital());
        assert!(r2.equity == 10_500.0);
    }
}