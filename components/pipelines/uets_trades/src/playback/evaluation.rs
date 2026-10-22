//! Fast, parallel evaluation of strategies over playback data.

use super::{MarketEvent, PlaybackBackend};
use crate::backend::BackendError;
use crate::calc::metrics::{drawdown_stats, returns_from_equity, sortino};
use crate::sim::SimConfig;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackMethod {
    ParametricMonteCarlo,
    #[serde(rename = "cpcv", alias = "combinatorial_purged_cv")]
    Cpcv,
    #[default]
    PurgedWalkForward,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlaybackSettings {
    pub method: PlaybackMethod,
    pub max_abs_correlation: f64,
    pub n_splits: usize,
    pub n_test_splits: usize,
    pub purge_bars: usize,
    pub max_paths: usize,
    pub periods_per_year: f64,
    pub strategy_id: String,
}

impl Default for PlaybackSettings {
    fn default() -> Self {
        Self {
            method: PlaybackMethod::PurgedWalkForward,
            max_abs_correlation: 0.5,
            n_splits: 6,
            n_test_splits: 2,
            purge_bars: 0,
            max_paths: 100_000,
            periods_per_year: 252.0,
            strategy_id: "playback".to_string(),
        }
    }
}

impl PlaybackSettings {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        if !self.max_abs_correlation.is_finite() || !(0.0..=1.0).contains(&self.max_abs_correlation)
        {
            return Err(EvaluationError::InvalidSettings(
                "max_abs_correlation must be finite and within [0, 1]".into(),
            ));
        }
        if !(2..=64).contains(&self.n_splits) {
            return Err(EvaluationError::InvalidSettings(
                "n_splits must be between 2 and 64".into(),
            ));
        }
        if self.n_test_splits == 0 || self.n_test_splits > self.n_splits {
            return Err(EvaluationError::InvalidSettings(
                "n_test_splits must be between 1 and n_splits".into(),
            ));
        }
        if self.max_paths == 0 {
            return Err(EvaluationError::InvalidSettings(
                "max_paths must be positive".into(),
            ));
        }
        if !self.periods_per_year.is_finite() || self.periods_per_year <= 0.0 {
            return Err(EvaluationError::InvalidSettings(
                "periods_per_year must be finite and positive".into(),
            ));
        }
        if self.strategy_id.trim().is_empty() {
            return Err(EvaluationError::InvalidSettings(
                "strategy_id must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Run the selected playback method. The factory creates independent
    /// strategy state for each fold or scenario. The playback loop itself is
    /// unpaced, so simulation throughput is not tied to wall-clock timestamps.
    pub fn run<S, F>(
        &self,
        input: PlaybackInput,
        sim_config: SimConfig,
        strategy_factory: F,
    ) -> Result<PlaybackReport, EvaluationError>
    where
        S: FnMut(&mut PlaybackBackend, &MarketEvent) -> Result<(), BackendError> + Send,
        F: Fn() -> S + Send + Sync,
    {
        self.validate()?;
        match (self.method, input) {
            (PlaybackMethod::ParametricMonteCarlo, PlaybackInput::Scenarios(paths)) => {
                self.run_monte_carlo(&paths, sim_config, &strategy_factory)
            }
            (PlaybackMethod::Cpcv, PlaybackInput::Events(events)) => {
                self.run_cpcv(&events, sim_config, &strategy_factory)
            }
            (PlaybackMethod::PurgedWalkForward, PlaybackInput::Events(events)) => {
                self.run_walk_forward(&events, sim_config, &strategy_factory)
            }
            (PlaybackMethod::ParametricMonteCarlo, PlaybackInput::Events(_)) => {
                Err(EvaluationError::InputMismatch(
                    "parametric_monte_carlo requires PlaybackInput::Scenarios".into(),
                ))
            }
            (PlaybackMethod::Cpcv, PlaybackInput::Scenarios(_))
            | (PlaybackMethod::PurgedWalkForward, PlaybackInput::Scenarios(_)) => {
                Err(EvaluationError::InputMismatch(
                    "CPCV and purged walk-forward require PlaybackInput::Events".into(),
                ))
            }
        }
    }

    fn run_monte_carlo<S, F>(
        &self,
        paths: &[Vec<MarketEvent>],
        sim_config: SimConfig,
        strategy_factory: &F,
    ) -> Result<PlaybackReport, EvaluationError>
    where
        S: FnMut(&mut PlaybackBackend, &MarketEvent) -> Result<(), BackendError> + Send,
        F: Fn() -> S + Send + Sync,
    {
        if paths.is_empty() {
            return Err(EvaluationError::InvalidInput(
                "Monte Carlo requires at least one scenario".into(),
            ));
        }
        for path in paths {
            validate_events(path)?;
        }
        let reference = &paths[0];
        if paths.iter().skip(1).any(|path| {
            path.len() != reference.len()
                || path
                    .iter()
                    .zip(reference)
                    .any(|(event, reference_event)| event.timestamp != reference_event.timestamp)
        }) {
            return Err(EvaluationError::InvalidInput(
                "Monte Carlo scenarios must share the same timestamp grid for correlation filtering"
                    .into(),
            ));
        }

        let selected = select_uncorrelated_paths(paths, self.max_abs_correlation, self.max_paths)?;
        let metrics = run_paths(
            selected
                .iter()
                .map(|&index| paths[index].as_slice())
                .collect(),
            &self.strategy_id,
            &sim_config,
            self.periods_per_year,
            strategy_factory,
        )?;
        Ok(PlaybackReport::new(
            self.method,
            metrics,
            paths.len(),
            selected.len(),
            0,
            self.max_abs_correlation,
        ))
    }

    fn run_walk_forward<S, F>(
        &self,
        events: &[MarketEvent],
        sim_config: SimConfig,
        strategy_factory: &F,
    ) -> Result<PlaybackReport, EvaluationError>
    where
        S: FnMut(&mut PlaybackBackend, &MarketEvent) -> Result<(), BackendError> + Send,
        F: Fn() -> S + Send + Sync,
    {
        validate_events(events)?;
        let symbol = events[0]
            .marks
            .keys()
            .next()
            .expect("validated event has a mark");
        if events
            .iter()
            .any(|event| event.marks.len() != 1 || !event.marks.contains_key(symbol))
        {
            return Err(EvaluationError::InvalidInput(
                "purged_walk_forward supports exactly one asset".into(),
            ));
        }
        if events.len() < self.n_splits {
            return Err(EvaluationError::InvalidInput(
                "event count must be at least n_splits".into(),
            ));
        }

        let fold_size = events.len().div_ceil(self.n_splits);
        let mut folds = Vec::with_capacity(self.n_splits);
        for fold in 0..self.n_splits {
            let raw_start = fold * fold_size;
            if raw_start >= events.len() {
                break;
            }
            let end = (raw_start + fold_size).min(events.len());
            let start = if fold == 0 {
                raw_start
            } else {
                raw_start.saturating_add(self.purge_bars)
            };
            if start < end {
                folds.push(&events[start..end]);
            }
        }
        if folds.is_empty() {
            return Err(EvaluationError::InvalidInput(
                "purge_bars removed all walk-forward data".into(),
            ));
        }
        let metrics = run_paths(
            folds,
            &self.strategy_id,
            &sim_config,
            self.periods_per_year,
            strategy_factory,
        )?;
        Ok(PlaybackReport::new(
            self.method,
            metrics,
            0,
            0,
            self.n_splits,
            self.max_abs_correlation,
        ))
    }

    fn run_cpcv<S, F>(
        &self,
        events: &[MarketEvent],
        sim_config: SimConfig,
        strategy_factory: &F,
    ) -> Result<PlaybackReport, EvaluationError>
    where
        S: FnMut(&mut PlaybackBackend, &MarketEvent) -> Result<(), BackendError> + Send,
        F: Fn() -> S + Send + Sync,
    {
        validate_events(events)?;
        if events.len() < self.n_splits * 4 {
            return Err(EvaluationError::InvalidInput(
                "CPCV needs at least four events per split".into(),
            ));
        }
        let chunks = split_ranges(events.len(), self.n_splits);
        let chunk_returns: Vec<Vec<f64>> = chunks
            .iter()
            .map(|&(start, end)| price_returns(&events[start..end]))
            .collect();
        let compatibility = correlation_mask(&chunk_returns, self.max_abs_correlation);
        let mut combinations = Vec::with_capacity(self.max_paths);
        let mut current = Vec::with_capacity(self.n_test_splits);
        collect_combinations(
            self.n_splits,
            self.n_test_splits,
            0,
            &mut current,
            &compatibility,
            self.max_paths,
            &mut combinations,
        );
        if combinations.is_empty() {
            return Err(EvaluationError::NoEligiblePaths(
                "no CPCV combination met the precomputed correlation mask".into(),
            ));
        }

        let metrics: Result<Vec<PathMetrics>, EvaluationError> = combinations
            .par_iter()
            .map(|combination| {
                let mut path = Vec::new();
                for (position, &chunk) in combination.iter().enumerate() {
                    let (start, end) = chunks[chunk];
                    let start = if position == 0 {
                        start
                    } else {
                        start.saturating_add(self.purge_bars)
                    };
                    let end = if position + 1 == combination.len() {
                        end
                    } else {
                        end.saturating_sub(self.purge_bars)
                    };
                    if start < end {
                        path.extend_from_slice(&events[start..end]);
                    }
                }
                if path.is_empty() {
                    return Err(EvaluationError::InvalidInput(
                        "purge_bars removed all events from a CPCV path".into(),
                    ));
                }
                run_one_path(
                    &path,
                    &self.strategy_id,
                    &sim_config,
                    self.periods_per_year,
                    strategy_factory,
                )
            })
            .collect();
        let metrics = metrics?;
        Ok(PlaybackReport::new(
            self.method,
            metrics,
            combinations.len(),
            0,
            combinations.len(),
            self.max_abs_correlation,
        ))
    }
}

/// Price data supplied to the selected evaluation method.
pub enum PlaybackInput {
    /// Timestamp-ordered market events, for CPCV and walk-forward.
    Events(Vec<MarketEvent>),
    /// Independently generated scenarios, typically sourced from the
    /// synthetic-data pipeline, for parametric Monte Carlo.
    Scenarios(Vec<Vec<MarketEvent>>),
}

#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    #[error("invalid playback settings: {0}")]
    InvalidSettings(String),
    #[error("invalid playback input: {0}")]
    InvalidInput(String),
    #[error("input does not match the selected playback method: {0}")]
    InputMismatch(String),
    #[error("no eligible scenarios: {0}")]
    NoEligiblePaths(String),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MetricSummary {
    pub mean: f64,
    /// Mean of the worst five percent of scenario outcomes.
    pub cvar05: f64,
    /// Mean of the best five percent of scenario outcomes.
    pub cvar95: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathMetrics {
    pub cumulative_return: f64,
    pub expectancy_per_trade: f64,
    pub max_drawdown: f64,
    pub sortino: f64,
    pub trades: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaybackReport {
    pub method: PlaybackMethod,
    pub paths_run: usize,
    pub paths_rejected: usize,
    pub folds_or_combinations: usize,
    pub max_abs_correlation: f64,
    pub cumulative_return: MetricSummary,
    pub expectancy_per_trade: MetricSummary,
    pub max_drawdown: MetricSummary,
    pub sortino: MetricSummary,
    pub path_metrics: Vec<PathMetrics>,
}

impl PlaybackReport {
    fn new(
        method: PlaybackMethod,
        path_metrics: Vec<PathMetrics>,
        candidates: usize,
        accepted: usize,
        folds_or_combinations: usize,
        max_abs_correlation: f64,
    ) -> Self {
        let summarize =
            |select: fn(&PathMetrics) -> f64| summarize(path_metrics.iter().map(select).collect());
        Self {
            method,
            paths_run: path_metrics.len(),
            paths_rejected: candidates.saturating_sub(accepted),
            folds_or_combinations,
            max_abs_correlation,
            cumulative_return: summarize(|metric| metric.cumulative_return),
            expectancy_per_trade: summarize(|metric| metric.expectancy_per_trade),
            max_drawdown: summarize(|metric| metric.max_drawdown),
            sortino: summarize(|metric| metric.sortino),
            path_metrics,
        }
    }
}

impl fmt::Display for PlaybackReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Playback method: {:?}", self.method)?;
        writeln!(
            f,
            "Paths run: {} ({} rejected); folds/combinations: {}; max |rho|: {:.3}",
            self.paths_run,
            self.paths_rejected,
            self.folds_or_combinations,
            self.max_abs_correlation
        )?;
        write_summary(f, "Cumulative return", self.cumulative_return)?;
        write_summary(f, "Expectancy per trade", self.expectancy_per_trade)?;
        write_summary(f, "Maximum drawdown", self.max_drawdown)?;
        write_summary(f, "Sortino", self.sortino)
    }
}

fn write_summary(f: &mut fmt::Formatter<'_>, label: &str, summary: MetricSummary) -> fmt::Result {
    writeln!(
        f,
        "{label}: {:.6} (CVaR05 {:.6}, CVaR95 {:.6})",
        summary.mean, summary.cvar05, summary.cvar95
    )
}

fn run_paths<S, F>(
    paths: Vec<&[MarketEvent]>,
    strategy_id: &str,
    sim_config: &SimConfig,
    periods_per_year: f64,
    strategy_factory: &F,
) -> Result<Vec<PathMetrics>, EvaluationError>
where
    S: FnMut(&mut PlaybackBackend, &MarketEvent) -> Result<(), BackendError> + Send,
    F: Fn() -> S + Send + Sync,
{
    paths
        .par_iter()
        .map(|path| {
            run_one_path(
                path,
                strategy_id,
                sim_config,
                periods_per_year,
                strategy_factory,
            )
        })
        .collect()
}

fn run_one_path<S, F>(
    path: &[MarketEvent],
    strategy_id: &str,
    sim_config: &SimConfig,
    periods_per_year: f64,
    strategy_factory: &F,
) -> Result<PathMetrics, EvaluationError>
where
    S: FnMut(&mut PlaybackBackend, &MarketEvent) -> Result<(), BackendError> + Send,
    F: Fn() -> S + Send + Sync,
{
    let mut backend = PlaybackBackend::with_config_capital(
        sim_config.clone(),
        strategy_id,
        sim_config.starting_capital,
        1.0,
    );
    let mut strategy = strategy_factory();
    let mut equity = Vec::with_capacity(path.len());
    backend.replay_slice_unpaced(path, |backend, event| {
        strategy(backend, event)?;
        let capital = backend
            .engine()
            .capital_ref(strategy_id)
            .expect("strategy capital is allocated before playback");
        equity.push(capital.equity);
        Ok(())
    })?;
    let capital = backend
        .engine()
        .capital_ref(strategy_id)
        .expect("strategy capital is allocated before playback");
    let (max_drawdown, _) = drawdown_stats(&equity);
    let returns = returns_from_equity(&equity);
    Ok(PathMetrics {
        cumulative_return: if capital.starting > 0.0 {
            (capital.equity - capital.starting) / capital.starting
        } else {
            0.0
        },
        expectancy_per_trade: if capital.trade_count > 0 {
            capital.realized_pnl / capital.trade_count as f64
        } else {
            0.0
        },
        max_drawdown,
        sortino: sortino(&returns, 0.0, periods_per_year),
        trades: capital.trade_count,
    })
}

fn validate_events(events: &[MarketEvent]) -> Result<(), EvaluationError> {
    if events.is_empty() {
        return Err(EvaluationError::InvalidInput(
            "market event sequence must not be empty".into(),
        ));
    }
    for (index, event) in events.iter().enumerate() {
        if event.marks.is_empty()
            || event
                .marks
                .values()
                .any(|price| !price.is_finite() || *price <= 0.0)
        {
            return Err(EvaluationError::InvalidInput(format!(
                "event {index} has no marks or a non-positive/non-finite price"
            )));
        }
        if index > 0 && events[index - 1].timestamp > event.timestamp {
            return Err(EvaluationError::InvalidInput(
                "market events must be timestamp ordered".into(),
            ));
        }
    }
    Ok(())
}

fn split_ranges(length: usize, splits: usize) -> Vec<(usize, usize)> {
    (0..splits)
        .map(|split| {
            let start = split * length / splits;
            let end = (split + 1) * length / splits;
            (start, end)
        })
        .collect()
}

fn price_returns(events: &[MarketEvent]) -> Vec<f64> {
    events
        .windows(2)
        .filter_map(|window| {
            let previous = &window[0].marks;
            let current = &window[1].marks;
            let mut count = 0usize;
            let mut sum = 0.0;
            for (symbol, previous_price) in previous {
                if let Some(current_price) = current.get(symbol) {
                    sum += current_price / previous_price - 1.0;
                    count += 1;
                }
            }
            (count > 0).then_some(sum / count as f64)
        })
        .collect()
}

fn correlation_mask(returns: &[Vec<f64>], threshold: f64) -> Vec<Vec<bool>> {
    let mut mask = vec![vec![false; returns.len()]; returns.len()];
    for left in 0..returns.len() {
        mask[left][left] = true;
        for right in (left + 1)..returns.len() {
            let valid = pearson(&returns[left], &returns[right])
                .is_some_and(|correlation| correlation.abs() <= threshold);
            mask[left][right] = valid;
            mask[right][left] = valid;
        }
    }
    mask
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    let length = left.len().min(right.len());
    if length < 3 {
        return None;
    }
    let n = length as f64;
    let mean_left = left[..length].iter().sum::<f64>() / n;
    let mean_right = right[..length].iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance_left = 0.0;
    let mut variance_right = 0.0;
    for index in 0..length {
        let delta_left = left[index] - mean_left;
        let delta_right = right[index] - mean_right;
        covariance += delta_left * delta_right;
        variance_left += delta_left * delta_left;
        variance_right += delta_right * delta_right;
    }
    let denominator = (variance_left * variance_right).sqrt();
    (denominator > 0.0).then_some(covariance / denominator)
}

fn select_uncorrelated_paths(
    paths: &[Vec<MarketEvent>],
    threshold: f64,
    max_paths: usize,
) -> Result<Vec<usize>, EvaluationError> {
    let mut selected: Vec<(usize, Vec<f64>)> = Vec::with_capacity(max_paths.min(paths.len()));
    for (index, path) in paths.iter().enumerate() {
        if selected.len() == max_paths {
            break;
        }
        let returns = price_returns(path);
        if returns.len() < 3 {
            continue;
        }
        let compatible = selected.iter().all(|(_, prior)| {
            pearson(&returns, prior).is_some_and(|value| value.abs() <= threshold)
        });
        if compatible {
            selected.push((index, returns));
        }
    }
    if selected.is_empty() {
        return Err(EvaluationError::NoEligiblePaths(
            "no Monte Carlo scenarios had enough valid returns and met the correlation threshold"
                .into(),
        ));
    }
    Ok(selected.into_iter().map(|(index, _)| index).collect())
}

fn collect_combinations(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    compatibility: &[Vec<bool>],
    maximum: usize,
    output: &mut Vec<Vec<usize>>,
) {
    if output.len() >= maximum {
        return;
    }
    if current.len() == k {
        output.push(current.clone());
        return;
    }
    let remaining = k - current.len();
    for candidate in start..=n.saturating_sub(remaining) {
        if current.iter().all(|&prior| compatibility[prior][candidate]) {
            current.push(candidate);
            collect_combinations(n, k, candidate + 1, current, compatibility, maximum, output);
            current.pop();
            if output.len() >= maximum {
                return;
            }
        }
    }
}

fn summarize(mut values: Vec<f64>) -> MetricSummary {
    if values.is_empty() {
        return MetricSummary::default();
    }
    values.sort_by(f64::total_cmp);
    let tail_count = values.len().div_ceil(20).max(1);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let cvar05 = values[..tail_count].iter().sum::<f64>() / tail_count as f64;
    let cvar95 = values[values.len() - tail_count..].iter().sum::<f64>() / tail_count as f64;
    MetricSummary {
        mean,
        cvar05,
        cvar95,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ActionContext, ExecutionBackend};
    use crate::types::OrderType;
    use chrono::{TimeZone, Utc};
    use std::collections::BTreeMap;

    fn events(prices: &[f64]) -> Vec<MarketEvent> {
        let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        prices
            .iter()
            .enumerate()
            .map(|(index, price)| {
                let mut marks = BTreeMap::new();
                marks.insert("BTCUSDT".to_string(), *price);
                MarketEvent {
                    timestamp: start + chrono::Duration::seconds(index as i64),
                    marks,
                }
            })
            .collect()
    }

    #[test]
    fn default_method_is_purged_walk_forward() {
        assert_eq!(
            PlaybackSettings::default().method,
            PlaybackMethod::PurgedWalkForward
        );
    }

    #[test]
    fn correlation_mask_precomputes_compatible_chunks() {
        let mask = correlation_mask(
            &[
                vec![0.01, -0.02, 0.03, -0.04],
                vec![-0.01, 0.02, -0.03, 0.04],
                vec![0.01, 0.02, 0.03, 0.04],
            ],
            0.5,
        );
        assert!(!mask[0][1]);
        assert!(mask[0][2]);
        assert!(mask[1][2]);
    }

    #[test]
    fn walk_forward_runs_each_chronological_fold_and_reports_metrics() {
        let settings = PlaybackSettings {
            n_splits: 2,
            ..PlaybackSettings::default()
        };
        let report = settings
            .run(
                PlaybackInput::Events(events(&[100.0, 101.0, 100.0, 102.0])),
                SimConfig::default(),
                || |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!(report.paths_run, 2);
        assert_eq!(report.folds_or_combinations, 2);
        assert!(report.cumulative_return.mean.is_finite());
    }

    #[test]
    fn callback_trades_are_included_in_expectancy_and_report() {
        let settings = PlaybackSettings {
            n_splits: 2,
            strategy_id: "test-strategy".into(),
            ..PlaybackSettings::default()
        };
        let report = settings
            .run(
                PlaybackInput::Events(events(&[100.0, 102.0, 100.0, 103.0])),
                SimConfig::default(),
                || {
                    let mut observations = 0;
                    move |backend: &mut PlaybackBackend, event: &MarketEvent| {
                        let symbol = event.marks.keys().next().unwrap();
                        let context = ActionContext::new("test-strategy", "synthetic", symbol, 1.0);
                        if observations == 0 {
                            backend.long(&context, OrderType::Market, None)?;
                        } else if observations == 1 {
                            backend.exit(&context)?;
                        }
                        observations += 1;
                        Ok(())
                    }
                },
            )
            .unwrap();
        assert_eq!(report.paths_run, 2);
        assert!(report.path_metrics.iter().all(|path| path.trades == 1));
        assert!(report.expectancy_per_trade.mean > 0.0);
        assert!(report.to_string().contains("CVaR05"));
    }

    #[test]
    fn monte_carlo_filters_correlated_paths_before_running() {
        let settings = PlaybackSettings {
            method: PlaybackMethod::ParametricMonteCarlo,
            n_splits: 2,
            max_paths: 10,
            ..PlaybackSettings::default()
        };
        let rising = events(&[100.0, 101.0, 103.0, 106.0, 110.0]);
        let falling = events(&[100.0, 99.0, 97.0, 94.0, 90.0]);
        let independent = events(&[100.0, 101.0, 99.0, 102.0, 100.0]);
        let report = settings
            .run(
                PlaybackInput::Scenarios(vec![rising.clone(), falling, independent]),
                SimConfig::default(),
                || |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!(report.paths_run, 2);
        assert_eq!(report.paths_rejected, 1);
    }

    #[test]
    fn cpcv_generates_only_compatible_combinations() {
        let settings = PlaybackSettings {
            method: PlaybackMethod::Cpcv,
            n_splits: 3,
            n_test_splits: 2,
            max_abs_correlation: 0.51,
            max_paths: 10,
            ..PlaybackSettings::default()
        };
        let report = settings
            .run(
                PlaybackInput::Events(events(&[
                    100.0, 101.0, 99.99, 99.99, 100.0, 100.0, 101.0, 99.99, 100.0, 101.0, 101.0,
                    99.99,
                ])),
                SimConfig::default(),
                || |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!(report.paths_run, report.folds_or_combinations);
        assert!(report.paths_run > 0);
        assert!(report.paths_run <= 3);
    }

    #[test]
    fn report_uses_tail_means_for_cvar() {
        let values = summarize((0..100).map(|value| value as f64).collect());
        assert_eq!(values.mean, 49.5);
        assert_eq!(values.cvar05, 2.0);
        assert_eq!(values.cvar95, 97.0);
    }
}
