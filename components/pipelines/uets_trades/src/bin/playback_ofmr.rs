use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;
use std::process::ExitCode;

use arrow::array::{Array, Float64Array, Int32Array, TimestampMicrosecondArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use uets::core::feature::FeatureRow;
use uets::core::strategy::{Decision, MarketState, Position};
use uets::live_strategy::ofmr::{
    features::OfmrFeatures,
    settings::Settings,
    strategy::{OfmrStrategy, barrier_fractions_from_features},
};
use uets::live_strategy::{ComputeFeatures, StateBuilder};
use uets_trades::playback::crypto_l2::{EventType, load_merged_parquet, prepare_playback};

const STATE_SPACE_FEATURES: [&str; 15] = [
    "spread_bps_z",
    "ma_slope_5",
    "rsi_x_ma",
    "ma_accel",
    "price_roc_3",
    "rsi_2",
    "sma_14",
    "ofi_z",
    "cvd_delta_z",
    "cvd_slope_z",
    "absorption_z",
    "tick_autocorr",
    "inst_ratio",
    "delta_divergence_z",
    "depth_imbalance_z",
];

struct StateSpaceRow {
    timestamp_ms: i64,
    session_id: i32,
    price: f64,
    volume: f64,
    bar_vol_std: f64,
    features: [f64; 15],
}

struct FoldSpec {
    start_ms: i64,
    end_ms: i64,
    model_path: PathBuf,
    expected_test_rows: Option<usize>,
    expected_train_rows: Option<usize>,
}

struct FoldModel {
    start_ms: i64,
    end_ms: i64,
    strategy: OfmrStrategy,
    expected_test_rows: Option<usize>,
    expected_train_rows: Option<usize>,
}

#[derive(Clone, Copy)]
struct ActiveBarriers {
    stop: f64,
    target: f64,
    risk_fraction: f64,
    long: bool,
}

impl ActiveBarriers {
    fn new(entry: f64, position: Position, stop_fraction: f64, target_fraction: f64) -> Self {
        let long = position == Position::Long;
        Self {
            stop: if long {
                entry * (1.0 - stop_fraction)
            } else {
                entry * (1.0 + stop_fraction)
            },
            target: if long {
                entry * (1.0 + target_fraction)
            } else {
                entry * (1.0 - target_fraction)
            },
            risk_fraction: stop_fraction,
            long,
        }
    }

    fn hit(self, price: f64) -> bool {
        if self.long {
            price <= self.stop || price >= self.target
        } else {
            price >= self.stop || price <= self.target
        }
    }
}

fn fold_at(models: &[FoldModel], timestamp_ms: i64) -> Option<usize> {
    models
        .iter()
        .position(|model| model.start_ms <= timestamp_ms && timestamp_ms <= model.end_ms)
}

fn read_state_space(path: &PathBuf) -> Result<Vec<StateSpaceRow>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let indices = |name: &str| {
        schema.index_of(name).map_err(|error| {
            format!("state-space Parquet is missing required column {name}: {error}")
        })
    };
    let datetime_index = indices("datetime")?;
    let session_index = indices("session_id")?;
    let price_index = indices("execution_price")?;
    let volume_index = indices("volume")?;
    let volatility_index = indices("bar_vol_std")?;
    let feature_indices = STATE_SPACE_FEATURES
        .iter()
        .map(|name| indices(name))
        .collect::<Result<Vec<_>, _>>()?;
    let mut reader = builder.with_batch_size(65_536).build()?;
    let mut rows = Vec::new();
    for batch in &mut reader {
        let batch = batch?;
        let timestamps = batch
            .column(datetime_index)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or("state-space datetime must be timestamp[us]")?;
        let sessions = batch
            .column(session_index)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("state-space session_id must be int32")?;
        let prices = batch
            .column(price_index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("state-space execution_price must be float64")?;
        let volumes = batch
            .column(volume_index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("state-space volume must be float64")?;
        let volatility = batch
            .column(volatility_index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("state-space bar_vol_std must be float64")?;
        let features = feature_indices
            .iter()
            .map(|index| {
                batch
                    .column(*index)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| format!("state-space feature {} must be float64", *index))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for row_index in 0..batch.num_rows() {
            if timestamps.is_null(row_index)
                || sessions.is_null(row_index)
                || prices.is_null(row_index)
                || volumes.is_null(row_index)
                || volatility.is_null(row_index)
                || features.iter().any(|column| column.is_null(row_index))
            {
                return Err(format!(
                    "state-space Parquet contains a null required value at batch row {row_index}"
                )
                .into());
            }
            let mut row_features = [0.0; 15];
            for (index, column) in features.iter().enumerate() {
                row_features[index] = column.value(row_index);
            }
            rows.push(StateSpaceRow {
                timestamp_ms: timestamps.value(row_index).div_euclid(1_000),
                session_id: sessions.value(row_index),
                price: prices.value(row_index),
                volume: volumes.value(row_index),
                bar_vol_std: volatility.value(row_index),
                features: row_features,
            });
        }
    }
    if rows.is_empty() {
        return Err(format!("state-space Parquet contains no rows: {}", path.display()).into());
    }
    if rows
        .windows(2)
        .any(|pair| pair[0].timestamp_ms > pair[1].timestamp_ms)
    {
        return Err("state-space Parquet rows must be ordered by datetime".into());
    }
    Ok(rows)
}

fn research_candidates(rows: &[StateSpaceRow], settings: &Settings) -> Vec<(i64, usize)> {
    let interval_ms = settings.feature.htf_timeframe_secs as i64 * 1_000;
    let mut candidates = Vec::new();
    let mut active_bucket: Option<i64> = None;
    let mut last_volume_row: Option<usize> = None;

    for (index, row) in rows.iter().enumerate() {
        if row.volume <= 0.0 {
            continue;
        }
        let bucket = row.timestamp_ms.div_euclid(interval_ms) * interval_ms;
        if active_bucket.is_some_and(|previous| previous != bucket) {
            if let (Some(previous_bucket), Some(previous_index)) =
                (active_bucket, last_volume_row.take())
            {
                candidates.push((previous_bucket, previous_index));
            }
        }
        active_bucket = Some(bucket);
        last_volume_row = Some(index);
    }
    if let (Some(bucket), Some(index)) = (active_bucket, last_volume_row) {
        candidates.push((bucket, index));
    }

    let mut session_ends = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        session_ends.insert(row.session_id, index + 1);
    }
    let horizon = settings.trade.label_horizon_seconds as usize;
    candidates
        .into_iter()
        .filter(|(_, index)| {
            let row = &rows[*index];
            let Some(session_end) = session_ends.get(&row.session_id) else {
                return false;
            };
            index.saturating_add(horizon) < *session_end
                && row.features[5] < settings.trade.rsi_filter_threshold
        })
        .collect()
}

#[derive(Clone, Copy)]
struct ExecutedTrade {
    entry_index: usize,
    exit_index: usize,
    side: f64,
    risk_fraction: f64,
}

#[derive(Clone, Copy)]
struct PathMetrics {
    cumulative_return: f64,
    expectancy_per_trade: f64,
    max_drawdown: f64,
    sortino: f64,
    trades: usize,
}

fn evaluate_state_space_fold(
    rows: &[StateSpaceRow],
    candidates: &[(i64, usize)],
    session_ends: &BTreeMap<i32, usize>,
    fold: &FoldModel,
    fold_index: usize,
    settings: &Settings,
    starting_capital: f64,
    fee_bps: f64,
    periods_per_year: f64,
) -> Result<PathMetrics, Box<dyn std::error::Error>> {
    if !starting_capital.is_finite() || starting_capital <= 0.0 {
        return Err("starting capital must be positive and finite".into());
    }
    let fold_candidates: Vec<_> = candidates
        .iter()
        .filter(|(timestamp_ms, _)| fold.start_ms <= *timestamp_ms && *timestamp_ms <= fold.end_ms)
        .copied()
        .collect();
    if let Some(expected) = fold.expected_test_rows
        && fold_candidates.len() != expected
    {
        return Err(format!(
            "state-space fold {} contains {} eligible test rows, expected {} from the fold catalog; data/catalog mismatch",
            fold_index + 1,
            fold_candidates.len(),
            expected
        )
        .into());
    }
    if fold_candidates.is_empty() {
        return Err(format!(
            "state-space data has no eligible research rows for fold {}",
            fold_index + 1
        )
        .into());
    }
    if fold_candidates[0].0 != fold.start_ms
        || fold_candidates.last().map(|candidate| candidate.0) != Some(fold.end_ms)
    {
        return Err(format!(
            "state-space fold {} timestamps do not match the catalog window {}..{}",
            fold_index + 1,
            fold.start_ms,
            fold.end_ms
        )
        .into());
    }

    let mut trades = Vec::new();
    let mut next_available_ms = i64::MIN;
    for (candidate_time_ms, entry_index) in fold_candidates {
        if candidate_time_ms < next_available_ms {
            continue;
        }
        let row = &rows[entry_index];
        if row.features.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "state-space candidate at row {entry_index} contains a non-finite model feature"
            )
            .into());
        }
        let stop_fraction = if row.bar_vol_std.is_finite() && row.bar_vol_std > 0.0 {
            row.bar_vol_std
        } else {
            0.001
        }
        .max(0.001)
            * 0.3;
        if !(settings.trade.min_barrier_pct..=settings.trade.max_barrier_pct)
            .contains(&stop_fraction)
        {
            continue;
        }
        let mut feature_row = FeatureRow::new();
        for (index, value) in row.features.iter().enumerate() {
            feature_row.insert(format!("Column_{index}"), *value);
        }
        let market = MarketState {
            features: feature_row,
            position: Position::Flat,
            sequence: candidate_time_ms.max(0) as u64,
        };
        let side = match fold.strategy.decide(&market) {
            Decision::EnterLong => 1.0,
            Decision::EnterShort => -1.0,
            _ => continue,
        };
        if !row.price.is_finite() || row.price <= 0.0 {
            return Err(format!("invalid state-space execution_price at row {entry_index}").into());
        }
        let target_fraction = stop_fraction * 10.0;
        let target = row.price * (1.0 + side * target_fraction);
        let stop = row.price * (1.0 - side * stop_fraction);
        let session_end = *session_ends
            .get(&row.session_id)
            .ok_or("state-space candidate has no session end")?;
        let exit_limit = entry_index
            .saturating_add(settings.trade.label_horizon_seconds as usize)
            .min(session_end.saturating_sub(1));
        if exit_limit <= entry_index {
            continue;
        }
        let mut exit_index = exit_limit;
        for (future_index, future) in rows
            .iter()
            .enumerate()
            .take(exit_limit + 1)
            .skip(entry_index + 1)
        {
            let hit_target = if side > 0.0 {
                future.price >= target
            } else {
                future.price <= target
            };
            let hit_stop = if side > 0.0 {
                future.price <= stop
            } else {
                future.price >= stop
            };
            if hit_target || hit_stop {
                exit_index = future_index;
                break;
            }
        }
        trades.push(ExecutedTrade {
            entry_index,
            exit_index,
            side,
            risk_fraction: stop_fraction,
        });
        let duration_seconds = rows[exit_index]
            .timestamp_ms
            .saturating_sub(row.timestamp_ms)
            .max(0);
        next_available_ms = candidate_time_ms
            .saturating_add(duration_seconds)
            .saturating_add(1_000);
    }

    let first_index = rows.partition_point(|row| row.timestamp_ms < fold.start_ms);
    let last_candidate_index = candidates
        .iter()
        .filter(|(timestamp_ms, _)| fold.start_ms <= *timestamp_ms && *timestamp_ms <= fold.end_ms)
        .map(|(_, index)| *index)
        .max()
        .unwrap_or(first_index);
    let last_index = trades
        .iter()
        .map(|trade| trade.exit_index)
        .max()
        .unwrap_or(last_candidate_index)
        .max(last_candidate_index);
    if last_index <= first_index {
        return Err(format!(
            "fold {} has fewer than two state-space prices",
            fold_index + 1
        )
        .into());
    }

    let fee_rate = fee_bps / 10_000.0;
    let mut equity = starting_capital;
    let mut peak = starting_capital;
    let mut max_drawdown = 0.0_f64;
    let mut daily_returns: Vec<f64> = Vec::new();
    let mut day_start_equity = starting_capital;
    let mut current_day = rows[first_index].timestamp_ms.div_euclid(86_400_000);
    let mut trade_returns = Vec::with_capacity(trades.len());
    for trade in &trades {
        let entry_price = rows[trade.entry_index].price;
        let exit_price = rows[trade.exit_index].price;
        let price_return = checked_relative_return(exit_price, entry_price, "trade price")?;
        if !trade.risk_fraction.is_finite() || trade.risk_fraction <= 0.0 {
            return Err("trade risk fraction must be positive and finite".into());
        }
        let net_return = trade.side * price_return - 2.0 * fee_rate;
        trade_returns.push(net_return / trade.risk_fraction);
    }
    let mut active_side = 0.0;
    let mut trade_cursor = 0;
    let mut previous_price = rows[first_index].price;
    if !previous_price.is_finite() || previous_price <= 0.0 {
        return Err(format!("invalid state-space price at row {first_index}").into());
    }
    for index in first_index + 1..=last_index {
        let day = rows[index].timestamp_ms.div_euclid(86_400_000);
        if day != current_day {
            daily_returns.push(checked_relative_return(equity, day_start_equity, "daily equity")?);
            day_start_equity = equity;
            current_day = day;
        }
        let mut current_side = active_side;
        if trade_cursor < trades.len() && trades[trade_cursor].entry_index == index {
            current_side = trades[trade_cursor].side;
        }
        if trade_cursor < trades.len() && trades[trade_cursor].exit_index == index {
            current_side = 0.0;
        }
        let price = rows[index].price;
        if !price.is_finite() || price <= 0.0 {
            return Err(format!("invalid state-space price at row {index}").into());
        }
        let asset_return = if active_side != 0.0 {
            active_side * checked_relative_return(price, previous_price, "market price")?
        } else {
            0.0
        };
        let turnover = (current_side - active_side).abs();
        let portfolio_return = (asset_return - turnover * fee_rate).max(-0.999_999);
        equity *= 1.0 + portfolio_return;
        peak = peak.max(equity);
        max_drawdown = max_drawdown.max((peak - equity) / peak);
        previous_price = price;
        active_side = current_side;
        if trade_cursor < trades.len() && trades[trade_cursor].exit_index == index {
            trade_cursor += 1;
        }
    }
    if active_side != 0.0 {
        equity *= 1.0 - fee_rate;
    }
    daily_returns.push(checked_relative_return(equity, day_start_equity, "daily equity")?);
    let downside_square_sum: f64 = daily_returns
        .iter()
        .map(|daily_return| daily_return.min(0.0).powi(2))
        .sum();
    let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len().max(1) as f64;
    let downside_deviation = (downside_square_sum / daily_returns.len().max(1) as f64).sqrt();
    let sortino = if downside_deviation > 0.0 {
        mean_return / downside_deviation * periods_per_year.sqrt()
    } else if mean_return > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    let expectancy = if trade_returns.is_empty() {
        0.0
    } else {
        trade_returns.iter().sum::<f64>() / trade_returns.len() as f64
    };
    Ok(PathMetrics {
        cumulative_return: checked_relative_return(equity, starting_capital, "starting capital")?,
        expectancy_per_trade: expectancy,
        max_drawdown,
        sortino,
        trades: trade_returns.len(),
    })
}

fn checked_relative_return(
    value: f64,
    base: f64,
    label: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    if !value.is_finite() || value <= 0.0 || !base.is_finite() || base <= 0.0 {
        return Err(format!("{label} numerator and denominator must be positive and finite").into());
    }
    let result = value / base - 1.0;
    if !result.is_finite() {
        return Err(format!("{label} return is not finite").into());
    }
    Ok(result)
}

fn run_state_space(
    path: &PathBuf,
    models: &[FoldModel],
    settings: &Settings,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = read_state_space(path)?;
    let candidates = research_candidates(&rows, settings);
    if models
        .iter()
        .any(|model| model.expected_train_rows.is_some())
    {
        let purge_window_seconds = settings
            .trade
            .label_horizon_seconds
            .checked_add(settings.trade.embargo_seconds)
            .and_then(|seconds| i64::try_from(seconds).ok())
            .and_then(|seconds| seconds.checked_mul(1_000))
            .ok_or("walk-forward purge window overflows milliseconds")?;
        let test_rows = models
            .first()
            .and_then(|model| model.expected_test_rows)
            .ok_or("fold catalog must include test_rows to validate purged splits")?;
        if models
            .iter()
            .any(|model| model.expected_test_rows != Some(test_rows))
        {
            return Err("state-space purged WF requires equal test_rows per fold".into());
        }
        let initial_train = (candidates.len() as f64 * 0.35) as usize;
        let available_test = candidates
            .len()
            .saturating_sub(initial_train)
            .saturating_sub(test_rows);
        if available_test == 0 {
            return Err("state-space data is too narrow for catalogued walk-forward folds".into());
        }
        let step = available_test / models.len();
        for (fold_index, model) in models.iter().enumerate() {
            let start = initial_train + fold_index * step;
            let end = start + test_rows;
            let test = candidates
                .get(start..end)
                .ok_or("state-space data does not contain the catalogued test split")?;
            if test.first().map(|candidate| candidate.0) != Some(model.start_ms)
                || test.last().map(|candidate| candidate.0) != Some(model.end_ms)
            {
                return Err(format!(
                    "state-space fold {} does not match its chronological split window; fold models may be stale",
                    fold_index + 1
                )
                .into());
            }
            let test_start = test[0].0;
            let purged_train_rows = candidates[..start]
                .iter()
                .filter(|(timestamp_ms, _)| {
                    timestamp_ms.saturating_add(purge_window_seconds) < test_start
                })
                .count();
            if model.expected_train_rows != Some(purged_train_rows) {
                return Err(format!(
                    "state-space fold {} has {} purged training rows; catalog expects {:?}; models and dataset do not match",
                    fold_index + 1,
                    purged_train_rows,
                    model.expected_train_rows
                )
                .into());
            }
        }
    }
    let mut session_ends = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        session_ends.insert(row.session_id, index + 1);
    }
    let starting_capital = std::env::var("UETS_STARTING_CAPITAL")
        .unwrap_or_else(|_| "10000".into())
        .parse::<f64>()?;
    let fee_bps = std::env::var("UETS_FEE_BPS")
        .unwrap_or_else(|_| "4".into())
        .parse::<f64>()?;
    let periods_per_year = std::env::var("UETS_PERIODS_PER_YEAR")
        .unwrap_or_else(|_| "252".into())
        .parse::<f64>()?;
    if !starting_capital.is_finite()
        || starting_capital <= 0.0
        || !fee_bps.is_finite()
        || fee_bps < 0.0
        || !periods_per_year.is_finite()
        || periods_per_year <= 0.0
    {
        return Err("invalid state-space playback capital, fees, or annualization periods".into());
    }
    println!("fold,cumulative_return,expectancy_per_trade,max_drawdown,sortino,trades");
    for (index, model) in models.iter().enumerate() {
        let metrics = evaluate_state_space_fold(
            &rows,
            &candidates,
            &session_ends,
            model,
            index,
            settings,
            starting_capital,
            fee_bps,
            periods_per_year,
        )?;
        println!(
            "{},{},{},{},{},{}",
            index + 1,
            metrics.cumulative_return,
            metrics.expectancy_per_trade,
            metrics.max_drawdown,
            metrics.sortino,
            metrics.trades
        );
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut current_symbol: Option<String> = None;
    let mut files: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut state_space_path = None;
    let mut fold_specs: Vec<FoldSpec> = Vec::new();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--state-space" => {
                state_space_path = Some(PathBuf::from(
                    args.next().ok_or("--state-space requires a Parquet path")?,
                ));
            }
            "--symbol" => {
                let symbol = args.next().ok_or("--symbol requires a symbol")?;
                current_symbol = Some(symbol.to_ascii_uppercase());
            }
            "--file" => {
                let path = args.next().ok_or("--file requires a Parquet path")?;
                let symbol = current_symbol
                    .as_ref()
                    .ok_or("--symbol must precede each --file group")?;
                files
                    .entry(symbol.clone())
                    .or_default()
                    .push(PathBuf::from(path));
            }
            "--fold-model" => {
                let start_ms = args
                    .next()
                    .ok_or("--fold-model requires START_MS END_MS MODEL_PATH")?
                    .parse::<i64>()?;
                let end_ms = args
                    .next()
                    .ok_or("--fold-model requires START_MS END_MS MODEL_PATH")?
                    .parse::<i64>()?;
                let model_path = args
                    .next()
                    .ok_or("--fold-model requires START_MS END_MS MODEL_PATH")?;
                if start_ms > end_ms {
                    return Err(
                        format!("invalid fold model interval: {start_ms} > {end_ms}").into(),
                    );
                }
                fold_specs.push(FoldSpec {
                    start_ms,
                    end_ms,
                    model_path: PathBuf::from(model_path),
                    expected_test_rows: None,
                    expected_train_rows: None,
                });
            }
            "--fold-test-rows" => {
                let expected = args.next().ok_or("--fold-test-rows requires a count")?;
                let last_fold = fold_specs
                    .last_mut()
                    .ok_or("--fold-test-rows must follow --fold-model")?;
                last_fold.expected_test_rows = Some(expected.parse::<usize>()?);
            }
            "--fold-train-rows" => {
                let expected = args.next().ok_or("--fold-train-rows requires a count")?;
                let last_fold = fold_specs
                    .last_mut()
                    .ok_or("--fold-train-rows must follow --fold-model")?;
                last_fold.expected_train_rows = Some(expected.parse::<usize>()?);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if fold_specs.is_empty() {
        return Err("OFMR playback requires purged walk-forward fold models; refusing to use a single all-data model".into());
    }
    if state_space_path.is_none() && files.is_empty() {
        return Err("usage: playback_ofmr (--state-space PATH | --symbol SYMBOL --file PATH [--file PATH ...]) --fold-model START_MS END_MS MODEL_PATH [...]".into());
    }
    if state_space_path.is_some() && !files.is_empty() {
        return Err(
            "choose either --state-space or CryptoL2 --symbol/--file input, not both".into(),
        );
    }
    fold_specs.sort_by_key(|fold| fold.start_ms);
    for pair in fold_specs.windows(2) {
        if pair[1].start_ms <= pair[0].end_ms {
            return Err(format!(
                "walk-forward model windows overlap: {}..{} and {}..{}",
                pair[0].start_ms, pair[0].end_ms, pair[1].start_ms, pair[1].end_ms
            )
            .into());
        }
    }

    let settings = Settings::try_load().map_err(|error| format!("load OFMR settings: {error}"))?;
    let mut models = Vec::with_capacity(fold_specs.len());
    for fold in fold_specs {
        let mut fold_settings = settings.clone();
        fold_settings.model.path = fold.model_path.to_string_lossy().into_owned();
        let strategy = OfmrStrategy::try_with_settings(fold_settings).map_err(|error| {
            format!(
                "load purged fold model {}: {error}",
                fold.model_path.display()
            )
        })?;
        models.push(FoldModel {
            start_ms: fold.start_ms,
            end_ms: fold.end_ms,
            strategy,
            expected_test_rows: fold.expected_test_rows,
            expected_train_rows: fold.expected_train_rows,
        });
    }
    if let Some(path) = state_space_path {
        return run_state_space(&path, &models, &settings);
    }
    println!("symbol,fold,timestamp_ms,price,position,risk_fraction");

    for (symbol, paths) in files {
        let mut rows = Vec::new();
        for path in paths {
            rows.extend(load_merged_parquet(&path, &symbol)?);
        }
        let playback = prepare_playback(rows, &symbol);
        let mut features = OfmrFeatures::with_settings(settings.clone());
        let mut position = Position::Flat;
        let mut barriers: Option<ActiveBarriers> = None;
        let mut active_fold: Option<usize> = None;
        let mut entry_timestamp_ms: Option<i64> = None;
        let mut last_bid_price = None;
        let mut last_bid_amount = None;
        let mut last_ask_price = None;
        let mut last_ask_amount = None;
        let mut current_second = None;
        let mut bar_features: Option<FeatureRow> = None;
        let mut bar_mid = 0.0;
        let mut bar_exit_price: Option<f64> = None;

        for row in &playback.rows {
            if row.event_type == Some(EventType::Lob) {
                if row.best_bid_price.is_some() {
                    last_bid_price = row.best_bid_price;
                }
                if row.best_bid_qty.is_some() {
                    last_bid_amount = row.best_bid_qty;
                }
                if row.best_ask_price.is_some() {
                    last_ask_price = row.best_ask_price;
                }
                if row.best_ask_qty.is_some() {
                    last_ask_amount = row.best_ask_qty;
                }
            }

            let second = row.timestamp_ms.div_euclid(1_000);
            if current_second.is_some_and(|previous| previous != second) {
                let bar_second = current_second
                    .ok_or("previous bar timestamp missing while closing a bar")?;
                if let (Some(computed), Some(fold_index)) = (
                    bar_features.take(),
                    current_second.and_then(|ms| fold_at(&models, ms * 1_000)),
                ) {
                    if active_fold != Some(fold_index) {
                        position = Position::Flat;
                        barriers = None;
                        entry_timestamp_ms = None;
                        active_fold = Some(fold_index);
                    }
                    let state = MarketState {
                        features: computed.clone(),
                        position,
                        sequence: second as u64,
                    };
                    let decision = if bar_exit_price.is_none() {
                        models[fold_index].strategy.decide(&state)
                    } else {
                        Decision::Hold
                    };
                    let entry_allowed = computed
                        .get("Column_5")
                        .is_some_and(|rsi| rsi < settings.trade.rsi_filter_threshold);
                    let (stop_fraction, target_fraction) =
                        barrier_fractions_from_features(&computed);
                    let barrier_allowed = (settings.trade.min_barrier_pct
                        ..=settings.trade.max_barrier_pct)
                        .contains(&stop_fraction);
                    match (position, decision) {
                        (Position::Flat, Decision::EnterLong)
                            if entry_allowed && barrier_allowed =>
                        {
                            position = Position::Long;
                            entry_timestamp_ms = Some(bar_second * 1_000);
                            barriers = Some(ActiveBarriers::new(
                                bar_mid,
                                position,
                                stop_fraction,
                                target_fraction,
                            ));
                        }
                        (Position::Flat, Decision::EnterShort)
                            if entry_allowed && barrier_allowed =>
                        {
                            position = Position::Short;
                            entry_timestamp_ms = Some(bar_second * 1_000);
                            barriers = Some(ActiveBarriers::new(
                                bar_mid,
                                position,
                                stop_fraction,
                                target_fraction,
                            ));
                        }
                        (Position::Long | Position::Short, Decision::Exit) => {
                            position = Position::Flat;
                            barriers = None;
                            entry_timestamp_ms = None;
                        }
                        _ => {}
                    }
                    let side = match position {
                        Position::Flat => 0,
                        Position::Long => 1,
                        Position::Short => -1,
                    };
                    let risk = barriers.map_or(0.0, |active| active.risk_fraction);
                    let report_price = bar_exit_price.unwrap_or(bar_mid);
                    println!(
                        "{symbol},{},{},{report_price},{side},{risk}",
                        fold_index + 1,
                        bar_second * 1_000
                    );
                } else {
                    position = Position::Flat;
                    barriers = None;
                    active_fold = None;
                    entry_timestamp_ms = None;
                }
                bar_exit_price = None;
            }
            current_second = Some(second);

            let mut raw = FeatureRow::new();
            raw.insert("event_time_ms", row.timestamp_ms as f64);
            if let Some(value) = last_bid_price {
                raw.insert("bid_price", value);
            }
            if let Some(value) = last_bid_amount {
                raw.insert("bid_amount", value);
            }
            if let Some(value) = last_ask_price {
                raw.insert("ask_price", value);
            }
            if let Some(value) = last_ask_amount {
                raw.insert("ask_amount", value);
            }
            if let Some(value) = row.trade_price {
                raw.insert("price", value);
            }
            if let Some(value) = row.trade_qty {
                raw.insert("amount", value);
            }
            if let Some(value) = row.is_buyer_maker {
                raw.insert("is_buyer_maker", if value { 1.0 } else { 0.0 });
            }
            let computed = features.compute(&raw).map_err(|error| {
                format!("OFMR feature computation failed for {symbol}: {error}")
            })?;
            let Some((bid, ask)) = last_bid_price.zip(last_ask_price) else {
                continue;
            };
            let mid = (bid + ask) / 2.0;
            if !mid.is_finite() || mid <= 0.0 {
                return Err(format!(
                    "invalid CryptoL2 mid price for {symbol} at {}",
                    row.timestamp_ms
                )
                .into());
            }
            bar_mid = mid;
            bar_features = Some(computed);

            if let Some(fold_index) = fold_at(&models, row.timestamp_ms) {
                if active_fold != Some(fold_index) {
                    position = Position::Flat;
                    barriers = None;
                    entry_timestamp_ms = None;
                    active_fold = Some(fold_index);
                }
                let price = row.trade_price.unwrap_or(mid);
                let barrier_hit = barriers.is_some_and(|active| active.hit(price));
                let horizon_hit = entry_timestamp_ms.is_some_and(|entry_ms| {
                    row.timestamp_ms.saturating_sub(entry_ms)
                        >= settings.trade.label_horizon_seconds as i64 * 1_000
                });
                if barrier_hit || horizon_hit {
                    position = Position::Flat;
                    barriers = None;
                    entry_timestamp_ms = None;
                    bar_exit_price = Some(price);
                }
            } else {
                position = Position::Flat;
                barriers = None;
                active_fold = None;
                entry_timestamp_ms = None;
                bar_exit_price = None;
            }
        }
        if let (Some(computed), Some(second), Some(fold_index)) = (
            bar_features,
            current_second,
            current_second.and_then(|ms| fold_at(&models, ms * 1_000)),
        ) {
            if active_fold != Some(fold_index) {
                position = Position::Flat;
                barriers = None;
            }
            let state = MarketState {
                features: computed.clone(),
                position,
                sequence: second as u64,
            };
            let decision = if bar_exit_price.is_none() {
                models[fold_index].strategy.decide(&state)
            } else {
                Decision::Hold
            };
            let entry_allowed = computed
                .get("Column_5")
                .is_some_and(|rsi| rsi < settings.trade.rsi_filter_threshold);
            let (stop_fraction, target_fraction) = barrier_fractions_from_features(&computed);
            let barrier_allowed = (settings.trade.min_barrier_pct..=settings.trade.max_barrier_pct)
                .contains(&stop_fraction);
            match (position, decision) {
                (Position::Flat, Decision::EnterLong) if entry_allowed && barrier_allowed => {
                    position = Position::Long;
                    barriers = Some(ActiveBarriers::new(
                        bar_mid,
                        position,
                        stop_fraction,
                        target_fraction,
                    ));
                }
                (Position::Flat, Decision::EnterShort) if entry_allowed && barrier_allowed => {
                    position = Position::Short;
                    barriers = Some(ActiveBarriers::new(
                        bar_mid,
                        position,
                        stop_fraction,
                        target_fraction,
                    ));
                }
                (Position::Long | Position::Short, Decision::Exit) => {
                    position = Position::Flat;
                    barriers = None;
                }
                _ => {}
            }
            let side = match position {
                Position::Flat => 0,
                Position::Long => 1,
                Position::Short => -1,
            };
            let risk = barriers.map_or(0.0, |active| active.risk_fraction);
            let report_price = bar_exit_price.unwrap_or(bar_mid);
            println!(
                "{symbol},{},{},{report_price},{side},{risk}",
                fold_index + 1,
                second * 1_000
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("OFMR CryptoL2 playback failed: {error}");
            ExitCode::FAILURE
        }
    }
}
