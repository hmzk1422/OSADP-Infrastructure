//! strategy1/compute_feature.rs — YOUR computation.
//!
//! Runs FIRST on every stream row: canonical FeatureRow in, computed
//! feature row out. Stateful: one instance per (strategy x symbol).
//!
//! Column_0 through Column_14 remain the existing model features.
//! bar_vol_std is emitted separately as a side feature.
//!
//! Missing semantics: emit insert_missing while history is too short,
//! never zero.

use crate::core::feature::FeatureRow;
use crate::live_strategy::ComputeFeatures;
use crate::live_strategy::ofmr::settings::{FeatureSettings, Settings};

pub const N_FEATURES: usize = 15;
const MAX_FEATURE_HISTORY_BARS: usize = 2_048;

pub fn output_column_names() -> Vec<String> {
    (0..N_FEATURES).map(|i| format!("Column_{i}")).collect()
}

#[derive(Clone, Debug)]
struct RawBar {
    close: Option<f64>,
    spread_bps: Option<f64>,
    depth_imbalance: Option<f64>,
    bid_low: Option<f64>,
    ask_high: Option<f64>,
    cvd_delta: f64,
    has_trade: bool,
    volume: f64,
    inst_buy: f64,
    inst_sell: f64,
    retail_buy: f64,
    retail_sell: f64,
    tick_rules: Vec<i32>,
}

impl RawBar {
    fn new(timeframe_secs: usize) -> Self {
        Self {
            close: None,
            spread_bps: None,
            depth_imbalance: None,
            bid_low: None,
            ask_high: None,
            cvd_delta: 0.0,
            has_trade: false,
            volume: 0.0,
            inst_buy: 0.0,
            inst_sell: 0.0,
            retail_buy: 0.0,
            retail_sell: 0.0,
            tick_rules: vec![0; timeframe_secs],
        }
    }

    fn empty(timeframe_secs: usize, previous_close: Option<f64>) -> Self {
        Self {
            close: previous_close,
            ..Self::new(timeframe_secs)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BarFeatures {
    values: [Option<f64>; N_FEATURES],
    bar_vol_std: Option<f64>,
}

impl Default for BarFeatures {
    fn default() -> Self {
        Self {
            values: [None; N_FEATURES],
            bar_vol_std: None,
        }
    }
}

struct ResearchFeatureState {
    timeframe_secs: usize,
    feature_settings: FeatureSettings,
    current_bar_index: Option<i64>,
    first_bar_index: Option<i64>,
    current_bar: Option<RawBar>,
    raw_bars: Vec<RawBar>,
    computed_bars: Vec<BarFeatures>,
    previous_trade_price: Option<f64>,
    terminal_bar_index: Option<i64>,
}

impl ResearchFeatureState {
    fn new(settings: &Settings) -> Self {
        Self {
            timeframe_secs: settings.feature.htf_timeframe_secs.max(1) as usize,
            feature_settings: settings.feature.clone(),
            current_bar_index: None,
            first_bar_index: None,
            current_bar: None,
            raw_bars: Vec::new(),
            computed_bars: Vec::new(),
            previous_trade_price: None,
            terminal_bar_index: None,
        }
    }

    fn observe(&mut self, row: &FeatureRow, event_time_ms: i64) -> Result<FeatureRow, String> {
        let timeframe_ms = self.timeframe_secs as i64 * 1_000;
        let bar_index = event_time_ms.div_euclid(timeframe_ms);
        if let Some(terminal_bar_index) = self.terminal_bar_index
            && bar_index >= terminal_bar_index
        {
            return Ok(self.output_for_bar(terminal_bar_index.saturating_sub(1)));
        }
        if self.current_bar_index != Some(bar_index) {
            self.close_current_bar(bar_index);
            self.current_bar_index = Some(bar_index);
            self.first_bar_index.get_or_insert(bar_index);
            self.current_bar = Some(RawBar::new(self.timeframe_secs));
        }

        let is_trade = row.get("price").is_some();
        let is_book = row
            .get("is_lob_event")
            .map(|value| value != 0.0)
            .unwrap_or(!is_trade);
        let bar = self
            .current_bar
            .as_mut()
            .ok_or_else(|| "OFMR current bar was not initialized".to_string())?;
        if is_book {
            if let (Some(bid), Some(ask)) = (row.get("bid_price"), row.get("ask_price")) {
                if bid > 0.0 && ask > 0.0 {
                    let mid = (bid + ask) / 2.0;
                    bar.close = Some(mid);
                    bar.bid_low = Some(bar.bid_low.map_or(bid, |value| value.min(bid)));
                    bar.ask_high = Some(bar.ask_high.map_or(ask, |value| value.max(ask)));
                    bar.spread_bps = Some((ask - bid) / mid * 10_000.0);
                    if let (Some(bid_amount), Some(ask_amount)) =
                        (row.get("bid_amount"), row.get("ask_amount"))
                    {
                        bar.depth_imbalance =
                            Some((bid_amount - ask_amount) / (bid_amount + ask_amount + 1e-9));
                    }
                }
            }
        }

        if let Some(price) = row.get("price") {
            let amount = row.get("amount").unwrap_or(0.0);
            let is_buyer_maker = row
                .get("is_buyer_maker")
                .map(|value| value != 0.0)
                .unwrap_or(false);
            let tick_rule = match self.previous_trade_price {
                Some(previous) if price > previous => 1,
                Some(previous) if price < previous => -1,
                _ => 0,
            };
            self.previous_trade_price = Some(price);
            let second_offset = event_time_ms
                .div_euclid(1_000)
                .rem_euclid(self.timeframe_secs as i64) as usize;
            if let Some(value) = bar.tick_rules.get_mut(second_offset) {
                *value = tick_rule;
            }

            let signed_amount = if is_buyer_maker { -amount } else { amount };
            bar.has_trade = true;
            bar.cvd_delta += signed_amount;
            bar.volume += amount;
            if amount >= 0.05 {
                if signed_amount >= 0.0 {
                    bar.inst_buy += amount;
                } else {
                    bar.inst_sell += amount;
                }
            } else if signed_amount >= 0.0 {
                bar.retail_buy += amount;
            } else {
                bar.retail_sell += amount;
            }
        }

        if let Some(close) = row.get("bar_close_price")
            && close.is_finite()
            && close > 0.0
        {
            bar.close = Some(close);
        }

        Ok(self.output_for_bar(bar_index))
    }

    fn close_current_bar(&mut self, next_index: i64) {
        let Some(current_index) = self.current_bar_index else {
            return;
        };
        let Some(bar) = self.current_bar.take() else {
            return;
        };
        if next_index.saturating_sub(current_index) > MAX_FEATURE_HISTORY_BARS as i64 {
            let previous_close = bar
                .close
                .or_else(|| self.raw_bars.last().and_then(|bar| bar.close));
            let first_retained = next_index.saturating_sub(1);
            self.raw_bars.clear();
            self.computed_bars.clear();
            self.first_bar_index = Some(first_retained);
            self.raw_bars
                .push(RawBar::empty(self.timeframe_secs, previous_close));
            self.computed_bars =
                calculate_research_features(&self.raw_bars, &self.feature_settings);
            return;
        }
        self.raw_bars.push(bar);
        let mut previous_close = self.raw_bars.last().and_then(|bar| bar.close);
        for _ in current_index.saturating_add(1)..next_index {
            let empty = RawBar::empty(self.timeframe_secs, previous_close);
            self.raw_bars.push(empty);
            previous_close = self.raw_bars.last().and_then(|bar| bar.close);
        }
        self.computed_bars = calculate_research_features(&self.raw_bars, &self.feature_settings);
        self.trim_history();
    }

    fn trim_history(&mut self) {
        let excess = self.raw_bars.len().saturating_sub(MAX_FEATURE_HISTORY_BARS);
        if excess == 0 {
            return;
        }
        let carried_close = self.raw_bars[..excess]
            .iter()
            .rev()
            .find_map(|bar| bar.close);
        self.raw_bars.drain(..excess);
        self.computed_bars.drain(..excess);
        if let (Some(close), Some(first)) = (carried_close, self.raw_bars.first_mut()) {
            if first.close.is_none() {
                first.close = Some(close);
            }
        }
        if let Some(first_bar_index) = &mut self.first_bar_index {
            *first_bar_index = first_bar_index.saturating_add(excess as i64);
        }
    }

    fn output_for_bar(&self, bar_index: i64) -> FeatureRow {
        let mut output = FeatureRow::new();
        for index in 0..N_FEATURES {
            output.insert_missing(format!("Column_{index}"));
        }
        output.insert_missing("bar_vol_std");
        let Some(first_bar_index) = self.first_bar_index else {
            return output;
        };
        let Some(price_index) = bar_index.checked_sub(1) else {
            return output;
        };
        let Some(orderflow_index) = bar_index.checked_sub(2) else {
            return output;
        };
        if price_index < first_bar_index || orderflow_index < first_bar_index {
            return output;
        }
        let price_offset = (price_index - first_bar_index) as usize;
        let orderflow_offset = (orderflow_index - first_bar_index) as usize;
        let (Some(price_features), Some(orderflow_features)) = (
            self.computed_bars.get(price_offset),
            self.computed_bars.get(orderflow_offset),
        ) else {
            return output;
        };
        for index in 0..N_FEATURES {
            let features = if (1..=6).contains(&index) {
                price_features
            } else {
                orderflow_features
            };
            let value = features.values[index];
            match value {
                Some(value) if value.is_finite() => output.insert(format!("Column_{index}"), value),
                _ => output.insert_missing(format!("Column_{index}")),
            }
        }
        match price_features.bar_vol_std {
            Some(value) if value.is_finite() => output.insert("bar_vol_std", value),
            _ => output.insert_missing("bar_vol_std"),
        }
        output
    }
}

fn rolling_mean(values: &[Option<f64>], index: usize, window: usize) -> Option<f64> {
    if window == 0 {
        return None;
    }
    let start = index.checked_add(1)?.checked_sub(window)?;
    let slice = &values[start..=index];
    let observations = slice.iter().copied().collect::<Option<Vec<_>>>()?;
    Some(observations.iter().sum::<f64>() / window as f64)
}

fn rolling_zscore_at(
    values: &[Option<f64>],
    index: usize,
    window: usize,
    epsilon: f64,
) -> Option<f64> {
    if window == 0 {
        return None;
    }
    let start = index.checked_add(1)?.checked_sub(window)?;
    let slice = &values[start..=index];
    let observations = slice.iter().copied().collect::<Option<Vec<_>>>()?;
    let mean = observations.iter().sum::<f64>() / window as f64;
    let variance = observations
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (window - 1).max(1) as f64;
    let std = variance.sqrt();
    let value = observations[window - 1];
    Some((value - mean) / (std + epsilon))
}

fn wilder_rsi(closes: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; closes.len()];
    if closes.len() < 2 {
        return output;
    }
    let mut differences = Vec::with_capacity(closes.len() - 1);
    for pair in closes.windows(2) {
        match (pair[0], pair[1]) {
            (Some(previous), Some(current)) => differences.push(Some(current - previous)),
            _ => differences.push(None),
        }
    }
    let Some(changes) = differences.into_iter().collect::<Option<Vec<_>>>() else {
        return output;
    };
    if changes.len() <= window {
        for index in 1..=changes.len() {
            output[index] = Some(0.0);
        }
        return output;
    }
    let gains = changes
        .iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    let losses = changes
        .iter()
        .map(|value| (-value).max(0.0))
        .collect::<Vec<_>>();
    let mut average_gain = vec![0.0; changes.len()];
    let mut average_loss = vec![0.0; changes.len()];
    average_gain[window] = gains[..window].iter().sum::<f64>() / window as f64;
    average_loss[window] = losses[..window].iter().sum::<f64>() / window as f64;
    for index in window + 1..changes.len() {
        average_gain[index] =
            (average_gain[index - 1] * (window - 1) as f64 + gains[index]) / window as f64;
        average_loss[index] =
            (average_loss[index - 1] * (window - 1) as f64 + losses[index]) / window as f64;
    }
    for index in window..changes.len() {
        let relative_strength = average_gain[index] / (average_loss[index] + 1e-10);
        output[index + 1] = Some(100.0 - 100.0 / (1.0 + relative_strength));
    }
    for index in 1..=window.min(output.len().saturating_sub(1)) {
        output[index] = Some(0.0);
    }
    output
}

fn tick_autocorrelation(ticks: &[i32]) -> f64 {
    if ticks.len() <= 5 {
        return 0.0;
    }
    let left = &ticks[..ticks.len() - 1];
    let right = &ticks[1..];
    let left_mean = left.iter().map(|value| *value as f64).sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().map(|value| *value as f64).sum::<f64>() / right.len() as f64;
    let covariance = left
        .iter()
        .zip(right)
        .map(|(left, right)| (*left as f64 - left_mean) * (*right as f64 - right_mean))
        .sum::<f64>();
    let left_variance = left
        .iter()
        .map(|value| (*value as f64 - left_mean).powi(2))
        .sum::<f64>();
    let right_variance = right
        .iter()
        .map(|value| (*value as f64 - right_mean).powi(2))
        .sum::<f64>();
    let denominator = (left_variance * right_variance).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        covariance / denominator
    }
}

fn calculate_research_features(bars: &[RawBar], settings: &FeatureSettings) -> Vec<BarFeatures> {
    let count = bars.len();
    let mut output = vec![BarFeatures::default(); count];
    let mut closes = Vec::with_capacity(count);
    let mut last_close = None;
    for bar in bars {
        if let Some(close) = bar.close {
            last_close = Some(close);
        }
        closes.push(last_close);
    }

    let rsi_2 = wilder_rsi(&closes, settings.rsi_2_window);
    let rsi_14 = wilder_rsi(&closes, settings.rsi_14_window);
    let sma_14 = (0..count)
        .map(|index| rolling_mean(&rsi_14, index, settings.sma_14_window))
        .collect::<Vec<_>>();
    let mut ma_slope = vec![Some(0.0); count];
    let mut ma_values = vec![None; count];
    for (index, value) in ma_values.iter_mut().enumerate() {
        *value = rolling_mean(&closes, index, settings.ma_slope_window);
    }
    for index in 0..count {
        if index > 0 {
            ma_slope[index] = match (ma_values[index], ma_values[index - 1]) {
                (Some(current), Some(previous)) => Some(current - previous),
                _ => Some(0.0),
            };
        }
    }
    let mut ma_accel = vec![Some(0.0); count];
    for index in 1..count {
        ma_accel[index] = Some(ma_slope[index].unwrap_or(0.0) - ma_slope[index - 1].unwrap_or(0.0));
    }
    let price_roc_3 = (0..count)
        .map(|index| {
            match (
                index.checked_sub(settings.price_roc_window_3),
                closes[index],
            ) {
                (Some(previous), Some(current)) => {
                    closes[previous].map(|old| (current / old) - 1.0)
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    let rsi_z = (0..count)
        .map(|index| {
            Some(rolling_zscore_at(&rsi_2, index, settings.rsi_z_window, 1e-9).unwrap_or(0.0))
        })
        .collect::<Vec<_>>();
    let spread_values = bars
        .iter()
        .map(|bar| Some(bar.spread_bps.unwrap_or(0.0)))
        .collect::<Vec<_>>();
    let depth_values = bars
        .iter()
        .map(|bar| Some(bar.depth_imbalance.unwrap_or(0.0)))
        .collect::<Vec<_>>();
    let cvd_values = bars
        .iter()
        .map(|bar| Some(bar.cvd_delta))
        .collect::<Vec<_>>();
    let first_trade_index = bars.iter().position(|bar| bar.has_trade);
    let mut cvd_cumulative = Vec::with_capacity(count);
    let mut cumulative = 0.0;
    for (index, value) in bars.iter().enumerate() {
        if first_trade_index.is_some_and(|first| index >= first) {
            cumulative += value.cvd_delta;
        }
        cvd_cumulative.push(cumulative);
    }
    let cvd_slope = (0..count)
        .map(|index| {
            first_trade_index
                .filter(|first| index >= first.saturating_add(5))
                .map(|_| cvd_cumulative[index] - cvd_cumulative[index - 5])
                .or(Some(0.0))
        })
        .collect::<Vec<_>>();
    let price_ranges = bars
        .iter()
        .map(|bar| bar.ask_high.zip(bar.bid_low).map(|(ask, bid)| ask - bid))
        .collect::<Vec<_>>();
    let mut last_price_range = None;
    let absorption = bars
        .iter()
        .zip(&price_ranges)
        .map(|(bar, price_range)| {
            if let Some(price_range) =
                price_range.filter(|range| range.is_finite() && *range >= 0.0)
            {
                last_price_range = Some(price_range);
            }
            let Some(price_range) = last_price_range else {
                return Some(0.0);
            };
            let scale = price_range + 1e-9;
            Some(
                if bar.cvd_delta.is_finite() && scale.is_finite() && scale > 0.0 {
                    bar.cvd_delta / scale
                } else {
                    0.0
                },
            )
        })
        .collect::<Vec<_>>();
    let delta_divergence = bars
        .iter()
        .map(|bar| Some((bar.inst_buy - bar.inst_sell) - (bar.retail_buy - bar.retail_sell)))
        .collect::<Vec<_>>();
    let inst_ratio = bars
        .iter()
        .map(|bar| Some((bar.inst_buy + bar.inst_sell) / (bar.volume + 1e-9)))
        .collect::<Vec<_>>();
    let mut close_returns = vec![None; count];
    for index in 1..count {
        close_returns[index] = match (closes[index - 1], closes[index]) {
            (Some(previous), Some(current)) if previous != 0.0 => Some(current / previous - 1.0),
            _ => None,
        };
    }

    for index in 0..count {
        let mut volatility_values = Vec::new();
        let start = index
            .saturating_add(1)
            .saturating_sub(settings.barrier_vol_bar_window);
        for value in &close_returns[start..=index] {
            if let Some(value) = value {
                volatility_values.push(*value);
            }
        }
        let bar_vol_std = if volatility_values.len() >= 2 {
            let mean = volatility_values.iter().sum::<f64>() / volatility_values.len() as f64;
            Some(
                (volatility_values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / volatility_values.len() as f64)
                    .sqrt(),
            )
        } else {
            None
        };
        output[index] = BarFeatures {
            values: [
                rolling_zscore_at(&spread_values, index, settings.deeplob_z_window, 1e-9)
                    .or(Some(0.0)),
                ma_slope[index],
                rsi_z[index]
                    .zip(ma_slope[index])
                    .map(|(rsi, slope)| rsi * slope),
                ma_accel[index],
                price_roc_3[index],
                rsi_2[index],
                sma_14[index],
                Some(0.0),
                rolling_zscore_at(&cvd_values, index, 10, 1e-9).or(Some(0.0)),
                rolling_zscore_at(&cvd_slope, index, 10, 1e-9).or(Some(0.0)),
                rolling_zscore_at(&absorption, index, 10, 1e-9).or(Some(0.0)),
                Some(tick_autocorrelation(&bars[index].tick_rules)),
                inst_ratio[index],
                rolling_zscore_at(&delta_divergence, index, 10, 1e-9).or(Some(0.0)),
                rolling_zscore_at(&depth_values, index, 10, 1e-9).or(Some(0.0)),
            ],
            bar_vol_std,
        };
    }
    output
}

/// Per-symbol streaming feature state.
/// One instance per (strategy x symbol).
pub struct OfmrFeatures {
    settings: Settings,
    research: ResearchFeatureState,
}

impl OfmrFeatures {
    pub fn new() -> Self {
        Self::with_settings(Settings::load())
    }

    pub fn with_settings(config: Settings) -> Self {
        Self {
            settings: config.clone(),
            research: ResearchFeatureState::new(&config),
        }
    }

    pub fn set_terminal_orderbook_time_ms(&mut self, timestamp_ms: i64) {
        let timeframe_ms = self.research.timeframe_secs as i64 * 1_000;
        self.research.terminal_bar_index = Some(timestamp_ms.div_euclid(timeframe_ms));
    }
}

impl Default for OfmrFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeFeatures for OfmrFeatures {
    fn compute(&mut self, row: &FeatureRow) -> Result<FeatureRow, String> {
        let event_time_ms = row
            .get("event_time_ms")
            .ok_or_else(|| "OFMR feature input is missing event_time_ms".to_string())?;
        if !event_time_ms.is_finite() || event_time_ms < 0.0 {
            return Err(format!("invalid OFMR event_time_ms: {event_time_ms}"));
        }
        self.research.observe(row, event_time_ms as i64)
    }

    fn output_names(&self) -> Vec<String> {
        output_column_names()
    }

    fn reset(&mut self) {
        self.research = ResearchFeatureState::new(&self.settings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book_tick(price: f64, event_time_ms: f64) -> FeatureRow {
        let mut row = FeatureRow::new();
        row.insert("bid_price", price - 0.01);
        row.insert("ask_price", price + 0.01);
        row.insert("bid_amount", 1.0);
        row.insert("ask_amount", 1.0);
        row.insert("is_lob_event", 1.0);
        row.insert("event_time_ms", event_time_ms);
        row
    }

    #[test]
    fn output_names_are_column_0_to_14() {
        let names = output_column_names();
        assert_eq!(names.len(), 15);
        assert_eq!(names[0], "Column_0");
        assert_eq!(names[14], "Column_14");
    }

    #[test]
    fn warmup_emits_missing_feature_columns() {
        let mut features = OfmrFeatures::with_settings(Settings::default());
        let row = features.compute(&book_tick(100.0, 1_000.0)).unwrap();
        for index in 0..N_FEATURES {
            assert!(row.contains(&format!("Column_{index}")));
            assert!(row.get(&format!("Column_{index}")).is_none());
        }
    }

    #[test]
    fn historical_bars_and_trades_seed_features_before_live_input() {
        let mut features = OfmrFeatures::with_settings(Settings::default());
        for minute in 0..120 {
            let close_time_ms = (minute + 1) * 60_000 - 1;
            let mut trade = FeatureRow::new();
            trade.insert("event_time_ms", (close_time_ms - 100) as f64);
            trade.insert("price", 100.0 + minute as f64 * 0.01);
            trade.insert("amount", 0.1);
            trade.insert("is_buyer_maker", if minute % 2 == 0 { 1.0 } else { 0.0 });
            features.compute(&trade).expect("compute historical trade");

            let mut candle = FeatureRow::new();
            candle.insert("event_time_ms", close_time_ms as f64);
            candle.insert("bar_close_price", 100.0 + minute as f64 * 0.01);
            features
                .compute(&candle)
                .expect("compute historical candle");
        }

        let live = book_tick(101.2, (120 * 60_000) as f64);
        let output = features.compute(&live).expect("compute first live row");
        for index in 0..N_FEATURES {
            assert!(
                output
                    .get(&format!("Column_{index}"))
                    .is_some_and(f64::is_finite),
                "Column_{index} should be ready immediately after history seeding"
            );
        }
    }

    #[test]
    fn rolling_zscore_requires_full_window() {
        let values = [Some(1.0), Some(2.0)];
        assert!(rolling_zscore_at(&values, 1, 10, 1e-9).is_none());
    }

    #[test]
    fn rolling_zscore_matches_zero_variance_research_behavior() {
        let values = [Some(5.0); 10];
        assert_eq!(rolling_zscore_at(&values, 9, 10, 1e-9).unwrap(), 0.0);
    }

    #[test]
    fn wilder_rsi_matches_research_seed_alignment() {
        let closes = [1.0, 2.0, 3.0, 8.0, 9.0].map(Some);
        let rsi = wilder_rsi(&closes, 2);
        assert_eq!(rsi[1], Some(0.0));
        assert_eq!(rsi[2], Some(0.0));
        assert!(rsi[3].unwrap() > 99.999);
    }

    #[test]
    fn output_uses_features_from_two_closed_bars_earlier() {
        let mut config = Settings::default();
        config.feature.htf_timeframe_secs = 1;
        let mut features = OfmrFeatures::with_settings(config);
        features.compute(&book_tick(100.0, 0.0)).unwrap();
        features.compute(&book_tick(101.0, 1_000.0)).unwrap();
        let row = features.compute(&book_tick(102.0, 2_000.0)).unwrap();
        assert_eq!(row.get("Column_7"), Some(0.0));
        assert_eq!(row.get("Column_1"), Some(0.0));
    }

    #[test]
    fn feature_history_is_bounded_and_carries_forward_the_previous_close() {
        let mut state = ResearchFeatureState::new(&Settings::default());
        state.first_bar_index = Some(0);
        state.raw_bars.push(RawBar {
            close: Some(100.0),
            ..RawBar::new(state.timeframe_secs)
        });
        state
            .raw_bars
            .extend((1..=MAX_FEATURE_HISTORY_BARS).map(|_| RawBar::new(state.timeframe_secs)));
        state
            .computed_bars
            .resize(state.raw_bars.len(), BarFeatures::default());

        state.trim_history();

        assert_eq!(state.raw_bars.len(), MAX_FEATURE_HISTORY_BARS);
        assert_eq!(state.computed_bars.len(), MAX_FEATURE_HISTORY_BARS);
        assert_eq!(state.first_bar_index, Some(1));
        assert_eq!(
            state.raw_bars.first().and_then(|bar| bar.close),
            Some(100.0)
        );
    }

    #[test]
    fn very_large_time_jump_resets_history_instead_of_allocating_gap_bars() {
        let mut state = ResearchFeatureState::new(&Settings::default());
        state.current_bar_index = Some(0);
        state.first_bar_index = Some(0);
        state.current_bar = Some(RawBar {
            close: Some(100.0),
            ..RawBar::new(state.timeframe_secs)
        });

        state.close_current_bar(MAX_FEATURE_HISTORY_BARS as i64 + 2);

        assert_eq!(state.raw_bars.len(), 1);
        assert_eq!(
            state.first_bar_index,
            Some(MAX_FEATURE_HISTORY_BARS as i64 + 1)
        );
        assert_eq!(state.raw_bars[0].close, Some(100.0));
    }
}
