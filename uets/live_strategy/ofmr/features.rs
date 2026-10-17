//! strategy1/compute_feature.rs — YOUR computation.
//!
//! Runs FIRST on every stream row: canonical `FeatureRow` in, computed
//! feature row out. Stateful: one instance per (strategy x symbol), so put
//! rolling windows / running sums here — they never leak to other strategies.
//!
//! Replace the body with your formula. Keep `Missing` semantics: emit
//! `insert_missing` while history is too short, never zero.

use std::collections::VecDeque;

use crate::core::feature::FeatureRow;
use crate::live_strategy::ofmr::settings::Settings;
use crate::live_strategy::ComputeFeatures;

pub const N_FEATURES: usize = 15;

pub fn output_column_names() -> Vec<String> {
    (0..N_FEATURES).map(|i| format!("Column_{i}")).collect()
}

fn rolling_zscore(window: &VecDeque<f64>, value: f64, min_size: usize) -> Option<f64> {
    if window.len() < min_size {
        return None;
    }
    let n = window.len() as f64;
    let mean: f64 = window.iter().sum::<f64>() / n;
    let var: f64 = window.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();
    if std == 0.0 {
        return Some(0.0);
    }
    Some((value - mean) / std)
}

fn push_window<T: Clone>(w: &mut VecDeque<T>, cap: usize, value: T) {
    if w.len() >= cap {
        w.pop_front();
    }
    w.push_back(value);
}


#[derive(Clone, Debug)]
struct RsiState {
    window: usize,
    avg_gain: Option<f64>,
    avg_loss: Option<f64>,
    prev_price: Option<f64>,
    seeded: bool,
    seed_count: usize,
}

impl RsiState {
    fn new(window: usize) -> Self {
        Self {
            window,
            avg_gain: None,
            avg_loss: None,
            prev_price: None,
            seeded: false,
            seed_count: 0,
        }
    }

    fn next(&mut self, price: f64) -> Option<f64> {
        let Some(prev) = self.prev_price else {
            self.prev_price = Some(price);
            return None;
        };
        let delta = price - prev;
        self.prev_price = Some(price);
        let gain = delta.max(0.0);
        let loss = (-delta).max(0.0);
        if !self.seeded {
            self.seed_count += 1;
            let g = self.avg_gain.unwrap_or(0.0) + gain;
            let l = self.avg_loss.unwrap_or(0.0) + loss;
            self.avg_gain = Some(g);
            self.avg_loss = Some(l);
            if self.seed_count >= self.window {
                self.avg_gain = Some(g / self.window as f64);
                self.avg_loss = Some(l / self.window as f64);
                self.seeded = true;
            }
            return None;
        }
        let ag = (self.avg_gain.unwrap() * (self.window as f64 - 1.0) + gain) / self.window as f64;
        let al = (self.avg_loss.unwrap() * (self.window as f64 - 1.0) + loss) / self.window as f64;
        self.avg_gain = Some(ag);
        self.avg_loss = Some(al);
        if al == 0.0 {
            return Some(100.0);
        }
        let rs = ag / al;
        Some(100.0 - 100.0 / (1.0 + rs))
    }
}

/// Per-symbol streaming feature state. One instance per (strategy x symbol).
pub struct OfmrFeatures {
    config: Settings,

    mid_prices: VecDeque<f64>,
    rsi_2: RsiState,
    rsi_14: RsiState,
    rsi_14_window: VecDeque<f64>,
    spread_bps_window: VecDeque<f64>,
    ma5_prices: VecDeque<f64>,
    prev_ma5: Option<f64>,
    prev_ma_slope: Option<f64>,
    ofi_window: VecDeque<f64>,
    prev_ofi: Option<f64>,
    cvd: f64,
    cvd_delta_window: VecDeque<f64>,
    cvd_slope_window: VecDeque<f64>,
    prev_cvd: Option<f64>,
    tick_rule_window: VecDeque<i32>,
    prev_trade_price: Option<f64>,
    inst_buy_vol: f64,
    inst_sell_vol: f64,
    retail_buy_vol: f64,
    retail_sell_vol: f64,
    total_volume: f64,
    depth_imbalance_window: VecDeque<f64>,
    abs_ratio_window: VecDeque<f64>,
    delta_divergence_window: VecDeque<f64>,
    sequence: u64,
    minute_bucket: Option<u64>,
    minute_close: Option<f64>,
    minute_returns: VecDeque<f64>,
}

impl OfmrFeatures {
    pub fn new() -> Self {
        Self::with_settings(Settings::load())
    }

    pub fn with_settings(config: Settings) -> Self {
        Self {
            config,
            mid_prices: VecDeque::with_capacity(200),
            rsi_2: RsiState::new(2),
            rsi_14: RsiState::new(14),
            rsi_14_window: VecDeque::with_capacity(14),
            spread_bps_window: VecDeque::with_capacity(60),
            ma5_prices: VecDeque::with_capacity(10),
            prev_ma5: None,
            prev_ma_slope: None,
            ofi_window: VecDeque::with_capacity(60),
            prev_ofi: None,
            cvd: 0.0,
            cvd_delta_window: VecDeque::with_capacity(60),
            cvd_slope_window: VecDeque::with_capacity(60),
            prev_cvd: None,
            tick_rule_window: VecDeque::with_capacity(60),
            prev_trade_price: None,
            inst_buy_vol: 0.0,
            inst_sell_vol: 0.0,
            retail_buy_vol: 0.0,
            retail_sell_vol: 0.0,
            total_volume: 0.0,
            depth_imbalance_window: VecDeque::with_capacity(60),
            abs_ratio_window: VecDeque::with_capacity(60),
            delta_divergence_window: VecDeque::with_capacity(60),
            sequence: 0,
            minute_bucket: None,
            minute_close: None,
            minute_returns: VecDeque::with_capacity(100),
        }
    }
}

impl Default for OfmrFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeFeatures for OfmrFeatures {
    fn compute(&mut self, row: &FeatureRow) -> Result<FeatureRow, String> {
        self.sequence += 1;
        let bid = row.get("bid_price");
        let ask = row.get("ask_price");
        let trade_price = row.get("price");
        let trade_amount = row.get("amount").unwrap_or(0.0);
        let is_buyer_maker = row.get("is_buyer_maker").map(|v| v != 0.0).unwrap_or(false);
        let mid = match (bid, ask) {
            (Some(b), Some(a)) if b > 0.0 && a > 0.0 => (b + a) / 2.0,
            _ => trade_price.unwrap_or(0.0),
        };
        let event_time_ms = row.get("event_time_ms").unwrap_or_default().max(0.0) as u64;
        let minute_bucket = event_time_ms / 60_000;
        if mid.is_finite() && mid > 0.0 {
            match self.minute_bucket {
                None => {
                    self.minute_bucket = Some(minute_bucket);
                    self.minute_close = Some(mid);
                }
                Some(previous_bucket) if minute_bucket == previous_bucket => {
                    self.minute_close = Some(mid);
                }
                Some(previous_bucket) if minute_bucket > previous_bucket => {
                    if minute_bucket == previous_bucket + 1 {
                        if let (Some(previous), Some(close)) = (self.minute_close, Some(mid)) {
                            let ret = (close - previous) / previous;
                            if ret.is_finite() {
                                push_window(&mut self.minute_returns, 100, ret);
                            }
                        }
                    }
                    self.minute_bucket = Some(minute_bucket);
                    self.minute_close = Some(mid);
                }
                _ => {}
            }
        }
        push_window(&mut self.mid_prices, 200, if mid > 0.0 { mid } else { 0.0 });
        let rsi_2_val = if mid > 0.0 { self.rsi_2.next(mid) } else { None };
        let rsi_14_val = if mid > 0.0 { self.rsi_14.next(mid) } else { None };
        push_window(&mut self.ma5_prices, 5, mid);
        let ma5 = if self.ma5_prices.len() == 5 {
            self.ma5_prices.iter().sum::<f64>() / 5.0
        } else { mid };
        let ma_slope = self.prev_ma5.map(|p| ma5 - p).unwrap_or(0.0);
        self.prev_ma5 = Some(ma5);
        let spread_bps = match (bid, ask) {
            (Some(b), Some(a)) if b > 0.0 && a > 0.0 && mid > 0.0 => (a - b) / mid * 10000.0,
            _ => 0.0,
        };
        push_window(&mut self.spread_bps_window, 60, spread_bps);
        let spread_bps_z = rolling_zscore(&self.spread_bps_window, spread_bps, 10);
        let price_roc_3 = if self.mid_prices.len() > 3 {
            let len = self.mid_prices.len();
            let prev_price = self.mid_prices[len - 4];
            if prev_price > 0.0 { (mid - prev_price) / prev_price } else { 0.0 }
        } else { 0.0 };
        push_window(&mut self.rsi_14_window, 14, rsi_14_val.unwrap_or(50.0));
        let sma_14 = if self.rsi_14_window.len() == 14 {
            self.rsi_14_window.iter().sum::<f64>() / 14.0
        } else { rsi_14_val.unwrap_or(50.0) };
        let ma_accel = self.prev_ma_slope.map(|p| ma_slope - p).unwrap_or(0.0);
        self.prev_ma_slope = Some(ma_slope);
        let rsi_z = rsi_2_val.map(|v| (v - 50.0) / 25.0).unwrap_or(0.0);
        let rsi_x_ma = rsi_z * ma_slope;
        let tick_rule = match trade_price {
            Some(p) => {
                let tr = match self.prev_trade_price {
                    Some(pp) if p > pp => 1,
                    Some(pp) if p < pp => -1,
                    _ => 0,
                };
                self.prev_trade_price = Some(p);
                tr
            }
            None => 0,
        };
        let ofi_raw = spread_bps;
        push_window(&mut self.ofi_window, 60, ofi_raw);
        let ofi_diff = self.prev_ofi.map(|p| ofi_raw - p).unwrap_or(0.0);
        self.prev_ofi = Some(ofi_raw);
        push_window(&mut self.cvd_delta_window, 60, ofi_diff);
        let ofi_z = rolling_zscore(&self.ofi_window, ofi_diff, 10);
        let cvd_delta = tick_rule as f64 * trade_amount;
        self.cvd += cvd_delta;
        push_window(&mut self.cvd_delta_window, 60, cvd_delta);
        let cvd_delta_z = rolling_zscore(&self.cvd_delta_window, cvd_delta, 10);
        let cvd_slope = if self.cvd_delta_window.len() >= 5 {
            let len = self.cvd_delta_window.len();
            let prev = self.cvd_delta_window[len - 5];
            self.cvd - prev
        } else { 0.0 };
        push_window(&mut self.cvd_slope_window, 60, cvd_slope);
        let cvd_slope_z = rolling_zscore(&self.cvd_slope_window, cvd_slope, 10);
        push_window(&mut self.tick_rule_window, 60, tick_rule);
        let tick_autocorr = if self.tick_rule_window.len() > 5 {
            let vals: Vec<f64> = self.tick_rule_window.iter().map(|&v| v as f64).collect();
            let n = vals.len();
            let a = &vals[..n - 1];
            let b = &vals[1..];
            let ma = a.iter().sum::<f64>() / n as f64;
            let mb = b.iter().sum::<f64>() / n as f64;
            let sa = a.iter().map(|v| (v - ma) * (v - ma)).sum::<f64>().sqrt();
            let sb = b.iter().map(|v| (v - mb) * (v - mb)).sum::<f64>().sqrt();
            if sa == 0.0 || sb == 0.0 { 0.0 } else {
                let cov: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - ma) * (y - mb)).sum();
                cov / (sa * sb)
            }
        } else { 0.0 };
        let retail_threshold = 0.05 * trade_amount.max(1.0);
        if trade_amount < retail_threshold {
            if is_buyer_maker { self.retail_sell_vol += trade_amount; } else { self.retail_buy_vol += trade_amount; }
        } else {
            if is_buyer_maker { self.inst_sell_vol += trade_amount; } else { self.inst_buy_vol += trade_amount; }
        }
        self.total_volume += trade_amount;
        let inst_ratio = if self.total_volume > 0.0 {
            (self.inst_buy_vol + self.inst_sell_vol) / (self.total_volume + 1e-9)
        } else { 0.0 };
        let depth_imbalance = match (bid, ask) {
            (Some(b), Some(a)) if b > 0.0 && a > 0.0 => {
                let ba = row.get("bid_amount").unwrap_or(0.0);
                let aa = row.get("ask_amount").unwrap_or(0.0);
                let denom = ba + aa + 1e-9;
                (ba - aa) / denom
            }
            _ => 0.0,
        };
        push_window(&mut self.depth_imbalance_window, 60, depth_imbalance);
        let depth_imbalance_z = rolling_zscore(&self.depth_imbalance_window, depth_imbalance, 10);
        let price_range = match (bid, ask) {
            (Some(b), Some(a)) if a > b => a - b,
            _ => 0.0,
        };
        let abs_ratio = if price_range > 0.0 {
            cvd_delta.abs() / (price_range + 1e-9)
        } else { 0.0 };
        push_window(&mut self.abs_ratio_window, 60, abs_ratio);
        let absorption_z = rolling_zscore(&self.abs_ratio_window, abs_ratio, 10);
        let delta_divergence = (self.inst_buy_vol - self.inst_sell_vol)
            - (self.retail_buy_vol - self.retail_sell_vol);
        push_window(&mut self.delta_divergence_window, 60, delta_divergence);
        let delta_divergence_z = rolling_zscore(&self.delta_divergence_window, delta_divergence, 10);
        let mut out = FeatureRow::new();
        out.insert("Column_0", spread_bps_z.unwrap_or(0.0));
        out.insert("Column_1", ma_slope);
        out.insert("Column_2", rsi_x_ma);
        out.insert("Column_3", ma_accel);
        out.insert("Column_4", price_roc_3);
        out.insert("Column_5", rsi_2_val.unwrap_or(50.0));
        out.insert("Column_6", sma_14);
        out.insert("Column_7", ofi_z.unwrap_or(0.0));
        out.insert("Column_8", cvd_delta_z.unwrap_or(0.0));
        out.insert("Column_9", cvd_slope_z.unwrap_or(0.0));
        out.insert("Column_10", absorption_z.unwrap_or(0.0));
        out.insert("Column_11", tick_autocorr);
        out.insert("Column_12", inst_ratio);
        out.insert("Column_13", delta_divergence_z.unwrap_or(0.0));
        out.insert("Column_14", depth_imbalance_z.unwrap_or(0.0));
        if self.minute_returns.len() >= 2 {
            let mean = self.minute_returns.iter().sum::<f64>() / self.minute_returns.len() as f64;
            let variance = self
                .minute_returns
                .iter()
                .map(|ret| (ret - mean).powi(2))
                .sum::<f64>()
                / self.minute_returns.len() as f64;
            out.insert("bar_volatility_1m", variance.sqrt());
        }
        Ok(out)
    }

    fn output_names(&self) -> Vec<String> {
        output_column_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tick(price: f64, bid: f64, ask: f64, amount: f64, buyer_maker: bool) -> FeatureRow {
        let mut r = FeatureRow::new();
        r.insert("price", price);
        r.insert("amount", amount);
        r.insert("bid_price", bid);
        r.insert("ask_price", ask);
        r.insert("bid_amount", 1.0);
        r.insert("ask_amount", 1.0);
        r.insert("is_buyer_maker", if buyer_maker { 1.0 } else { 0.0 });
        r
    }

    #[test]
    fn output_names_are_column_0_to_14() {
        let names = output_column_names();
        assert_eq!(names.len(), 15);
        assert_eq!(names[0], "Column_0");
        assert_eq!(names[14], "Column_14");
    }

    #[test]
    fn first_tick_returns_all_columns() {
        let mut f = OfmrFeatures::new();
        let row = f.compute(&tick(100.0, 99.99, 100.01, 1.0, false)).unwrap();
        for i in 0..15 {
            assert!(row.contains(&format!("Column_{i}")), "missing Column_{i}");
        }
    }

    #[test]
    fn rolling_zscore_returns_none_when_window_short() {
        let w: VecDeque<f64> = VecDeque::from([1.0, 2.0]);
        assert!(rolling_zscore(&w, 3.0, 10).is_none());
    }

    #[test]
    fn rolling_zscore_returns_zero_when_std_zero() {
        let w: VecDeque<f64> = VecDeque::from([5.0; 10]);
        assert_eq!(rolling_zscore(&w, 5.0, 10).unwrap(), 0.0);
    }
}


