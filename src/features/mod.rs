/// Streaming feature engine — 1:1 port of `main/features.py`
/// `LiveFeatureEngine`, which itself mirrors `research/feature_builder.py`.
///
/// Produces the exact 15-feature vector expected by the ONNX model on every
/// 1-second tick. All rolling z-scores use population std (ddof=0), matching
/// the numpy (`np.nanstd`) and polars (`rolling_std(ddof=0)`) reference
/// implementations.
use crate::config::Config;

pub const EPS: f32 = 1e-9;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Wilder's RSI over a diff array, returning the full-length RSI vector
/// exactly like `utils/indicators.get_wilders_rsi`.
pub fn get_wilders_rsi(delta: &[f32], window: usize) -> Vec<f32> {
    let n = delta.len();
    let mut avg_gain = vec![0.0f32; n];
    let mut avg_loss = vec![0.0f32; n];
    if n < window {
        return vec![0.0; n];
    }
    let mean_gain: f32 = delta[..window].iter().map(|d| if *d > 0.0 { *d } else { 0.0 }).sum::<f32>() / window as f32;
    let mean_loss: f32 = delta[..window].iter().map(|d| if *d < 0.0 { -*d } else { 0.0 }).sum::<f32>() / window as f32;
    avg_gain[window] = mean_gain;
    avg_loss[window] = mean_loss;
    for i in window + 1..n {
        let g = if delta[i] > 0.0 { delta[i] } else { 0.0 };
        let l = if delta[i] < 0.0 { -delta[i] } else { 0.0 };
        avg_gain[i] = (avg_gain[i - 1] * (window as f32 - 1.0) + g) / window as f32;
        avg_loss[i] = (avg_loss[i - 1] * (window as f32 - 1.0) + l) / window as f32;
    }
    let n_s = n as f32;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let rs = avg_gain[i] / (avg_loss[i] + 1e-10);
        out[i] = 100.0 - 100.0 / (1.0 + rs);
    }
    let _ = n_s;
    out
}

/// Population std (ddof=0).
fn pop_std(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / values.len() as f32;
    var.sqrt()
}

/// Fixed-capacity FIFO buffer (chronological ordering), mirroring the
/// RingBuffer class used by `LiveFeatureEngine`.
pub struct Ring {
    data: Vec<f32>,
    cap: usize,
    start: usize, // cursor for insertion index when full
}

impl Ring {
    pub fn new(cap: usize) -> Self {
        Self { data: Vec::with_capacity(cap), cap: cap.max(1), start: 0 }
    }

    pub fn push(&mut self, v: f32) {
        if self.data.len() < self.cap {
            self.data.push(v);
        } else {
            self.data[self.start] = v;
            self.start = (self.start + 1) % self.cap;
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn values(&self) -> Vec<f32> {
        if self.data.len() < self.cap {
            return self.data.clone();
        }
        let mut out = Vec::with_capacity(self.cap);
        out.extend_from_slice(&self.data[self.start..]);
        out.extend_from_slice(&self.data[..self.start]);
        out
    }
}
// ---------------------------------------------------------------------------
// FeatureEngine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MacroFeatures {
    pub rsi_2: f32,
    pub sma_14: f32,
    pub ma_slope_5: f32,
    pub ma_accel: f32,
    pub spread_bps_z: f32,
    pub rsi_x_ma: f32,
    pub bar_vol_std: f32,
    // orderflow (present when use_orderflow)
    pub ofi_z: f32,
    pub cvd_delta_z: f32,
    pub cvd_slope_z: f32,
    pub vpin: f32,
    pub absorption_z: f32,
    pub tick_autocorr: f32,
    pub inst_ratio: f32,
    pub delta_divergence_z: f32,
    pub depth_imbalance_z: f32,
}

impl MacroFeatures {
    pub fn defaults() -> Self {
        Self {
            rsi_2: 50.0, sma_14: 50.0, ma_slope_5: 0.0, ma_accel: 0.0,
            spread_bps_z: 0.0, rsi_x_ma: 0.0, bar_vol_std: 0.0,
            ofi_z: 0.0, cvd_delta_z: 0.0, cvd_slope_z: 0.0, vpin: 0.0,
            absorption_z: 0.0, tick_autocorr: 0.0, inst_ratio: 0.0,
            delta_divergence_z: 0.0, depth_imbalance_z: 0.0,
        }
    }
}

pub struct FeatureEngine {
    cfg: Config,

    // --- 1-second micro bar state ---
    current_second: Option<i64>,
    second_open: Option<f32>,
    second_high: Option<f32>,
    second_low: Option<f32>,
    second_close: Option<f32>,
    second_buy_vol: f32,
    second_sell_vol: f32,

    // --- 1s price buffer (price_roc_3) ---
    price_buffer: Ring,

    // --- macro (TIMEFRAME_BARRIER) bar state ---
    minute_open: Option<f32>,
    minute_high: Option<f32>,
    minute_low: Option<f32>,
    minute_close: Option<f32>,
    minute_buy_vol: f32,
    minute_sell_vol: f32,
    minute_volume: f32,
    minute_largest_buy: f32,
    minute_largest_sell: f32,

    // --- latest orderbook ---
    last_bid_price: f32,
    last_bid_amount: f32,
    last_ask_price: f32,
    last_ask_amount: f32,
    last_mid_price: f32,

    // --- macro feature buffers ---
    close_prices: Ring,
    price_diffs: Ring,
    rsi_2_buffer: Ring,
    spread_bps_buffer: Ring,

    // --- ofi state ---
    ofi_prev_bid_price: Option<f32>,
    ofi_prev_bid_amount: Option<f32>,
    ofi_prev_ask_price: Option<f32>,
    ofi_prev_ask_amount: Option<f32>,
    bar_ofi_sum: f32,
    prev_bar_ofi: f32,
    ofi_diff_buffer: Ring,

    // --- cvd ---
    bar_cvd_net: f32,
    cvd_delta_buffer: Ring,
    cvd_slope_buffer: Ring,

    // --- vpin ---
    vpin_bucket_buy: f32,
    vpin_bucket_sell: f32,
    vpin_bucket_vol: f32,
    vpin_bucket_size: Option<f32>,
    vpin_bar_volumes: Ring,
    vpin_ratio_buffer: Ring,
    vpin_smoothed: f32,

    // --- absorption ---
    bar_ask_high: Option<f32>,
    bar_bid_low: Option<f32>,
    absorption_buffer: Ring,

    // --- tick sign autocorr ---
    last_trade_price: Option<f32>,
    second_last_tick_sign: i32,
    bar_tick_signs: Vec<i32>,

    // --- retail / institutional ---
    bar_retail_buy: f32,
    bar_retail_sell: f32,
    bar_inst_buy: f32,
    bar_inst_sell: f32,
    delta_divergence_buffer: Ring,

    // --- depth imbalance ---
    depth_imbalance_buffer: Ring,

    // --- computed macro features (forward-filled) ---
    macro_features: MacroFeatures,

    // --- timeframe bookkeeping ---
    current_minute_key: Option<i64>,
    last_macro_update_minute: i64,
    second_count: u64,
}
impl FeatureEngine {
    pub fn new(cfg: Config) -> Self {
        let vpin_lookback_bars = (cfg.vpin_lookback_seconds / cfg.timeframe_barrier.max(1)).max(3) as usize;
        let of_buf_cap = (30usize).max(cfg.orderflow_z_window * 3);
        Self {
            cfg: cfg.clone(),
            current_second: None,
            second_open: None, second_high: None, second_low: None, second_close: None,
            second_buy_vol: 0.0, second_sell_vol: 0.0,
            price_buffer: Ring::new(300),
            minute_open: None, minute_high: None, minute_low: None, minute_close: None,
            minute_buy_vol: 0.0, minute_sell_vol: 0.0, minute_volume: 0.0,
            minute_largest_buy: 0.0, minute_largest_sell: 0.0,
            last_bid_price: 0.0, last_bid_amount: 0.0,
            last_ask_price: 0.0, last_ask_amount: 0.0, last_mid_price: 0.0,
            close_prices: Ring::new(200),
            price_diffs: Ring::new(300),
            rsi_2_buffer: Ring::new(50),
            spread_bps_buffer: Ring::new(50),
            ofi_prev_bid_price: None, ofi_prev_bid_amount: None,
            ofi_prev_ask_price: None, ofi_prev_ask_amount: None,
            bar_ofi_sum: 0.0, prev_bar_ofi: 0.0,
            ofi_diff_buffer: Ring::new(of_buf_cap),
            bar_cvd_net: 0.0,
            cvd_delta_buffer: Ring::new(of_buf_cap),
            cvd_slope_buffer: Ring::new(of_buf_cap),
            vpin_bucket_buy: 0.0, vpin_bucket_sell: 0.0, vpin_bucket_vol: 0.0,
            vpin_bucket_size: None,
            vpin_bar_volumes: Ring::new(vpin_lookback_bars),
            vpin_ratio_buffer: Ring::new((30usize).max(cfg.vpin_window_buckets as usize * 3)),
            vpin_smoothed: 0.0,
            bar_ask_high: None, bar_bid_low: None,
            absorption_buffer: Ring::new(of_buf_cap),
            last_trade_price: None, second_last_tick_sign: 0, bar_tick_signs: Vec::new(),
            bar_retail_buy: 0.0, bar_retail_sell: 0.0, bar_inst_buy: 0.0, bar_inst_sell: 0.0,
            delta_divergence_buffer: Ring::new(of_buf_cap),
            depth_imbalance_buffer: Ring::new(of_buf_cap),
            macro_features: MacroFeatures::defaults(),
            current_minute_key: None,
            last_macro_update_minute: -1,
            second_count: 0,
        }
    }

    pub fn last_mid(&self) -> f32 { self.last_mid_price }
    pub fn macro_features(&self) -> &MacroFeatures { &self.macro_features }
    pub fn second_count(&self) -> u64 { self.second_count }
            pub fn bar_vol_std(&self) -> f32 { self.macro_features.bar_vol_std }

    /// Number of completed 1-second closes buffered (1s micro-bar series).
    /// Used to gate the clock: the decision loop stays idle until the
    /// price_roc_3 window + RSI(2) warmup are available.
    pub fn price_buffer_len(&self) -> usize { self.price_buffer.len() }

    pub fn seed_bar_series(&mut self, closes: &[f32]) {
        // Warm up close_prices + price_diffs from historical data
        for (i, &close) in closes.iter().enumerate() {
            if i > 0 {
                let prev = closes[i - 1];
                self.price_diffs.push(close - prev);
            }
            self.close_prices.push(close);
        }
        // Force an initial macro compute so RSI/MA are ready immediately.
        if let Some(&last) = closes.last() {
            self.last_mid_price = last;
            self.minute_close = Some(last);
            self.compute_macro_features();
        }
    }
}
    impl FeatureEngine {
    pub fn on_trade(&mut self, ts_ms: i64, price: f32, amount: f32, side: &str) {
        let second_key = ts_ms / 1000;
        if let Some(cur) = self.current_second {
            if second_key != cur { self.flush_second(); }
        }
        self.current_second = Some(second_key);

        if self.second_open.is_none() { self.second_open = Some(price); }
        self.second_high = Some(self.second_high.map_or(price, |h: f32| h.max(price)));
        self.second_low = Some(self.second_low.map_or(price, |l: f32| l.min(price)));
        self.second_close = Some(price);
        if side == "buy" { self.second_buy_vol += amount; } else { self.second_sell_vol += amount; }

        // macro bar aggregates
        if self.minute_open.is_none() { self.minute_open = Some(price); }
        self.minute_high = Some(self.minute_high.map_or(price, |h: f32| h.max(price)));
        self.minute_low = Some(self.minute_low.map_or(price, |l: f32| l.min(price)));
        self.minute_close = Some(price);
        self.minute_volume += amount;
        if side == "buy" {
            self.minute_buy_vol += amount;
            self.minute_largest_buy = self.minute_largest_buy.max(amount);
        } else {
            self.minute_sell_vol += amount;
            self.minute_largest_sell = self.minute_largest_sell.max(amount);
        }

        if !self.cfg.use_orderflow { return; }
        self.bar_cvd_net += if side == "buy" { amount } else { -amount };
        if amount < self.cfg.retail_threshold {
            if side == "buy" { self.bar_retail_buy += amount; } else { self.bar_retail_sell += amount; }
        } else if side == "buy" {
            self.bar_inst_buy += amount;
        } else {
            self.bar_inst_sell += amount;
        }

        if let Some(prev) = self.last_trade_price {
            self.second_last_tick_sign = if price > prev { 1 } else if price < prev { -1 } else { 0 };
        }
        self.last_trade_price = Some(price);

        self.vpin_bucket_buy += if side == "buy" { amount } else { 0.0 };
        self.vpin_bucket_sell += if side == "sell" { amount } else { 0.0 };
        self.vpin_bucket_vol += amount;
        if let Some(size) = self.vpin_bucket_size {
            if self.vpin_bucket_vol >= size {
                let vpin_raw = (self.vpin_bucket_buy - self.vpin_bucket_sell).abs() / (self.vpin_bucket_vol + EPS);
                self.vpin_ratio_buffer.push(vpin_raw);
                if self.vpin_ratio_buffer.len() >= self.cfg.vpin_min_periods as usize {
                    let vals = self.vpin_ratio_buffer.values();
                    let win = vals.len().min(self.cfg.vpin_window_buckets as usize);
                    let slice = &vals[vals.len() - win..];
                    self.vpin_smoothed = slice.iter().sum::<f32>() / win as f32;
                }
                self.vpin_bucket_buy = 0.0; self.vpin_bucket_sell = 0.0; self.vpin_bucket_vol = 0.0;
            }
        }
    }

    pub fn on_orderbook(&mut self, _ts_ms: i64, bid_price: f32, bid_amount: f32, ask_price: f32, ask_amount: f32) {
        if self.cfg.use_orderflow {
            if let Some(prev_bid_px) = self.ofi_prev_bid_price {
                let delta_bid = if bid_price > prev_bid_px { bid_amount }
                    else if bid_price < prev_bid_px { -self.ofi_prev_bid_amount.unwrap_or(0.0) }
                    else { bid_amount - self.ofi_prev_bid_amount.unwrap_or(0.0) };
                let delta_ask = if ask_price < self.ofi_prev_ask_price.unwrap_or(0.0) { ask_amount }
                    else if ask_price > self.ofi_prev_ask_price.unwrap_or(0.0) { -self.ofi_prev_ask_amount.unwrap_or(0.0) }
                    else { ask_amount - self.ofi_prev_ask_amount.unwrap_or(0.0) };
                self.bar_ofi_sum += delta_bid - delta_ask;
            }
            self.ofi_prev_bid_price = Some(bid_price); self.ofi_prev_bid_amount = Some(bid_amount);
            self.ofi_prev_ask_price = Some(ask_price); self.ofi_prev_ask_amount = Some(ask_amount);
            self.bar_ask_high = Some(self.bar_ask_high.map_or(ask_price, |h: f32| h.max(ask_price)));
            self.bar_bid_low = Some(self.bar_bid_low.map_or(bid_price, |l: f32| l.min(bid_price)));
        }
        self.last_bid_price = bid_price; self.last_bid_amount = bid_amount;
        self.last_ask_price = ask_price; self.last_ask_amount = ask_amount;
        self.last_mid_price = (bid_price + ask_price) / 2.0;
    }

    pub fn flush_second(&mut self) {
        let close = self.second_close.unwrap_or(self.last_mid_price);
        self.price_buffer.push(close);
        self.second_count += 1;

        if self.cfg.use_orderflow {
            self.bar_tick_signs.push(self.second_last_tick_sign);
            self.second_last_tick_sign = 0;
        }

        let tf_sec = self.cfg.timeframe_barrier as i64;
        let key = self.current_second.map(|s| s / tf_sec).unwrap_or(0);
        if key != self.last_macro_update_minute && self.minute_close.is_some() {
            self.compute_macro_features();
            self.reset_minute();
        }

        self.second_open = None; self.second_high = None; self.second_low = None; self.second_close = None;
        self.second_buy_vol = 0.0; self.second_sell_vol = 0.0;
    }

    fn reset_minute(&mut self) {
        self.minute_open = None; self.minute_high = None; self.minute_low = None; self.minute_close = None;
        self.minute_buy_vol = 0.0; self.minute_sell_vol = 0.0; self.minute_volume = 0.0;
        self.minute_largest_buy = 0.0; self.minute_largest_sell = 0.0;
        if self.cfg.use_orderflow {
            self.bar_ofi_sum = 0.0; self.bar_cvd_net = 0.0;
            self.bar_ask_high = None; self.bar_bid_low = None;
            self.bar_tick_signs.clear();
            self.bar_retail_buy = 0.0; self.bar_retail_sell = 0.0;
            self.bar_inst_buy = 0.0; self.bar_inst_sell = 0.0;
        }
    }

}
impl FeatureEngine {
    /// Append `value` to the rolling buffer and return its z-score over the
    /// trailing `window` values (population std, ddof=0) — exact mirror of
    /// `LiveFeatureEngine._rolling_zscore`.
    fn push_z(buffer: &mut Ring, value: f32, window: usize) -> f32 {
        buffer.push(value);
        let vals = buffer.values();
        if vals.len() < window {
            return 0.0;
        }
        let win = &vals[vals.len() - window..];
        let mean = win.iter().sum::<f32>() / win.len() as f32;
        let std = pop_std(win);
        (value - mean) / (std + EPS)
    }

    pub fn compute_macro_features(&mut self) {
        let close = self.minute_close.unwrap_or(self.last_mid_price);
        if close <= 0.0 {
            return;
        }

        // ---- append close to the macro close/diff series ----
        let prev_close = self.close_prices.values().last().copied();
        let delta = match prev_close {
            Some(pc) => close - pc,
            None => 0.0,
        };
        self.price_diffs.push(delta);
        self.close_prices.push(close);

        // ---- RSI(2), RSI(14), SMA(14) ----
        let diffs = self.price_diffs.values();
        let mut rsi_2_val = f32::NAN;
        let mut rsi_14_arr: Vec<f32> = Vec::new();
        if diffs.len() >= self.cfg.rsi_2_window + 1 {
            let r2 = get_wilders_rsi(&diffs, self.cfg.rsi_2_window);
            rsi_2_val = *r2.last().unwrap_or(&f32::NAN);
        }
        if diffs.len() >= self.cfg.rsi_14_window + 1 {
            rsi_14_arr = get_wilders_rsi(&diffs, self.cfg.rsi_14_window);
        }
        let sma_14_val = if rsi_14_arr.len() >= self.cfg.sma_14_window {
            let n = rsi_14_arr.len();
            let slice = &rsi_14_arr[n - self.cfg.sma_14_window..];
            slice.iter().sum::<f32>() / slice.len() as f32
        } else {
            50.0
        };
        self.rsi_2_buffer.push(rsi_2_val);

        // ---- bar_vol_std (barrier sizing, NOT an ML feature) ----
        let bar_closes = self.close_prices.values();
        let mut bar_vol_std = 0.0f32;
        if bar_closes.len() >= 3 {
            let mut bar_rets = Vec::with_capacity(bar_closes.len() - 1);
            for i in 1..bar_closes.len() {
                bar_rets.push((bar_closes[i] - bar_closes[i - 1]) / (bar_closes[i - 1] + EPS));
            }
            let window = bar_rets.len().min(self.cfg.barrier_vol_bar_window);
            if window >= 2 {
                let slice = &bar_rets[bar_rets.len() - window..];
                bar_vol_std = pop_std(slice);
            }
        }

        // ---- MA slope(5) / accel ----
        let close_arr = self.close_prices.values();
        let w = self.cfg.ma_slope_window;
        let mut ma_slope_5 = 0.0f32;
        let mut ma_accel = 0.0f32;
        if close_arr.len() >= w + 1 {
            let n_valid = close_arr.len() - w + 1;
            let ma5: Vec<f32> = (0..n_valid)
                .map(|i| close_arr[i..i + w].iter().sum::<f32>() / w as f32)
                .collect();
            ma_slope_5 = if ma5.len() >= 2 { ma5[ma5.len() - 1] - ma5[ma5.len() - 2] } else { 0.0 };
            if ma5.len() >= 4 {
                ma_accel = (ma5[ma5.len() - 1] - ma5[ma5.len() - 2])
                    - (ma5[ma5.len() - 2] - ma5[ma5.len() - 3]);
            }
        }

        // ---- DeepLOB spread z-score ----
        let spread_bps = if self.last_bid_price > 0.0 && self.last_ask_price > 0.0 {
            let mid = (self.last_bid_price + self.last_ask_price) / 2.0;
            (self.last_ask_price - self.last_bid_price) / mid * 10000.0
        } else {
            0.0
        };
        let spread_bps_z = Self::push_z(&mut self.spread_bps_buffer, spread_bps, self.cfg.deeplob_z_window);

        // ---- RSI x MA interaction ----
        let rsi_vals = self.rsi_2_buffer.values();
        let rsi_z = if rsi_vals.len() >= self.cfg.rsi_z_window {
            let win = &rsi_vals[rsi_vals.len() - self.cfg.rsi_z_window..];
            let mean = win.iter().sum::<f32>() / win.len() as f32;
            let std = pop_std(win);
            (rsi_2_val - mean) / (std + EPS)
        } else {
            0.0
        };
        let rsi_x_ma = rsi_z * ma_slope_5;

        self.macro_features = MacroFeatures {
            rsi_2: rsi_2_val,
            sma_14: sma_14_val,
            ma_slope_5,
            ma_accel,
            spread_bps_z,
            rsi_x_ma,
            bar_vol_std,
            ofi_z: 0.0, cvd_delta_z: 0.0, cvd_slope_z: 0.0, vpin: 0.0,
            absorption_z: 0.0, tick_autocorr: 0.0, inst_ratio: 0.0,
            delta_divergence_z: 0.0, depth_imbalance_z: 0.0,
        };

        if self.cfg.use_orderflow {
            let zw = self.cfg.orderflow_z_window;

            // OFI: this bar's total minus last bar's total, z-scored.
            let ofi_diff = self.bar_ofi_sum - self.prev_bar_ofi;
            let ofi_z = Self::push_z(&mut self.ofi_diff_buffer, ofi_diff, zw);
            self.prev_bar_ofi = self.bar_ofi_sum;

            // CVD: per-bar net delta + 5-bar slope, z-scored.
            let cvd_delta_z = Self::push_z(&mut self.cvd_delta_buffer, self.bar_cvd_net, zw);
            let cvd_hist = self.cvd_delta_buffer.values();
            let cvd_slope_5 = if cvd_hist.len() >= 5 {
                cvd_hist[cvd_hist.len() - 5..].iter().sum::<f32>()
            } else {
                0.0
            };
            let cvd_slope_z = Self::push_z(&mut self.cvd_slope_buffer, cvd_slope_5, zw);

            // VPIN bucket recalibration from trailing realized volume.
            self.vpin_bar_volumes.push(self.minute_volume);
            if self.vpin_bar_volumes.len() >= 3 {
                let vol = self.vpin_bar_volumes.values().iter().sum::<f32>();
                self.vpin_bucket_size = Some((vol / self.cfg.vpin_target_buckets as f32).max(EPS));
            }
            let vpin = self.vpin_smoothed;

            // Absorption: bar CVD / orderbook price range.
            let price_range = match (self.bar_ask_high, self.bar_bid_low) {
                (Some(h), Some(l)) => h - l,
                _ => 0.0,
            };
            let abs_ratio_raw = self.bar_cvd_net / (price_range + EPS);
            let absorption_z = Self::push_z(&mut self.absorption_buffer, abs_ratio_raw, zw);

            // Tick-sign lag-1 autocorrelation (Pearson corr of x[:-1], x[1:]).
            let signs: Vec<f32> = self.bar_tick_signs.iter().map(|&s| s as f32).collect();
            let mut tick_autocorr = 0.0f32;
            if signs.len() > 5 && pop_std(&signs[..signs.len() - 1]) > 0.0 && pop_std(&signs[1..]) > 0.0 {
                let a = &signs[..signs.len() - 1];
                let b = &signs[1..];
                let ma = a.iter().sum::<f32>() / a.len() as f32;
                let mb = b.iter().sum::<f32>() / b.len() as f32;
                let mut num = 0.0f32;
                let mut da = 0.0f32;
                let mut db = 0.0f32;
                for i in 0..a.len() {
                    num += (a[i] - ma) * (b[i] - mb);
                    da += (a[i] - ma) * (a[i] - ma);
                    db += (b[i] - mb) * (b[i] - mb);
                }
                let denom = (da * db).sqrt();
                if denom > 0.0 {
                    tick_autocorr = num / denom;
                }
            }

            // Retail vs institutional flow.
            let bar_volume = self.bar_retail_buy + self.bar_retail_sell
                + self.bar_inst_buy + self.bar_inst_sell;
            let inst_delta = self.bar_inst_buy - self.bar_inst_sell;
            let retail_delta = self.bar_retail_buy - self.bar_retail_sell;
            let delta_divergence = inst_delta - retail_delta;
            let inst_ratio = (self.bar_inst_buy + self.bar_inst_sell) / (bar_volume + EPS);
            let delta_divergence_z = Self::push_z(&mut self.delta_divergence_buffer, delta_divergence, zw);

            // Depth imbalance from latest OB snapshot.
            let depth_imbalance = (self.last_bid_amount - self.last_ask_amount)
                / (self.last_bid_amount + self.last_ask_amount + EPS);
            let depth_imbalance_z = Self::push_z(&mut self.depth_imbalance_buffer, depth_imbalance, zw);

            self.macro_features.ofi_z = ofi_z;
            self.macro_features.cvd_delta_z = cvd_delta_z;
            self.macro_features.cvd_slope_z = cvd_slope_z;
            self.macro_features.vpin = vpin;
            self.macro_features.absorption_z = absorption_z;
            self.macro_features.tick_autocorr = tick_autocorr;
            self.macro_features.inst_ratio = inst_ratio;
            self.macro_features.delta_divergence_z = delta_divergence_z;
            self.macro_features.depth_imbalance_z = depth_imbalance_z;
        }

        let tf_sec = self.cfg.timeframe_barrier as i64;
        self.last_macro_update_minute = self.current_second.map(|s| s / tf_sec).unwrap_or(0);
    }

    // ------------------------------------------------------------------
    // 1-second tick: build the full feature vector for inference.
    // ------------------------------------------------------------------
    pub fn compute_1s_feature_vector(&self) -> [f32; 15] {
        let close = self.second_close.unwrap_or(self.last_mid_price);

        // Micro feature: price ROC(3) on 1-second closes.
        let price_vals = self.price_buffer.values();
        let n = price_vals.len();
        let price_roc_3 = if n >= self.cfg.price_roc_window_3 + 1 {
            let ref3 = price_vals[n - self.cfg.price_roc_window_3];
            (close - ref3) / (ref3 + EPS)
        } else {
            0.0
        };

        let m = &self.macro_features;
        [
            m.spread_bps_z,
            m.ma_slope_5,
            m.rsi_x_ma,
            m.ma_accel,
            price_roc_3,
            m.rsi_2,
            m.sma_14,
            m.ofi_z,
            m.cvd_delta_z,
            m.cvd_slope_z,
            m.absorption_z,
            m.tick_autocorr,
            m.inst_ratio,
            m.delta_divergence_z,
            m.depth_imbalance_z,
        ]
    }

    /// Hunting-zone value: RSI(2) from the latest macro update.
    pub fn rsi_2(&self) -> f32 {
        self.macro_features.rsi_2
    }
}