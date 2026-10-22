//! Timeframe reconstruction: websocket event stream -> timeframe bars.
//!
//! This is the Rust streaming counterpart of
//! `components/pipelines/event_merger/matched/reconstruct.py` and the
//! `dense_resample()` helper in `ml_state_builder.py`. Its whole job is to
//! turn the *event-time* stream (one sample per websocket message) into
//! *bar-time* data the strategy layer can compute features on:
//!
//! * **Time bars** — `"1s"`, `"30s"`, `"1min"`/`"1m"`, `"5m"`, `"300s"`, ...
//!   Bars are right-labelled like pandas `resample(..., label="right")`: the
//!   bucket `(0, T]` is emitted with `close_ms == T`. A bar is only emitted
//!   once a sample at or past its close has been observed — incomplete bars
//!   are never fabricated.
//! * **Tick bars** — an integer `N` means "one bar per N aggregate trades",
//!   labelled with the timestamp of the Nth trade (mirrors the integer path
//!   of the Python `reconstruct()`, whose causal column is `agg_trade_id`).
//!
//! When the stream jumps across empty periods, the gaps are emitted as
//! bars flagged [`Bar::empty`]:
//!
//! * the OFMR **macro** path (`drop_empty=True` in Python) simply skips them;
//! * the OFMR **micro** path (`dense_resample`) consumes them and
//!   forward-fills the price / zero-fills the volumes.
//!
//! Everything here is deterministic, allocation-light, and I/O-free so the
//! same code can drive a live websocket loop or a recorded-feed replay.

/// Maximum number of empty bars one timestamp jump may emit. Bounds the work
/// a multi-hour feed gap can cause in a single `push()`; beyond this the
/// jump is collapsed into one wide empty bar.
const MAX_GAP_BARS: u64 = 86_400;

/// A bar size: `N` trades, or `S` seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timeframe {
    /// One bar per `N` aggregate trades (tick bars).
    Ticks(usize),
    /// One bar per `S` seconds (wall-clock bars).
    Seconds(u64),
}

impl Timeframe {
    /// Parse `"1s"`, `"30s"`, `"1min"`, `"1m"`, `"5m"`, `"1h"`, `"1d"`, or a
    /// bare integer (tick count). Same vocabulary as the Python
    /// `_normalize_timeframe()`.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let text = spec.trim().to_ascii_lowercase();
        let number = |raw: &str| -> Result<u64, String> {
            raw.parse::<u64>()
                .map_err(|_| format!("invalid timeframe {spec:?}"))
                .and_then(|n| {
                    if n == 0 {
                        Err(format!("timeframe {spec:?} must be > 0"))
                    } else {
                        Ok(n)
                    }
                })
        };
        if let Some(raw) = text.strip_suffix("ms") {
            let ms = number(raw)?;
            if ms < 1_000 || ms % 1_000 != 0 {
                return Err(format!(
                    "timeframe {spec:?}: sub-second bars are not supported (use whole seconds)"
                ));
            }
            return Ok(Self::Seconds(ms / 1_000));
        }
        for (suffix, seconds) in [
            ("min", 60),
            ("s", 1),
            ("m", 60),
            ("h", 3_600),
            ("d", 86_400),
        ] {
            if let Some(raw) = text.strip_suffix(suffix) {
                return Ok(Self::Seconds(number(raw)? * seconds));
            }
        }
        if let Ok(ticks) = text.parse::<usize>() {
            if ticks == 0 {
                return Err("timeframe must be > 0".into());
            }
            return Ok(Self::Ticks(ticks));
        }
        Err(format!(
            "invalid timeframe {spec:?}; expected e.g. \"1s\", \"1min\", \"5m\", \"1h\", or a tick count"
        ))
    }

    /// Length of one time bar in seconds, if this is a time timeframe.
    pub fn seconds(&self) -> Option<u64> {
        match self {
            Self::Seconds(s) => Some(*s),
            Self::Ticks(_) => None,
        }
    }
}

/// One event sample fed into a [`Reconstructor`].
///
/// The caller decides what `price` means: the mid price for a macro (book)
/// reconstruction, the trade price for a micro (trade) reconstruction.
/// `has_trade` marks rows that actually carried a trade print so volume
/// splits and trade counts stay correct when book-only rows share the
/// same series.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sample {
    /// Event time in epoch **milliseconds**.
    pub timestamp_ms: i64,
    /// The value whose OHLC this bar aggregates (mid or trade price).
    pub price: f64,
    /// Whether this sample carried a trade print.
    pub has_trade: bool,
    /// Trade size (0.0 on book-only rows).
    pub quantity: f64,
    /// Taker side marker (true = taker sold); only read when `has_trade`.
    pub is_buyer_maker: bool,
    /// Top-of-book carried on this row (absent book -> `None`).
    pub bid_price: Option<f64>,
    pub bid_amount: f64,
    pub ask_price: Option<f64>,
    pub ask_amount: f64,
}

/// One completed bar. `close_ms` is the right-edge label.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bar {
    /// Right-edge label in epoch milliseconds (bar covers `(close_ms - len, close_ms]`).
    pub close_ms: i64,
    /// True when the bar covers a silent period (no samples), i.e. OHLC come
    /// from no observation at all. Micro consumers forward-fill; macro
    /// consumers skip.
    pub empty: bool,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// Sum of trade quantities inside the bar.
    pub volume: f64,
    /// Taker-side split of `volume` (is_buyer_maker -> sell).
    pub buy_volume: f64,
    pub sell_volume: f64,
    /// Number of trade prints in the bar.
    pub trade_count: u64,
    /// Last observed top-of-book inside the bar.
    pub bid_price: f64,
    pub bid_amount: f64,
    pub ask_price: f64,
    pub ask_amount: f64,
    /// Sum of per-row top-of-book quantities inside the bar.
    pub bid_amount_sum: f64,
    pub ask_amount_sum: f64,
}

impl Bar {
    /// `(ask - bid) / mid` as a fraction at the bar's last book state.
    pub fn spread_pct(&self) -> Option<f64> {
        if self.bid_price > 0.0 && self.ask_price > 0.0 {
            let mid = (self.bid_price + self.ask_price) / 2.0;
            (mid > 0.0).then_some((self.ask_price - self.bid_price) / mid)
        } else {
            None
        }
    }

    /// `(bid_qty - ask_qty) / (bid_qty + ask_qty)` at the bar's last book state.
    pub fn depth_imbalance(&self) -> Option<f64> {
        let denom = self.bid_amount + self.ask_amount;
        (denom > 0.0).then_some((self.bid_amount - self.ask_amount) / denom)
    }
}

/// Streaming event-stream -> bars state machine.
#[derive(Clone, Debug)]
pub struct Reconstructor {
    timeframe: Timeframe,
    /// Index of the bucket currently accumulating (time bars only).
    bucket: Option<i64>,
    /// Trades seen in the current bar (tick bars).
    trades_in_bar: usize,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    buy_volume: f64,
    sell_volume: f64,
    trade_count: u64,
    last_bid_price: f64,
    last_bid_amount: f64,
    last_ask_price: f64,
    last_ask_amount: f64,
    bid_amount_sum: f64,
    ask_amount_sum: f64,
    has_data: bool,
    /// Samples dropped for arriving before the open bar (late arrivals).
    pub dropped: u64,
}

impl Reconstructor {
    pub fn new(timeframe: Timeframe) -> Self {
        Self {
            timeframe,
            bucket: None,
            trades_in_bar: 0,
            open: 0.0,
            high: f64::MIN,
            low: f64::MAX,
            close: 0.0,
            volume: 0.0,
            buy_volume: 0.0,
            sell_volume: 0.0,
            trade_count: 0,
            last_bid_price: 0.0,
            last_bid_amount: 0.0,
            last_ask_price: 0.0,
            last_ask_amount: 0.0,
            bid_amount_sum: 0.0,
            ask_amount_sum: 0.0,
            has_data: false,
            dropped: 0,
        }
    }

    pub fn timeframe(&self) -> Timeframe {
        self.timeframe
    }

    /// Whether the current (incomplete) bar holds any data.
    pub fn has_open_bar(&self) -> bool {
        self.has_data
    }

    /// Feed one sample; returns every bar completed by it, in order.
    /// Usually 0 or 1 bars; a timestamp gap can complete several.
    pub fn push(&mut self, sample: Sample) -> Vec<Bar> {
        let mut completed = Vec::new();
        match self.timeframe {
            Timeframe::Seconds(seconds) => {
                let len_ms = seconds.saturating_mul(1_000) as i64;
                // Right-closed bucketing, matching pandas
                // `resample(..., closed="right", label="right")`: a sample at
                // exactly T belongs to the bar whose right edge is T.
                let q = sample.timestamp_ms.div_euclid(len_ms);
                let r = sample.timestamp_ms.rem_euclid(len_ms);
                let bucket = if r == 0 { q } else { q + 1 };
                match self.bucket {
                    None => {
                        self.bucket = Some(bucket);
                        self.accumulate(sample);
                    }
                    Some(open) if bucket == open => self.accumulate(sample),
                    Some(open) if bucket > open => {
                        // Finalize the open bar at its own right edge, then
                        // emit empty bars for every skipped bucket.
                        if self.has_data {
                            completed.push(self.finish(open * len_ms, false));
                        }
                        let mut skipped = (bucket - open - 1).min(MAX_GAP_BARS as i64);
                        // Wide jumps collapse into one wide empty bar so a
                        // long outage does not fan out into millions of rows.
                        if bucket - open - 1 > MAX_GAP_BARS as i64 {
                            skipped = 1;
                        }
                        for gap in (open + 1)..(open + 1 + skipped) {
                            let empty = Bar {
                                close_ms: gap * len_ms,
                                empty: true,
                                ..Default::default()
                            };
                            completed.push(empty);
                        }
                        self.reset(Some(bucket));
                        self.accumulate(sample);
                    }
                    // Late arrival from before the open bucket: drop.
                    Some(_) => self.dropped += 1,
                }
            }
            Timeframe::Ticks(ticks) => {
                if !sample.has_trade {
                    return completed;
                }
                self.accumulate(sample);
                self.trades_in_bar += 1;
                if self.trades_in_bar >= ticks {
                    completed.push(self.finish(sample.timestamp_ms, false));
                    self.reset(None);
                }
            }
        }
        completed
    }

    fn reset(&mut self, bucket: Option<i64>) {
        self.bucket = bucket;
        self.trades_in_bar = 0;
        self.open = 0.0;
        self.high = f64::MIN;
        self.low = f64::MAX;
        self.close = 0.0;
        self.volume = 0.0;
        self.buy_volume = 0.0;
        self.sell_volume = 0.0;
        self.trade_count = 0;
        self.last_bid_price = 0.0;
        self.last_bid_amount = 0.0;
        self.last_ask_price = 0.0;
        self.last_ask_amount = 0.0;
        self.bid_amount_sum = 0.0;
        self.ask_amount_sum = 0.0;
        self.has_data = false;
    }

    fn accumulate(&mut self, sample: Sample) {
        if !self.has_data {
            self.open = sample.price;
            self.high = sample.price;
            self.low = sample.price;
        } else {
            self.high = self.high.max(sample.price);
            self.low = self.low.min(sample.price);
        }
        self.close = sample.price;
        self.has_data = true;

        if sample.has_trade {
            self.trade_count += 1;
            self.volume += sample.quantity;
            if sample.is_buyer_maker {
                self.sell_volume += sample.quantity;
            } else {
                self.buy_volume += sample.quantity;
            }
        }

        if let Some(bid) = sample.bid_price.filter(|p| *p > 0.0) {
            self.last_bid_price = bid;
            self.last_bid_amount = sample.bid_amount;
            self.bid_amount_sum += sample.bid_amount;
        }
        if let Some(ask) = sample.ask_price.filter(|p| *p > 0.0) {
            self.last_ask_price = ask;
            self.last_ask_amount = sample.ask_amount;
            self.ask_amount_sum += sample.ask_amount;
        }
    }

    fn finish(&mut self, close_ms: i64, empty: bool) -> Bar {
        Bar {
            close_ms,
            empty,
            open: self.open,
            high: if self.high.is_finite() { self.high } else { 0.0 },
            low: if self.low.is_finite() { self.low } else { 0.0 },
            close: self.close,
            volume: self.volume,
            buy_volume: self.buy_volume,
            sell_volume: self.sell_volume,
            trade_count: self.trade_count,
            bid_price: self.last_bid_price,
            bid_amount: self.last_bid_amount,
            ask_price: self.last_ask_price,
            ask_amount: self.last_ask_amount,
            bid_amount_sum: self.bid_amount_sum,
            ask_amount_sum: self.ask_amount_sum,
        }
    }
}

/// Batch convenience: reconstruct every sample over one timeframe.
pub fn reconstruct<'a, I>(timeframe: Timeframe, samples: I) -> Vec<Bar>
where
    I: IntoIterator<Item = &'a Sample>,
{
    let mut reconstructor = Reconstructor::new(timeframe);
    let mut bars = Vec::new();
    for sample in samples {
        bars.extend(reconstructor.push(*sample));
    }
    bars
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(ts: i64, price: f64, qty: f64, sell: bool) -> Sample {
        Sample {
            timestamp_ms: ts,
            price,
            has_trade: true,
            quantity: qty,
            is_buyer_maker: sell,
            bid_price: Some(price - 0.5),
            bid_amount: 1.0,
            ask_price: Some(price + 0.5),
            ask_amount: 2.0,
        }
    }

    fn book(ts: i64, mid: f64) -> Sample {
        Sample {
            timestamp_ms: ts,
            price: mid,
            bid_price: Some(mid - 0.5),
            bid_amount: 1.0,
            ask_price: Some(mid + 0.5),
            ask_amount: 2.0,
            ..Sample::default()
        }
    }

    #[test]
    fn parses_the_python_timeframe_vocabulary() {
        assert_eq!(Timeframe::parse("1s").unwrap(), Timeframe::Seconds(1));
        assert_eq!(Timeframe::parse("30s").unwrap(), Timeframe::Seconds(30));
        assert_eq!(Timeframe::parse("1min").unwrap(), Timeframe::Seconds(60));
        assert_eq!(Timeframe::parse("1m").unwrap(), Timeframe::Seconds(60));
        assert_eq!(Timeframe::parse("5m").unwrap(), Timeframe::Seconds(300));
        assert_eq!(Timeframe::parse("300s").unwrap(), Timeframe::Seconds(300));
        assert_eq!(Timeframe::parse("1h").unwrap(), Timeframe::Seconds(3_600));
        assert_eq!(Timeframe::parse("10").unwrap(), Timeframe::Ticks(10));
        assert!(Timeframe::parse("bogus").is_err());
        assert!(Timeframe::parse("0s").is_err());
    }

    #[test]
    fn time_bars_are_right_labelled_and_only_emit_when_closed() {
        let mut rec = Reconstructor::new(Timeframe::Seconds(60));
        // Bar covering (0, 60_000] — right-closed, so the print at exactly
        // 60_000 still belongs to it and the bar only completes once a
        // strictly later sample proves the bucket is final.
        assert!(rec.push(trade(1_000, 100.0, 1.0, false)).is_empty());
        assert!(rec.push(trade(30_000, 101.0, 2.0, true)).is_empty());
        assert!(rec.push(trade(60_000, 102.0, 3.0, false)).is_empty());
        let bars = rec.push(trade(61_000, 103.0, 1.0, false));
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];
        assert_eq!(bar.close_ms, 60_000);
        assert!(!bar.empty);
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 102.0);
        assert_eq!(bar.low, 100.0);
        assert_eq!(bar.close, 102.0);
        assert_eq!(bar.volume, 6.0);
        assert_eq!(bar.buy_volume, 4.0);
        assert_eq!(bar.sell_volume, 2.0);
        assert_eq!(bar.trade_count, 3);
    }

    #[test]
    fn gaps_emit_empty_bars_that_macro_consumers_can_skip() {
        let mut rec = Reconstructor::new(Timeframe::Seconds(60));
        rec.push(trade(1_000, 100.0, 1.0, false));
        // Jump three full minutes ahead: (0,60s] closes, 60-120 and
        // 120-180 are silent, then (180,240] starts accumulating.
        let bars = rec.push(trade(240_000, 105.0, 1.0, false));
        assert_eq!(bars.len(), 3);
        assert!(!bars[0].empty);
        assert_eq!(bars[0].close_ms, 60_000);
        assert!(bars[1].empty);
        assert_eq!(bars[1].close_ms, 120_000);
        assert!(bars[2].empty);
        assert_eq!(bars[2].close_ms, 180_000);
    }

    #[test]
    fn book_only_samples_still_build_macro_bars() {
        let mut rec = Reconstructor::new(Timeframe::Seconds(60));
        rec.push(book(1_000, 100.0));
        rec.push(book(30_000, 104.0));
        let bars = rec.push(book(61_000, 106.0));
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 104.0);
        assert_eq!(bar.low, 100.0);
        // The print at 61_000 belongs to the NEXT (still open) bar; this
        // bar's last book state is the mid-104 sample.
        assert_eq!(bar.close, 104.0);
        assert_eq!(bar.volume, 0.0);
        assert_eq!(bar.trade_count, 0);
        assert!((bar.spread_pct().unwrap() - 1.0 / 104.0).abs() < 1e-12);
        assert_eq!(bar.depth_imbalance(), Some((1.0 - 2.0) / 3.0));
    }

    #[test]
    fn tick_bars_close_on_the_nth_trade_and_ignore_book_rows() {
        let mut rec = Reconstructor::new(Timeframe::Ticks(3));
        assert!(rec.push(trade(10, 100.0, 1.0, false)).is_empty());
        assert!(rec.push(book(11, 100.5)).is_empty());
        assert!(rec.push(trade(12, 101.0, 1.0, true)).is_empty());
        let bars = rec.push(trade(14, 102.0, 1.0, false));
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].close_ms, 14);
        assert_eq!(bars[0].trade_count, 3);
        assert_eq!(bars[0].volume, 3.0);
    }

    #[test]
    fn late_samples_are_dropped_not_rolled_back() {
        let mut rec = Reconstructor::new(Timeframe::Seconds(60));
        rec.push(trade(61_000, 100.0, 1.0, false));
        let bars = rec.push(trade(1_000, 99.0, 1.0, false));
        assert!(bars.is_empty());
        assert_eq!(rec.dropped, 1);
    }

    #[test]
    fn batch_reconstruct_matches_the_streaming_path() {
        let samples = [
            trade(1_000, 100.0, 1.0, false),
            trade(60_000, 101.0, 2.0, true),
            trade(120_000, 102.0, 3.0, false),
        ];
        let bars = reconstruct(Timeframe::Seconds(60), samples.iter());
        // Only the first bar is provably final within this window; the
        // second stays open. Trades at 1_000 and 60_000 fall in bar 1.
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].close_ms, 60_000);
        assert_eq!(bars[0].trade_count, 2);
        assert_eq!(bars[0].volume, 3.0);
    }
}





