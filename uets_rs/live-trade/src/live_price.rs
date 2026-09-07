//! Independent live price state for UI display.
//!
//! This is updated by the feed reader and read by the UI ~1 second.
//! Completely decoupled from the strategy timeframe.
//!
//! Uses a simple RwLock since UI only reads once per second - simplicity
//! over micro-optimization. All three fields (price, timestamp, received_at)
//! are updated atomically as a coherent snapshot.

use std::sync::RwLock;
use std::time::Instant;

/// Read-only view of live price state.
#[derive(Debug, Clone)]
pub struct LivePrice {
    pub mid: f64,
    pub exchange_ts_ms: i64,
    pub received_at: Instant,
}

impl LivePrice {
    /// Age of this price data (how long ago it was received).
    pub fn age_secs(&self) -> f64 {
        self.received_at.elapsed().as_secs_f64()
    }
}

/// Per-asset live price state.
pub struct LivePriceState {
    inner: RwLock<Option<LivePrice>>,
}

impl LivePriceState {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(None),
        }
    }

    /// Update with a new trade price. Coherent snapshot - all fields updated together.
    pub fn update_trade(&self, price: f64, exchange_ts_ms: i64) {
        let now = Instant::now();
        let new_price = LivePrice {
            mid: price,
            exchange_ts_ms,
            received_at: now,
        };
        
        if let Ok(mut guard) = self.inner.write() {
            *guard = Some(new_price);
        }
    }

    /// Read the current live price (non-blocking, returns immediately).
    pub fn read(&self) -> Option<LivePrice> {
        self.inner.read().ok()?.clone()
    }

    /// Check if the price is "fresh" - received within the last N seconds.
    pub fn is_fresh(&self, max_age_secs: f64) -> bool {
        match self.read() {
            Some(price) => price.age_secs() < max_age_secs,
            None => false,
        }
    }
}

/// Multi-asset live price state container.
pub struct PortfolioPriceState {
    pub btcusdt: LivePriceState,
    pub ethusdt: LivePriceState,
    pub solusdt: LivePriceState,
}

impl PortfolioPriceState {
    pub fn new() -> Self {
        Self {
            btcusdt: LivePriceState::new(),
            ethusdt: LivePriceState::new(),
            solusdt: LivePriceState::new(),
        }
    }

    pub fn update_trade(&self, ticker: &str, price: f64, exchange_ts_ms: i64) {
        let state = match ticker {
            "btcusdt" => &self.btcusdt,
            "ethusdt" => &self.ethusdt,
            "solusdt" => &self.solusdt,
            _ => return,
        };
        state.update_trade(price, exchange_ts_ms);
    }

    pub fn read(&self, ticker: &str) -> Option<LivePrice> {
        match ticker {
            "btcusdt" => self.btcusdt.read(),
            "ethusdt" => self.ethusdt.read(),
            "solusdt" => self.solusdt.read(),
            _ => None,
        }
    }
}
