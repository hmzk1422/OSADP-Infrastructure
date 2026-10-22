//! Offline backtesting using historical or synthetic data.
//!
//! Replays market events through the same deterministic [`crate::sim::SimEngine`]
//! that paper trading uses, with a configurable acceleration factor (the
//! replay driver feeds marks at accelerated pace). Never submits real orders.

use crate::backend::{BackendError, ExecutionBackend};
use crate::sim::{SimConfig, SimEngine};
use crate::types::{Capital, ExecutionEvent, FillEvent, Order, OrderId, OrderRequest, TradeIntent};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

pub mod crypto_l2;
pub mod evaluation;

/// One ordered item in a deterministic market-data replay.
#[derive(Clone, Debug)]
pub struct MarketEvent {
    pub timestamp: DateTime<Utc>,
    pub marks: BTreeMap<String, f64>,
}

/// Playback backend — simulated execution against replayed events.
pub struct PlaybackBackend {
    engine: SimEngine,
    acceleration: f64,
    connected: bool,
}

impl PlaybackBackend {
    /// `acceleration`: replay speed multiplier (1.0 = real time, 10 = 10x).
    pub fn new(acceleration: f64) -> Self {
        Self::with_config(SimConfig::default(), acceleration)
    }

    pub fn with_config(config: SimConfig, acceleration: f64) -> Self {
        Self {
            engine: SimEngine::new(config),
            acceleration,
            connected: true,
        }
    }

    pub fn with_capital(strategy_id: impl Into<String>, starting: f64, acceleration: f64) -> Self {
        Self::with_config_capital(SimConfig::default(), strategy_id, starting, acceleration)
    }

    pub fn with_config_capital(
        config: SimConfig,
        strategy_id: impl Into<String>,
        starting: f64,
        acceleration: f64,
    ) -> Self {
        Self {
            engine: SimEngine::with_capital(config, strategy_id, starting),
            acceleration,
            connected: true,
        }
    }

    /// Shared read access to the accounting engine (reporting integration).
    pub fn engine(&self) -> &SimEngine {
        &self.engine
    }

    /// Replay every input event in order, applying its marks before invoking
    /// strategy processing. Events are paced by the configured acceleration;
    /// no event is skipped to meet a speed target.
    pub fn replay_events<F>(
        &mut self,
        events: impl IntoIterator<Item = MarketEvent>,
        process: F,
    ) -> Result<usize, BackendError>
    where
        F: FnMut(&mut Self, &MarketEvent) -> Result<(), BackendError>,
    {
        self.replay_events_inner(events, true, process)
    }

    /// Replay events as fast as possible, without wall-clock pacing.
    pub fn replay_events_unpaced<F>(
        &mut self,
        events: impl IntoIterator<Item = MarketEvent>,
        process: F,
    ) -> Result<usize, BackendError>
    where
        F: FnMut(&mut Self, &MarketEvent) -> Result<(), BackendError>,
    {
        self.replay_events_inner(events, false, process)
    }

    /// Replay a borrowed event slice without cloning event marks.
    pub fn replay_slice_unpaced<F>(
        &mut self,
        events: &[MarketEvent],
        mut process: F,
    ) -> Result<usize, BackendError>
    where
        F: FnMut(&mut Self, &MarketEvent) -> Result<(), BackendError>,
    {
        let mut previous_timestamp = None;
        for event in events {
            if previous_timestamp.is_some_and(|previous| event.timestamp < previous) {
                return Err(BackendError::Internal(
                    "playback events must be timestamp ordered".into(),
                ));
            }
            self.on_market_update(&event.marks, event.timestamp);
            process(self, event)?;
            previous_timestamp = Some(event.timestamp);
        }
        Ok(events.len())
    }

    fn replay_events_inner<F>(
        &mut self,
        events: impl IntoIterator<Item = MarketEvent>,
        paced: bool,
        mut process: F,
    ) -> Result<usize, BackendError>
    where
        F: FnMut(&mut Self, &MarketEvent) -> Result<(), BackendError>,
    {
        if !self.acceleration.is_finite() || self.acceleration <= 0.0 {
            return Err(BackendError::Internal(
                "playback acceleration must be finite and positive".into(),
            ));
        }
        let mut count = 0usize;
        let mut first_timestamp = None;
        let mut previous_timestamp = None;
        let wall_start = Instant::now();
        for event in events {
            if previous_timestamp.is_some_and(|previous| event.timestamp < previous) {
                return Err(BackendError::Internal(
                    "playback events must be timestamp ordered".into(),
                ));
            }
            let first = *first_timestamp.get_or_insert(event.timestamp);
            if paced {
                let elapsed_ms = event
                    .timestamp
                    .signed_duration_since(first)
                    .num_milliseconds()
                    .max(0) as u64;
                let target_ms = (elapsed_ms as f64 / self.acceleration).min(u64::MAX as f64) as u64;
                let wait = Duration::from_millis(target_ms).saturating_sub(wall_start.elapsed());
                if !wait.is_zero() {
                    std::thread::sleep(wait);
                }
            }
            self.on_market_update(&event.marks, event.timestamp);
            process(self, &event)?;
            previous_timestamp = Some(event.timestamp);
            count += 1;
        }
        Ok(count)
    }
}

impl Default for PlaybackBackend {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ExecutionBackend for PlaybackBackend {
    fn submit_intent(&mut self, _intent: TradeIntent) -> Result<OrderId, BackendError> {
        Err(BackendError::Internal(
            "use long()/short()/exit() or submit_order with a full OrderRequest".into(),
        ))
    }

    fn submit_order(&mut self, request: OrderRequest) -> Result<OrderId, BackendError> {
        self.engine.submit(request, Utc::now())
    }

    fn cancel_order(&mut self, order_id: &OrderId) -> Result<(), BackendError> {
        self.engine.cancel(order_id, Utc::now())
    }

    fn order_state(&self, order_id: &OrderId) -> Option<&Order> {
        self.engine.order_state(order_id)
    }

    fn orders(&self) -> Vec<&Order> {
        self.engine.orders()
    }

    fn drain_events(&mut self) -> Vec<ExecutionEvent> {
        self.engine.drain_events()
    }

    fn capital(&self, strategy_id: &str) -> Option<Capital> {
        self.engine.capital(strategy_id)
    }

    /// Apply an **externally sourced** fill (fills drained from this backend
    /// were already applied internally).
    fn apply_fill(&mut self, fill: &FillEvent) {
        let (strategy, barrier) = match self.engine.order_state(&fill.order_id) {
            Some(order) => (
                order.request.strategy_id.clone(),
                order.request.barrier.clone(),
            ),
            None => (String::new(), None),
        };
        if let Err(err) = self.engine.book.apply_fill(
            &strategy,
            &fill.symbol,
            fill.side,
            fill.qty,
            fill.price,
            fill.fee,
            barrier.as_ref(),
        ) {
            tracing::error!(%err, order = %fill.order_id, "apply_fill failed");
        }
    }

    fn on_market_update(&mut self, marks: &BTreeMap<String, f64>, now: DateTime<Utc>) {
        self.engine.on_market_update(marks, now);
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn acceleration(&self) -> f64 {
        self.acceleration
    }

    fn position_qty(&self, strategy_id: &str, symbol: &str) -> Option<f64> {
        self.engine.book.position_qty(strategy_id, symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ActionContext;
    use crate::triple_barrier::TripleBarrier;
    use crate::types::{OrderSide, OrderStatus, OrderType};
    use chrono::TimeZone;

    fn marks(price: f64) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        m.insert("BTCUSDT".to_string(), price);
        m
    }

    #[test]
    fn playback_never_sends_real_orders() {
        let mut backend = PlaybackBackend::new(10.0);
        assert!(backend.is_connected());
        let _ = backend.submit_order(OrderRequest::new(
            OrderId::new("test-1"),
            "BTCUSDT",
            OrderSide::Buy,
            1.0,
            "ofmr@btcusdt",
            "btcusdt",
        ));
        let events = backend.drain_events();
        assert_eq!(events.len(), 1);
        // Only simulated acceptance, no real fill.
        assert!(matches!(events[0], ExecutionEvent::OrderAccepted { .. }));
    }

    #[test]
    fn acceleration_factor_is_exposed() {
        let backend = PlaybackBackend::new(25.0);
        assert_eq!(backend.acceleration(), 25.0);
        let realtime = PlaybackBackend::default();
        assert_eq!(ExecutionBackend::acceleration(&realtime), 1.0);
    }

    #[test]
    fn barrier_take_profit_produces_exit_trade() {
        let mut backend = PlaybackBackend::new(10.0);
        let base = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();

        // Entry with a 5 bps TP / 5 bps SL / 300 s vertical barrier.
        let barrier = TripleBarrier::long(100.0, base, 5.0, 5.0, 300);
        let ctx =
            ActionContext::new("ofmr@btcusdt", "btcusdt", "BTCUSDT", 1.0).with_barrier(barrier);
        backend.on_market_update(&marks(100.0), base);
        backend.long(&ctx, OrderType::Market, None).unwrap();
        assert_eq!(backend.position_qty("ofmr@btcusdt", "BTCUSDT"), Some(1.0));

        // Mark crosses the take-profit barrier.
        let later = base + chrono::Duration::seconds(5);
        backend.on_market_update(&marks(100.6), later);

        assert_eq!(
            backend.position_qty("ofmr@btcusdt", "BTCUSDT"),
            None,
            "TP exit must close the position"
        );
        let capital = backend.capital("ofmr@btcusdt").unwrap();
        assert_eq!(capital.trade_count, 1);
        assert_eq!(capital.win_count, 1);
        assert_eq!(capital.r_trades, 1, "barrier risk makes R known");
        assert!(
            capital.expectancy_rr() > 0.9,
            "TP at ~+6 bps over 5 bps risk ≈ 1.1R, got {}",
            capital.expectancy_rr()
        );

        let events = backend.drain_events();
        let exit_fills = events
            .iter()
            .filter(
                |e| matches!(e, ExecutionEvent::Fill(f) if f.order_id.0.starts_with("barrier-")),
            )
            .count();
        assert_eq!(exit_fills, 1, "barrier exit emits exactly one fill");
    }

    #[test]
    fn barrier_vertical_exit_fires_on_time_without_price_move() {
        let mut backend = PlaybackBackend::new(10.0);
        let base = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let barrier = TripleBarrier::long(100.0, base, 5.0, 5.0, 10);
        let ctx =
            ActionContext::new("ofmr@btcusdt", "btcusdt", "BTCUSDT", 1.0).with_barrier(barrier);
        backend.on_market_update(&marks(100.0), base);
        backend.long(&ctx, OrderType::Market, None).unwrap();

        // 11 seconds later, price unchanged: vertical barrier fires.
        let late = base + chrono::Duration::seconds(11);
        backend.on_market_update(&marks(100.0), late);
        assert_eq!(backend.position_qty("ofmr@btcusdt", "BTCUSDT"), None);
        let orders = backend.orders();
        let exit = orders
            .iter()
            .find(|o| o.id.0.starts_with("barrier-"))
            .unwrap();
        assert_eq!(exit.status, OrderStatus::Filled);
    }

    #[test]
    fn accelerated_replay_preserves_every_event_and_rejects_time_reversal() {
        use super::MarketEvent;
        let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let events = [0, 1, 2].map(|second| MarketEvent {
            timestamp: start + chrono::Duration::seconds(second),
            marks: marks(100.0 + second as f64),
        });
        let mut backend = PlaybackBackend::new(100.0);
        let mut seen = Vec::new();
        let count = backend
            .replay_events(events.clone(), |_, event| {
                seen.push(event.timestamp);
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 3);
        assert_eq!(
            seen,
            events
                .iter()
                .map(|event| event.timestamp)
                .collect::<Vec<_>>()
        );
        assert_eq!(backend.engine().book.mark("BTCUSDT"), Some(102.0));

        let reversed = [events[1].clone(), events[0].clone()];
        assert!(backend.replay_events(reversed, |_, _| Ok(())).is_err());
    }
}
