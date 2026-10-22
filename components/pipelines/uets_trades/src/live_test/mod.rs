//! Paper trading using live market data.
//!
//! Never submits real orders: fills are simulated by the shared
//! [`crate::sim::SimEngine`] with configurable spread, slippage and fees —
//! the same engine `playback` uses, so paper and replay produce identical
//! results for identical inputs.

use crate::backend::{BackendError, ExecutionBackend};
use crate::sim::{SimConfig, SimEngine};
use crate::types::{Capital, ExecutionEvent, FillEvent, Order, OrderId, OrderRequest, TradeIntent};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;

/// Paper trading backend — simulated execution against live market data.
pub struct LiveTestBackend {
    engine: SimEngine,
    connected: bool,
}

impl LiveTestBackend {
    pub fn new() -> Self {
        Self::with_config(SimConfig::default())
    }

    /// Paper backend with explicit execution assumptions.
    pub fn with_config(config: SimConfig) -> Self {
        Self {
            engine: SimEngine::new(config),
            connected: true,
        }
    }

    /// Paper backend with pre-allocated strategy capital.
    pub fn with_capital(strategy_id: impl Into<String>, starting: f64) -> Self {
        Self::with_config_capital(SimConfig::default(), strategy_id, starting)
    }

    pub fn with_config_capital(
        config: SimConfig,
        strategy_id: impl Into<String>,
        starting: f64,
    ) -> Self {
        Self {
            engine: SimEngine::with_capital(config, strategy_id, starting),
            connected: true,
        }
    }

    /// Shared read access to the accounting engine (reporting integration).
    pub fn engine(&self) -> &SimEngine {
        &self.engine
    }
}

impl Default for LiveTestBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionBackend for LiveTestBackend {
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

    fn position_qty(&self, strategy_id: &str, symbol: &str) -> Option<f64> {
        self.engine.book.position_qty(strategy_id, symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ActionContext;
    use crate::types::{OrderSide, OrderStatus, OrderType};

    fn marks(price: f64) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        m.insert("BTCUSDT".to_string(), price);
        m
    }

    #[test]
    fn paper_backend_never_sends_real_orders() {
        let mut backend = LiveTestBackend::new();
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
        // Only an acceptance event, no fill (no mark known yet).
        assert!(matches!(events[0], ExecutionEvent::OrderAccepted { .. }));
    }

    #[test]
    fn market_order_fills_at_mark_with_slippage_and_fees() {
        let mut backend = LiveTestBackend::new();
        backend.on_market_update(&marks(100.0), Utc::now());
        let ctx = ActionContext::new("ofmr@btcusdt", "btcusdt", "BTCUSDT", 1.0);
        backend.long(&ctx, OrderType::Market, None).unwrap();

        let capital = backend.capital("ofmr@btcusdt").unwrap();
        // Buy executes at mark + (spread/2 + slippage) = +1.5 bps.
        let px = 100.0 * 1.00015;
        let fee = 0.0004 * px;
        assert!((capital.cash - (10_000.0 - px - fee)).abs() < 1e-9);
        assert!(capital.fees_paid > 0.0);
        assert!(capital.gross_exposure > 0.0);

        let events = backend.drain_events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ExecutionEvent::OrderFilled { .. }))
        );
        assert!(events.iter().any(|e| matches!(e, ExecutionEvent::Fill(_))));
    }

    #[test]
    fn partial_fill_ratio_completes_on_next_update() {
        let config = SimConfig { partial_fill_ratio: Some(0.5), ..Default::default() };
        let mut backend = LiveTestBackend::with_config(config);
        backend.on_market_update(&marks(100.0), Utc::now());
        let ctx = ActionContext::new("ofmr@btcusdt", "btcusdt", "BTCUSDT", 1.0);
        let id = backend.long(&ctx, OrderType::Market, None).unwrap();

        assert_eq!(backend.order_state(&id).unwrap().filled_qty, 0.5);
        let events = backend.drain_events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ExecutionEvent::OrderPartiallyFilled { .. }))
        );

        backend.on_market_update(&marks(100.0), Utc::now());
        assert_eq!(
            backend.order_state(&id).unwrap().status,
            OrderStatus::Filled
        );
    }

    #[test]
    fn cancel_before_fill_marks_cancelled_and_prevents_fill() {
        let mut backend = LiveTestBackend::new();
        let ctx = ActionContext::new("ofmr@btcusdt", "btcusdt", "BTCUSDT", 1.0);
        // No mark yet: order waits.
        let id = backend.long(&ctx, OrderType::Market, None).unwrap();
        backend.cancel_order(&id).unwrap();
        assert_eq!(
            backend.order_state(&id).unwrap().status,
            OrderStatus::Cancelled
        );

        // A later mark must not fill the cancelled order.
        backend.on_market_update(&marks(100.0), Utc::now());
        assert_eq!(backend.order_state(&id).unwrap().filled_qty, 0.0);
        let capital = backend.capital("ofmr@btcusdt").unwrap();
        assert_eq!(capital.cash, 10_000.0);
    }

    #[test]
    fn insufficient_cash_rejects_with_state_update() {
        let mut backend = LiveTestBackend::with_capital("ofmr@btcusdt", 50.0);
        backend.on_market_update(&marks(100.0), Utc::now());
        let ctx = ActionContext::new("ofmr@btcusdt", "btcusdt", "BTCUSDT", 1.0);
        assert!(
            backend.long(&ctx, OrderType::Market, None).is_err(),
            "1 BTC at ~100 must exceed 50 cash"
        );

        let events = backend.drain_events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ExecutionEvent::OrderRejected { .. }))
        );
        let order = backend.orders();
        assert_eq!(order[0].status, OrderStatus::Rejected);
    }

    #[test]
    fn exit_action_closes_the_open_position() {
        let mut backend = LiveTestBackend::new();
        backend.on_market_update(&marks(100.0), Utc::now());
        let ctx = ActionContext::new("ofmr@btcusdt", "btcusdt", "BTCUSDT", 1.0);
        backend.long(&ctx, OrderType::Market, None).unwrap();
        assert_eq!(backend.position_qty("ofmr@btcusdt", "BTCUSDT"), Some(1.0));

        let exit_ctx = ActionContext::new("ofmr@btcusdt", "btcusdt", "BTCUSDT", 1.0);
        backend.exit(&exit_ctx).unwrap();
        assert_eq!(
            backend.position_qty("ofmr@btcusdt", "BTCUSDT"),
            None,
            "closed position is removed from the book"
        );
        let capital = backend.capital("ofmr@btcusdt").unwrap();
        assert_eq!(capital.trade_count, 1, "round trip counts as one trade");
    }

    #[test]
    fn paper_and_playback_share_identical_accounting() {
        use crate::playback::PlaybackBackend;

        let config = SimConfig::default();
        let mut paper =
            LiveTestBackend::with_config_capital(config.clone(), "ofmr@btcusdt", 10_000.0);
        let mut playback =
            PlaybackBackend::with_config_capital(config, "ofmr@btcusdt", 10_000.0, 1.0);
        let start = Utc::now();
        let initial_marks = marks(100.0);
        paper.on_market_update(&initial_marks, start);
        playback.on_market_update(&initial_marks, start);
        let entry = OrderRequest::new(
            "entry",
            "BTCUSDT",
            OrderSide::Buy,
            1.0,
            "ofmr@btcusdt",
            "btcusdt",
        );
        paper.submit_order(entry.clone()).unwrap();
        playback.submit_order(entry).unwrap();
        let exit_marks = marks(101.0);
        paper.on_market_update(&exit_marks, start + chrono::Duration::seconds(1));
        playback.on_market_update(&exit_marks, start + chrono::Duration::seconds(1));
        let exit = OrderRequest::new(
            "exit",
            "BTCUSDT",
            OrderSide::Sell,
            1.0,
            "ofmr@btcusdt",
            "btcusdt",
        );
        paper.submit_order(exit.clone()).unwrap();
        playback.submit_order(exit).unwrap();

        let paper_capital = paper.capital("ofmr@btcusdt").unwrap();
        let playback_capital = playback.capital("ofmr@btcusdt").unwrap();
        assert_eq!(paper_capital.trade_count, playback_capital.trade_count);
        assert_eq!(paper_capital.win_count, playback_capital.win_count);
        assert!((paper_capital.cash - playback_capital.cash).abs() < 1e-9);
        assert!((paper_capital.equity - playback_capital.equity).abs() < 1e-9);
        assert!((paper_capital.realized_pnl - playback_capital.realized_pnl).abs() < 1e-9);
        assert!((paper_capital.fees_paid - playback_capital.fees_paid).abs() < 1e-9);
        assert_eq!(
            paper.position_qty("ofmr@btcusdt", "BTCUSDT"),
            playback.position_qty("ofmr@btcusdt", "BTCUSDT")
        );
    }
}
