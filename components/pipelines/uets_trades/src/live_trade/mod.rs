//! Real trading using live market data and broker/exchange execution.
//!
//! Every order is routed through a [`crate::backend::LiveRouter`] — the port
//! implemented by `uets_manager::ManagerRouter`, which owns the venue
//! connector and all validation (duplicate ids, stale market data, request
//! sanity). This backend has **no other path to a venue** and never falls
//! back to paper execution: without a router, every submission fails with
//! [`BackendError::NotConnected`].

use crate::backend::{BackendError, ExecutionBackend, LiveRouter, RouterError};
use crate::sim::Book;
use crate::types::{Capital, ExecutionEvent, FillEvent, Order, OrderId, OrderRequest, TradeIntent};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// Live trading backend — routes through `uets_manager` to a real broker.
pub struct LiveTradeBackend {
    router: Option<Box<dyn LiveRouter>>,
    orders: BTreeMap<OrderId, Order>,
    events: Vec<ExecutionEvent>,
    book: Book,
    starting_capital: f64,
    barrier_exits_pending: BTreeSet<(String, String)>,
    barrier_order_seq: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{LiveRouter, RouterError};
    use crate::triple_barrier::TripleBarrier;
    use crate::types::{ExecutionEvent, OrderSide};
    use chrono::TimeZone;
    use std::sync::{Arc, Mutex};

    struct RecordingRouter(Arc<Mutex<Vec<OrderRequest>>>);
    impl LiveRouter for RecordingRouter {
        fn note_market_tick(&mut self, _ts: DateTime<Utc>) {}
        fn route(
            &mut self,
            request: &OrderRequest,
            now: DateTime<Utc>,
        ) -> Result<Vec<ExecutionEvent>, RouterError> {
            self.0.lock().unwrap().push(request.clone());
            Ok(vec![ExecutionEvent::OrderAccepted {
                order_id: request.id.clone(),
                symbol: request.symbol.clone(),
                at: now,
            }])
        }
        fn cancel(&mut self, _id: &OrderId, _symbol: &str) -> Result<(), RouterError> {
            Ok(())
        }
        fn drain_events(&mut self) -> Vec<ExecutionEvent> {
            Vec::new()
        }
    }

    struct UncertainRouter;
    impl LiveRouter for UncertainRouter {
        fn note_market_tick(&mut self, _ts: DateTime<Utc>) {}
        fn route(
            &mut self,
            _request: &OrderRequest,
            _now: DateTime<Utc>,
        ) -> Result<Vec<ExecutionEvent>, RouterError> {
            Err(RouterError::Transport(
                "connection lost after submit".into(),
            ))
        }
        fn cancel(&mut self, _id: &OrderId, _symbol: &str) -> Result<(), RouterError> {
            Err(RouterError::Transport("not reachable".into()))
        }
        fn drain_events(&mut self) -> Vec<ExecutionEvent> {
            Vec::new()
        }
    }

    #[test]
    fn live_triple_barrier_routes_exit_through_live_router() {
        let routed = Arc::new(Mutex::new(Vec::new()));
        let mut backend = LiveTradeBackend::with_router(Box::new(RecordingRouter(routed.clone())));
        let entered_at = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let mut marks = BTreeMap::new();
        marks.insert("BTCUSDT".to_string(), 100.0);
        backend.on_market_update(&marks, entered_at);
        let barrier = TripleBarrier::long(100.0, entered_at, 5.0, 5.0, 300);
        let request = OrderRequest::new(
            "entry",
            "BTCUSDT",
            OrderSide::Buy,
            1.0,
            "ofmr@btcusdt",
            "btcusdt",
        )
        .with_barrier(barrier);
        let order_id = backend.submit_order(request).unwrap();
        backend.apply_fill(&FillEvent {
            order_id,
            symbol: "BTCUSDT".into(),
            side: OrderSide::Buy,
            qty: 1.0,
            price: 100.0,
            fee: 0.04,
            executed_at: entered_at,
        });

        marks.insert("BTCUSDT".to_string(), 100.06);
        backend.on_market_update(&marks, entered_at + chrono::Duration::seconds(1));

        let routes = routed.lock().unwrap();
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[1].side, OrderSide::Sell);
        assert_eq!(routes[1].qty, 1.0);
        assert!(routes[1].barrier.is_none());
        assert!(routes[1].reduce_only);
    }

    #[test]
    fn lost_venue_response_is_tracked_as_uncertain_not_rejected() {
        let mut backend = LiveTradeBackend::with_router(Box::new(UncertainRouter));
        let request = OrderRequest::new(
            "uncertain",
            "BTCUSDT",
            OrderSide::Buy,
            1.0,
            "ofmr@btcusdt",
            "btcusdt",
        );
        assert!(backend.submit_order(request).is_err());
        assert_eq!(
            backend
                .order_state(&OrderId::new("uncertain"))
                .unwrap()
                .status,
            crate::types::OrderStatus::Uncertain
        );
        assert!(
            backend
                .drain_events()
                .iter()
                .any(|event| matches!(event, ExecutionEvent::OrderUncertain { .. }))
        );
    }
}

impl LiveTradeBackend {
    /// A live backend **without** a router: every order is refused
    /// (explicit `NotConnected` — never a silent paper fallback).
    pub fn new() -> Self {
        Self {
            router: None,
            orders: BTreeMap::new(),
            events: Vec::new(),
            book: Book::new(),
            starting_capital: 10_000.0,
            barrier_exits_pending: BTreeSet::new(),
            barrier_order_seq: 0,
        }
    }

    /// A live backend wired to `uets_manager` (required for live trading).
    pub fn with_router(router: Box<dyn LiveRouter>) -> Self {
        Self {
            router: Some(router),
            orders: BTreeMap::new(),
            events: Vec::new(),
            book: Book::new(),
            starting_capital: 10_000.0,
            barrier_exits_pending: BTreeSet::new(),
            barrier_order_seq: 0,
        }
    }

    /// Pre-allocate capital for a strategy instance.
    pub fn with_capital(mut self, strategy_id: impl Into<String>, starting: f64) -> Self {
        self.book.set_capital(strategy_id, starting);
        self
    }

    /// Mutable access to the router (market-tick notes from the pipeline).
    pub fn router_mut(&mut self) -> Option<&mut dyn LiveRouter> {
        self.router
            .as_mut()
            .map(|r| r.as_mut() as &mut dyn LiveRouter)
    }

    /// Apply one venue execution event to local order state.
    fn apply_event(&mut self, event: &ExecutionEvent, now: DateTime<Utc>) {
        match event {
            ExecutionEvent::OrderAccepted { order_id, .. } => {
                if let Some(order) = self.orders.get_mut(order_id) {
                    order.status = crate::types::OrderStatus::Accepted;
                    order.updated_at = now;
                }
            }
            ExecutionEvent::OrderPartiallyFilled {
                order_id,
                filled_qty,
                fill_price,
                fee,
                at,
            } => {
                if let Some(order) = self.orders.get_mut(order_id) {
                    order.record_fill(*filled_qty, *fill_price, *fee, *at);
                }
            }
            ExecutionEvent::OrderFilled {
                order_id,
                filled_qty,
                fill_price,
                fee,
                at,
            } => {
                if let Some(order) = self.orders.get_mut(order_id) {
                    order.record_fill(*filled_qty, *fill_price, *fee, *at);
                }
            }
            ExecutionEvent::OrderRejected { order_id, at, .. } => {
                if let Some(order) = self.orders.get_mut(order_id) {
                    order.status = crate::types::OrderStatus::Rejected;
                    order.updated_at = *at;
                }
            }
            ExecutionEvent::OrderCancelled { order_id, at } => {
                if let Some(order) = self.orders.get_mut(order_id) {
                    order.status = crate::types::OrderStatus::Cancelled;
                    order.updated_at = *at;
                }
            }
            ExecutionEvent::OrderUncertain { order_id, at, .. } => {
                if let Some(order) = self.orders.get_mut(order_id) {
                    order.status = crate::types::OrderStatus::Uncertain;
                    order.updated_at = *at;
                }
            }
            ExecutionEvent::Fill(fill) => {
                if let Some(order) = self.orders.get_mut(&fill.order_id) {
                    order.record_fill(fill.qty, fill.price, fill.fee, fill.executed_at);
                }
                self.apply_fill(fill);
            }
        }
    }
}

impl Default for LiveTradeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionBackend for LiveTradeBackend {
    fn submit_intent(&mut self, _intent: TradeIntent) -> Result<OrderId, BackendError> {
        Err(BackendError::Internal(
            "use long()/short()/exit() or submit_order with a full OrderRequest".into(),
        ))
    }

    /// Route the order through `uets_manager` (the only live path).
    fn submit_order(&mut self, request: OrderRequest) -> Result<OrderId, BackendError> {
        let now = Utc::now();
        let Some(router) = self.router.as_mut() else {
            // Explicit failure: no router means no venue access. There is
            // no paper fallback anywhere in this backend.
            return Err(BackendError::NotConnected);
        };
        if self.orders.contains_key(&request.id) {
            return Err(BackendError::DuplicateOrderId(request.id.0.clone()));
        }
        self.book
            .ensure_capital(&request.strategy_id, self.starting_capital);

        let id = request.id.clone();
        let mut order = Order::new(id.clone(), request.clone(), now);
        order.status = crate::types::OrderStatus::Submitted;
        self.orders.insert(id.clone(), order);

        match router.route(&request, now) {
            Ok(events) => {
                for event in &events {
                    self.apply_event(event, now);
                }
                self.events.extend(events);
                Ok(id)
            }
            Err(err) => {
                if let Some(order) = self.orders.get_mut(&id) {
                    order.status = if matches!(&err, RouterError::Transport(_)) {
                        crate::types::OrderStatus::Uncertain
                    } else {
                        crate::types::OrderStatus::Rejected
                    };
                    order.updated_at = now;
                }
                if matches!(&err, RouterError::Transport(_)) {
                    self.events.push(ExecutionEvent::OrderUncertain {
                        order_id: id.clone(),
                        reason: err.to_string(),
                        at: now,
                    });
                    Err(BackendError::Internal(format!(
                        "execution state uncertain for {id}: {err}"
                    )))
                } else {
                    self.events.push(ExecutionEvent::OrderRejected {
                        order_id: id.clone(),
                        reason: err.to_string(),
                        at: now,
                    });
                    Err(BackendError::OrderRejected(err.to_string()))
                }
            }
        }
    }

    fn cancel_order(&mut self, order_id: &OrderId) -> Result<(), BackendError> {
        let now = Utc::now();
        let Some(router) = self.router.as_mut() else {
            return Err(BackendError::NotConnected);
        };
        let Some(order) = self.orders.get(order_id) else {
            return Err(BackendError::Internal(format!("unknown order {order_id}")));
        };
        if order.status.is_terminal() {
            return Err(BackendError::Internal(format!(
                "order {order_id} already {:?}",
                order.status
            )));
        }
        let symbol = order.request.symbol.clone();
        router.cancel(order_id, &symbol).map_err(|e| match e {
            RouterError::NotConfigured => BackendError::NotConnected,
            RouterError::DuplicateOrderId(id) => BackendError::DuplicateOrderId(id),
            other => BackendError::OrderRejected(other.to_string()),
        })?;
        if let Some(order) = self.orders.get_mut(order_id) {
            order.status = crate::types::OrderStatus::Cancelled;
            order.updated_at = now;
        }
        self.events.push(ExecutionEvent::OrderCancelled {
            order_id: order_id.clone(),
            at: now,
        });
        Ok(())
    }

    fn order_state(&self, order_id: &OrderId) -> Option<&Order> {
        self.orders.get(order_id)
    }

    fn orders(&self) -> Vec<&Order> {
        self.orders.values().collect()
    }

    fn drain_events(&mut self) -> Vec<ExecutionEvent> {
        let mut drained = std::mem::take(&mut self.events);
        if let Some(router) = self.router.as_mut() {
            drained.extend(router.drain_events());
        }
        drained
    }

    fn capital(&self, strategy_id: &str) -> Option<Capital> {
        self.book.capital(strategy_id)
    }

    /// Apply an **externally sourced** fill (fills drained from this backend
    /// were already applied internally by `apply_event`).
    fn apply_fill(&mut self, fill: &FillEvent) {
        let (strategy, barrier) = match self.orders.get(&fill.order_id) {
            Some(order) => (
                order.request.strategy_id.clone(),
                order.request.barrier.clone(),
            ),
            None => (String::new(), None),
        };
        if let Err(err) = self.book.apply_fill(
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
        self.book.set_marks(marks);
        self.book.mark_to_market();
        // Feed the router so it can enforce stale-market-data checks.
        if let Some(router) = self.router.as_mut() {
            router.note_market_tick(now);
        }

        let active_positions: BTreeSet<(String, String)> = self
            .book
            .barrier_positions()
            .into_iter()
            .map(|(strategy, symbol, _, _, _)| (strategy, symbol))
            .collect();
        self.barrier_exits_pending
            .retain(|key| active_positions.contains(key));
        let triggered: Vec<_> = self
            .book
            .barrier_positions()
            .into_iter()
            .filter_map(|(strategy, symbol, qty, is_long, barrier)| {
                let mark = marks.get(&symbol).copied()?;
                let check = barrier.check(mark, now);
                (check.should_exit
                    && !self
                        .barrier_exits_pending
                        .contains(&(strategy.clone(), symbol.clone())))
                .then_some((strategy, symbol, qty, is_long, check.exit_price))
            })
            .collect();

        for (strategy_id, symbol, qty, is_long, _exit_price) in triggered {
            self.barrier_order_seq += 1;
            let key = (strategy_id.clone(), symbol.clone());
            let side = if is_long {
                crate::types::OrderSide::Sell
            } else {
                crate::types::OrderSide::Buy
            };
            let request = OrderRequest::new(
                OrderId::new(format!(
                    "live-barrier-{}-{}",
                    self.barrier_order_seq,
                    symbol.to_ascii_lowercase()
                )),
                symbol.clone(),
                side,
                qty.abs(),
                strategy_id,
                symbol.to_ascii_lowercase(),
            )
            .reducing();
            match self.submit_order(request) {
                Ok(_) => {
                    self.barrier_exits_pending.insert(key);
                }
                Err(err) => {
                    tracing::error!(%err, symbol, "live triple-barrier exit routing failed; position remains open")
                }
            }
        }
    }

    fn is_connected(&self) -> bool {
        self.router.is_some()
    }

    fn position_qty(&self, strategy_id: &str, symbol: &str) -> Option<f64> {
        self.book.position_qty(strategy_id, symbol)
    }
}

// __END__
