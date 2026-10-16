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
use crate::types::{
    Capital, ExecutionEvent, FillEvent, Order, OrderId, OrderRequest, TradeIntent,
};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;

/// Live trading backend — routes through `uets_manager` to a real broker.
pub struct LiveTradeBackend {
    router: Option<Box<dyn LiveRouter>>,
    orders: BTreeMap<OrderId, Order>,
    events: Vec<ExecutionEvent>,
    book: Book,
    starting_capital: f64,
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
            ExecutionEvent::Fill(fill) => {
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
                    order.status = crate::types::OrderStatus::Rejected;
                    order.updated_at = now;
                }
                self.events.push(ExecutionEvent::OrderRejected {
                    order_id: id.clone(),
                    reason: err.to_string(),
                    at: now,
                });
                Err(BackendError::OrderRejected(err.to_string()))
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
    }

    fn is_connected(&self) -> bool {
        self.router.is_some()
    }

    fn position_qty(&self, strategy_id: &str, symbol: &str) -> Option<f64> {
        self.book.position_qty(strategy_id, symbol)
    }
}

// __END__


