//! Execution backend trait — the interface between strategy decisions and
//! the selected execution mode (live_test / live_trade / playback).
//!
//! The strategy never talks to an exchange directly. It emits `TradeIntent`s
//! through this trait, and the backend decides how to handle them.

use crate::types::{
    Capital, ExecutionEvent, FillEvent, Order, OrderId, OrderRequest, OrderSide, TradeIntent,
};
use chrono::{DateTime, Utc};

/// Error returned by an execution backend.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("order rejected: {0}")]
    OrderRejected(String),
    #[error("insufficient margin: required {required}, available {available}")]
    InsufficientMargin { required: f64, available: f64 },
    #[error("backend not connected")]
    NotConnected,
    #[error("duplicate order id: {0}")]
    DuplicateOrderId(String),
    #[error("internal backend error: {0}")]
    Internal(String),
}

/// The execution backend trait. Each mode (live_test, live_trade, playback)
/// implements this trait to handle trade intents its own way.
pub trait ExecutionBackend: Send + Sync {
    /// Submit a trade intent. Returns the order id assigned by the backend.
    fn submit_intent(&mut self, intent: TradeIntent) -> Result<OrderId, BackendError>;

    /// Submit a full order request (with barrier, size, etc.).
    fn submit_order(&mut self, request: OrderRequest) -> Result<OrderId, BackendError>;

    /// Cancel an order by id.
    fn cancel_order(&mut self, order_id: &OrderId) -> Result<(), BackendError>;

    /// Retrieve the current state of a tracked order.
    fn order_state(&self, order_id: &OrderId) -> Option<&Order>;

    /// All orders tracked by this backend.
    fn orders(&self) -> Vec<&Order>;

    /// Drain execution events since the last call.
    fn drain_events(&mut self) -> Vec<ExecutionEvent>;

    /// Current capital snapshot for a strategy instance.
    fn capital(&self, strategy_id: &str) -> Option<Capital>;

    /// Apply a fill event to the backend's internal position tracking.
    fn apply_fill(&mut self, fill: &FillEvent);

    /// Mark-to-market update: latest prices plus the update timestamp
    /// (timestamps drive barrier checks and stale-data tracking).
    fn on_market_update(
        &mut self,
        marks: &std::collections::BTreeMap<String, f64>,
        now: DateTime<Utc>,
    );

    /// Check if the backend is connected and healthy.
    fn is_connected(&self) -> bool;

    /// Acceleration factor for playback (1.0 = real-time, 10.0 = 10x, etc.).
    /// Returns 1.0 for non-playback backends.
    fn acceleration(&self) -> f64 {
        1.0
    }

    /// Open quantity for (strategy, symbol), signed (positive = long).
    fn position_qty(&self, strategy_id: &str, symbol: &str) -> Option<f64>;

    // ---- Strategy action interface (consistent across all modes) ----
    //
    // The strategy calls `long`/`short`/`exit`; the selected backend
    // decides how each action is handled (simulated vs venue-routed).

    /// Go long: submits a market order with the attached barrier.
    fn long(&mut self, ctx: &ActionContext) -> Result<OrderId, BackendError> {
        let mut request = OrderRequest::new(
            ctx.order_id.clone(),
            ctx.symbol.clone(),
            OrderSide::Buy,
            ctx.qty,
            ctx.strategy_id.clone(),
            ctx.asset_id.clone(),
        );
        request.barrier = ctx.barrier.clone();
        self.submit_order(request)
    }

    /// Go short: submits a market order with the attached barrier.
    fn short(&mut self, ctx: &ActionContext) -> Result<OrderId, BackendError> {
        let mut request = OrderRequest::new(
            ctx.order_id.clone(),
            ctx.symbol.clone(),
            OrderSide::Sell,
            ctx.qty,
            ctx.strategy_id.clone(),
            ctx.asset_id.clone(),
        );
        request.barrier = ctx.barrier.clone();
        self.submit_order(request)
    }

    /// Exit the open position for (strategy, symbol): opposite side, full
    /// quantity, no barrier. Errors when there is nothing to exit.
    fn exit(&mut self, ctx: &ActionContext) -> Result<OrderId, BackendError> {
        let Some(qty) = self.position_qty(&ctx.strategy_id, &ctx.symbol) else {
            return Err(BackendError::Internal(format!(
                "no open position to exit for {}/{}",
                ctx.strategy_id, ctx.symbol
            )));
        };
        if qty == 0.0 {
            return Err(BackendError::Internal(format!(
                "no open position to exit for {}/{}",
                ctx.strategy_id, ctx.symbol
            )));
        }
        let side = if qty > 0.0 {
            OrderSide::Sell
        } else {
            OrderSide::Buy
        };
        let request = OrderRequest::new(
            ctx.order_id.clone(),
            ctx.symbol.clone(),
            side,
            qty.abs(),
            ctx.strategy_id.clone(),
            ctx.asset_id.clone(),
        );
        self.submit_order(request)
    }
}

/// Context for the `long`/`short`/`exit` strategy actions.
#[derive(Clone, Debug)]
pub struct ActionContext {
    /// Caller-assigned order id (deduplication key).
    pub order_id: OrderId,
    /// Strategy instance identity, e.g. `ofmr@btcusdt`.
    pub strategy_id: String,
    /// Asset id, e.g. `btcusdt`.
    pub asset_id: String,
    /// Traded symbol, e.g. `BTCUSDT`.
    pub symbol: String,
    /// Order quantity (positive).
    pub qty: f64,
    /// Triple barrier attached to entries.
    pub barrier: Option<crate::triple_barrier::TripleBarrier>,
}

impl ActionContext {
    /// Entry/exit context with a fresh unique order id.
    pub fn new(
        strategy_id: impl Into<String>,
        asset_id: impl Into<String>,
        symbol: impl Into<String>,
        qty: f64,
    ) -> Self {
        Self {
            order_id: OrderId::new(format!("act-{}", uuid::Uuid::new_v4())),
            strategy_id: strategy_id.into(),
            asset_id: asset_id.into(),
            symbol: symbol.into(),
            qty,
            barrier: None,
        }
    }

    pub fn with_barrier(mut self, barrier: crate::triple_barrier::TripleBarrier) -> Self {
        self.barrier = Some(barrier);
        self
    }

    pub fn with_order_id(mut self, id: OrderId) -> Self {
        self.order_id = id;
        self
    }
}

/// Error returned when routing a live order through `uets_manager`.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    /// Live routing is not configured (no venue connector). Never falls back
    /// to paper execution — the caller must configure live execution.
    #[error("live routing not configured through uets_manager")]
    NotConfigured,
    /// Duplicate order id — protects against double submissions.
    #[error("duplicate order id: {0}")]
    DuplicateOrderId(String),
    /// Market data is older than the allowed threshold.
    #[error("stale market data: last tick {age_ms} ms old (threshold {threshold_ms} ms)")]
    StaleMarketData { age_ms: u64, threshold_ms: u64 },
    /// The venue rejected the order.
    #[error("order rejected: {0}")]
    Rejected(String),
    /// Transport/rate-limit failure at the venue.
    #[error("venue error: {0}")]
    Transport(String),
    /// Locally invalid request (failed validation before any round trip).
    #[error("invalid request: {0}")]
    Invalid(String),
}

/// Port through which live orders flow to `uets_manager`.
///
/// Implemented by `uets_manager::ManagerRouter`. A live backend holds this
/// router and has **no other path to a venue** — every live order is
/// validated (duplicate ids, stale market data, request sanity) and routed
/// by `uets_manager`, satisfying the execution-ownership requirement.
pub trait LiveRouter: Send + Sync {
    /// Note the latest market-data timestamp (drives stale-data checks).
    fn note_market_tick(&mut self, ts: DateTime<Utc>);

    /// Validate and route one order request. Returns the execution events
    /// the venue produced (acceptance, fills) so the backend can update
    /// local order state.
    fn route(
        &mut self,
        request: &OrderRequest,
        now: DateTime<Utc>,
    ) -> Result<Vec<ExecutionEvent>, RouterError>;

    /// Cancel a previously routed order.
    fn cancel(&mut self, order_id: &OrderId, symbol: &str) -> Result<(), RouterError>;

    /// Async execution events since the last call (cancel acks, etc.).
    fn drain_events(&mut self) -> Vec<ExecutionEvent>;
}