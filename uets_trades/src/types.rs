//! Core trading types shared by all execution modes.
//!
//! Order lifecycle: submitted → accepted → [partially filled]* → filled
//!                                                      → rejected / cancelled
//!
//! A `TradeIntent` is the strategy's request; an `Order` is the tracked
//! lifecycle object; a `FillEvent` is an execution report from the backend.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uets::core::order::Side;
use uets::core::strategy::Decision;

/// Strategy action: what the strategy wants to do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeIntent {
    Long,
    Short,
    Exit,
    Hold,
}

impl TradeIntent {
    pub fn from_decision(decision: Decision) -> Self {
        match decision {
            Decision::EnterLong => TradeIntent::Long,
            Decision::EnterShort => TradeIntent::Short,
            Decision::Exit => TradeIntent::Exit,
            Decision::Hold => TradeIntent::Hold,
        }
    }
}

impl fmt::Display for TradeIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TradeIntent::Long => write!(f, "LONG"),
            TradeIntent::Short => write!(f, "SHORT"),
            TradeIntent::Exit => write!(f, "EXIT"),
            TradeIntent::Hold => write!(f, "HOLD"),
        }
    }
}

/// Which side of the book an order takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn from_side(side: Side) -> Self {
        match side {
            Side::Buy => OrderSide::Buy,
            Side::Sell => OrderSide::Sell,
        }
    }

    pub fn to_side(self) -> Side {
        match self {
            OrderSide::Buy => Side::Buy,
            OrderSide::Sell => Side::Sell,
        }
    }

    pub fn opposite(self) -> Self {
        match self {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        }
    }
}

/// How the order is priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
}

/// How long a resting order may wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    Gtc,
    Ioc,
    Fok,
}

/// Order lifecycle status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Submitted,
    Accepted,
    PartiallyFilled,
    Filled,
    Rejected,
    Cancelled,
}

impl OrderStatus {
    pub fn is_live(self) -> bool {
        matches!(
            self,
            OrderStatus::Submitted | OrderStatus::Accepted | OrderStatus::PartiallyFilled
        )
    }

    pub fn is_terminal(self) -> bool {
        !self.is_live()
    }
}

/// A client-side order identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct OrderId(pub String);

impl OrderId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for OrderId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for OrderId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// What the strategy hands to the execution backend to open or close exposure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderRequest {
    pub id: OrderId,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub qty: f64,
    pub price: Option<f64>,
    pub time_in_force: TimeInForce,
    /// Strategy instance identity, e.g. `ofmr@btcusdt`.
    pub strategy_id: String,
    pub asset_id: String,
    /// Triple-barrier parameters attached to the entry order (OFMR).
    pub barrier: Option<crate::triple_barrier::TripleBarrier>,
}

impl OrderRequest {
    pub fn new(
        id: impl Into<OrderId>,
        symbol: impl Into<String>,
        side: OrderSide,
        qty: f64,
        strategy_id: impl Into<String>,
        asset_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            symbol: symbol.into(),
            side,
            order_type: OrderType::Market,
            qty,
            price: None,
            time_in_force: TimeInForce::Gtc,
            strategy_id: strategy_id.into(),
            asset_id: asset_id.into(),
            barrier: None,
        }
    }

    pub fn with_barrier(mut self, barrier: crate::triple_barrier::TripleBarrier) -> Self {
        self.barrier = Some(barrier);
        self
    }
}

/// An order as tracked locally, fed by the backend's fill events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Order {
    pub id: OrderId,
    pub request: OrderRequest,
    pub status: OrderStatus,
    pub filled_qty: f64,
    pub avg_fill_price: Option<f64>,
    pub fee_paid: f64,
    pub submitted_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Order {
    pub fn new(id: OrderId, request: OrderRequest, submitted_at: DateTime<Utc>) -> Self {
        Self {
            id: id.clone(),
            request,
            status: OrderStatus::Submitted,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: 0.0,
            submitted_at,
            updated_at: submitted_at,
        }
    }

    pub fn remaining(&self) -> f64 {
        (self.request.qty - self.filled_qty).max(0.0)
    }

    pub fn record_fill(&mut self, qty: f64, price: f64, fee: f64, now: DateTime<Utc>) {
        let prior_notional = self.avg_fill_price.unwrap_or(0.0) * self.filled_qty;
        self.filled_qty += qty;
        self.fee_paid += fee;
        self.avg_fill_price = if self.filled_qty > 0.0 {
            Some((prior_notional + price * qty) / self.filled_qty)
        } else {
            None
        };
        self.status = if self.filled_qty + 1e-12 >= self.request.qty {
            OrderStatus::Filled
        } else {
            OrderStatus::PartiallyFilled
        };
        self.updated_at = now;
    }
}

/// One execution report from the backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FillEvent {
    pub order_id: OrderId,
    pub symbol: String,
    pub side: OrderSide,
    pub qty: f64,
    pub price: f64,
    pub fee: f64,
    pub executed_at: DateTime<Utc>,
}

/// One execution event emitted by a backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExecutionEvent {
    OrderAccepted {
        order_id: OrderId,
        symbol: String,
        at: DateTime<Utc>,
    },
    OrderPartiallyFilled {
        order_id: OrderId,
        filled_qty: f64,
        fill_price: f64,
        fee: f64,
        at: DateTime<Utc>,
    },
    OrderFilled {
        order_id: OrderId,
        filled_qty: f64,
        fill_price: f64,
        fee: f64,
        at: DateTime<Utc>,
    },
    OrderRejected {
        order_id: OrderId,
        reason: String,
        at: DateTime<Utc>,
    },
    OrderCancelled {
        order_id: OrderId,
        at: DateTime<Utc>,
    },
    Fill(FillEvent),
}

/// Capital snapshot for a strategy instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capital {
    pub starting: f64,
    pub cash: f64,
    pub equity: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub fees_paid: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub margin_used: f64,
    pub available_margin: f64,
    pub effective_leverage: f64,
    /// Total trade count.
    pub trade_count: usize,
    /// Winning trade count.
    pub win_count: usize,
    /// Losing trade count.
    pub loss_count: usize,
    /// Sum of realised R-multiples (net pnl / initial risk) over closed
    /// trades whose entry risk was known (i.e. a triple barrier was set).
    pub r_multiple_sum: f64,
    /// Number of closed trades with a known risk (R) — denominator for
    /// expectancy in RR units.
    pub r_trades: usize,
}

impl Capital {
    /// A fresh allocation: everything flat at `starting`.
    pub fn new(starting: f64) -> Self {
        Self {
            starting,
            cash: starting,
            equity: starting,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            fees_paid: 0.0,
            gross_exposure: 0.0,
            net_exposure: 0.0,
            margin_used: 0.0,
            available_margin: starting,
            effective_leverage: 0.0,
            trade_count: 0,
            win_count: 0,
            loss_count: 0,
            r_multiple_sum: 0.0,
            r_trades: 0,
        }
    }

    pub fn trades(&self) -> usize {
        self.trade_count
    }

    pub fn wins(&self) -> usize {
        self.win_count
    }

    pub fn losses(&self) -> usize {
        self.loss_count
    }

    /// Expectancy per closed trade in RR units (R multiple of the average
    /// trade whose initial risk was known). 0.0 when no R is known.
    pub fn expectancy_rr(&self) -> f64 {
        if self.r_trades > 0 {
            self.r_multiple_sum / self.r_trades as f64
        } else {
            0.0
        }
    }
}

/// Decision intent wrapper: what the strategy decided plus context.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionIntent {
    pub strategy_id: String,
    pub asset_id: String,
    pub symbol: String,
    pub intent: TradeIntent,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub barrier: Option<crate::triple_barrier::TripleBarrier>,
    pub quantity: Option<f64>,
}