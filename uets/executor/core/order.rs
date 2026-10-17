//! Exchange-agnostic order vocabulary.
//!
//! Binance spells order states `NEW`/`FILLED` in uppercase strings, Bybit uses
//! its own `OrderStatus` enum, Bitfinex numeric flags. All of that translation
//! belongs in `connectors/<venue>`; everything above it speaks the types below,
//! so the executor never matches on a venue string.

use std::fmt;

/// Which side of the book an order takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Buy => "buy",
            Self::Sell => "sell",
        }
    }

    /// The side that *closes* a position held on `self`.
    ///
    /// Every venue closes by trading the opposite side; encoding that once here
    /// is what keeps `ExecutionConnector::exit` venue-independent.
    pub fn opposite(self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }

    pub fn is_buy(self) -> bool {
        matches!(self, Self::Buy)
    }

    /// How much this side adds to the signed position size.
    pub fn signed(self, qty: f64) -> f64 {
        if self.is_buy() { qty } else { -qty }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the order is priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
}

impl OrderType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Market => "market",
            Self::Limit => "limit",
        }
    }
}

/// How long a resting order may wait for its opposite flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeInForce {
    Gtc,
    Ioc,
    Fok,
}

/// Where an order is in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

impl OrderStatus {
    /// Still capable of producing another fill.
    pub fn is_live(self) -> bool {
        matches!(self, Self::Pending | Self::Open | Self::PartiallyFilled)
    }

    /// No further change is possible.
    pub fn is_terminal(self) -> bool {
        !self.is_live()
    }
}

/// A client-side order id.
///
/// A `String`, not a `u64`: Binance only reveals its numeric `order_id` after
/// the round trip, and a client id is what survives a reconnect or a restart.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

/// What the executor hands a venue to open or close exposure.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderRequest {
    pub client_order_id: Option<String>,
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub order_type: OrderType,
    /// Required for [`OrderType::Limit`], ignored for [`OrderType::Market`].
    pub limit_price: Option<f64>,
    pub time_in_force: TimeInForce,
    /// Venue hint: refuse to *increase* exposure, only reduce it. Set by exits.
    pub reduce_only: bool,
}

impl OrderRequest {
    /// A market order — what enter/exit use.
    pub fn market(symbol: impl Into<String>, side: Side, qty: f64) -> Self {
        Self {
            client_order_id: None,
            symbol: symbol.into(),
            side,
            qty,
            order_type: OrderType::Market,
            limit_price: None,
            time_in_force: TimeInForce::Gtc,
            reduce_only: false,
        }
    }

    /// A resting limit order.
    ///
    /// The price is snapped to `tick_size` with `(price / tick).round() *
    /// tick` at construction, so an order built from a dusty float (e.g.
    /// `108613.20000000001`) can never reach the venue off-tick and every
    /// `==` comparison downstream sees the tick-grid value. Non-finite or
    /// non-positive prices are left for `validate()` to reject.
    pub fn limit(
        symbol: impl Into<String>,
        side: Side,
        qty: f64,
        price: f64,
        tick_size: f64,
    ) -> Self {
        let snapped =
            if tick_size.is_finite() && tick_size > 0.0 && price.is_finite() && price > 0.0 {
                (price / tick_size).round() * tick_size
            } else {
                price
            };
        Self {
            client_order_id: None,
            symbol: symbol.into(),
            side,
            qty,
            order_type: OrderType::Limit,
            limit_price: Some(snapped),
            time_in_force: TimeInForce::Gtc,
            reduce_only: false,
        }
    }

    /// Backwards-compatible constructor without a tick grid: prefers the
    /// canonical BTCUSDT cent tick (0.01). Prefer `limit(.., tick_size)`.
    pub fn limit_tick(symbol: impl Into<String>, side: Side, qty: f64, price: f64) -> Self {
        Self::limit(symbol, side, qty, price, 0.01)
    }

    /// Mark the request as an exit: the venue must not flip the position.
    pub fn reducing(mut self) -> Self {
        self.reduce_only = true;
        self
    }

    pub fn with_client_order_id(mut self, id: impl Into<String>) -> Self {
        self.client_order_id = Some(id.into());
        self
    }

    /// The notional, when a price is known (limits only; a market order is
    /// priced by the venue at fill time).
    pub fn notional(&self) -> Option<f64> {
        self.limit_price.map(|p| p * self.qty)
    }

    /// Reject a malformed request before it ever reaches a venue.
    ///
    /// Cheap, total, and venue-independent: a zero or `NaN` quantity costs a
    /// round trip and can leave a real order resting on the book.
    pub fn validate(&self) -> Result<(), OrderError> {
        if self.symbol.trim().is_empty() {
            return Err(OrderError::EmptySymbol);
        }
        if !self.qty.is_finite() || self.qty <= 0.0 {
            return Err(OrderError::NonPositiveQuantity { qty: self.qty });
        }
        match (self.order_type, self.limit_price) {
            (OrderType::Limit, None) => Err(OrderError::MissingLimitPrice),
            (OrderType::Limit, Some(price)) if !price.is_finite() || price <= 0.0 => {
                Err(OrderError::LimitPriceNotPositive { price })
            }
            _ => Ok(()),
        }
    }

    /// Venue-aware validation: as `validate()`, plus rejects a limit price
    /// that is not on the venue tick grid (within 1e-9). Call this at every
    /// order-routing boundary with the destination venue's `price_tick`, so
    /// an off-tick price fails locally instead of as INVALID_TICK_SIZE.
    pub fn validate_on_tick(&self, price_tick: Option<f64>) -> Result<(), OrderError> {
        self.validate()?;
        if self.order_type == OrderType::Limit {
            if let (Some(tick), Some(price)) = (price_tick, self.limit_price) {
                if tick.is_finite() && tick > 0.0 && price.is_finite() && price > 0.0 {
                    let snapped = (price / tick).round() * tick;
                    if (price - snapped).abs() > 1e-9 {
                        return Err(OrderError::OffTickPrice { price, tick });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Why a request is malformed. Caught locally, never by the venue.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderError {
    EmptySymbol,
    NonPositiveQuantity {
        qty: f64,
    },
    MissingLimitPrice,
    LimitPriceNotPositive {
        price: f64,
    },
    /// Limit price is not on the venue tick grid: reject locally instead of
    /// paying a round trip for the venue's INVALID_TICK_SIZE.
    OffTickPrice {
        price: f64,
        tick: f64,
    },
}

impl fmt::Display for OrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySymbol => write!(f, "order symbol must not be empty"),
            Self::NonPositiveQuantity { qty } => {
                write!(f, "order quantity must be finite and positive, got {qty}")
            }
            Self::MissingLimitPrice => write!(f, "a limit order needs a price"),
            Self::LimitPriceNotPositive { price } => {
                write!(f, "limit price must be positive, got {price}")
            }
            Self::OffTickPrice { price, tick } => {
                write!(f, "limit price {price} is off the {tick} tick grid")
            }
        }
    }
}

impl std::error::Error for OrderError {}

/// An order as tracked locally, fed by the venue's [`Fill`]s.
#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    pub id: OrderId,
    pub request: OrderRequest,
    pub status: OrderStatus,
    pub filled_qty: f64,
    pub avg_fill_price: Option<f64>,
    pub fee_paid: f64,
}

impl Order {
    pub fn new(id: impl Into<OrderId>, request: OrderRequest) -> Self {
        Self {
            id: id.into(),
            request,
            status: OrderStatus::Pending,
            filled_qty: 0.0,
            avg_fill_price: None,
            fee_paid: 0.0,
        }
    }

    /// Quantity still working on the book.
    pub fn remaining(&self) -> f64 {
        (self.request.qty - self.filled_qty).max(0.0)
    }

    /// Fold an execution into this order.
    ///
    /// The average fill price is a quantity-weighted mean of the executions,
    /// not of the last one: a partially filled order that completes in two
    /// prints has one meaningful entry price, and this is it.
    pub fn record_fill(&mut self, qty: f64, price: f64, fee: f64) {
        let prior_notional = self.avg_fill_price.unwrap_or(0.0) * self.filled_qty;
        self.filled_qty += qty;
        self.fee_paid += fee;
        self.avg_fill_price = if self.filled_qty > 0.0 {
            Some((prior_notional + price * qty) / self.filled_qty)
        } else {
            None
        };
        // `+ 1e-12`: a venue that rounds the final lot down must not leave the
        // order stuck at `PartiallyFilled` forever.
        self.status = if self.filled_qty + 1e-12 >= self.request.qty {
            OrderStatus::Filled
        } else {
            OrderStatus::PartiallyFilled
        };
    }
}

/// One execution report from the venue.
#[derive(Clone, Debug, PartialEq)]
pub struct Fill {
    pub order_id: OrderId,
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub price: f64,
    pub fee: f64,
}

impl Fill {
    pub fn new(
        order_id: impl Into<OrderId>,
        symbol: impl Into<String>,
        side: Side,
        qty: f64,
        price: f64,
        fee: f64,
    ) -> Self {
        Self {
            order_id: order_id.into(),
            symbol: symbol.into(),
            side,
            qty,
            price,
            fee,
        }
    }
}
