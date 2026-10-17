//! The contract every venue must satisfy.
//!
//! This file is the seam between the custom engine (`crate::core`) and the
//! outside world: everything about *communicating with a broker* — buying,
//! selling, marks, the account snapshot — is expressed here. It speaks
//! [`OrderRequest`], [`Fill`] and [`Holding`] from `core`'s vocabulary, and
//! never a wire format. `connectors/<venue>` supplies the transport and the
//! translation.
//!
//! Three narrow traits rather than one wide one, because a venue rarely
//! implements all three honestly:
//!
//! | trait | asks | who needs it |
//! |-------|------|--------------|
//! | [`MarketDataConnector`] | "what is this worth right now?" | risk, marks, pnl |
//! | [`ExecutionConnector`] | "open / close / cancel" | the executor loop |
//! | [`AccountConnector`] | "what does the venue say I hold?" | reconciliation |
//!
//! [`ExchangeConnector`] is the blanket-implemented union, so a component that
//! genuinely needs everything can say so in one bound without listing three.
//!
//! Note what is *not* here: any `impl` for a real exchange. The contract must
//! not know that `binance::errors::Error` exists, which is why every venue
//! error is carried as a [`ConnectorError`] built from a `Display` string.

use std::collections::BTreeMap;
use std::fmt;

use crate::core::order::{Fill, Order, OrderError, OrderId, OrderRequest};
use crate::core::position::{Holding, PositionBook};

/// What a venue declares about itself.
///
/// Anything the executor would otherwise hard-code per exchange: the settlement
/// asset, the fee schedule, and the tick/step its filters will accept. A market
/// order sent at an off-tick price is rejected by Binance, so rounding belongs
/// to the venue description rather than to a call site.
#[derive(Clone, Debug, PartialEq)]
pub struct VenueSpec {
    pub name: &'static str,
    /// Asset the account is margined and charged in, e.g. `"USDT"`.
    pub settlement: String,
    pub maker_fee: f64,
    pub taker_fee: f64,
    /// Minimum price increment, when the venue publishes one.
    pub price_tick: Option<f64>,
    /// Minimum quantity increment, when the venue publishes one.
    pub qty_step: Option<f64>,
}

impl VenueSpec {
    /// A spec with no filters and zero fees. The defaults are deliberately
    /// *empty* rather than guessed: an invented tick size silently corrupts
    /// every price it touches.
    pub fn new(name: &'static str, settlement: impl Into<String>) -> Self {
        Self {
            name,
            settlement: settlement.into(),
            maker_fee: 0.0,
            taker_fee: 0.0,
            price_tick: None,
            qty_step: None,
        }
    }

    pub fn with_fees(mut self, maker_fee: f64, taker_fee: f64) -> Self {
        self.maker_fee = maker_fee;
        self.taker_fee = taker_fee;
        self
    }

    pub fn with_filters(mut self, price_tick: f64, qty_step: f64) -> Self {
        self.price_tick = Some(price_tick);
        self.qty_step = Some(qty_step);
        self
    }

    /// Fee fraction for a fill, given whether it rested on the book.
    pub fn fee(&self, maker: bool) -> f64 {
        if maker {
            self.maker_fee
        } else {
            self.taker_fee
        }
    }

    /// Snap a price to the venue's tick. Pass through when unspecified.
    pub fn round_price(&self, price: f64) -> f64 {
        match self.price_tick {
            Some(step) if step > 0.0 && price.is_finite() => (price / step).round() * step,
            _ => price,
        }
    }

    /// Snap a quantity *down* to the venue's step, so a rounded-up lot is never
    /// submitted at a size the venue would reject.
    pub fn round_qty(&self, qty: f64) -> f64 {
        match self.qty_step {
            Some(step) if step > 0.0 && qty.is_finite() => (qty / step).floor() * step,
            _ => qty,
        }
    }
}

/// One asset's free/locked split on the venue.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct AssetBalance {
    pub free: f64,
    pub locked: f64,
}

impl AssetBalance {
    pub fn total(&self) -> f64 {
        self.free + self.locked
    }
}

/// The venue's own view of the account, at one instant.
///
/// Kept separate from `core::portfolio::Portfolio` rather than merged into it:
/// the venue knows what it actually holds, `Portfolio` knows what the strategy
/// *thinks* it holds, and a difference between the two is exactly the signal
/// worth showing on the dashboard. Reconciliation compares; it does not
/// overwrite one with the other.
#[derive(Clone, Debug, PartialEq)]
pub struct AccountSnapshot {
    pub venue: &'static str,
    /// Free and locked quantities per asset, keyed by the venue's spelling.
    pub balances: BTreeMap<String, AssetBalance>,
    pub can_trade: bool,
    pub maker_fee: f64,
    pub taker_fee: f64,
}

impl AccountSnapshot {
    pub fn new(venue: &'static str) -> Self {
        Self {
            venue,
            balances: BTreeMap::new(),
            can_trade: false,
            maker_fee: 0.0,
            taker_fee: 0.0,
        }
    }

    pub fn free(&self, asset: &str) -> f64 {
        self.balances.get(asset).map_or(0.0, |b| b.free)
    }

    pub fn total(&self, asset: &str) -> f64 {
        self.balances.get(asset).map_or(0.0, AssetBalance::total)
    }

    /// The quote asset of a pair by convention: everything after the base.
    /// `BTCUSDT` -> `USDT`. Without an exchange-info call this is the only split
    /// available, so it is stated rather than hidden.
    pub fn quote_asset(symbol: &str) -> &str {
        const QUOTES: [&str; 5] = ["USDT", "BUSD", "USDC", "BTC", "ETH"];
        for quote in QUOTES {
            if let Some(base) = symbol.strip_suffix(quote) {
                if !base.is_empty() {
                    return quote;
                }
            }
        }
        ""
    }

    /// The base asset of a pair: `BTCUSDT` -> `BTC`.
    pub fn base_asset(symbol: &str) -> &str {
        let quote = Self::quote_asset(symbol);
        &symbol[..symbol.len() - quote.len()]
    }
}

/// Why a connector call failed, in terms `core` can reason about.
#[derive(Clone, Debug, PartialEq)]
pub enum ConnectorError {
    /// The request never left, or the answer never arrived.
    Transport(String),
    /// The venue answered with something this adapter could not read.
    Decode(String),
    /// The venue answered, and said no (bad filters, unknown symbol, ...).
    Rejected(String),
    /// Throttled. Distinct from `Rejected`: backing off and retrying is correct
    /// here, and wrong there.
    RateLimited(String),
    /// The adapter does not implement that operation for this venue.
    Unsupported(String),
    /// Caught locally by `OrderRequest::validate`, before any round trip.
    Malformed(OrderError),
}

impl ConnectorError {
    pub fn transport(error: impl fmt::Display) -> Self {
        Self::Transport(error.to_string())
    }

    pub fn decode(error: impl fmt::Display) -> Self {
        Self::Decode(error.to_string())
    }

    pub fn rejected(error: impl fmt::Display) -> Self {
        Self::Rejected(error.to_string())
    }

    /// Safe to retry unchanged after a pause.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Transport(_) | Self::RateLimited(_))
    }
}

impl fmt::Display for ConnectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(m) => write!(f, "transport: {m}"),
            Self::Decode(m) => write!(f, "decode: {m}"),
            Self::Rejected(m) => write!(f, "rejected by venue: {m}"),
            Self::RateLimited(m) => write!(f, "rate limited: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::Malformed(e) => write!(f, "malformed request: {e}"),
        }
    }
}

impl std::error::Error for ConnectorError {}

impl From<OrderError> for ConnectorError {
    fn from(error: OrderError) -> Self {
        Self::Malformed(error)
    }
}
/// What every connector must say about its venue.
///
/// A supertrait rather than a method on each of the three, so there is exactly
/// one answer to "which exchange is this?" no matter which trait you hold.
pub trait Venue {
    fn venue(&self) -> VenueSpec;

    /// Short label for logs and the dashboard header.
    fn name(&self) -> &'static str {
        self.venue().name
    }
}

/// Prices. The only thing `core` needs to mark a book to market.
pub trait MarketDataConnector: Venue {
    /// Last traded price for `symbol`.
    fn mark_price(&self, symbol: &str) -> Result<f64, ConnectorError>;

    /// One mark per symbol, as risk and pnl consume them.
    ///
    /// Sequential rather than concurrent: an intraday strategy holds one or two
    /// symbols, and a fan-out would put the venue's rate limit at risk for a
    /// gain that does not exist at that size.
    fn marks(&self, symbols: &[String]) -> Result<BTreeMap<String, f64>, ConnectorError> {
        let mut marks = BTreeMap::new();
        for symbol in symbols {
            marks.insert(symbol.clone(), self.mark_price(symbol)?);
        }
        Ok(marks)
    }
}

/// Orders. Every method is `&mut self` even where a venue's client is
/// `&self`-only, because a connector that must track order state (sequencing,
/// open-order caches) cannot be asked to do so through a shared reference.
pub trait ExecutionConnector: Venue {
    /// Send the order. Returns whatever executed immediately.
    ///
    /// An empty `Vec` is a legitimate answer: a resting limit order has not
    /// filled yet, and pretending otherwise would invent a fill.
    fn submit(&mut self, request: &OrderRequest) -> Result<Vec<Fill>, ConnectorError>;

    fn cancel(&mut self, symbol: &str, order_id: &OrderId) -> Result<(), ConnectorError>;

    /// Orders the venue still considers live. `None` means every symbol.
    fn open_orders(&mut self, symbol: Option<&str>) -> Result<Vec<Order>, ConnectorError>;

    /// Close `holding` at market, on the opposing side.
    ///
    /// Venue-independent because closing is always the opposite side at the
    /// size that exists: that rule lives here once instead of in every adapter.
    /// A flat holding produces no order and no fills.
    fn exit(&mut self, symbol: &str, holding: &Holding) -> Result<Vec<Fill>, ConnectorError> {
        let side = match holding.side() {
            Some(side) => side.opposite(),
            None => return Ok(Vec::new()),
        };
        let request = OrderRequest::market(symbol, side, holding.size.abs()).reducing();
        self.submit(&request)
    }

    /// Kill switch: close every open holding in the book.
    ///
    /// Stops at the first failure and returns it, discarding the fills already
    /// collected. Continuing to send orders after a transport error risks
    /// trading a book this process no longer has an accurate view of.
    fn flatten(&mut self, book: &PositionBook) -> Result<Vec<Fill>, ConnectorError> {
        let open: Vec<(String, Holding)> = book
            .open_holdings()
            .map(|(symbol, holding)| (symbol.clone(), *holding))
            .collect();
        let mut fills = Vec::new();
        for (symbol, holding) in open {
            fills.extend(self.exit(&symbol, &holding)?);
        }
        Ok(fills)
    }
}

/// The venue's own account view, for reconciliation against `Portfolio`.
pub trait AccountConnector: Venue {
    fn snapshot(&self) -> Result<AccountSnapshot, ConnectorError>;
}

/// A connector that can do all three. Blanket-implemented, so an adapter gets
/// it by implementing the pieces it supports.
pub trait ExchangeConnector: MarketDataConnector + ExecutionConnector + AccountConnector {}

impl<T> ExchangeConnector for T where T: MarketDataConnector + ExecutionConnector + AccountConnector {}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::order::{Order, OrderId, Side};

    /// A connector with no network. The traits' *default* methods are the part
    /// `core` owns, so those are what these tests exercise.
    struct Stub {
        spec: VenueSpec,
        prices: BTreeMap<String, f64>,
        sent: Vec<OrderRequest>,
        fail_on: Option<String>,
    }

    impl Stub {
        fn new(prices: &[(&str, f64)]) -> Self {
            Self {
                spec: VenueSpec::new("stub", "USDT").with_fees(0.0002, 0.0004),
                prices: prices.iter().map(|(s, p)| (s.to_string(), *p)).collect(),
                sent: Vec::new(),
                fail_on: None,
            }
        }
    }

    impl Venue for Stub {
        fn venue(&self) -> VenueSpec {
            self.spec.clone()
        }
    }

    impl MarketDataConnector for Stub {
        fn mark_price(&self, symbol: &str) -> Result<f64, ConnectorError> {
            self.prices
                .get(symbol)
                .copied()
                .ok_or_else(|| ConnectorError::rejected(format!("no market for {symbol}")))
        }
    }

    impl ExecutionConnector for Stub {
        fn submit(&mut self, request: &OrderRequest) -> Result<Vec<Fill>, ConnectorError> {
            request.validate()?;
            if self.fail_on.as_deref() == Some(request.symbol.as_str()) {
                return Err(ConnectorError::transport("socket closed"));
            }
            self.sent.push(request.clone());
            let price = self.mark_price(&request.symbol)?;
            Ok(vec![Fill::new(
                OrderId::new("venue-1"),
                request.symbol.clone(),
                request.side,
                request.qty,
                price,
                self.spec.taker_fee * price * request.qty,
            )])
        }

        fn cancel(&mut self, _symbol: &str, _order_id: &OrderId) -> Result<(), ConnectorError> {
            Ok(())
        }

        fn open_orders(&mut self, _symbol: Option<&str>) -> Result<Vec<Order>, ConnectorError> {
            Ok(Vec::new())
        }
    }

    impl AccountConnector for Stub {
        fn snapshot(&self) -> Result<AccountSnapshot, ConnectorError> {
            Err(ConnectorError::Unsupported("stub has no account".into()))
        }
    }

    #[test]
    fn an_unfiltered_spec_never_rewrites_a_price() {
        let spec = VenueSpec::new("binance", "USDT");
        assert_eq!(spec.price_tick, None);
        assert_eq!(spec.round_price(64_123.4567), 64_123.4567);
        assert_eq!(spec.round_qty(0.0001234), 0.0001234);
        assert_eq!(spec.fee(true), 0.0, "unset fees stay unset");
    }

    #[test]
    fn filters_snap_prices_to_a_tick_and_sizes_down_to_a_step() {
        let spec = VenueSpec::new("binance", "USDT").with_filters(0.01, 0.001);
        assert!((spec.round_price(64_123.456) - 64_123.46).abs() < 1e-9);
        assert!(
            (spec.round_qty(0.0015) - 0.001).abs() < 1e-12,
            "never rounds up"
        );
        assert_eq!(
            spec.round_qty(0.0005),
            0.0,
            "below one step is no size at all"
        );
    }

    #[test]
    fn dusty_floats_snap_to_the_exact_cent_tick() {
        use crate::core::order::{OrderRequest, OrderType, Side, TimeInForce};
        // The needle-fix float dust: must become exactly 108613.2 on-grid.
        let req = OrderRequest::limit("BTCUSDT", Side::Buy, 0.001, 108613.20000000001, 0.01);
        let price = req.limit_price.unwrap();
        assert!((price - 108613.2).abs() < 1e-9, "got {price}");
        assert!(req.validate_on_tick(Some(0.01)).is_ok());
        // Off-tick limit (built raw, bypassing the snapping constructor)
        // fails locally, not as a venue INVALID_TICK_SIZE.
        let bad = OrderRequest {
            client_order_id: None,
            symbol: "BTCUSDT".to_string(),
            side: Side::Buy,
            qty: 0.001,
            order_type: OrderType::Limit,
            limit_price: Some(108613.2051),
            time_in_force: TimeInForce::Gtc,
            reduce_only: false,
        };
        assert!(bad.validate_on_tick(Some(0.01)).is_err());
    }

    #[test]
    fn a_pair_is_split_by_its_quote_convention() {
        assert_eq!(AccountSnapshot::base_asset("BTCUSDT"), "BTC");
        assert_eq!(AccountSnapshot::quote_asset("BTCUSDT"), "USDT");
        assert_eq!(AccountSnapshot::base_asset("ETHBTC"), "ETH");
        assert_eq!(AccountSnapshot::quote_asset("ETHBTC"), "BTC");
        // No known quote: the whole symbol is the base, and the quote is empty.
        assert_eq!(AccountSnapshot::quote_asset("WEIRD"), "");
        assert_eq!(AccountSnapshot::base_asset("WEIRD"), "WEIRD");
    }

    #[test]
    fn default_marks_queries_every_symbol() {
        let stub = Stub::new(&[("BTCUSDT", 100.0), ("ETHUSDT", 10.0)]);
        let marks = stub
            .marks(&["BTCUSDT".to_string(), "ETHUSDT".to_string()])
            .unwrap();
        assert_eq!(marks.len(), 2);
        assert_eq!(marks["ETHUSDT"], 10.0);
        assert_eq!(stub.name(), "stub");
        assert!(stub.marks(&["DOGEUSDT".to_string()]).is_err());
    }

    fn holding(size: f64, avg_entry: f64) -> Holding {
        Holding {
            size,
            avg_entry,
            ..Default::default()
        }
    }

    #[test]
    fn exit_trades_the_opposite_side_at_the_size_held() {
        let mut stub = Stub::new(&[("BTCUSDT", 100.0)]);
        let fills = stub.exit("BTCUSDT", &holding(2.0, 90.0)).unwrap();
        assert_eq!(stub.sent.len(), 1);
        assert_eq!(stub.sent[0].side, Side::Sell);
        assert_eq!(stub.sent[0].qty, 2.0);
        assert!(stub.sent[0].reduce_only, "an exit must never flip the book");
        assert_eq!(fills.len(), 1);

        // A short closes by buying.
        stub.exit("BTCUSDT", &holding(-3.0, 110.0)).unwrap();
        assert_eq!(stub.sent[1].side, Side::Buy);
        assert_eq!(stub.sent[1].qty, 3.0);
    }

    #[test]
    fn a_flat_holding_is_not_an_order() {
        let mut stub = Stub::new(&[("BTCUSDT", 100.0)]);
        let fills = stub.exit("BTCUSDT", &holding(0.0, 0.0)).unwrap();
        assert!(fills.is_empty());
        assert!(stub.sent.is_empty(), "no round trip for nothing to close");
    }

    #[test]
    fn flatten_closes_every_open_holding_and_stops_at_the_first_failure() {
        let mut book = PositionBook::new();
        book.apply(&Fill::new(
            OrderId::new("a"),
            "BTCUSDT",
            Side::Buy,
            1.0,
            100.0,
            0.0,
        ));
        book.apply(&Fill::new(
            OrderId::new("b"),
            "ETHUSDT",
            Side::Sell,
            2.0,
            10.0,
            0.0,
        ));

        let mut stub = Stub::new(&[("BTCUSDT", 100.0), ("ETHUSDT", 10.0)]);
        let fills = stub.flatten(&book).unwrap();
        assert_eq!(fills.len(), 2);
        assert_eq!(stub.sent.len(), 2, "BTCUSDT then ETHUSDT, both closed");

        let mut failing = Stub::new(&[("BTCUSDT", 100.0), ("ETHUSDT", 10.0)]);
        failing.fail_on = Some("ETHUSDT".to_string());
        let error = failing.flatten(&book).unwrap_err();
        assert!(error.is_retryable());
        assert_eq!(failing.sent.len(), 1, "stopped instead of trading blind");
    }

    #[test]
    fn only_transport_and_throttling_are_retryable() {
        assert!(ConnectorError::transport("timeout").is_retryable());
        assert!(ConnectorError::RateLimited("429".into()).is_retryable());
        assert!(!ConnectorError::rejected("MIN_NOTIONAL").is_retryable());
        assert!(!ConnectorError::Malformed(OrderError::EmptySymbol).is_retryable());
        assert_eq!(
            ConnectorError::from(OrderError::EmptySymbol).to_string(),
            "malformed request: order symbol must not be empty"
        );
    }
}
