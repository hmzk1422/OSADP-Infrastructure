//! Binance spot: REST execution and marks.
//!
//! Implements `ExecutionConnector` and `MarketDataConnector` over the `binance`
//! crate's synchronous REST client. The streams under `websocket/` are a
//! separate concern; nothing here touches a socket.
//!
//! # Order identity
//!
//! The `binance` crate always generates its own `newClientOrderId` (a uuid) and
//! offers no way to supply one, so a `Fill` carries the *venue's* client id
//! rather than an id chosen here. That is accepted rather than worked around:
//! the venue's id is the only one a later cancel or reconciliation can be sure
//! refers to the same order, so the executor keys on what the venue reports.
//!
//! # Fees
//!
//! Commission is taken from each fill's `commission` field, but only counted
//! when it was charged in the pair's quote asset. With BNB fee payment enabled
//! Binance charges commission *in BNB*, which needs a BNB mark to convert; this
//! adapter reports `0.0` for such a fill instead of inventing a conversion.
//!
//! # `reduce_only` and `time_in_force`
//!
//! Spot has no reduce-only flag, so that hint is dropped here: a caller that
//! must not flip a position has to size the order itself, which is exactly what
//! `ExecutionConnector::exit` does. Time in force is GTC for every limit order
//! because the crate hard-codes it on that path — a limit that must not rest
//! would need `custom_order`, and pretending otherwise would be worse than the
//! limitation.

use binance::account::Account;
use binance::api::Binance;
use binance::errors::{Error as BinanceError, ErrorKind as BinanceErrorKind};
use binance::market::Market;
use binance::model::{Order as VenueOrder, Transaction};

use crate::core::order::{Fill, Order, OrderId, OrderRequest, OrderStatus, OrderType, Side};
use crate::exchange::{
    AccountSnapshot, ConnectorError, ExecutionConnector, MarketDataConnector, Venue, VenueSpec,
};

/// Binance spot over REST.
pub struct BinanceTrade {
    account: Account,
    market: Market,
    spec: VenueSpec,
}

impl BinanceTrade {
    /// Market data needs no credentials, so only the account client is keyed.
    pub fn new(api_key: Option<String>, secret_key: Option<String>) -> Self {
        Self {
            account: Binance::new(api_key, secret_key),
            market: Binance::new(None, None),
            spec: super::spot_spec(),
        }
    }

    /// Point both clients at testnet, where orders are validated by the real
    /// matching engine's rules but never filled against the book.
    pub fn testnet(mut self, on: bool) -> Self {
        self.account.set_testnet(on);
        self.market.set_testnet(on);
        self
    }

    /// Replace the declared schedule, e.g. with one an account snapshot read.
    pub fn with_spec(mut self, spec: VenueSpec) -> Self {
        self.spec = spec;
        self
    }
}

impl Venue for BinanceTrade {
    fn venue(&self) -> VenueSpec {
        self.spec.clone()
    }
}

impl MarketDataConnector for BinanceTrade {
    fn mark_price(&self, symbol: &str) -> Result<f64, ConnectorError> {
        let price = self
            .market
            .get_price(symbol.to_string())
            .map_err(map_error)?;
        Ok(price.price)
    }
}

impl ExecutionConnector for BinanceTrade {
    fn submit(&mut self, request: &OrderRequest) -> Result<Vec<Fill>, ConnectorError> {
        request.validate()?;
        let symbol = request.symbol.clone();
        // Snap to the venue's step *before* sending: an oversized lot is
        // rejected outright, and a rejection costs a round trip that the check
        // below makes unnecessary.
        let qty = self.spec.round_qty(request.qty);
        if qty <= 0.0 {
            return Err(ConnectorError::rejected(format!(
                "{qty} is below the venue quantity step"
            )));
        }
        let quote = AccountSnapshot::quote_asset(&symbol).to_string();
        let transaction = match (request.order_type, request.side) {
            (OrderType::Market, Side::Buy) => self.account.market_buy(symbol, qty),
            (OrderType::Market, Side::Sell) => self.account.market_sell(symbol, qty),
            (OrderType::Limit, Side::Buy) => {
                let price = self.spec.round_price(
                    request
                        .limit_price
                        .ok_or(crate::core::order::OrderError::MissingLimitPrice)?,
                );
                self.account.limit_buy(symbol, qty, price)
            }
            (OrderType::Limit, Side::Sell) => {
                let price = self.spec.round_price(
                    request
                        .limit_price
                        .ok_or(crate::core::order::OrderError::MissingLimitPrice)?,
                );
                self.account.limit_sell(symbol, qty, price)
            }
        }
        .map_err(map_error)?;
        fills_from_transaction(&transaction, &quote)
    }

    fn cancel(&mut self, symbol: &str, order_id: &OrderId) -> Result<(), ConnectorError> {
        // A numeric id is the venue's own; anything else is a client id. Both
        // are accepted because both are what a caller can hold after a restart.
        match order_id.as_str().parse::<u64>() {
            Ok(numeric) => self.account.cancel_order(symbol.to_string(), numeric),
            Err(_) => self
                .account
                .cancel_order_with_client_id(symbol.to_string(), order_id.as_str().to_string()),
        }
        .map(|_| ())
        .map_err(map_error)
    }

    fn open_orders(&mut self, symbol: Option<&str>) -> Result<Vec<Order>, ConnectorError> {
        let orders = match symbol {
            Some(symbol) => self.account.get_open_orders(symbol.to_string()),
            None => self.account.get_all_open_orders(),
        }
        .map_err(map_error)?;
        orders.iter().map(order_from_venue).collect()
    }
}
/// Binance's "too many requests" code, which arrives inside a 400 body rather
/// than as an HTTP 429 on some endpoints.
const RATE_LIMIT_CODE: i16 = -1003;

/// Translate a venue error into the one thing the caller needs to know:
/// whether to back off and retry, or to give up on this order.
pub fn map_error(error: BinanceError) -> ConnectorError {
    if let BinanceErrorKind::BinanceError(content) = error.kind() {
        let text = format!("[{}] {}", content.code, content.msg);
        return if content.code == RATE_LIMIT_CODE {
            ConnectorError::RateLimited(text)
        } else {
            ConnectorError::Rejected(text)
        };
    }
    if let BinanceErrorKind::Msg(msg) = error.kind() {
        return classify_message(msg);
    }
    let text = error.to_string();
    match error.kind() {
        BinanceErrorKind::ReqError(_)
        | BinanceErrorKind::IoError(_)
        | BinanceErrorKind::Tungstenite(_)
        | BinanceErrorKind::TimestampError(_) => ConnectorError::Transport(text),
        // A Json/ParseFloat/Url error means the answer arrived and could not be
        // read, which is a decoding defect on this side, not a quiet venue.
        _ => ConnectorError::Decode(text),
    }
}

/// Classify a status-level failure. The crate reports HTTP 5xx, 401 and 429 as
/// bare messages, so the distinction has to be recovered from the text.
fn classify_message(msg: &str) -> ConnectorError {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("429") || lower.contains("too many") {
        ConnectorError::RateLimited(msg.to_string())
    } else if lower.contains("received response") || lower.contains("unauthorized") {
        ConnectorError::Rejected(msg.to_string())
    } else {
        // "Internal Server Error" / "Service Unavailable": the venue is broken,
        // and the same request may well succeed shortly.
        ConnectorError::Transport(msg.to_string())
    }
}

/// A venue acknowledgement as the fills `core` understands.
pub fn fills_from_transaction(
    transaction: &Transaction,
    quote_asset: &str,
) -> Result<Vec<Fill>, ConnectorError> {
    let side = side_from_str(&transaction.side)?;
    let order_id = OrderId::new(transaction.client_order_id.clone());
    let Some(fills) = transaction.fills.as_ref().filter(|f| !f.is_empty()) else {
        // No executions: a resting limit order. `executed_qty` is zero, and
        // emitting a fill for it would invent a trade that has not happened.
        if transaction.executed_qty <= 0.0 {
            return Ok(Vec::new());
        }
        // A market order can report a fill without the array; the volume-weighted
        // price is then the only honest price available.
        let price = if transaction.price > 0.0 {
            transaction.price
        } else {
            transaction.cummulative_quote_qty / transaction.executed_qty
        };
        return Ok(vec![Fill::new(
            order_id,
            transaction.symbol.clone(),
            side,
            transaction.executed_qty,
            price,
            0.0,
        )]);
    };
    Ok(fills
        .iter()
        .map(|fill| {
            Fill::new(
                order_id.clone(),
                transaction.symbol.clone(),
                side,
                fill.qty,
                fill.price,
                fee_in_quote(fill.commission, &fill.commission_asset, quote_asset),
            )
        })
        .collect())
}

/// Commission, counted only when it was charged in the quote asset.
///
/// A discount-schedule fill is commissioned in BNB, and converting that needs
/// the BNB mark. Reporting `0.0` (and so slightly overstating pnl) is a stated
/// gap; converting at a guessed rate would be an unstated error.
fn fee_in_quote(commission: f64, commission_asset: &str, quote_asset: &str) -> f64 {
    if commission_asset.eq_ignore_ascii_case(quote_asset) {
        commission
    } else {
        0.0
    }
}

/// `"BUY"` / `"SELL"` as a [`Side`].
pub fn side_from_str(side: &str) -> Result<Side, ConnectorError> {
    match side.to_ascii_uppercase().as_str() {
        "BUY" => Ok(Side::Buy),
        "SELL" => Ok(Side::Sell),
        other => Err(ConnectorError::decode(format!(
            "unknown order side {other:?}"
        ))),
    }
}

/// The venue's order status as a local [`OrderStatus`].
pub fn status_from_str(status: &str) -> Result<OrderStatus, ConnectorError> {
    match status.to_ascii_uppercase().as_str() {
        "NEW" => Ok(OrderStatus::Open),
        "PENDING_CANCEL" => Ok(OrderStatus::Pending),
        "PARTIALLY_FILLED" => Ok(OrderStatus::PartiallyFilled),
        "FILLED" => Ok(OrderStatus::Filled),
        "CANCELED" | "EXPIRED" => Ok(OrderStatus::Cancelled),
        "REJECTED" => Ok(OrderStatus::Rejected),
        other => Err(ConnectorError::decode(format!(
            "unknown order status {other:?}"
        ))),
    }
}

/// A quantity the venue reports as a string.
///
/// Unparseable text is `0.0` rather than an error: the venue sends an empty
/// string for "no quantity", and failing on it would turn every unfilled order
/// into a spurious failure.
fn qty_from_str(text: &str) -> f64 {
    text.parse().unwrap_or(0.0)
}

/// One of the venue's open orders as a locally tracked [`Order`].
pub fn order_from_venue(venue: &VenueOrder) -> Result<Order, ConnectorError> {
    let side = side_from_str(&venue.side)?;
    let order_type = match venue.type_name.to_ascii_uppercase().as_str() {
        "LIMIT" | "LIMIT_MAKER" | "STOP_LOSS_LIMIT" | "TAKE_PROFIT_LIMIT" => OrderType::Limit,
        "MARKET" => OrderType::Market,
        other => {
            return Err(ConnectorError::decode(format!(
                "unsupported order type {other:?}"
            )));
        }
    };
    let qty = qty_from_str(&venue.orig_qty);
    let request = match order_type {
        OrderType::Limit => OrderRequest::limit(venue.symbol.clone(), side, qty, venue.price),
        OrderType::Market => OrderRequest::market(venue.symbol.clone(), side, qty),
    };
    let mut order = Order::new(OrderId::new(venue.client_order_id.clone()), request);
    order.status = status_from_str(&venue.status)?;
    order.filled_qty = qty_from_str(&venue.executed_qty);
    // The venue leaves `price` at zero for market orders; an average of zero
    // would be a wrong entry price on every such order, so it stays absent.
    order.avg_fill_price = (order.filled_qty > 0.0 && venue.price > 0.0).then_some(venue.price);
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use binance::errors::BinanceContentError;
    use binance::model::FillInfo;

    /// A venue acknowledgement. Every field is set explicitly: a defaulted
    /// struct would hide the difference between "zero" and "the venue sent 0".
    fn ack(side: &str, executed_qty: f64, fills: Option<Vec<FillInfo>>) -> Transaction {
        Transaction {
            symbol: "BTCUSDT".to_string(),
            order_id: 42,
            order_list_id: None,
            client_order_id: "uuid-1".to_string(),
            transact_time: 1_700_000_000_000,
            price: 0.0,
            orig_qty: 1.0,
            executed_qty,
            cummulative_quote_qty: 100.0 * executed_qty,
            stop_price: 0.0,
            status: "FILLED".to_string(),
            time_in_force: "GTC".to_string(),
            type_name: "MARKET".to_string(),
            side: side.to_string(),
            fills,
        }
    }

    fn fill_info(price: f64, qty: f64, commission: f64, asset: &str) -> FillInfo {
        FillInfo {
            price,
            qty,
            commission,
            commission_asset: asset.to_string(),
            trade_id: None,
        }
    }

    #[test]
    fn an_acknowledgement_with_executions_becomes_one_fill_per_print() {
        let transaction = ack(
            "BUY",
            1.0,
            Some(vec![
                fill_info(100.0, 0.6, 0.06, "USDT"),
                fill_info(101.0, 0.4, 0.04, "USDT"),
            ]),
        );
        let fills = fills_from_transaction(&transaction, "USDT").unwrap();
        assert_eq!(fills.len(), 2, "one Fill per print, not one per order");
        assert_eq!(fills[0].side, Side::Buy);
        assert_eq!(fills[0].order_id, OrderId::new("uuid-1"));
        assert_eq!(fills[1].qty, 0.4);
        assert_eq!(fills[1].price, 101.0);
        assert_eq!(fills[1].fee, 0.04);
        assert_eq!(fills[1].symbol, "BTCUSDT");
    }

    #[test]
    fn commission_paid_in_bnb_is_not_counted_as_quote() {
        let transaction = ack(
            "SELL",
            1.0,
            Some(vec![fill_info(100.0, 1.0, 0.0004, "BNB")]),
        );
        let fills = fills_from_transaction(&transaction, "USDT").unwrap();
        assert_eq!(
            fills[0].fee, 0.0,
            "a BNB fee needs the BNB mark, which this adapter does not fetch"
        );

        let transaction = ack("SELL", 1.0, Some(vec![fill_info(100.0, 1.0, 0.1, "usdt")]));
        let fills = fills_from_transaction(&transaction, "USDT").unwrap();
        assert_eq!(
            fills[0].fee, 0.1,
            "asset comparison ignores case: the venue is inconsistent about it"
        );
    }

    #[test]
    fn a_resting_limit_acknowledgement_produces_no_fills() {
        let mut transaction = ack("BUY", 0.0, None);
        transaction.status = "NEW".to_string();
        assert!(
            fills_from_transaction(&transaction, "USDT")
                .unwrap()
                .is_empty(),
            "an order waiting on the book has traded nothing"
        );

        // A fill reported without the array still counts, priced by the
        // volume-weighted total the venue reports.
        let mut transaction = ack("BUY", 2.0, None);
        transaction.price = 0.0;
        transaction.cummulative_quote_qty = 204.0;
        let fills = fills_from_transaction(&transaction, "USDT").unwrap();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].qty, 2.0);
        assert_eq!(fills[0].price, 102.0, "204 / 2");
    }
    #[test]
    fn an_open_order_is_tracked_with_its_own_quantity_and_status() {
        let venue = VenueOrder {
            symbol: "ETHUSDT".to_string(),
            order_id: 7,
            order_list_id: -1,
            client_order_id: "uuid-2".to_string(),
            price: 2000.0,
            orig_qty: "0.500".to_string(),
            executed_qty: "0.250".to_string(),
            cummulative_quote_qty: "500.0".to_string(),
            status: "PARTIALLY_FILLED".to_string(),
            time_in_force: "GTC".to_string(),
            type_name: "LIMIT".to_string(),
            side: "sell".to_string(),
            stop_price: 0.0,
            iceberg_qty: "0".to_string(),
            time: 0,
            update_time: 0,
            is_working: true,
            orig_quote_order_qty: "0".to_string(),
        };
        let order = order_from_venue(&venue).unwrap();
        assert_eq!(order.status, OrderStatus::PartiallyFilled);
        assert_eq!(order.filled_qty, 0.25);
        assert_eq!(order.avg_fill_price, Some(2000.0));
        assert_eq!(order.request.side, Side::Sell);
        assert_eq!(order.request.qty, 0.5);
        assert_eq!(order.request.order_type, OrderType::Limit);
        assert_eq!(order.remaining(), 0.25, "0.5 ordered, 0.25 filled");

        // A market order has no resting price; reporting 0.0 as an average
        // fill price would poison any entry price that reads it.
        let mut market = venue.clone();
        market.type_name = "MARKET".to_string();
        market.price = 0.0;
        market.status = "NEW".to_string();
        let order = order_from_venue(&market).unwrap();
        assert_eq!(order.request.order_type, OrderType::Market);
        assert_eq!(order.avg_fill_price, None);
    }

    #[test]
    fn an_unreadable_side_or_status_is_a_decoding_defect() {
        assert!(matches!(
            side_from_str("HOLD"),
            Err(ConnectorError::Decode(_))
        ));
        assert!(matches!(
            status_from_str("MAYBE"),
            Err(ConnectorError::Decode(_))
        ));
        assert_eq!(status_from_str("canceled").unwrap(), OrderStatus::Cancelled);
        assert_eq!(status_from_str("NEW").unwrap(), OrderStatus::Open);
    }

    #[test]
    fn venue_errors_classify_by_what_the_caller_should_do() {
        let rate_limited: BinanceError = BinanceErrorKind::BinanceError(BinanceContentError {
            code: RATE_LIMIT_CODE,
            msg: "Too many requests".to_string(),
        })
        .into();
        assert!(map_error(rate_limited).is_retryable());

        let rejected: BinanceError = BinanceErrorKind::BinanceError(BinanceContentError {
            code: -2010,
            msg: "Account has insufficient balance.".to_string(),
        })
        .into();
        assert!(matches!(map_error(rejected), ConnectorError::Rejected(_)));

        let unauthorized: BinanceError = BinanceErrorKind::Msg("Unauthorized".into()).into();
        assert!(matches!(
            map_error(unauthorized),
            ConnectorError::Rejected(_)
        ));

        let throttled: BinanceError = BinanceErrorKind::Msg("429 Too Many".into()).into();
        assert!(map_error(throttled).is_retryable());

        let unavailable: BinanceError = BinanceErrorKind::Msg("Service Unavailable".into()).into();
        assert!(
            map_error(unavailable).is_retryable(),
            "a broken venue is worth retrying; a rejected order is not"
        );

        let unreadable: BinanceError =
            BinanceErrorKind::ParseFloatError("x".parse::<f64>().unwrap_err()).into();
        assert!(matches!(map_error(unreadable), ConnectorError::Decode(_)));
    }
}
