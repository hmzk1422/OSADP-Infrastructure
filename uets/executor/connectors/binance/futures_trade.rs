//! Binance USD-M Futures execution over the Futures REST API.

use binance::account::OrderSide as BinanceSide;
use binance::api::Binance;
use binance::errors::{Error as BinanceError, ErrorKind as BinanceErrorKind};
use binance::futures::account::{
    CustomOrderRequest, FuturesAccount, OrderType as FuturesOrderType,
    TimeInForce as BinanceTimeInForce,
};
use binance::futures::model::Order as FuturesOrder;

use crate::core::order::{
    Fill, Order, OrderId, OrderRequest, OrderStatus, OrderType, Side, TimeInForce,
};
use crate::exchange::{ConnectorError, ExecutionConnector, Venue, VenueSpec};

/// USD-M Futures REST execution. This is intentionally separate from the Spot
/// adapter so the execution venue matches the `fstream.binance.com` market feed.
pub struct BinanceFuturesTrade {
    account: FuturesAccount,
    quote_asset: String,
}

impl BinanceFuturesTrade {
    pub fn new(api_key: Option<String>, secret_key: Option<String>) -> Self {
        let mut account: FuturesAccount = Binance::new(api_key, secret_key);
        account.set_testnet(true);
        Self {
            account,
            quote_asset: "USDT".to_string(),
        }
    }

    /// Explicitly opt into production endpoints; `new` is testnet-only.
    pub fn production(api_key: Option<String>, secret_key: Option<String>) -> Self {
        Self::new(api_key, secret_key).testnet(false)
    }

    /// Use Binance's USD-M Futures test environment unless explicitly disabled.
    pub fn testnet(mut self, on: bool) -> Self {
        self.account.set_testnet(on);
        self
    }

    pub fn with_quote_asset(mut self, quote_asset: impl Into<String>) -> Self {
        self.quote_asset = quote_asset.into().to_ascii_uppercase();
        self
    }
}

impl Venue for BinanceFuturesTrade {
    fn venue(&self) -> VenueSpec {
        VenueSpec::new("binance-usdm", self.quote_asset.clone())
    }
}

impl ExecutionConnector for BinanceFuturesTrade {
    fn submit(&mut self, request: &OrderRequest) -> Result<Vec<Fill>, ConnectorError> {
        request.validate()?;
        if request.order_type == OrderType::Limit {
            return Err(ConnectorError::Unsupported(
                "USD-M limit orders need asynchronous user-stream order tracking".into(),
            ));
        }
        if let Some(client_order_id) = &request.client_order_id {
            let valid = !client_order_id.is_empty()
                && client_order_id.len() <= 36
                && client_order_id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:/".contains(&byte));
            if !valid {
                return Err(ConnectorError::rejected(
                    "USD-M client order ID must be 1-36 supported ASCII characters",
                ));
            }
        }
        let order_type = match request.order_type {
            OrderType::Market => FuturesOrderType::Market,
            OrderType::Limit => FuturesOrderType::Limit,
        };
        let time_in_force = match (request.order_type, request.time_in_force) {
            (OrderType::Limit, TimeInForce::Gtc) => Some(BinanceTimeInForce::GTC),
            (OrderType::Limit, TimeInForce::Ioc) => Some(BinanceTimeInForce::IOC),
            (OrderType::Limit, TimeInForce::Fok) => Some(BinanceTimeInForce::FOK),
            (OrderType::Market, _) => None,
        };
        let response = self
            .account
            .custom_order(CustomOrderRequest {
                symbol: request.symbol.clone(),
                side: match request.side {
                    Side::Buy => BinanceSide::Buy,
                    Side::Sell => BinanceSide::Sell,
                },
                position_side: None,
                order_type,
                time_in_force,
                qty: Some(request.qty),
                reduce_only: Some(request.reduce_only),
                price: request.limit_price,
                stop_price: None,
                close_position: None,
                activation_price: None,
                callback_rate: None,
                working_type: None,
                price_protect: None,
                new_client_order_id: request.client_order_id.clone(),
            })
            .map_err(map_error)?;

        if response.executed_qty <= 0.0 {
            return Err(ConnectorError::Transport(format!(
                "USD-M order {} is not filled yet; asynchronous order updates are required",
                response.client_order_id
            )));
        }

        let history = self
            .account
            .get_user_trades(
                request.symbol.clone(),
                None::<u64>,
                None::<u64>,
                None::<u64>,
                Some(1000_u16),
            )
            .map_err(map_error)?;
        let mut fills = Vec::new();
        for trade in history
            .into_iter()
            .filter(|trade| trade.order_id == response.order_id)
        {
            if !trade
                .commission_asset
                .eq_ignore_ascii_case(&self.quote_asset)
            {
                return Err(ConnectorError::Decode(format!(
                    "USD-M commission asset {} is not the configured quote asset {}; fee conversion is required",
                    trade.commission_asset, self.quote_asset
                )));
            }
            let side = if trade.side.eq_ignore_ascii_case("BUY") {
                Side::Buy
            } else if trade.side.eq_ignore_ascii_case("SELL") {
                Side::Sell
            } else {
                return Err(ConnectorError::Decode(format!(
                    "unknown USD-M fill side {:?}",
                    trade.side
                )));
            };
            if !trade.qty.is_finite()
                || trade.qty <= 0.0
                || !trade.price.is_finite()
                || trade.price <= 0.0
                || !trade.commission.is_finite()
                || trade.commission < 0.0
            {
                return Err(ConnectorError::Decode(
                    "USD-M returned invalid execution data".into(),
                ));
            }
            fills.push(Fill::new(
                OrderId::new(response.client_order_id.clone()),
                trade.symbol,
                side,
                trade.qty,
                trade.price,
                trade.commission,
            ));
        }
        if fills.is_empty() {
            return Err(ConnectorError::Transport(format!(
                "USD-M order {} reports {} executed but fill history is not yet available; reconcile before retry",
                response.order_id, response.executed_qty
            )));
        }
        let filled_qty: f64 = fills.iter().map(|fill| fill.qty).sum();
        if (filled_qty - response.executed_qty).abs() > request.qty.max(1.0) * 1e-10 {
            return Err(ConnectorError::Transport(format!(
                "USD-M order {} reports {} executed but trade history sums to {}; reconcile before retry",
                response.order_id, response.executed_qty, filled_qty
            )));
        }
        Ok(fills)
    }

    fn cancel(&mut self, symbol: &str, order_id: &OrderId) -> Result<(), ConnectorError> {
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
        let venue_orders = match symbol {
            Some(symbol) => self.account.get_all_open_orders(symbol.to_string()),
            None => {
                return Err(ConnectorError::Unsupported(
                    "Binance USD-M open-order query requires a symbol".into(),
                ));
            }
        }
        .map_err(map_error)?;
        venue_orders.iter().map(order_from_venue).collect()
    }
}

fn order_from_venue(venue: &FuturesOrder) -> Result<Order, ConnectorError> {
    let side = match venue.side.to_ascii_uppercase().as_str() {
        "BUY" => Side::Buy,
        "SELL" => Side::Sell,
        other => {
            return Err(ConnectorError::Decode(format!(
                "unknown USD-M order side {other:?}"
            )));
        }
    };
    let order_type = match venue.order_type.to_ascii_uppercase().as_str() {
        "LIMIT" => OrderType::Limit,
        "MARKET" => OrderType::Market,
        other => {
            return Err(ConnectorError::Unsupported(format!(
                "open USD-M order type {other:?}"
            )));
        }
    };
    let request = match order_type {
        OrderType::Market => OrderRequest::market(venue.symbol.clone(), side, venue.orig_qty),
        OrderType::Limit => {
            let mut request =
                OrderRequest::limit(venue.symbol.clone(), side, venue.orig_qty, venue.price, 0.0);
            request.reduce_only = venue.reduce_only;
            request
        }
    };
    let mut order = Order::new(OrderId::new(venue.client_order_id.clone()), request);
    order.status = match venue.status.to_ascii_uppercase().as_str() {
        "NEW" => OrderStatus::Open,
        "PARTIALLY_FILLED" => OrderStatus::PartiallyFilled,
        "FILLED" => OrderStatus::Filled,
        "CANCELED" | "EXPIRED" => OrderStatus::Cancelled,
        "REJECTED" => OrderStatus::Rejected,
        other => {
            return Err(ConnectorError::Decode(format!(
                "unknown USD-M order status {other:?}"
            )));
        }
    };
    order.filled_qty = venue.executed_qty;
    order.avg_fill_price =
        (venue.executed_qty > 0.0 && venue.avg_price > 0.0).then_some(venue.avg_price);
    Ok(order)
}

fn map_error(error: BinanceError) -> ConnectorError {
    if let BinanceErrorKind::BinanceError(content) = error.kind() {
        return ConnectorError::Rejected(format!("[{}] {}", content.code, content.msg));
    }
    let message = error.to_string();
    match error.kind() {
        BinanceErrorKind::ReqError(_)
        | BinanceErrorKind::IoError(_)
        | BinanceErrorKind::Tungstenite(_)
        | BinanceErrorKind::TimestampError(_) => ConnectorError::Transport(message),
        _ => ConnectorError::Decode(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn venue_order() -> FuturesOrder {
        FuturesOrder {
            client_order_id: "client-1".into(),
            cum_qty: 0.25,
            cum_quote: 25.0,
            executed_qty: 0.25,
            order_id: 42,
            avg_price: 100.0,
            orig_qty: 0.5,
            price: 99.5,
            side: "SELL".into(),
            reduce_only: true,
            position_side: "BOTH".into(),
            status: "PARTIALLY_FILLED".into(),
            stop_price: 0.0,
            close_position: false,
            symbol: "BTCUSDT".into(),
            time_in_force: "GTC".into(),
            order_type: "LIMIT".into(),
            orig_type: "LIMIT".into(),
            activation_price: 0.0,
            price_rate: 0.0,
            update_time: 1_700_000_000_000,
            working_type: "CONTRACT_PRICE".into(),
            price_protect: false,
        }
    }

    #[test]
    fn open_order_mapping_preserves_usdm_state_and_reduce_only() {
        let order = order_from_venue(&venue_order()).unwrap();
        assert_eq!(order.id, OrderId::new("client-1"));
        assert_eq!(order.request.symbol, "BTCUSDT");
        assert_eq!(order.request.side, Side::Sell);
        assert_eq!(order.request.qty, 0.5);
        assert_eq!(order.request.limit_price, Some(99.5));
        assert!(order.request.reduce_only);
        assert_eq!(order.status, OrderStatus::PartiallyFilled);
        assert_eq!(order.filled_qty, 0.25);
        assert_eq!(order.avg_fill_price, Some(100.0));
    }

    #[test]
    fn unknown_usdm_status_is_a_decode_error_not_a_guessed_state() {
        let mut order = venue_order();
        order.status = "SOMETHING_NEW".into();
        assert!(matches!(
            order_from_venue(&order),
            Err(ConnectorError::Decode(_))
        ));
    }
}
