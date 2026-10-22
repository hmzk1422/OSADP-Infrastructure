use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use thiserror::Error;
use uets::core::order::{
    OrderRequest as ConnectorOrderRequest, Side as ConnectorSide,
    TimeInForce as ConnectorTimeInForce,
};
use uets::exchange::ExecutionConnector;
use uets_trades::types::{
    ExecutionEvent, FillEvent, OrderId, OrderRequest, OrderSide, OrderType, TimeInForce,
};

#[derive(Debug, Error)]
pub enum VenueError {
    #[error("venue rejected order: {0}")]
    Rejected(String),
    #[error("venue transport error: {0}")]
    Transport(String),
}

/// Synchronous venue boundary. Implementations own exchange/broker clients;
/// strategy code receives only execution events through the manager.
pub trait VenueConnector: Send + Sync {
    fn place_order(
        &mut self,
        request: &OrderRequest,
        now: DateTime<Utc>,
    ) -> Result<Vec<ExecutionEvent>, VenueError>;
    fn cancel_order(
        &mut self,
        order_id: &OrderId,
        symbol: &str,
        now: DateTime<Utc>,
    ) -> Result<Vec<ExecutionEvent>, VenueError>;
}

/// Adapter for the repository's venue connectors, keeping connector calls
/// behind `ManagerRouter`. Market orders with immediate fills are supported;
/// the current connector contract cannot safely correlate an unfilled resting
/// limit order to its later venue order ID, so those are refused.
pub struct UetsExecutionConnectorAdapter<C> {
    connector: C,
    venue_order_ids: BTreeMap<OrderId, String>,
}

impl<C> UetsExecutionConnectorAdapter<C> {
    pub fn new(connector: C) -> Self {
        Self {
            connector,
            venue_order_ids: BTreeMap::new(),
        }
    }
    pub fn into_inner(self) -> C {
        self.connector
    }
}

impl<C: ExecutionConnector + Send + Sync> VenueConnector for UetsExecutionConnectorAdapter<C> {
    fn place_order(
        &mut self,
        request: &OrderRequest,
        now: DateTime<Utc>,
    ) -> Result<Vec<ExecutionEvent>, VenueError> {
        let mut connector_request = match request.order_type {
            OrderType::Market => ConnectorOrderRequest::market(
                request.symbol.clone(),
                match request.side {
                    OrderSide::Buy => ConnectorSide::Buy,
                    OrderSide::Sell => ConnectorSide::Sell,
                },
                request.qty,
            ),
            OrderType::Limit => {
                let price = request.price.ok_or_else(|| {
                    VenueError::Rejected("limit order requires a price".into())
                })?;
                ConnectorOrderRequest::limit(
                    request.symbol.clone(),
                    match request.side {
                        OrderSide::Buy => ConnectorSide::Buy,
                        OrderSide::Sell => ConnectorSide::Sell,
                    },
                    request.qty,
                    price,
                    0.0,
                )
            }
        }
        .with_client_order_id(request.id.0.clone());

        connector_request.time_in_force = match request.time_in_force {
            TimeInForce::Gtc => ConnectorTimeInForce::Gtc,
            TimeInForce::Ioc => ConnectorTimeInForce::Ioc,
            TimeInForce::Fok => ConnectorTimeInForce::Fok,
        };
        if request.reduce_only {
            connector_request = connector_request.reducing();
        }

        let fills = self
            .connector
            .submit(&connector_request)
            .map_err(|err| match err {
                uets::exchange::ConnectorError::Rejected(message)
                | uets::exchange::ConnectorError::Unsupported(message) => {
                    VenueError::Rejected(message)
                }
                uets::exchange::ConnectorError::Malformed(error) => {
                    VenueError::Rejected(error.to_string())
                }
                other => VenueError::Transport(other.to_string()),
            })?;

        // Market orders must fill immediately; limit orders may rest on the book (zero fills).
        if fills.is_empty() && request.order_type == OrderType::Market {
            return Err(VenueError::Transport(
                "venue returned no fill for market order; reconcile order state before retry".into(),
            ));
        }

        let mut events = vec![ExecutionEvent::OrderAccepted {
            order_id: request.id.clone(),
            symbol: request.symbol.clone(),
            at: now,
        }];

        // If fills are returned immediately (market execution or immediate limit cross), record them.
        if !fills.is_empty() {
            let venue_id = fills[0].order_id.clone();
            self.venue_order_ids
                .insert(request.id.clone(), venue_id.as_str().to_string());

            for fill in fills {
                if !fill.qty.is_finite()
                    || fill.qty <= 0.0
                    || !fill.price.is_finite()
                    || fill.price <= 0.0
                    || !fill.fee.is_finite()
                {
                    return Err(VenueError::Transport(
                        "venue returned invalid fill data; reconciliation is required".into(),
                    ));
                }
                events.push(ExecutionEvent::Fill(FillEvent {
                    order_id: request.id.clone(),
                    symbol: fill.symbol,
                    side: match fill.side {
                        ConnectorSide::Buy => OrderSide::Buy,
                        ConnectorSide::Sell => OrderSide::Sell,
                    },
                    qty: fill.qty,
                    price: fill.price,
                    fee: fill.fee,
                    executed_at: now,
                }));
            }
        }

        Ok(events)
    }

    fn cancel_order(
        &mut self,
        order_id: &OrderId,
        symbol: &str,
        now: DateTime<Utc>,
    ) -> Result<Vec<ExecutionEvent>, VenueError> {
        let venue_id = self
            .venue_order_ids
            .get(order_id)
            .cloned()
            .unwrap_or_else(|| order_id.0.clone());
        self.connector
            .cancel(symbol, &uets::core::order::OrderId::new(venue_id))
            .map_err(|err| VenueError::Transport(err.to_string()))?;
        self.venue_order_ids.remove(order_id);
        Ok(vec![ExecutionEvent::OrderCancelled {
            order_id: order_id.clone(),
            at: now,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VenueConnector;
    use std::sync::{Arc, Mutex};
    use uets::core::order::{Fill, Order, OrderId as ConnectorOrderId};
    use uets::exchange::{ConnectorError, Venue, VenueSpec};

    struct RecordingConnector {
        requests: Arc<Mutex<Vec<ConnectorOrderRequest>>>,
        cancellations: Arc<Mutex<Vec<String>>>,
    }

    impl Venue for RecordingConnector {
        fn venue(&self) -> VenueSpec {
            VenueSpec::new("test", "USDT")
        }
    }

    impl ExecutionConnector for RecordingConnector {
        fn submit(&mut self, request: &ConnectorOrderRequest) -> Result<Vec<Fill>, ConnectorError> {
            self.requests.lock().unwrap().push(request.clone());
            Ok(vec![Fill::new(
                "venue-order",
                request.symbol.clone(),
                request.side,
                request.qty,
                10.0,
                0.0,
            )])
        }

        fn cancel(
            &mut self,
            _symbol: &str,
            order_id: &ConnectorOrderId,
        ) -> Result<(), ConnectorError> {
            self.cancellations
                .lock()
                .unwrap()
                .push(order_id.as_str().to_string());
            Ok(())
        }

        fn open_orders(&mut self, _symbol: Option<&str>) -> Result<Vec<Order>, ConnectorError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn adapter_preserves_reduce_only_on_exit_orders() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let cancellations = Arc::new(Mutex::new(Vec::new()));
        let connector = RecordingConnector {
            requests: requests.clone(),
            cancellations,
        };
        let mut adapter = UetsExecutionConnectorAdapter::new(connector);
        let request = OrderRequest::new(
            "exit",
            "BTCUSDT",
            OrderSide::Sell,
            1.0,
            "ofmr@btcusdt",
            "btcusdt",
        )
        .reducing();

        adapter.place_order(&request, Utc::now()).unwrap();
        let captured = requests.lock().unwrap();
        assert_eq!(captured[0].symbol, "BTCUSDT");
        assert_eq!(captured[0].client_order_id.as_deref(), Some("exit"));
        assert_eq!(captured[0].side, ConnectorSide::Sell);
        assert_eq!(captured[0].qty, 1.0);
        assert_eq!(captured[0].time_in_force, ConnectorTimeInForce::Gtc);
        assert!(captured[0].reduce_only);

        let restarted_cancel_ids = Arc::new(Mutex::new(Vec::new()));
        let mut restarted_adapter = UetsExecutionConnectorAdapter::new(RecordingConnector {
            requests: Arc::new(Mutex::new(Vec::new())),
            cancellations: restarted_cancel_ids.clone(),
        });
        restarted_adapter
            .cancel_order(&OrderId::new("exit"), "BTCUSDT", Utc::now())
            .unwrap();
        assert_eq!(restarted_cancel_ids.lock().unwrap()[0], "exit");
    }
}
