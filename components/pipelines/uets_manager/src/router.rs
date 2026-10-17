use chrono::{DateTime, Utc};
use std::collections::BTreeSet;
use std::time::Duration;
use uets_trades::backend::{LiveRouter, RouterError};
use uets_trades::types::{ExecutionEvent, OrderId, OrderRequest};

use crate::{VenueConnector, VenueError};

#[derive(Clone, Debug)]
pub struct ManagerConfig {
    pub max_market_data_age: Duration,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            max_market_data_age: Duration::from_secs(2),
        }
    }
}

/// Validates and forwards live orders to the configured venue connector.
pub struct ManagerRouter {
    venue: Option<Box<dyn VenueConnector>>,
    config: ManagerConfig,
    last_market_tick: Option<DateTime<Utc>>,
    order_ids: BTreeSet<OrderId>,
    events: Vec<ExecutionEvent>,
}

impl ManagerRouter {
    pub fn new(venue: Box<dyn VenueConnector>, config: ManagerConfig) -> Self {
        Self {
            venue: Some(venue),
            config,
            last_market_tick: None,
            order_ids: BTreeSet::new(),
            events: Vec::new(),
        }
    }
    pub fn unconfigured(config: ManagerConfig) -> Self {
        Self {
            venue: None,
            config,
            last_market_tick: None,
            order_ids: BTreeSet::new(),
            events: Vec::new(),
        }
    }
}

impl LiveRouter for ManagerRouter {
    fn note_market_tick(&mut self, ts: DateTime<Utc>) {
        self.last_market_tick = Some(ts);
    }

    fn route(
        &mut self,
        request: &OrderRequest,
        now: DateTime<Utc>,
    ) -> Result<Vec<ExecutionEvent>, RouterError> {
        if !self.order_ids.insert(request.id.clone()) {
            return Err(RouterError::DuplicateOrderId(request.id.0.clone()));
        }
        if request.symbol.trim().is_empty()
            || request.strategy_id.trim().is_empty()
            || request.asset_id.trim().is_empty()
            || !request.qty.is_finite()
            || request.qty <= 0.0
        {
            self.order_ids.remove(&request.id);
            return Err(RouterError::Invalid(
                "symbol, strategy, asset and positive finite quantity are required".into(),
            ));
        }
        let Some(tick) = self.last_market_tick else {
            self.order_ids.remove(&request.id);
            return Err(RouterError::StaleMarketData {
                age_ms: u64::MAX,
                threshold_ms: self.config.max_market_data_age.as_millis() as u64,
            });
        };
        let age_ms = now.signed_duration_since(tick).num_milliseconds().max(0) as u64;
        let threshold_ms = self
            .config
            .max_market_data_age
            .as_millis()
            .min(u64::MAX as u128) as u64;
        if age_ms > threshold_ms {
            self.order_ids.remove(&request.id);
            return Err(RouterError::StaleMarketData {
                age_ms,
                threshold_ms,
            });
        }
        let Some(venue) = self.venue.as_mut() else {
            self.order_ids.remove(&request.id);
            return Err(RouterError::NotConfigured);
        };
        match venue.place_order(request, now) {
            Ok(events) => Ok(events),
            Err(VenueError::Rejected(reason)) => {
                self.order_ids.remove(&request.id);
                Err(RouterError::Rejected(reason))
            }
            Err(err) => Err(RouterError::Transport(err.to_string())),
        }
    }

    fn cancel(&mut self, order_id: &OrderId, symbol: &str) -> Result<(), RouterError> {
        let Some(venue) = self.venue.as_mut() else {
            return Err(RouterError::NotConfigured);
        };
        let events = venue
            .cancel_order(order_id, symbol, Utc::now())
            .map_err(|e| RouterError::Transport(e.to_string()))?;
        self.events.extend(events);
        Ok(())
    }

    fn drain_events(&mut self) -> Vec<ExecutionEvent> {
        std::mem::take(&mut self.events)
    }
}
