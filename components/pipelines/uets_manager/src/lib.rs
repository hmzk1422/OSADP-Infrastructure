//! Execution ownership and mode selection for UETS.

mod router;
mod venue;

pub use router::{ManagerConfig, ManagerRouter};
pub use venue::{UetsExecutionConnectorAdapter, VenueConnector, VenueError};

use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use uets_trades::backend::{BackendError, ExecutionBackend};
use uets_trades::live_test::LiveTestBackend;
use uets_trades::live_trade::LiveTradeBackend;
use uets_trades::playback::PlaybackBackend;
use uets_trades::sim::SimConfig;
use uets_trades::types::{
    Capital, ExecutionEvent, FillEvent, Order, OrderId, OrderRequest, TradeIntent,
};

/// Explicitly selected execution mode. Live mode cannot be created without a venue.
pub enum ExecutionMode {
    LiveTest {
        config: SimConfig,
    },
    LiveTrade {
        router: ManagerRouter,
    },
    Playback {
        config: SimConfig,
        acceleration: f64,
    },
}

/// Owns the selected backend and exposes the shared trading interface.
pub struct ManagerEngine {
    backend: Box<dyn ExecutionBackend>,
}

impl ManagerEngine {
    pub fn new(mode: ExecutionMode) -> Self {
        let backend: Box<dyn ExecutionBackend> = match mode {
            ExecutionMode::LiveTest { config } => Box::new(LiveTestBackend::with_config(config)),
            ExecutionMode::LiveTrade { router } => {
                Box::new(LiveTradeBackend::with_router(Box::new(router)))
            }
            ExecutionMode::Playback {
                config,
                acceleration,
            } => Box::new(PlaybackBackend::with_config(config, acceleration)),
        };
        Self { backend }
    }

    pub fn live_test(config: SimConfig) -> Self {
        Self::new(ExecutionMode::LiveTest { config })
    }
    pub fn live_trade(router: ManagerRouter) -> Self {
        Self::new(ExecutionMode::LiveTrade { router })
    }
    pub fn playback(config: SimConfig, acceleration: f64) -> Self {
        Self::new(ExecutionMode::Playback {
            config,
            acceleration,
        })
    }

    pub fn backend(&self) -> &dyn ExecutionBackend {
        self.backend.as_ref()
    }
    pub fn backend_mut(&mut self) -> &mut dyn ExecutionBackend {
        self.backend.as_mut()
    }
}

// Delegation keeps all modes on the one uets_trades contract.
impl ExecutionBackend for ManagerEngine {
    fn submit_intent(&mut self, intent: TradeIntent) -> Result<OrderId, BackendError> {
        self.backend.submit_intent(intent)
    }
    fn submit_order(&mut self, request: OrderRequest) -> Result<OrderId, BackendError> {
        self.backend.submit_order(request)
    }
    fn cancel_order(&mut self, id: &OrderId) -> Result<(), BackendError> {
        self.backend.cancel_order(id)
    }
    fn order_state(&self, id: &OrderId) -> Option<&Order> {
        self.backend.order_state(id)
    }
    fn orders(&self) -> Vec<&Order> {
        self.backend.orders()
    }
    fn drain_events(&mut self) -> Vec<ExecutionEvent> {
        self.backend.drain_events()
    }
    fn capital(&self, strategy: &str) -> Option<Capital> {
        self.backend.capital(strategy)
    }
    fn apply_fill(&mut self, fill: &FillEvent) {
        self.backend.apply_fill(fill)
    }
    fn on_market_update(&mut self, marks: &BTreeMap<String, f64>, now: DateTime<Utc>) {
        self.backend.on_market_update(marks, now)
    }
    fn is_connected(&self) -> bool {
        self.backend.is_connected()
    }
    fn acceleration(&self) -> f64 {
        self.backend.acceleration()
    }
    fn position_qty(&self, strategy: &str, symbol: &str) -> Option<f64> {
        self.backend.position_qty(strategy, symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ManagerConfig, ManagerRouter, VenueConnector, VenueError};
    use chrono::Duration;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration as StdDuration;
    use uets_trades::backend::{LiveRouter, RouterError};
    use uets_trades::types::{ExecutionEvent, OrderId, OrderRequest, OrderSide};

    struct FakeVenue(Arc<AtomicUsize>);
    impl VenueConnector for FakeVenue {
        fn place_order(
            &mut self,
            request: &OrderRequest,
            now: DateTime<Utc>,
        ) -> Result<Vec<ExecutionEvent>, VenueError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(vec![ExecutionEvent::OrderAccepted {
                order_id: request.id.clone(),
                symbol: request.symbol.clone(),
                at: now,
            }])
        }
        fn cancel_order(
            &mut self,
            id: &OrderId,
            _symbol: &str,
            now: DateTime<Utc>,
        ) -> Result<Vec<ExecutionEvent>, VenueError> {
            Ok(vec![ExecutionEvent::OrderCancelled {
                order_id: id.clone(),
                at: now,
            }])
        }
    }

    fn request() -> OrderRequest {
        OrderRequest::new(
            "one",
            "BTCUSDT",
            OrderSide::Buy,
            1.0,
            "ofmr@btcusdt",
            "btcusdt",
        )
    }

    #[test]
    fn manager_router_routes_once_and_rejects_duplicate_ids() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut router =
            ManagerRouter::new(Box::new(FakeVenue(calls.clone())), ManagerConfig::default());
        let now = Utc::now();
        router.note_market_tick(now);
        assert_eq!(router.route(&request(), now).unwrap().len(), 1);
        assert!(matches!(
            router.route(&request(), now),
            Err(RouterError::DuplicateOrderId(_))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn live_routing_rejects_missing_connector_and_stale_data() {
        let now = Utc::now();
        let mut missing = ManagerRouter::unconfigured(ManagerConfig::default());
        missing.note_market_tick(now);
        assert!(matches!(
            missing.route(&request(), now),
            Err(RouterError::NotConfigured)
        ));

        let mut stale = ManagerRouter::new(
            Box::new(FakeVenue(Arc::new(AtomicUsize::new(0)))),
            ManagerConfig {
                max_market_data_age: StdDuration::from_millis(50),
            },
        );
        stale.note_market_tick(now - Duration::seconds(1));
        assert!(matches!(
            stale.route(&request(), now),
            Err(RouterError::StaleMarketData { .. })
        ));
    }
}
