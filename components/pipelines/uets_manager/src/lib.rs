//! Execution ownership and mode selection for UETS.

mod router;
mod venue;

pub use router::{ManagerConfig, ManagerRouter};
pub use venue::{UetsExecutionConnectorAdapter, VenueConnector, VenueError};

use chrono::{DateTime, Utc};
use serde::Deserialize;
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

/// Errors raised while validating or deserializing execution settings.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("unknown execution mode: {0}")]
    UnknownMode(String),
    #[error("live_trade requires a configured venue router")]
    LiveTradeMissingRouter,
    #[error("invalid acceleration: {0}")]
    InvalidAcceleration(String),
    #[error("invalid spread_bps: {0}")]
    InvalidSpreadBps(String),
    #[error("invalid slippage_bps: {0}")]
    InvalidSlippageBps(String),
    #[error("invalid fee_bps: {0}")]
    InvalidFeeBps(String),
    #[error("invalid starting_capital: {0}")]
    InvalidStartingCapital(String),
    #[error("yaml deserialization failed: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// Typed execution mode accepted from YAML. Using an enum (rather than a free
/// string) means typos like `play_back` fail during deserialization with a
/// normal serde error instead of a runtime panic.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionModeSetting {
    LiveTest,
    LiveTrade,
    Playback,
}

/// YAML-deserializable execution settings. Mirrors [`ExecutionMode`] variants
/// but uses only plain data so it can be loaded from `settings.yaml`.
///
/// This struct is intended for the `execution:` section of a settings file, not
/// necessarily the entire file — `#[serde(deny_unknown_fields)]` will reject any
/// key that does not match a field below. If `settings.yaml` holds other
/// top-level sections, deserialize a nested `execution` block instead.
///
/// Changing `mode` to `playback` (and setting `acceleration`) makes
/// [`ManagerEngine::from_settings`] build a `PlaybackBackend` that replays
/// merged Parquet through the same `SimEngine` paper trading uses.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionSettings {
    /// `live_test`, `live_trade`, or `playback`.
    pub mode: ExecutionModeSetting,
    /// Replay speed multiplier (playback only). 1.0 = real time, 100 = 100x.
    /// Ignored for `live_test` and `live_trade`.
    #[serde(default = "default_acceleration")]
    pub acceleration: f64,
    /// Half-spread applied at execution, in bps.
    #[serde(default)]
    pub spread_bps: Option<f64>,
    /// Adverse slippage applied at execution, in bps.
    #[serde(default)]
    pub slippage_bps: Option<f64>,
    /// Taker fee per fill notional, in bps.
    #[serde(default)]
    pub fee_bps: Option<f64>,
    /// Capital allocated to strategies with no explicit allocation.
    #[serde(default)]
    pub starting_capital: Option<f64>,
}

fn default_acceleration() -> f64 {
    1.0
}

impl ExecutionSettings {
    /// Validate numeric fields and return `Ok(())` when all are within range.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.acceleration.is_finite() || self.acceleration <= 0.0 {
            return Err(ConfigError::InvalidAcceleration(format!(
                "acceleration must be finite and positive, got {}",
                self.acceleration
            )));
        }
        if let Some(v) = self.spread_bps {
            if !v.is_finite() || v < 0.0 {
                return Err(ConfigError::InvalidSpreadBps(format!(
                    "spread_bps must be finite and nonnegative, got {v}"
                )));
            }
        }
        if let Some(v) = self.slippage_bps {
            if !v.is_finite() || v < 0.0 {
                return Err(ConfigError::InvalidSlippageBps(format!(
                    "slippage_bps must be finite and nonnegative, got {v}"
                )));
            }
        }
        if let Some(v) = self.fee_bps {
            if !v.is_finite() || v < 0.0 {
                return Err(ConfigError::InvalidFeeBps(format!(
                    "fee_bps must be finite and nonnegative, got {v}"
                )));
            }
        }
        if let Some(v) = self.starting_capital {
            if !v.is_finite() || v <= 0.0 {
                return Err(ConfigError::InvalidStartingCapital(format!(
                    "starting_capital must be finite and positive, got {v}"
                )));
            }
        }
        Ok(())
    }

    /// Build the [`ExecutionMode`] for this settings block after validation.
    ///
    /// `live_trade` requires `Some(router)`; passing `None` returns
    /// [`ConfigError::LiveTradeMissingRouter`] rather than silently building an
    /// unconfigured router that would reject every live order.
    pub fn into_mode(self, router: Option<ManagerRouter>) -> Result<ExecutionMode, ConfigError> {
        self.validate()?;

        let mut config = SimConfig::default();
        if let Some(v) = self.spread_bps {
            config.spread_bps = v;
        }
        if let Some(v) = self.slippage_bps {
            config.slippage_bps = v;
        }
        if let Some(v) = self.fee_bps {
            config.fee_bps = v;
        }
        if let Some(v) = self.starting_capital {
            config.starting_capital = v;
        }

        match self.mode {
            ExecutionModeSetting::LiveTest => Ok(ExecutionMode::LiveTest { config }),
            ExecutionModeSetting::LiveTrade => {
                let router = router.ok_or(ConfigError::LiveTradeMissingRouter)?;
                Ok(ExecutionMode::LiveTrade { router })
            }
            ExecutionModeSetting::Playback => Ok(ExecutionMode::Playback {
                config,
                acceleration: self.acceleration,
            }),
        }
    }

    /// Load execution settings from a YAML string.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, ConfigError> {
        serde_yaml::from_str(yaml).map_err(ConfigError::Yaml)
    }

    /// Load execution settings from a YAML file.
    pub fn from_yaml_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            ConfigError::Yaml(serde_yaml::Error::custom(e.to_string()))
        })?;
        Self::from_yaml_str(&contents)
    }
}

impl ManagerEngine {
    /// Build an engine from YAML-loaded execution settings.
    ///
    /// Pass `Some(router)` when `mode` is `live_trade`; otherwise `None` is fine
    /// (playback / live_test never touch a venue). Returns an error for invalid
    /// settings or a missing live-trade router.
    pub fn from_settings(
        settings: ExecutionSettings,
        router: Option<ManagerRouter>,
    ) -> Result<Self, ConfigError> {
        Ok(Self::new(settings.into_mode(router)?))
    }
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

    // ---- ExecutionSettings / ConfigError tests ----

    #[test]
    fn valid_yaml_selects_live_test() {
        let yaml = "mode: live_test";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        assert_eq!(settings.mode, ExecutionModeSetting::LiveTest);
        assert_eq!(settings.acceleration, 1.0);
        let mode = settings.into_mode(None).unwrap();
        assert!(matches!(mode, ExecutionMode::LiveTest { .. }));
    }

    #[test]
    fn valid_yaml_selects_live_trade_with_router() {
        let yaml = "mode: live_trade";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let router = ManagerRouter::unconfigured(ManagerConfig::default());
        let mode = settings.into_mode(Some(router)).unwrap();
        assert!(matches!(mode, ExecutionMode::LiveTrade { .. }));
    }

    #[test]
    fn live_trade_without_router_returns_error() {
        let yaml = "mode: live_trade";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let err = settings.into_mode(None).unwrap_err();
        assert!(matches!(err, ConfigError::LiveTradeMissingRouter));
    }

    #[test]
    fn valid_yaml_selects_playback() {
        let yaml = "mode: playback\nacceleration: 100";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        assert_eq!(settings.mode, ExecutionModeSetting::Playback);
        assert_eq!(settings.acceleration, 100.0);
        let mode = settings.into_mode(None).unwrap();
        assert!(matches!(
            mode,
            ExecutionMode::Playback {
                acceleration: 100.0,
                ..
            }
        ));
    }

    #[test]
    fn omitted_optional_fields_receive_defaults() {
        let yaml = "mode: playback";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        assert_eq!(settings.acceleration, 1.0);
        assert!(settings.spread_bps.is_none());
        assert!(settings.slippage_bps.is_none());
        assert!(settings.fee_bps.is_none());
        assert!(settings.starting_capital.is_none());
    }

    #[test]
    fn sim_config_overrides_reach_selected_backend() {
        let yaml = "mode: live_test\nspread_bps: 2.5\nslippage_bps: 1.5\nfee_bps: 8.0\nstarting_capital: 50000.0";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let mode = settings.into_mode(None).unwrap();
        match mode {
            ExecutionMode::LiveTest { config } => {
                assert_eq!(config.spread_bps, 2.5);
                assert_eq!(config.slippage_bps, 1.5);
                assert_eq!(config.fee_bps, 8.0);
                assert_eq!(config.starting_capital, 50000.0);
            }
            _ => panic!("expected LiveTest"),
        }
    }

    #[test]
    fn unknown_mode_deserialization_fails_instead_of_panic() {
        let yaml = "mode: play_back";
        let err = ExecutionSettings::from_yaml_str(yaml).unwrap_err();
        assert!(err.to_string().contains("play_back"));
    }

    #[test]
    fn unknown_key_is_rejected() {
        let yaml = "mode: live_test\nbogus_key: 42";
        let err = ExecutionSettings::from_yaml_str(yaml).unwrap_err();
        assert!(err
            .to_string()
            .contains("bogus_key") || err.to_string().contains("unknown field"));
    }

    #[test]
    fn invalid_acceleration_is_rejected() {
        let yaml = "mode: playback\nacceleration: 0";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let err = settings.into_mode(None).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidAcceleration(_)));
    }

    #[test]
    fn negative_acceleration_is_rejected() {
        let yaml = "mode: playback\nacceleration: -5";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let err = settings.into_mode(None).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidAcceleration(_)));
    }

    #[test]
    fn negative_spread_bps_is_rejected() {
        let yaml = "mode: live_test\nspread_bps: -1";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let err = settings.into_mode(None).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidSpreadBps(_)));
    }

    #[test]
    fn negative_starting_capital_is_rejected() {
        let yaml = "mode: live_test\nstarting_capital: 0";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let err = settings.into_mode(None).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidStartingCapital(_)));
    }

    #[test]
    fn negative_fee_bps_is_rejected() {
        let yaml = "mode: live_test\nfee_bps: -0.5";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let err = settings.into_mode(None).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidFeeBps(_)));
    }

    #[test]
    fn from_settings_returns_result_for_live_trade_missing_router() {
        let yaml = "mode: live_trade";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let err = ManagerEngine::from_settings(settings, None).unwrap_err();
        assert!(matches!(err, ConfigError::LiveTradeMissingRouter));
    }

    #[test]
    fn from_settings_succeeds_for_playback_without_router() {
        let yaml = "mode: playback\nacceleration: 10";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let engine = ManagerEngine::from_settings(settings, None).unwrap();
        assert_eq!(engine.acceleration(), 10.0);
    }

    #[test]
    fn from_settings_succeeds_for_live_test_without_router() {
        let yaml = "mode: live_test";
        let settings = ExecutionSettings::from_yaml_str(yaml).unwrap();
        let engine = ManagerEngine::from_settings(settings, None).unwrap();
        assert_eq!(engine.acceleration(), 1.0);
    }
}
