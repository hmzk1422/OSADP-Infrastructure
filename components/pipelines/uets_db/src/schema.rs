//! Record schemas and serialization for the uets_db store.
//!
//! Uses compact, versioned binary serialization (bincode). JSON is avoided
//! for high-volume records.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Strategy action: what the strategy wants to do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeIntent {
    Long,
    Short,
    Exit,
    Hold,
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
    Uncertain,
}

/// Schema version — increment when the record format changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaVersion {
    pub major: u16,
    pub minor: u16,
}

impl Default for SchemaVersion {
    fn default() -> Self {
        Self { major: 1, minor: 0 }
    }
}

/// Strategy-instance identity: `strategy_id@asset_id`, e.g. `ofmr@btcusdt`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StrategyInstance {
    pub strategy_id: String,
    pub asset_id: String,
}

impl StrategyInstance {
    pub fn new(strategy_id: impl Into<String>, asset_id: impl Into<String>) -> Self {
        Self {
            strategy_id: strategy_id.into(),
            asset_id: asset_id.into(),
        }
    }

    /// Composite key: `ofmr@btcusdt`.
    pub fn composite(&self) -> String {
        format!("{}@{}", self.strategy_id, self.asset_id)
    }
}

/// A trade record — what the strategy observed, decided, and executed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeRecord {
    /// Schema version for forward compatibility.
    pub schema: SchemaVersion,
    /// Stable session ID.
    pub session_id: String,
    /// Strategy instance identity.
    pub instance: StrategyInstance,
    /// Asset/ticker (stored separately from strategy identity).
    pub symbol: String,
    /// Event timestamp.
    pub event_time: DateTime<Utc>,
    /// The feature values and market inputs consumed by compute_features.
    ///
    /// Stored as JSON text inside the binary envelope: bincode serializes
    /// the record structure compactly, while `serde_json::Value` cannot
    /// round-trip through bincode (it requires `deserialize_any`). Use
    /// [`TradeRecord::set_feature_inputs`] / [`TradeRecord::feature_inputs_json`].
    pub feature_inputs: String,
    /// Strategy decision and metadata (JSON text; see [`TradeRecord::feature_inputs`]).
    pub decision: TradeIntent,
    pub decision_metadata: String,
    /// Trade intent (what was requested).
    pub intent: TradeIntent,
    /// Order lifecycle events.
    pub order_events: Vec<OrderEvent>,
    /// Entry and exit prices.
    pub entry_price: Option<f64>,
    pub exit_price: Option<f64>,
    /// Triple-barrier information (OFMR).
    pub tp_price: Option<f64>,
    pub sl_price: Option<f64>,
    pub vertical_time: Option<DateTime<Utc>>,
    /// Fill quantity, fees, realized PnL.
    pub fill_qty: f64,
    pub fees: f64,
    pub realized_pnl: f64,
    /// Strategy/model version and configuration identifiers.
    pub model_version: String,
    pub config_id: String,
}

impl TradeRecord {
    /// Replace the feature payload (encoded to JSON text).
    pub fn set_feature_inputs(&mut self, value: &serde_json::Value) {
        self.feature_inputs = value.to_string();
    }

    /// Decode the feature payload. Errors are returned, never swallowed.
    pub fn feature_inputs_json(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::from_str(&self.feature_inputs)
    }

    /// Replace the decision metadata payload (encoded to JSON text).
    pub fn set_decision_metadata(&mut self, value: &serde_json::Value) {
        self.decision_metadata = value.to_string();
    }

    /// Decode the decision metadata payload. Errors are returned, never swallowed.
    pub fn decision_metadata_json(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::from_str(&self.decision_metadata)
    }
}

/// One order lifecycle event within a trade record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderEvent {
    pub order_id: String,
    pub status: OrderStatus,
    pub filled_qty: f64,
    pub fill_price: Option<f64>,
    pub fee: f64,
    pub timestamp: DateTime<Utc>,
}

/// A portfolio snapshot — calculated reporting metrics persisted at cadence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortfolioRecord {
    pub schema: SchemaVersion,
    pub session_id: String,
    /// Strategy instance identity.
    pub instance: StrategyInstance,
    /// Asset ID.
    pub asset_id: String,
    /// Snapshot timestamp.
    pub timestamp: DateTime<Utc>,
    pub allocated_capital: f64,
    pub cash: f64,
    pub equity: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub cumulative_return: f64,
    pub sortino: f64,
    pub drawdown: f64,
    pub max_drawdown: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub margin_used: f64,
    pub available_margin: f64,
    pub effective_leverage: f64,
    pub fees_paid: f64,
    pub trade_count: usize,
    pub win_count: usize,
    pub loss_count: usize,
    pub win_rate: f64,
}

/// One session-wide live resource and strategy-latency sample.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryRecord {
    pub schema: SchemaVersion,
    pub session_id: String,
    pub timestamp_ms: u64,
    pub connection_count: usize,
    /// Sum of per-connection RSS in MiB at this sample time.
    pub total_ram_mb: f64,
    /// Sum of per-connection CPU, in percent of one logical core.
    pub total_cpu_percent: f64,
    /// Arithmetic mean across all strategy-compute observations represented.
    pub latency_mean_ms: Option<f64>,
    pub latency_sum_ms: f64,
    pub latency_observations: u64,
    /// Per-connection readings preserve enough detail to audit each total.
    pub connections: Vec<ConnectionTelemetry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectionTelemetry {
    pub strategy_id: String,
    pub asset_id: String,
    pub ram_mb: Option<f64>,
    pub cpu_percent: Option<f64>,
    pub latency_mean_ms: Option<f64>,
    pub latency_sum_ms: f64,
    pub latency_observations: u64,
    pub latency_p95_ms: Option<f64>,
    pub latency_p99_ms: Option<f64>,
}

/// Session metadata — enables independent querying, reporting, archival
/// and recovery per session without separate physical database instances.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub started_at: DateTime<Utc>,
    /// Set when the session is finalized.
    pub ended_at: Option<DateTime<Utc>>,
    pub active: bool,
}

/// Final summary written when a session ends.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionSummary {
    pub schema: SchemaVersion,
    pub session_id: String,
    pub finalized_at: DateTime<Utc>,
    pub trade_count: usize,
    pub portfolio_snapshots: usize,
    pub final_equity: f64,
    pub cumulative_return: f64,
    pub sortino: f64,
    pub max_drawdown: f64,
}

/// Integrity check result.
#[derive(Clone, Debug, Default)]
pub struct IntegrityReport {
    /// Number of decodable trade records.
    pub trade_records: usize,
    /// Number of decodable portfolio records.
    pub portfolio_records: usize,
    /// Keys that failed to decode (corruption detected, not silently discarded).
    pub corrupt_keys: Vec<String>,
    /// Number of active sessions.
    pub active_sessions: usize,
}

impl IntegrityReport {
    pub fn is_clean(&self) -> bool {
        self.corrupt_keys.is_empty()
    }
}

impl PortfolioRecord {}

/// Query filter for retrieving records.
#[derive(Clone, Debug)]
pub struct QueryFilter {
    pub session_id: Option<String>,
    pub strategy_id: Option<String>,
    pub asset_id: Option<String>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}

impl QueryFilter {
    pub fn new() -> Self {
        Self {
            session_id: None,
            strategy_id: None,
            asset_id: None,
            start_time: None,
            end_time: None,
        }
    }

    pub fn session(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    pub fn strategy(mut self, id: impl Into<String>) -> Self {
        self.strategy_id = Some(id.into());
        self
    }

    pub fn asset(mut self, id: impl Into<String>) -> Self {
        self.asset_id = Some(id.into());
        self
    }

    pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_composite_key() {
        let inst = StrategyInstance::new("ofmr", "btcusdt");
        assert_eq!(inst.composite(), "ofmr@btcusdt");
    }
}
