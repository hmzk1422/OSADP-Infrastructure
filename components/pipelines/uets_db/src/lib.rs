//! `uets_db` — persistence layer for the UETS trading system.
//!
//! Owns the embedded RocksDB key-value store, record
//! schemas, serialization, CRUD operations, indexed queries, batched writes,
//! session management, schema versioning, and recovery.
//!
//! Logical structure (shared store, session-aware keys):
//! ```text
//! live_db/
//!   trades/     -> trade records (session, strategy, asset, time)
//!   portfolio/  -> portfolio snapshots (session, strategy, asset, time)
//! ```
//!
//! Other components must use this API and must not open or manipulate the
//! store directly.

pub mod persistence;
pub mod schema;
pub mod store;

pub use persistence::{
    PersistError, PersistItem, PersistStats, PersistenceConfig, PersistenceHandle, StatsHandle,
    SubmitOutcome,
};
pub use schema::{
    IntegrityReport, OrderEvent, OrderStatus, PortfolioRecord, QueryFilter, SchemaVersion,
    SessionMeta, SessionSummary, StrategyInstance, TradeIntent, TradeRecord,
};
pub use store::{DbError, UetsDb};
