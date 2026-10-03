//! Binance websocket reads: market streams and the user data stream.
//!
//! Three separate responsibilities, deliberately not merged:
//!
//! | file | role |
//! |------|------|
//! | [`live`] | the canonical capture pipeline: streams -> reorder -> merge -> CSV rows |
//! | [`diagnostic`] | one row per event, `update_type` = aggTrades/orderbook, the other stream `NaN` |
//! | [`monitor`] | the single-symbol live frame `display.py` renders |
//! | [`reader`] | standalone stream listeners, one per stream type |
//!
//! Nothing here translates into `crate::core` types: a websocket frame is market
//! *data*, and its consumers are the feature engine (`core::feature`) and the
//! recorders, not the order path. Keeping the streams out of the connector
//! traits is what lets `crate::exchange` stay synchronous and testable.

pub mod diagnostic;
pub mod live;
pub mod monitor;
pub mod reader;
