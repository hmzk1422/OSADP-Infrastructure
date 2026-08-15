//! Intraday Momentum Composite — Rust executor (library crate).
//!
//! Re-exports the data-feed client, streaming feature engine, ONNX model
//! loader, position tracker, and strategy gate. The `executor` binary in
//! `src/main.rs` wires these together: it spawns a Python CCXT sidecar
//! (`tools/ccxt_sidecar.py`), replays the trade/orderbook feed through the
//! feature engine, and runs the ONNX model + strategy gates every second.

pub mod config;
pub mod exchange;
pub mod features;
pub mod model;
pub mod positions;
pub mod strategy;

pub use config::Config;
pub use features::{FeatureEngine, MacroFeatures};
pub use model::OnnxModel;
pub use positions::{ExitResult, Position};
pub use strategy::{Action, EnterReceipt, StrategyExecutor};
