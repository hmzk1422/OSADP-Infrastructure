//! `aggregator` — the fan-out point between exchange feeds and consumers.
//!
//! ```text
//!   exchange WS (per symbol) --+--> sockets/exchanges/<venue>.sock
//!                               |
//!                          aggregator (this module)
//!                               |
//!              +-----------------+-----------------+
//!              |                 |                 |
//!    sockets/strategies/   recorder.rs        health socket
//!    strategyN.sock        (CSV now,            (latency/
//!                           RocksDB later)      resources)
//! ```
//!
//! Today the `record` path writes CSVs directly; this module is the socket
//! scaffolding that lets strategies attach without touching that path.
//! Everything here is `#[cfg(unix)]`-gated: Unix-domain sockets do not exist
//! on Windows, and the crate must keep building on the win32 dev machine.
//! The Linux runtime (Docker) uses the real implementation.

pub mod cleanup;
pub mod ipc;
pub mod latency;
pub mod recorder;
pub mod resources;
pub mod risk;
pub mod supervisor;
