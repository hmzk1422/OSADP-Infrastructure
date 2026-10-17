//! Read-only monitoring TUI. The application supplies authoritative snapshots;
//! this crate only renders, selects, and transports them over local sockets.

pub mod controls;
pub mod gui;
pub mod socket;

pub use gui::{
    DashboardSnapshot, InstanceSnapshot, Monitor, MonitorError, SnapshotSource, Statistics,
};
pub use socket::{SocketSnapshotSource, publish_instance};
