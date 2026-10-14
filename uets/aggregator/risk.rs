//! Risk: the only write path to the exchange. Owns limits + kill switch.
//!
//! Correction applied: a stale-data watchdog must NOT blindly flatten --
//! flattening into a dead book or a halted venue turns a data problem into an
//! execution disaster. Policy: staleness *blocks new orders* immediately and
//! follows a defined emergency policy (cancel-open, then flatten only when
//! the book is verifiably tradeable). The kill switch is reachable from the
//! control socket so an operator can halt from outside the process.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// Staleness that blocks new orders.
pub const STALE_BLOCK_AFTER: Duration = Duration::from_secs(5);

/// What the watchdog decided.
#[derive(Debug, PartialEq, Eq)]
pub enum RiskDecision {
    /// Healthy: orders flow.
    Ok,
    /// Feed stale: no NEW orders; working orders stand.
    BlockNew,
    /// Operator kill switch: nothing leaves this process.
    Killed,
}

/// Owns the exchange write handle's gate. Hot path: two atomic loads.
pub struct RiskGate {
    killed: AtomicBool,
    last_event_ms: AtomicU64,
    now_ms: AtomicU64,
}

impl RiskGate {
    pub fn new() -> Self {
        Self {
            killed: AtomicBool::new(false),
            last_event_ms: AtomicU64::new(0),
            now_ms: AtomicU64::new(0),
        }
    }

    /// Operator kill switch (control socket). Irreversible within the process.
    pub fn kill(&self) {
        self.killed.store(true, Ordering::SeqCst);
    }

    /// Hot path: called per decision. Blocks new orders when stale/killed.
    pub fn check(&self) -> RiskDecision {
        if self.killed.load(Ordering::SeqCst) {
            return RiskDecision::Killed;
        }
        let age = self
            .now_ms
            .load(Ordering::Relaxed)
            .saturating_sub(self.last_event_ms.load(Ordering::Relaxed));
        if age > STALE_BLOCK_AFTER.as_millis() as u64 {
            return RiskDecision::BlockNew;
        }
        RiskDecision::Ok
    }
}

impl Default for RiskGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_blocks_but_does_not_flatten() {
        let g = RiskGate::new();
        g.now_ms.store(10_000, Ordering::Relaxed);
        g.last_event_ms.store(0, Ordering::Relaxed);
        // Blocks new orders -- the emergency flatten is a separate,
        // operator-confirmed step, never an automatic side effect.
        assert_eq!(g.check(), RiskDecision::BlockNew);
    }

    #[test]
    fn kill_is_sticky() {
        let g = RiskGate::new();
        g.kill();
        g.now_ms.store(1, Ordering::Relaxed);
        g.last_event_ms.store(1, Ordering::Relaxed);
        assert_eq!(g.check(), RiskDecision::Killed);
    }
}
