//! Supervisor: process lifecycle, health, restart-with-backoff.
//!
//! Spawns one child per symbol-capture (the `record` workers graduate into
//! supervised children here), restarts with backoff, and reaps the child's
//! sockets on death so no garbage accumulates. Three missed heartbeats =
//! restart; restart storms are capped by the backoff ceiling.

use std::time::Duration;

/// Miss this many heartbeats and the child is restarted.
pub const MISS_LIMIT: u32 = 3;
/// Backoff ceiling between restarts.
pub const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// Health of one supervised child.
#[derive(Debug, PartialEq, Eq)]
pub enum ChildHealth {
    Healthy,
    Missed(u32),
    Down,
}

/// Backoff for restart attempt `n` (1-based): 1s, 2s, 4s ... capped.
pub fn backoff(n: u32) -> Duration {
    let secs = 1u64.checked_shl(n.saturating_sub(1)).unwrap_or(u64::MAX);
    Duration::from_secs(secs.min(BACKOFF_MAX.as_secs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_caps() {
        assert_eq!(backoff(1), Duration::from_secs(1));
        assert_eq!(backoff(3), Duration::from_secs(4));
        assert_eq!(backoff(100), BACKOFF_MAX);
    }
}
