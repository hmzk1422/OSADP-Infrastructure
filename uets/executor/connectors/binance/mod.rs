//! Binance spot: one adapter per trait of the contract in `crate::exchange`.
//!
//! | file | trait | transport |
//! |------|-------|-----------|
//! | [`trade`] | `ExecutionConnector` + `MarketDataConnector` | REST (`/api/v3/order`, `/api/v3/ticker/price`) |
//! | [`account`] | `AccountConnector` | REST (`/api/v3/account`) |
//! | [`websocket`] | none — it feeds the feature engine | websockets |
//!
//! `trade` and `account` are separate files for the same reason the traits are
//! separate: the order path runs on every decision and must stay short enough to
//! audit, while the balance path runs once a session and is mostly a
//! translation table.
//!
//! The default fee schedule below is Binance spot's published 0.1%: it is a
//! *default*, replaced per-account by whatever `account::BinanceAccount` reads
//! from the venue. Nothing here assumes 0.1% is what any given account pays.

pub mod account;
pub mod trade;
pub mod websocket;

pub use account::BinanceAccount;
pub use trade::BinanceTrade;

use crate::exchange::VenueSpec;

/// The published spot schedule, used until the account reports its own.
pub fn spot_spec() -> VenueSpec {
    VenueSpec::new("binance", "USDT").with_fees(0.001, 0.001)
}
