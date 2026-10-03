//! Venue adapters.
//!
//! Each module under here is a *translation layer* and nothing more: it knows
//! the venue's wire format, its REST endpoints and its error codes, and it
//! converts all of that into the vocabulary in `crate::core`. The split is
//! enforced by what each side is allowed to import —
//!
//! ```text
//!   core      never imports connectors   (no venue name appears above this line)
//!   connectors never import pnl/risk     (an adapter does not decide what is safe)
//! ```
//!
//! A connector is therefore *auditable in isolation*: if a wrong number reaches
//! the ledger, the question narrows to "did this adapter decode the frame
//! correctly?", and answering it needs no knowledge of the strategy.
//!
//! | venue | market data | execution | account | streams |
//! |-------|-------------|-----------|---------|---------|
//! | [`binance`] | REST `ticker/price` | REST `order` | REST `account` | `websocket/` |

pub mod binance;
