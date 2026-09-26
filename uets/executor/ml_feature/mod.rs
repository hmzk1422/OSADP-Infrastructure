//! Runtime feature computation.
//!
//! The module is split by responsibility so each half can be replaced without
//! touching the other:
//!
//! | file | role |
//! |------|------|
//! | [`ingest`] | where numbers come from (websocket frame -> row) |
//! | [`spec`] | the Python -> Rust bridge (JSON spec -> executable [`Expr`]) |
//! | [`compute`] | what is computed from them (row -> features) |
//! | [`window`] | causal trailing-window state |
//! | [`row`] | the row type shared by both halves |
//!
//! `ingest` and `compute` are deliberately independent: the same
//! [`FeatureComputer`] serves a live socket, a recorded file, and the parity
//! harness in `researcher-toolkit/stresstest/`.

mod compute;
mod ingest;
mod row;
mod spec;
mod window;

pub use compute::{FeatureComputer, FeatureError};
pub use ingest::{
    BookTop, CANON_AGG_IS_BUYER_MAKER, CANON_AGG_PRICE, CANON_AGG_QUANTITY, CANON_AGG_TRADE_ID,
    CANON_AGG_TRANSACT_TIME, CANON_BEST_ASK_PRICE, CANON_BEST_ASK_QTY, CANON_BEST_BID_PRICE,
    CANON_BEST_BID_QTY, CANON_EVENT_SEQUENCE, CANON_EVENT_TYPE, CANON_LOB_EVENT_TIME,
    CANON_LOB_FINAL_UPDATE_ID, CANON_TIMESTAMP_NS, COL_AGG_TRADE_ID, COL_BEST_ASK_PRICE,
    COL_BEST_ASK_QTY, COL_BEST_BID_PRICE, COL_BEST_BID_QTY, COL_EVENT_TIME, COL_PRICE,
    COL_QUANTITY, CanonicalIntegers, CanonicalTop, CanonicalValues, EVENT_TYPE_LOB,
    EVENT_TYPE_TRADE, ask_price_column, ask_qty_column, book_price_column, book_qty_column, num,
    parse_levels, row_from_canonical, row_from_frame,
};
pub use row::{FeatureRow, FeatureValue};
pub use spec::{Expr, FeatureSpec, SpecError, parse_expr, parse_spec};
pub use window::{Aggregation, RollingWindow, aggregate, std_dev};
