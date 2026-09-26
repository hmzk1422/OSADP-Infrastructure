//! Compile the real executor feature modules into this harness.
//!
//! `uets/executor/lib.rs` is outside the allowed edit scope and does not
//! declare `ml_feature`, so including the sources by path is what actually
//! type-checks them. Without this, a change to `uets/executor/ml_feature/`
//! would never be compiled by anything.
//!
//! The paths are relative to `components/rust/researcher-toolkit/stresstest/`.

#[path = "../../../../uets/executor/ml_feature/mod.rs"]
pub mod ml_feature;

#[path = "../../../../uets/executor/strategy/mod.rs"]
pub mod strategy;
