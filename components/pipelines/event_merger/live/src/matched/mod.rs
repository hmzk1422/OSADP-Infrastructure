//! `matched` — the bar-time half of the stream handler.
//!
//! Where [`crate::event_merger`] turns raw events into canonical *rows*
//! (event time), this module turns those rows into timeframe *bars*
//! (bar time) via [`reconstruct`]. Strategies that train on resampled
//! research frames (OFMR's micro 1s / macro 300s grid, VDC's HTF/LTF
//! splits) consume this to keep live and research on the same grid.

pub mod reconstruct;

pub use reconstruct::{Bar, Reconstructor, Sample, Timeframe, reconstruct};
