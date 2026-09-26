//! Rolling window state for causal features.
//!
//! The window is strictly **causal and trailing**: after pushing the value for
//! row `i`, it holds the values for rows `max(0, i - size + 1) ..= i`. A short
//! window at the start of a stream aggregates over whatever exists so far
//! (equivalent to `min_periods = 1`), so a feature is defined from the very
//! first row rather than emitting nothing.
//!
//! This is the single definition of window semantics shared with the Python
//! side; `stresstest/` asserts the two agree exactly.

/// Aggregation applied to the values currently inside a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggregation {
    Mean,
    Sum,
    Min,
    Max,
    Last,
    /// Sample standard deviation (Bessel-corrected), matching pandas `.std()`.
    Std,
}

impl Aggregation {
    /// Parse the wire name emitted by the Python compiler.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "mean" => Ok(Self::Mean),
            "sum" => Ok(Self::Sum),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "last" => Ok(Self::Last),
            "std" => Ok(Self::Std),
            other => Err(format!("unsupported window aggregation: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
            Self::Last => "last",
            Self::Std => "std",
        }
    }
}

/// Sample standard deviation (Bessel-corrected), matching pandas `.std()`.
///
/// A single sample has no spread, so it yields `NaN` rather than panicking;
/// callers propagate that as a missing feature value.
pub fn std_dev(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let sq: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    (sq / (n as f64 - 1.0)).sqrt()
}

/// A fixed-capacity trailing window.
///
/// Storage is a ring buffer, so `push` is `O(1)` amortized and never reallocates
/// after the first `size` pushes.
#[derive(Clone, Debug)]
pub struct RollingWindow {
    size: usize,
    aggregation: Aggregation,
    values: Vec<f64>,
    /// Next slot to overwrite once the window is full.
    head: usize,
    filled: bool,
}

impl RollingWindow {
    pub fn new(size: usize, aggregation: Aggregation) -> Result<Self, String> {
        if size == 0 {
            return Err("window size must be positive".to_string());
        }
        Ok(Self {
            size,
            aggregation,
            values: vec![0.0; size],
            head: 0,
            filled: false,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn aggregation(&self) -> Aggregation {
        self.aggregation
    }

    /// Number of values currently held, `<= size`.
    pub fn len(&self) -> usize {
        if self.filled { self.size } else { self.head }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append the value for the current row and return the aggregation.
    ///
    /// `O(1)`: the outgoing value is overwritten in place once full.
    pub fn push(&mut self, value: f64) -> f64 {
        self.values[self.head] = value;
        self.head += 1;
        if self.head == self.size {
            self.head = 0;
            self.filled = true;
        }
        self.aggregate()
    }

    /// The values in chronological order (oldest first).
    pub fn snapshot(&self) -> Vec<f64> {
        let len = self.len();
        if !self.filled {
            return self.values[..len].to_vec();
        }
        let mut out = Vec::with_capacity(self.size);
        // `head` is the oldest slot once the window is full.
        out.extend_from_slice(&self.values[self.head..]);
        out.extend_from_slice(&self.values[..self.head]);
        out
    }

    /// Current aggregate over the values currently in the window.
    pub fn aggregate(&self) -> f64 {
        aggregate(&self.snapshot(), self.aggregation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_over_partial_then_full_window() {
        let mut w = RollingWindow::new(3, Aggregation::Mean).unwrap();
        assert_eq!(w.push(1.0), 1.0);
        assert_eq!(w.push(2.0), 1.5);
        assert_eq!(w.push(3.0), 2.0);
        // 1.0 falls out; 2, 3, 4 remain.
        assert_eq!(w.push(4.0), 3.0);
    }

    #[test]
    fn ring_order_is_chronological() {
        let mut w = RollingWindow::new(3, Aggregation::Sum).unwrap();
        for v in [1.0, 2.0, 3.0, 4.0] {
            w.push(v);
        }
        assert_eq!(w.snapshot(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn last_min_max_track_trailing_window() {
        let mut w = RollingWindow::new(2, Aggregation::Last).unwrap();
        w.push(5.0);
        w.push(7.0);
        assert_eq!(w.aggregate(), 7.0);

        let mut w = RollingWindow::new(2, Aggregation::Min).unwrap();
        w.push(5.0);
        w.push(7.0);
        assert_eq!(w.aggregate(), 5.0);

        let mut w = RollingWindow::new(2, Aggregation::Max).unwrap();
        w.push(5.0);
        w.push(7.0);
        assert_eq!(w.aggregate(), 7.0);
    }

    #[test]
    fn std_matches_sample_formula() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&values) - 2.13808993529939).abs() < 1e-12);
    }

    #[test]
    fn single_sample_std_is_nan() {
        assert!(std_dev(&[1.0]).is_nan());
    }

    #[test]
    fn zero_size_is_rejected() {
        assert!(RollingWindow::new(0, Aggregation::Mean).is_err());
    }
}
/// Aggregate a slice in chronological order.
///
/// The single reduction shared by [`RollingWindow::aggregate`] and
/// `compute.rs`, so the summation order â€” and therefore the floating-point
/// result â€” is identical in both places.
pub fn aggregate(values: &[f64], aggregation: Aggregation) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    match aggregation {
        Aggregation::Mean => values.iter().sum::<f64>() / values.len() as f64,
        Aggregation::Sum => values.iter().sum(),
        Aggregation::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        Aggregation::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        Aggregation::Last => values[values.len() - 1],
        Aggregation::Std => std_dev(values),
    }
}
