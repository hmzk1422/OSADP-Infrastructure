//! Evaluate a compiled feature spec, one row at a time.
//!
//! This is the "compute" half of `core::feature`. It takes a [`FeatureRow`]
//! (produced by `ingest`) and returns a row of computed features. It has no
//! transport dependency, so it can be driven from a websocket, a replay file,
//! or the parity harness in `researcher-toolkit/stresstest/`.
//!
//! State is per feature: any `window` / `zscore` / `reconstruct` node owns a
//! window that advances by exactly one row per `compute` call, which is what
//! keeps the evaluation causal.

use super::row::FeatureRow;
use super::spec::{Expr, FeatureSpec, SpecError, parse_spec};
use super::window::{Aggregation, RollingWindow, aggregate, std_dev};

/// Anything that can stop a feature from being produced.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureError {
    /// The spec could not be compiled.
    Spec(SpecError),
    /// A column the spec references is absent from the row.
    MissingColumn(String),
    /// Division by an exact zero. Python raises `ZeroDivisionError` here, so
    /// both sides agree this is a hard failure rather than `inf`.
    DivisionByZero,
}

impl std::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spec(e) => write!(f, "{e}"),
            Self::MissingColumn(c) => write!(f, "missing required column: {c:?}"),
            Self::DivisionByZero => write!(f, "division by zero"),
        }
    }
}

impl std::error::Error for FeatureError {}

impl From<SpecError> for FeatureError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

/// Per-feature mutable state.
enum FeatureState {
    /// No history needed; evaluate directly.
    Stateless,
    /// A trailing window over the inner expression.
    Window { expr: Expr, window: RollingWindow },
    /// A trailing window plus the mean/std needed for a z-score.
    ZScore { expr: Expr, window: RollingWindow },
    /// A group accumulator for bar reconstruction.
    Reconstruct {
        expr: Expr,
        size: usize,
        aggregation: Aggregation,
        buffer: Vec<f64>,
    },
}

/// Stateful evaluator over a compiled spec.
pub struct FeatureComputer {
    features: Vec<(String, Expr)>,
    state: Vec<FeatureState>,
    rows_seen: u64,
}

impl FeatureComputer {
    /// Build from an already-parsed spec.
    pub fn new(specs: Vec<FeatureSpec>) -> Result<Self, FeatureError> {
        let mut state = Vec::with_capacity(specs.len());
        for spec in &specs {
            state.push(build_state(&spec.expr)?);
        }
        Ok(Self {
            features: specs.into_iter().map(|s| (s.name, s.expr)).collect(),
            state,
            rows_seen: 0,
        })
    }

    /// Compile a spec document and build from it. The normal entry point:
    /// hand it the JSON produced by the Python `FeaturePipeline.compile()`.
    pub fn from_spec_json(json: &str) -> Result<Self, FeatureError> {
        Self::new(parse_spec(json)?)
    }

    /// Feature names, in declaration order.
    pub fn feature_names(&self) -> Vec<&str> {
        self.features.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn rows_seen(&self) -> u64 {
        self.rows_seen
    }

    /// Advance every feature by one row and return the computed values.
    ///
    /// A value that is legitimately undefined (for example the standard
    /// deviation of a single sample) is recorded as `Missing` rather than as
    /// `NaN` or `0.0`, so "not enough history" is never mistaken for a real
    /// measurement.
    pub fn compute(&mut self, row: &FeatureRow) -> Result<FeatureRow, FeatureError> {
        let mut out = FeatureRow::new();
        for index in 0..self.features.len() {
            let name = self.features[index].0.clone();
            match self.step(index, row)? {
                Some(value) => out.insert(name, value),
                None => out.insert_missing(name),
            }
        }
        self.rows_seen += 1;
        Ok(out)
    }

    /// Advance one feature, returning `None` when the value is undefined.
    fn step(&mut self, index: usize, row: &FeatureRow) -> Result<Option<f64>, FeatureError> {
        let expr = self.features[index].1.clone();
        match &mut self.state[index] {
            FeatureState::Stateless => eval(&expr, row).map(Some),
            FeatureState::Window { expr, window } => {
                let value = eval(expr, row)?;
                Ok(Some(window.push(value)))
            }
            FeatureState::ZScore { expr, window } => {
                let value = eval(expr, row)?;
                window.push(value);
                // The window now holds the values up to and including this
                // row, so the mean and std-dev are causal.
                let snapshot = window.snapshot();
                let mean = aggregate(&snapshot, Aggregation::Mean);
                let sd = std_dev(&snapshot);
                if sd.is_nan() {
                    return Ok(None);
                }
                Ok(Some((value - mean) / sd))
            }
            FeatureState::Reconstruct {
                expr,
                size,
                aggregation,
                buffer,
            } => {
                buffer.push(eval(expr, row)?);
                // Only completed groups are emitted, matching the Python
                // rule that an incomplete trailing bar is dropped.
                if buffer.len() < *size {
                    return Ok(None);
                }
                let value = aggregate(buffer, *aggregation);
                buffer.clear();
                Ok(Some(value))
            }
        }
    }
}

fn build_state(expr: &Expr) -> Result<FeatureState, FeatureError> {
    Ok(match expr {
        Expr::Window {
            expr,
            size,
            aggregation,
        } => FeatureState::Window {
            expr: (**expr).clone(),
            window: RollingWindow::new(*size, *aggregation)
                .map_err(|e| FeatureError::Spec(SpecError(e)))?,
        },
        Expr::ZScore { expr, size } => FeatureState::ZScore {
            expr: (**expr).clone(),
            // The window only accumulates; the z-score reduces it itself.
            window: RollingWindow::new(*size, Aggregation::Last)
                .map_err(|e| FeatureError::Spec(SpecError(e)))?,
        },
        Expr::Reconstruct {
            expr,
            size,
            aggregation,
        } => FeatureState::Reconstruct {
            expr: (**expr).clone(),
            size: *size,
            aggregation: *aggregation,
            buffer: Vec::with_capacity(*size),
        },
        _ => FeatureState::Stateless,
    })
}

/// Evaluate a stateless expression against one row.
fn eval(expr: &Expr, row: &FeatureRow) -> Result<f64, FeatureError> {
    match expr {
        Expr::Column(name) => row
            .get(name)
            .ok_or_else(|| FeatureError::MissingColumn(name.clone())),
        Expr::Literal(v) => Ok(*v),
        Expr::Add(a, b) => Ok(eval(a, row)? + eval(b, row)?),
        Expr::Sub(a, b) => Ok(eval(a, row)? - eval(b, row)?),
        Expr::Mul(a, b) => Ok(eval(a, row)? * eval(b, row)?),
        Expr::Div(a, b) => {
            let d = eval(b, row)?;
            if d == 0.0 {
                return Err(FeatureError::DivisionByZero);
            }
            Ok(eval(a, row)? / d)
        }
        // Stateful nodes are folded into a `FeatureState` at build time, so
        // reaching one here means the spec nests them.
        Expr::Window { .. } | Expr::ZScore { .. } | Expr::Reconstruct { .. } => {
            Err(FeatureError::Spec(SpecError(
                "nested stateful expressions are not supported".to_string(),
            )))
        }
    }
}
