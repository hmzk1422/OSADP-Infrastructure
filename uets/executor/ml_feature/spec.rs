//! Parse the feature spec compiled by the Python side.
//!
//! This is the bridge that makes the pipeline lossless. Python builds a
//! deterministic JSON document (`FeaturePipeline.compile()`); this module
//! turns it back into an executable [`Expr`]. The op vocabulary is mirrored
//! exactly from `researcher-toolkit/live/feature.py`, and `stresstest/`
//! asserts the two evaluate identically.
//!
//! An unknown op is a hard error rather than a silent skip: a feature that
//! quietly evaluates to nothing is far worse than a startup failure.

use super::window::Aggregation;

/// A compiled feature expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Column(String),
    Literal(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    /// Causal trailing window over the inner expression.
    Window {
        expr: Box<Expr>,
        size: usize,
        aggregation: Aggregation,
    },
    /// Causal z-score of the inner expression over a trailing window.
    ZScore {
        expr: Box<Expr>,
        size: usize,
    },
    /// Bar reconstruction: group `size` consecutive rows, aggregate, and emit
    /// only completed groups. Mirrors `FeaturePipeline.reconstruct()`.
    Reconstruct {
        expr: Box<Expr>,
        size: usize,
        aggregation: Aggregation,
    },
}

/// Why a spec could not be compiled.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecError(pub String);

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid feature spec: {}", self.0)
    }
}

impl std::error::Error for SpecError {}

fn err<T>(message: impl Into<String>) -> Result<T, SpecError> {
    Err(SpecError(message.into()))
}

/// One `(name, expression)` pair from the spec.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureSpec {
    pub name: String,
    pub expr: Expr,
}

/// Compile a whole spec document into executable features.
pub fn parse_spec(json: &str) -> Result<Vec<FeatureSpec>, SpecError> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| SpecError(e.to_string()))?;

    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);
    if version != 1 {
        return err(format!("unsupported spec version {version}; expected 1"));
    }

    let features = value
        .get("features")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| SpecError("spec is missing a 'features' array".to_string()))?;

    let mut out = Vec::with_capacity(features.len());
    for entry in features {
        let name = entry
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| SpecError("feature is missing 'name'".to_string()))?
            .to_string();
        if name.is_empty() {
            return err("feature name must be non-empty");
        }
        let expr = parse_expr(
            entry
                .get("expr")
                .ok_or_else(|| SpecError(format!("feature '{name}' is missing 'expr'")))?,
        )?;
        out.push(FeatureSpec { name, expr });
    }
    Ok(out)
}

/// Compile one expression node.
pub fn parse_expr(value: &serde_json::Value) -> Result<Expr, SpecError> {
    let op = value
        .get("op")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| SpecError("expression is missing 'op'".to_string()))?;

    let args = value
        .get("args")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| SpecError(format!("op '{op}' is missing 'args'")))?;

    match op {
        "column" => {
            if args.len() != 1 {
                return err(format!("column expects 1 arg, got {}", args.len()));
            }
            let name = args[0]
                .as_str()
                .ok_or_else(|| SpecError("column name must be a string".to_string()))?;
            if name.is_empty() {
                return err("column name must be non-empty");
            }
            Ok(Expr::Column(name.to_string()))
        }
        "literal" => {
            if args.len() != 1 {
                return err(format!("literal expects 1 arg, got {}", args.len()));
            }
            let v = args[0]
                .as_f64()
                .ok_or_else(|| SpecError("literal must be a number".to_string()))?;
            Ok(Expr::Literal(v))
        }
        "add" | "sub" | "mul" | "div" => binary(op, args),
        "window" => {
            if args.len() != 3 {
                return err(format!("window expects 3 args, got {}", args.len()));
            }
            Ok(Expr::Window {
                expr: Box::new(parse_expr(&args[0])?),
                size: size_arg(&args[1])?,
                aggregation: aggregation_arg(&args[2])?,
            })
        }
        "zscore" => {
            if args.len() != 2 {
                return err(format!("zscore expects 2 args, got {}", args.len()));
            }
            Ok(Expr::ZScore {
                expr: Box::new(parse_expr(&args[0])?),
                size: size_arg(&args[1])?,
            })
        }
        "reconstruct" => {
            if args.len() != 3 {
                return err(format!("reconstruct expects 3 args, got {}", args.len()));
            }
            Ok(Expr::Reconstruct {
                expr: Box::new(parse_expr(&args[0])?),
                size: size_arg(&args[1])?,
                aggregation: aggregation_arg(&args[2])?,
            })
        }
        other => err(format!("unknown op '{other}'")),
    }
}

fn binary(op: &str, args: &[serde_json::Value]) -> Result<Expr, SpecError> {
    if args.len() != 2 {
        return err(format!("{op} expects 2 args, got {}", args.len()));
    }
    let lhs = Box::new(parse_expr(&args[0])?);
    let rhs = Box::new(parse_expr(&args[1])?);
    Ok(match op {
        "add" => Expr::Add(lhs, rhs),
        "sub" => Expr::Sub(lhs, rhs),
        "mul" => Expr::Mul(lhs, rhs),
        _ => Expr::Div(lhs, rhs),
    })
}

fn size_arg(value: &serde_json::Value) -> Result<usize, SpecError> {
    let n = value
        .as_u64()
        .ok_or_else(|| SpecError("window size must be a non-negative integer".to_string()))?;
    if n == 0 {
        return err("window size must be positive");
    }
    Ok(n as usize)
}

fn aggregation_arg(value: &serde_json::Value) -> Result<Aggregation, SpecError> {
    let name = value
        .as_str()
        .ok_or_else(|| SpecError("aggregation must be a string".to_string()))?;
    Aggregation::parse(name).map_err(SpecError)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_bid_ask_example() {
        let json = r#"{"version":1,"features":[
            {"name":"bid_ask","expr":{"op":"sub","args":[
                {"op":"column","args":["ask_price"]},
                {"op":"column","args":["bid_price"]}]}}]}"#;
        let specs = parse_spec(json).expect("valid spec");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "bid_ask");
        assert_eq!(
            specs[0].expr,
            Expr::Sub(
                Box::new(Expr::Column("ask_price".into())),
                Box::new(Expr::Column("bid_price".into()))
            )
        );
    }

    #[test]
    fn parses_window_and_reconstruct() {
        let json = r#"{"version":1,"features":[
            {"name":"w","expr":{"op":"window","args":[
                {"op":"column","args":["bid_size"]},3,"mean"]}},
            {"name":"r","expr":{"op":"reconstruct","args":[
                {"op":"column","args":["price"]},2,"last"]}}]}"#;
        let specs = parse_spec(json).expect("valid spec");
        assert!(matches!(specs[0].expr, Expr::Window { size: 3, .. }));
        assert!(matches!(specs[1].expr, Expr::Reconstruct { size: 2, .. }));
    }

    #[test]
    fn unknown_op_is_an_error() {
        let json = r#"{"version":1,"features":[
            {"name":"x","expr":{"op":"skew","args":[]}}]}"#;
        assert!(parse_spec(json).is_err());
    }

    #[test]
    fn zero_window_size_is_an_error() {
        let json = r#"{"version":1,"features":[
            {"name":"x","expr":{"op":"window","args":[
                {"op":"column","args":["a"]},0,"mean"]}}]}"#;
        assert!(parse_spec(json).is_err());
    }

    #[test]
    fn unknown_version_is_an_error() {
        assert!(parse_spec(r#"{"version":99,"features":[]}"#).is_err());
    }
}
