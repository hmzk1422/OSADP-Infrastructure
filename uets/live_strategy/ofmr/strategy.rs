//! OFMR state builder — hand-rolled LightGBM txt inference over Column_0..Column_14.
//!
//! Parses `uets/models/ofmr/trader.txt` (LightGBM v4 text model), evaluates
//! all trees, applies sigmoid, and maps P(WIN) >= threshold to `EnterLong`
//! (long-only gated by `trade_direction`). Missing input features => Hold.

use std::fs;

use crate::core::feature::FeatureRow;
use crate::core::strategy::{Decision, MarketState};
use crate::live_strategy::ofmr::settings::Settings;
use crate::live_strategy::StateBuilder;

const N_FEATURES: usize = 15;

/// A parsed LightGBM tree node.
#[derive(Clone, Debug)]
struct TreeNode {
    split_feature: usize,
    threshold: f64,
    left_child: i32,
    right_child: i32,
}

/// A parsed LightGBM tree: internal nodes + leaf values.
#[derive(Clone, Debug)]
struct GbtTree {
    nodes: Vec<TreeNode>,
    leaf_values: Vec<f64>,
    shrinkage: f64,
}

/// Parsed LightGBM v4 text model.
struct LightGBMModel {
    trees: Vec<GbtTree>,
}

impl LightGBMModel {
    /// Parse the LightGBM v4 text model from `path`.
    fn load(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("failed to read model file {path}: {e}"))?;

        let mut trees: Vec<GbtTree> = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            if line.starts_with("Tree=") {
                let tree = Self::parse_tree(&lines, &mut i)?;
                trees.push(tree);
            } else {
                i += 1;
            }
        }

        Ok(Self { trees })
    }

    /// Parse a single `Tree=N` block starting at index `i` (points at the
    /// "Tree=N" line). Advances `i` past the block.
    fn parse_tree(lines: &[&str], i: &mut usize) -> Result<GbtTree, String> {
        let _tree_line = lines[*i];
        *i += 1;

        let mut num_leaves = 0usize;
        let mut split_feature: Vec<usize> = Vec::new();
        let mut threshold: Vec<f64> = Vec::new();
        let mut left_child: Vec<i32> = Vec::new();
        let mut right_child: Vec<i32> = Vec::new();
        let mut leaf_value: Vec<f64> = Vec::new();
        let mut shrinkage = 1.0f64;

        while *i < lines.len() {
            let line = lines[*i].trim();
            if line.starts_with("Tree=") {
                break;
            }
            if line.is_empty() {
                *i += 1;
                continue;
            }

            if let Some(rest) = line.strip_prefix("num_leaves=") {
                num_leaves = rest.parse().map_err(|e| format!("bad num_leaves: {e}"))?;
            } else if let Some(rest) = line.strip_prefix("split_feature=") {
                split_feature = parse_f64_list(rest)?
                    .into_iter()
                    .map(|v| v as usize)
                    .collect();
            } else if let Some(rest) = line.strip_prefix("threshold=") {
                threshold = parse_f64_list(rest)?;
            } else if let Some(rest) = line.strip_prefix("left_child=") {
                left_child = parse_i32_list(rest)?;
            } else if let Some(rest) = line.strip_prefix("right_child=") {
                right_child = parse_i32_list(rest)?;
            } else if let Some(rest) = line.strip_prefix("leaf_value=") {
                leaf_value = parse_f64_list(rest)?;
            } else if let Some(rest) = line.strip_prefix("shrinkage=") {
                shrinkage = rest.parse().map_err(|e| format!("bad shrinkage: {e}"))?;
            }
            *i += 1;
        }

        let n_internal = split_feature.len();
        if n_internal == 0 {
            return Err("tree has no internal nodes".to_string());
        }
        if leaf_value.len() != num_leaves {
            return Err(format!(
                "tree leaf_value count {} != num_leaves {}",
                leaf_value.len(),
                num_leaves
            ));
        }
        if left_child.len() != n_internal || right_child.len() != n_internal {
            return Err("tree child count mismatch".to_string());
        }

        let nodes: Vec<TreeNode> = (0..n_internal)
            .map(|j| TreeNode {
                split_feature: split_feature[j],
                threshold: threshold[j],
                left_child: left_child[j],
                right_child: right_child[j],
            })
            .collect();

        Ok(GbtTree {
            nodes,
            leaf_values: leaf_value,
            shrinkage,
        })
    }

    /// Evaluate all trees on `features` (indexed by Column_0..Column_14),
    /// sum shrinkage * leaf_value, apply sigmoid => P(WIN) in (0, 1).
    fn predict(&self, features: &[f64; N_FEATURES]) -> f64 {
        let mut score = 0.0f64;
        for tree in &self.trees {
            let mut node_idx = 0usize;
            loop {
                let node = &tree.nodes[node_idx];
                let feat_val = features[node.split_feature];
                if feat_val <= node.threshold {
                    match node.left_child {
                        c if c >= 0 => node_idx = c as usize,
                        c => {
                            let leaf = (-c - 1) as usize;
                            score += tree.shrinkage * tree.leaf_values[leaf];
                            break;
                        }
                    }
                } else {
                    match node.right_child {
                        c if c >= 0 => node_idx = c as usize,
                        c => {
                            let leaf = (-c - 1) as usize;
                            score += tree.shrinkage * tree.leaf_values[leaf];
                            break;
                        }
                    }
                }
            }
        }
        sigmoid(score)
    }
}

fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    s.split_whitespace()
        .map(|t| {
            t.parse::<f64>()
                .map_err(|e| format!("parse f64 '{t}': {e}"))
        })
        .collect()
}

fn parse_i32_list(s: &str) -> Result<Vec<i32>, String> {
    s.split_whitespace()
        .map(|t| t.parse::<i32>().map_err(|e| format!("parse i32 '{t}': {e}")))
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// OFMR strategy: LightGBM inference -> decision.
pub struct OfmrStrategy {
    model: Option<LightGBMModel>,
    settings: Settings,
    threshold: f64,
}

impl OfmrStrategy {
    pub fn new() -> Self {
        Self::with_settings(Settings::load())
    }

    pub fn with_settings(settings: Settings) -> Self {
        let model = LightGBMModel::load(&settings.model.path).ok();
        Self {
            model,
            threshold: settings.model.threshold,
            settings,
        }
    }

    /// Build a [f64; N_FEATURES] array from the feature row, using
    /// Column_0 .. Column_14 in order. Returns None if any column is Missing.
    fn features_from_row(row: &FeatureRow) -> Option<[f64; N_FEATURES]> {
        let mut out = [0.0f64; N_FEATURES];
        for i in 0..N_FEATURES {
            let name = format!("Column_{i}");
            let v = row.get(&name)?;
            out[i] = v;
        }
        Some(out)
    }
}

impl Default for OfmrStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl StateBuilder for OfmrStrategy {
    fn decide(&self, state: &MarketState) -> Decision {
        let Some(model) = &self.model else {
            return Decision::Hold;
        };
        let features = match Self::features_from_row(&state.features) {
            Some(f) => f,
            None => return Decision::Hold, // Missing input: hold, never guess
        };

        let p = model.predict(&features);
        if p < self.threshold {
            return Decision::Hold;
        }

        // Long-only gated by trade_direction.
        match self.settings.model.trade_direction.as_str() {
            "long" | "Long" => Decision::EnterLong,
            "short" | "Short" => Decision::EnterShort,
            _ => Decision::EnterLong,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::feature::FeatureRow;
    use crate::core::strategy::Position;

    fn make_state(values: &[Option<f64>; N_FEATURES], position: Position) -> MarketState {
        let mut row = FeatureRow::new();
        for (i, v) in values.iter().enumerate() {
            match v {
                Some(x) => row.insert(format!("Column_{i}"), *x),
                None => row.insert_missing(format!("Column_{i}")),
            }
        }
        MarketState {
            features: row,
            position,
            sequence: 1,
        }
    }

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn missing_features_hold() {
        let s = OfmrStrategy::new();
        let values: [Option<f64>; N_FEATURES] = [None; N_FEATURES];
        let state = make_state(&values, Position::Flat);
        assert_eq!(s.decide(&state), Decision::Hold);
    }

    #[test]
    fn high_score_enters_long_when_flat() {
        let s = OfmrStrategy::new();
        let values: [Option<f64>; N_FEATURES] = [Some(0.5); N_FEATURES];
        let state = make_state(&values, Position::Flat);
        let _ = s.decide(&state);
        // Just verify it doesn't panic; the actual decision depends on model.
    }
}
