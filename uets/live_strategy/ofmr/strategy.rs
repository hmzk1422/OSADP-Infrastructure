//! OFMR state builder — hand-rolled LightGBM text-model inference over
//! Column_0..Column_14.
//!
//! This parser intentionally supports binary LightGBM models with ordinary
//! numerical splits. It rejects unsupported model features instead of silently
//! producing potentially incorrect trading predictions.

use std::fs;

use crate::core::feature::FeatureRow;
use crate::core::strategy::{Decision, MarketState};
use crate::live_strategy::ofmr::settings::Settings;
use crate::live_strategy::StateBuilder;

const N_FEATURES: usize = 15;
const BARRIER_MULTIPLIER: f64 = 0.3;
const REWARD_RISK_RATIO: f64 = 10.0;
const MIN_VOLATILITY_FRACTION: f64 = 0.001;

// LightGBM's numerical tree implementation treats values in this range as zero
// when that split was trained with MissingType::Zero.
const LIGHTGBM_ZERO_THRESHOLD: f64 = 1.0e-35;

/// Convert trained timeframe close-return volatility to stop and target
/// fractions, matching the research label builder.
///
/// `bar_vol_std` is already calculated on the configured higher-timeframe
/// closes (300 seconds by default); scaling it by sqrt(5) again would make
/// playback barriers wider than the research labels.
pub fn barrier_fractions(bar_vol_std: Option<f64>) -> (f64, f64) {
    let volatility = bar_vol_std
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(MIN_VOLATILITY_FRACTION)
        .max(MIN_VOLATILITY_FRACTION);
    let stop = volatility * BARRIER_MULTIPLIER;
    (stop, stop * REWARD_RISK_RATIO)
}

/// Calculate barrier fractions from the one-minute population-volatility
/// feature produced by feature computation.
///
/// `bar_vol_std` must match the training pipeline's rolling population
/// standard deviation of configured higher-timeframe close returns.
pub fn barrier_fractions_from_features(features: &FeatureRow) -> (f64, f64) {
    barrier_fractions(features.get("bar_vol_std"))
}

/// Parsed LightGBM numerical split.
#[derive(Clone, Debug)]
struct TreeNode {
    split_feature: usize,
    threshold: f64,
    left_child: i32,
    right_child: i32,
    default_left: bool,
    missing_type: u8,
}

/// Parsed LightGBM tree. LightGBM serializes the leaf values after learning-rate
/// shrinkage has been applied, so inference must sum leaf values directly and
/// must NOT multiply them by the serialized `shrinkage` field a second time.
#[derive(Clone, Debug)]
struct GbtTree {
    nodes: Vec<TreeNode>,
    leaf_values: Vec<f64>,
}

/// Parsed binary LightGBM text model.
struct LightGBMModel {
    trees: Vec<GbtTree>,
    sigmoid_scale: f64,
}

impl LightGBMModel {
    /// Load a LightGBM text model and reject model types this parser cannot
    /// faithfully evaluate.
    fn load(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("failed to read model file {path}: {e}"))?;
        let lines: Vec<&str> = content.lines().collect();

        let objective = header_value(&lines, "objective=")
            .ok_or_else(|| "model header is missing objective".to_string())?;
        let objective_kind = objective.split_whitespace().next().unwrap_or("");
        if objective_kind != "binary" {
            return Err(format!(
                "unsupported LightGBM objective '{objective_kind}'; expected binary"
            ));
        }

        let mut sigmoid_scale = 1.0_f64;
        for token in objective.split_whitespace().skip(1) {
            if let Some(value) = token.strip_prefix("sigmoid:") {
                sigmoid_scale = value
                    .parse::<f64>()
                    .map_err(|e| format!("bad objective sigmoid value: {e}"))?;
            }
        }
        if !sigmoid_scale.is_finite() || sigmoid_scale <= 0.0 {
            return Err(format!(
                "invalid LightGBM sigmoid scale: {sigmoid_scale}"
            ));
        }

        if let Some(value) = header_value(&lines, "num_class=") {
            let num_class = value
                .parse::<usize>()
                .map_err(|e| format!("bad num_class: {e}"))?;
            if num_class != 1 {
                return Err(format!(
                    "unsupported num_class={num_class}; expected 1 for binary classification"
                ));
            }
        }

        if let Some(value) = header_value(&lines, "num_tree_per_iteration=") {
            let count = value
                .parse::<usize>()
                .map_err(|e| format!("bad num_tree_per_iteration: {e}"))?;
            if count != 1 {
                return Err(format!(
                    "unsupported num_tree_per_iteration={count}; expected 1"
                ));
            }
        }

        if let Some(value) = header_value(&lines, "max_feature_idx=") {
            let max_feature_idx = value
                .parse::<usize>()
                .map_err(|e| format!("bad max_feature_idx: {e}"))?;
            if max_feature_idx != N_FEATURES - 1 {
                return Err(format!(
                    "model expects max feature index {max_feature_idx}, but this strategy provides {} features (expected max index {})",
                    N_FEATURES,
                    N_FEATURES - 1
                ));
            }
        }

        if let Some(value) = header_value(&lines, "average_output=") {
            let average_output = value
                .parse::<u8>()
                .map_err(|e| format!("bad average_output: {e}"))?;
            if average_output != 0 {
                return Err(
                    "average_output models are not supported by this binary classifier parser"
                        .to_string(),
                );
            }
        }

        let mut trees = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            if lines[i].trim().starts_with("Tree=") {
                trees.push(Self::parse_tree(&lines, &mut i)?);
            } else {
                i += 1;
            }
        }

        if trees.is_empty() {
            return Err(format!("no trees found in model file: {path}"));
        }

        Ok(Self {
            trees,
            sigmoid_scale,
        })
    }

    /// Parse one Tree=N block. `i` points to the Tree=N line on entry.
    fn parse_tree(lines: &[&str], i: &mut usize) -> Result<GbtTree, String> {
        let tree_line = lines
            .get(*i)
            .ok_or_else(|| "unexpected end of model while reading tree".to_string())?
            .trim();
        if !tree_line.starts_with("Tree=") {
            return Err(format!(
                "expected Tree=N at line index {}, got: {}",
                *i, tree_line
            ));
        }
        tree_line["Tree=".len()..]
            .parse::<usize>()
            .map_err(|e| format!("bad tree index in '{tree_line}': {e}"))?;
        *i += 1;

        let mut num_leaves: Option<usize> = None;
        let mut num_cat = 0_usize;
        let mut is_linear = false;
        let mut split_feature: Vec<usize> = Vec::new();
        let mut threshold: Vec<f64> = Vec::new();
        let mut decision_type: Option<Vec<i32>> = None;
        let mut left_child: Vec<i32> = Vec::new();
        let mut right_child: Vec<i32> = Vec::new();
        let mut leaf_values: Vec<f64> = Vec::new();

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
                num_leaves = Some(
                    rest.parse::<usize>()
                        .map_err(|e| format!("bad num_leaves: {e}"))?,
                );
            } else if let Some(rest) = line.strip_prefix("num_cat=") {
                num_cat = rest
                    .parse::<usize>()
                    .map_err(|e| format!("bad num_cat: {e}"))?;
            } else if let Some(rest) = line.strip_prefix("is_linear=") {
                is_linear = match rest.parse::<u8>() {
                    Ok(0) => false,
                    Ok(1) => true,
                    Ok(other) => {
                        return Err(format!("invalid is_linear value: {other}"));
                    }
                    Err(e) => return Err(format!("bad is_linear: {e}")),
                };
            } else if let Some(rest) = line.strip_prefix("split_feature=") {
                split_feature = parse_usize_list(rest)?;
            } else if let Some(rest) = line.strip_prefix("threshold=") {
                threshold = parse_f64_list(rest)?;
            } else if let Some(rest) = line.strip_prefix("decision_type=") {
                decision_type = Some(parse_i32_list(rest)?);
            } else if let Some(rest) = line.strip_prefix("left_child=") {
                left_child = parse_i32_list(rest)?;
            } else if let Some(rest) = line.strip_prefix("right_child=") {
                right_child = parse_i32_list(rest)?;
            } else if let Some(rest) = line.strip_prefix("leaf_value=") {
                leaf_values = parse_f64_list(rest)?;
            }

            // Deliberately do not apply the serialized `shrinkage=` value.
            // LightGBM's stored leaf_value entries have already been shrunk.
            *i += 1;
        }

        let num_leaves = num_leaves
            .ok_or_else(|| "tree is missing num_leaves".to_string())?;
        if num_leaves == 0 {
            return Err("tree has zero leaves".to_string());
        }
        if leaf_values.len() != num_leaves {
            return Err(format!(
                "tree leaf_value count {} != num_leaves {}",
                leaf_values.len(),
                num_leaves
            ));
        }
        if leaf_values.iter().any(|v| !v.is_finite()) {
            return Err("tree contains a non-finite leaf value".to_string());
        }
        if num_cat != 0 {
            return Err(format!(
                "categorical LightGBM splits are not supported (num_cat={num_cat})"
            ));
        }
        if is_linear {
            return Err("linear LightGBM trees are not supported".to_string());
        }

        let n_internal = split_feature.len();
        if n_internal != num_leaves.saturating_sub(1) {
            return Err(format!(
                "tree structure mismatch: {} internal nodes for {} leaves",
                n_internal, num_leaves
            ));
        }
        if threshold.len() != n_internal {
            return Err(format!(
                "tree threshold count {} != split count {}",
                threshold.len(), n_internal
            ));
        }
        if left_child.len() != n_internal || right_child.len() != n_internal {
            return Err("tree child count mismatch".to_string());
        }

        // LightGBM v4 writes decision_type. Falling back to zero matches
        // LightGBM's own tree-text loader for older text models.
        let decision_type = decision_type.unwrap_or_else(|| vec![0; n_internal]);
        if decision_type.len() != n_internal {
            return Err(format!(
                "tree decision_type count {} != split count {}",
                decision_type.len(), n_internal
            ));
        }

        let mut nodes = Vec::with_capacity(n_internal);
        for index in 0..n_internal {
            let feature = split_feature[index];
            let threshold_value = threshold[index];
            let decision = decision_type[index];

            if feature >= N_FEATURES {
                return Err(format!(
                    "tree node {index} references feature {feature}, but only {} features are provided",
                    N_FEATURES
                ));
            }
            if !threshold_value.is_finite() {
                return Err(format!(
                    "tree node {index} has a non-finite threshold"
                ));
            }
            if !(0..=15).contains(&decision) {
                return Err(format!(
                    "tree node {index} has unsupported decision_type={decision}"
                ));
            }

            // LightGBM decision_type bit 0 marks categorical splits.
            // They are rejected above/ here rather than evaluated as numeric.
            if decision & 1 != 0 {
                return Err(format!(
                    "categorical split at tree node {index} is not supported"
                ));
            }

            // Bits 2-3 encode MissingType: None=0, NaN=1, Zero=2.
            let missing_type = ((decision >> 2) & 0b11) as u8;
            if missing_type > 2 {
                return Err(format!(
                    "tree node {index} has unsupported missing type {missing_type}"
                ));
            }

            nodes.push(TreeNode {
                split_feature: feature,
                threshold: threshold_value,
                left_child: left_child[index],
                right_child: right_child[index],
                default_left: decision & 0b10 != 0,
                missing_type,
            });
        }

        // LightGBM stores internal children as non-negative node indexes and
        // leaves as negative indexes encoded as -leaf_index - 1.
        for child in left_child.iter().chain(right_child.iter()) {
            if *child >= 0 {
                if (*child as usize) >= n_internal {
                    return Err(format!(
                        "tree contains invalid internal child index {child}"
                    ));
                }
            } else {
                let leaf_index = -(i64::from(*child)) - 1;
                if leaf_index < 0 || leaf_index as usize >= num_leaves {
                    return Err(format!(
                        "tree contains invalid leaf child index {child}"
                    ));
                }
            }
        }

        Ok(GbtTree { nodes, leaf_values })
    }

    /// Return P(class 1) for the supported binary objective.
    ///
    /// A malformed tree returns None instead of panicking or looping forever.
    fn predict(&self, features: &[f64; N_FEATURES]) -> Option<f64> {
        let mut score = 0.0_f64;

        for tree in &self.trees {
            let leaf_value = if tree.nodes.is_empty() {
                // LightGBM supports constant trees with one leaf and no splits.
                if tree.leaf_values.len() != 1 {
                    return None;
                }
                tree.leaf_values[0]
            } else {
                let mut node_index = 0_usize;
                let mut found_leaf: Option<f64> = None;

                // A valid tree cannot require more than nodes.len() + 1 steps.
                // The bound also prevents an invalid child cycle from hanging.
                for _ in 0..=tree.nodes.len() {
                    let node = tree.nodes.get(node_index)?;
                    let feature_value = *features.get(node.split_feature)?;

                    // This strategy rejects non-finite model inputs before
                    // prediction. Still implement the zero-as-missing behavior
                    // used by LightGBM for splits trained with MissingType::Zero.
                    let zero_is_missing = node.missing_type == 2
                        && feature_value.abs() <= LIGHTGBM_ZERO_THRESHOLD;

                    let child = if zero_is_missing {
                        if node.default_left {
                            node.left_child
                        } else {
                            node.right_child
                        }
                    } else if feature_value <= node.threshold {
                        node.left_child
                    } else {
                        node.right_child
                    };

                    if child < 0 {
                        let leaf_index = (-(i64::from(child)) - 1) as usize;
                        found_leaf = Some(*tree.leaf_values.get(leaf_index)?);
                        break;
                    }

                    node_index = child as usize;
                }

                found_leaf?
            };

            score += leaf_value;
            if !score.is_finite() {
                return None;
            }
        }

        let scaled_score = score * self.sigmoid_scale;
        if !scaled_score.is_finite() {
            return None;
        }

        Some(sigmoid(scaled_score))
    }
}

fn header_value<'a>(lines: &[&'a str], key: &str) -> Option<&'a str> {
    for line in lines {
        let line = (*line).trim();
        if line.starts_with("Tree=") {
            break;
        }
        if let Some(value) = line.strip_prefix(key) {
            return Some(value.trim());
        }
    }
    None
}

fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    s.split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|e| format!("parse f64 '{token}': {e}"))
        })
        .collect()
}

fn parse_i32_list(s: &str) -> Result<Vec<i32>, String> {
    s.split_whitespace()
        .map(|token| {
            token
                .parse::<i32>()
                .map_err(|e| format!("parse i32 '{token}': {e}"))
        })
        .collect()
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>, String> {
    s.split_whitespace()
        .map(|token| {
            token
                .parse::<usize>()
                .map_err(|e| format!("parse feature index '{token}': {e}"))
        })
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
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
        let model = match LightGBMModel::load(&settings.model.path) {
            Ok(model) => Some(model),
            Err(error) => {
                eprintln!("OFMR model could not be loaded: {error}");
                None
            }
        };

        let threshold = settings.model.threshold;
        Self {
            model,
            settings,
            threshold,
        }
    }

    /// Construct OFMR only when its configured LightGBM model can be loaded.
    /// Offline playback uses this to avoid reporting a successful all-Hold
    /// backtest when model loading failed.
    pub fn try_with_settings(settings: Settings) -> Result<Self, String> {
        let model = LightGBMModel::load(&settings.model.path)?;
        let threshold = settings.model.threshold;
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(format!("invalid OFMR model threshold: {threshold}"));
        }
        Ok(Self {
            model: Some(model),
            settings,
            threshold,
        })
    }

    /// Build the model input in Column_0 .. Column_14 order.
    /// Missing and non-finite values are rejected rather than guessed.
    fn features_from_row(row: &FeatureRow) -> Option<[f64; N_FEATURES]> {
        let mut output = [0.0_f64; N_FEATURES];
        for (index, slot) in output.iter_mut().enumerate() {
            let name = format!("Column_{index}");
            let value = row.get(&name)?;
            if !value.is_finite() {
                return None;
            }
            *slot = value;
        }
        Some(output)
    }

    /// Return volatility-derived stop and target fractions for the execution
    /// / risk-management layer. These are fractions of entry price, not prices.
    /// The existing Decision API cannot carry stop/target values by itself.
    pub fn barrier_fractions_for_state(state: &MarketState) -> (f64, f64) {
        barrier_fractions_from_features(&state.features)
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

        // A malformed threshold must not accidentally allow every trade.
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            return Decision::Hold;
        }

        let Some(features) = Self::features_from_row(&state.features) else {
            // Original behavior preserved: missing input means Hold.
            return Decision::Hold;
        };

        let Some(probability) = model.predict(&features) else {
            return Decision::Hold;
        };

        if probability < self.threshold {
            return Decision::Hold;
        }

        // Preserve the original trade-direction behavior exactly.
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
    use crate::core::strategy::Position;

    fn make_state(values: &[Option<f64>; N_FEATURES], position: Position) -> MarketState {
        let mut row = FeatureRow::new();
        for (index, value) in values.iter().enumerate() {
            match value {
                Some(x) => row.insert(format!("Column_{index}"), *x),
                None => row.insert_missing(format!("Column_{index}")),
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
    fn sigmoid_scale_is_applied_to_raw_score() {
        let model = LightGBMModel {
            trees: vec![GbtTree {
                nodes: Vec::new(),
                leaf_values: vec![0.25],
            }],
            sigmoid_scale: 2.0,
        };
        let features = [0.0_f64; N_FEATURES];
        let predicted = model.predict(&features).unwrap();
        assert!((predicted - sigmoid(0.5)).abs() < 1e-12);
    }

    #[test]
    fn leaf_values_are_not_shrunk_a_second_time() {
        let model = LightGBMModel {
            trees: vec![GbtTree {
                nodes: Vec::new(),
                // LightGBM serializes this value after training shrinkage.
                leaf_values: vec![0.2],
            }],
            sigmoid_scale: 1.0,
        };
        let features = [0.0_f64; N_FEATURES];
        assert!((model.predict(&features).unwrap() - sigmoid(0.2)).abs() < 1e-12);
    }

    #[test]
    fn zero_as_missing_uses_the_serialized_default_direction() {
        let model = LightGBMModel {
            trees: vec![GbtTree {
                nodes: vec![TreeNode {
                    split_feature: 0,
                    threshold: -1.0,
                    left_child: -1,
                    right_child: -2,
                    default_left: true,
                    missing_type: 2,
                }],
                leaf_values: vec![0.2, 0.8],
            }],
            sigmoid_scale: 1.0,
        };
        let features = [0.0_f64; N_FEATURES];
        assert!((model.predict(&features).unwrap() - sigmoid(0.2)).abs() < 1e-12);
    }

    #[test]
    fn missing_features_hold() {
        let strategy = OfmrStrategy::new();
        let values: [Option<f64>; N_FEATURES] = [None; N_FEATURES];
        let state = make_state(&values, Position::Flat);
        assert_eq!(strategy.decide(&state), Decision::Hold);
    }

    #[test]
    fn populated_features_do_not_panic_without_a_valid_model() {
        let strategy = OfmrStrategy::new();
        let values: [Option<f64>; N_FEATURES] = [Some(0.5); N_FEATURES];
        let state = make_state(&values, Position::Flat);
        let _ = strategy.decide(&state);
    }

    #[test]
    fn barriers_match_research_timeframe_volatility_and_ten_to_one_rr() {
        let (stop, target) = barrier_fractions(Some(0.002));
        let expected_stop = 0.002 * 0.3;
        assert!((stop - expected_stop).abs() < 1e-12);
        assert!((target / stop - 10.0).abs() < 1e-12);
    }

    #[test]
    fn barriers_use_the_training_minimum_when_volatility_is_unavailable() {
        let (stop, target) = barrier_fractions(None);
        assert!((stop - 0.001 * 0.3).abs() < 1e-12);
        assert!((target - stop * 10.0).abs() < 1e-12);
    }
}
