//! LightGBM text-model parser + predictor (zero-dependency, std only).
//!
//! Reads the `main/models/lgb_model.txt` export produced by
//! `research/ml_state_optimized.py` STEP 12 and reproduces
//! `lightgbm.Booster.predict` for `objective=binary sigmoid:1` — the exact
//! P(WIN) the live `main/executor.py` feeds into its entry gate.
//!
//! Text-format semantics (empirically verified against lightgbm 4.6.0 on the
//! shipped model, see the `real_model_matches_lightgbm` test):
//!   * `split_feature` / `threshold` / `decision_type` / `left_child` /
//!     `right_child` have `num_leaves - 1` entries (internal nodes), and
//!     `leaf_value` has `num_leaves` entries;
//!   * a child index `>= 0` is another internal node; a negative child index
//!     encodes a leaf with `leaf_idx = -child - 1`;
//!   * `value <= threshold` goes LEFT (this model's `decision_type=2` behaves as
//!     the numerical default `<=`; no categorical splits, `num_cat=0`);
//!   * the raw margin is the sum of the reached `leaf_value`s (the dump already
//!     bakes in the per-tree learning-rate / shrinkage), and
//!     `P(WIN) = 1 / (1 + exp(-raw))`.

use std::path::Path;

/// One decision tree (internal + leaf nodes in the LightGBM compact layout).
#[derive(Debug, Clone)]
struct LgbTree {
    #[allow(dead_code)] // kept for structural validation
    num_leaves: usize,
    split_feature: Vec<usize>,
    threshold: Vec<f64>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    leaf_value: Vec<f64>,
}

/// A parsed LightGBM binary-classification model.
#[derive(Debug, Clone)]
pub struct LightGbm {
    num_features: usize,
    trees: Vec<LgbTree>,
}

impl LightGbm {
    /// Parse a LightGBM v4 text dump.
    pub fn from_str(text: &str) -> Result<Self, String> {
        let mut num_features = 0usize;
        let mut trees: Vec<LgbTree> = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        // --- header until the first Tree= block ---
        while i < lines.len() && !lines[i].trim_start().starts_with("Tree=") {
            if let Some((k, v)) = lines[i].trim().split_once('=') {
                if k.trim() == "max_feature_idx" {
                    let idx: i64 = v
                        .trim()
                        .parse()
                        .map_err(|_| "bad max_feature_idx".to_string())?;
                    num_features = idx as usize + 1;
                }
            }
            i += 1;
        }

        // --- one block per tree ---
        while i < lines.len() {
            let ln = lines[i].trim();
            if !ln.starts_with("Tree=") {
                i += 1;
                continue;
            }
            let mut block: Vec<(String, String)> = Vec::new();
            i += 1;
            while i < lines.len() {
                let l = lines[i].trim();
                if l.is_empty() {
                    i += 1;
                    break;
                }
                if let Some((k, v)) = l.split_once('=') {
                    block.push((k.trim().to_string(), v.trim().to_string()));
                }
                i += 1;
            }
            let get = |key: &str| -> Result<&str, String> {
                block
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.as_str())
                    .ok_or_else(|| format!("tree {} missing '{key}'", trees.len()))
            };

            let num_leaves: usize = get("num_leaves")?
                .parse()
                .map_err(|_| "bad num_leaves".to_string())?;
            let parse_usize = |key: &str| -> Result<Vec<usize>, String> {
                get(key)?
                    .split_whitespace()
                    .map(|s| {
                        s.parse::<i64>()
                            .map(|v| v as usize)
                            .map_err(|_| format!("bad {key} value {s}"))
                    })
                    .collect()
            };
            let parse_f64 = |key: &str| -> Result<Vec<f64>, String> {
                get(key)?
                    .split_whitespace()
                    .map(|s| s.parse::<f64>().map_err(|_| format!("bad {key} value {s}")))
                    .collect()
            };
            let parse_i32 = |key: &str| -> Result<Vec<i32>, String> {
                get(key)?
                    .split_whitespace()
                    .map(|s| s.parse::<i32>().map_err(|_| format!("bad {key} value {s}")))
                    .collect()
            };

            let split_feature = parse_usize("split_feature")?;
            let threshold = parse_f64("threshold")?;
            let left_child = parse_i32("left_child")?;
            let right_child = parse_i32("right_child")?;
            let leaf_value = parse_f64("leaf_value")?;

            let n_internal = num_leaves.saturating_sub(1);
            if split_feature.len() != n_internal
                || threshold.len() != n_internal
                || left_child.len() != n_internal
                || right_child.len() != n_internal
            {
                return Err(format!(
                    "tree {}: expected {n_internal} internal nodes, got split_feature={}, threshold={}, left_child={}, right_child={}",
                    trees.len(),
                    split_feature.len(),
                    threshold.len(),
                    left_child.len(),
                    right_child.len(),
                ));
            }
            if leaf_value.len() != num_leaves {
                return Err(format!(
                    "tree {}: expected {num_leaves} leaf values, got {}",
                    trees.len(),
                    leaf_value.len()
                ));
            }
            // Track the widest feature index actually used.
            if let Some(&f) = split_feature.iter().max() {
                num_features = num_features.max(f + 1);
            }

            trees.push(LgbTree {
                num_leaves,
                split_feature,
                threshold,
                left_child,
                right_child,
                leaf_value,
            });
        }

        if trees.is_empty() {
            return Err("no Tree= blocks found in model".to_string());
        }
        Ok(LightGbm {
            num_features,
            trees,
        })
    }

    /// Load from a `.txt` model file on disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read model: {e}"))?;
        Self::from_str(&text)
    }

    /// Number of features the model was trained on.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Raw margin before the sigmoid (lightgbm `predict(raw_score=True)`).
    pub fn predict_raw(&self, features: &[f64]) -> f64 {
        let mut raw = 0.0;
        for tree in &self.trees {
            let mut node: i64 = 0;
            loop {
                debug_assert!(node >= 0, "leaf reached before terminal child");
                let ni = node as usize;
                let f = tree.split_feature[ni];
                let val = features.get(f).copied().unwrap_or(0.0);
                let child = if val <= tree.threshold[ni] {
                    tree.left_child[ni]
                } else {
                    tree.right_child[ni]
                };
                if child < 0 {
                    let leaf = (-child - 1) as usize;
                    raw += tree.leaf_value[leaf];
                    break;
                }
                node = child as i64;
            }
        }
        raw
    }

    /// P(class = 1) = sigmoid(raw), i.e. `Booster.predict` for binary objective.
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        let raw = self.predict_raw(features);
        1.0 / (1.0 + (-raw).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny hand-built 3-leaf tree exercises every encoding rule.
    #[test]
    fn synthetic_tree_prediction() {
        let text = r#"tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=1
objective=binary sigmoid:1
feature_names=Column_0 Column_1
Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
threshold=0.5 0.5
decision_type=2 2
left_child=1 -1
right_child=-2 -3
leaf_value=0.1 0.2 -0.3
internal_value=0 0
internal_weight=1 1
internal_count=3 2
is_linear=0
shrinkage=1
"#;
        let m = LightGbm::from_str(text).expect("parse");
        assert_eq!(m.num_features(), 2);
        assert_eq!(m.num_trees(), 1);
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        // f0<=0.5 & f1<=0.5 -> node1 left, child -1 -> leaf 0 (+0.1)
        assert!((m.predict_raw(&[0.0, 0.0]) - 0.1).abs() < 1e-12);
        assert!((m.predict_proba(&[0.0, 0.0]) - sig(0.1)).abs() < 1e-12);
        // f0<=0.5 & f1>0.5 -> node1 right, child -3 -> leaf 2 (-0.3)
        assert!((m.predict_raw(&[0.0, 1.0]) - -0.3).abs() < 1e-12);
        // f0>0.5 -> node0 right, child -2 -> leaf 1 (+0.2)
        assert!((m.predict_raw(&[1.0, 0.0]) - 0.2).abs() < 1e-12);
        // missing feature index -> treated as 0.0 (goes left at node 0)
        assert!((m.predict_raw(&[0.0]) - 0.1).abs() < 1e-12);
    }

    /// Resolve a file by walking up from CWD (tests inherit the package dir).
    fn find_upwards(paths: &[&str]) -> Option<std::path::PathBuf> {
        let cwd = std::env::current_dir().ok()?;
        let mut dir: Option<&std::path::Path> = Some(cwd.as_path());
        while let Some(d) = dir {
            for p in paths {
                let cand = d.join(p);
                if cand.exists() {
                    return Some(cand);
                }
            }
            dir = d.parent();
        }
        None
    }

    /// Cross-check the Rust parser against ground truth captured from the Python
    /// `lightgbm` library (build/gen_lgbm_cases.py) on the REAL research model.
    /// Skips gracefully when the model/cases files are absent.
    #[test]
    fn real_model_matches_lightgbm() {
        let model_path = match find_upwards(&["main/models/lgb_model.txt"]) {
            Some(p) => p,
            None => {
                eprintln!("skipping: lgb_model.txt not found");
                return;
            }
        };
        let cases_path = match find_upwards(&["build/lgbm_cases.json"]) {
            Some(p) => p,
            None => {
                eprintln!("skipping: lgbm_cases.json not found");
                return;
            }
        };
        let model = LightGbm::from_path(&model_path).expect("load real model");
        let cases: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cases_path).expect("read cases"))
                .expect("parse cases");
        let feats = cases["features"].as_array().expect("features");
        let raw_expected = cases["raw_score"].as_array().expect("raw_score");
        let prob_expected = cases["prob"].as_array().expect("prob");
        let leaves_expected = cases["pred_leaf"].as_array().expect("pred_leaf");

        assert_eq!(feats.len(), raw_expected.len());
        let mut leaf_mismatches = 0usize;
        for (i, row) in feats.iter().enumerate() {
            let x: Vec<f64> = row
                .as_array()
                .expect("row")
                .iter()
                .map(|v| v.as_f64().expect("num"))
                .collect();
            assert_eq!(x.len(), model.num_features());
            let raw = model.predict_raw(&x);
            let prob = model.predict_proba(&x);
            let exp_raw = raw_expected[i].as_f64().expect("num");
            let exp_prob = prob_expected[i].as_f64().expect("num");
            assert!(
                (raw - exp_raw).abs() < 1e-9,
                "case {i}: raw {raw:.12} != {exp_raw:.12}"
            );
            assert!(
                (prob - exp_prob).abs() < 1e-9,
                "case {i}: prob {prob:.12} != {exp_prob:.12}"
            );
            // leaf index per tree must also match lightgbm's pred_leaf exactly
            let my_leaves = model
                .trees
                .iter()
                .map(|t| {
                    let mut node: i64 = 0;
                    loop {
                        let ni = node as usize;
                        let f = t.split_feature[ni];
                        let val = x.get(f).copied().unwrap_or(0.0);
                        let child = if val <= t.threshold[ni] {
                            t.left_child[ni]
                        } else {
                            t.right_child[ni]
                        };
                        if child < 0 {
                            return -child - 1;
                        }
                        node = child as i64;
                    }
                })
                .collect::<Vec<i32>>();
            let exp_leaves = leaves_expected[i]
                .as_array()
                .expect("leaves")
                .iter()
                .map(|v| v.as_i64().expect("num") as i32)
                .collect::<Vec<i32>>();
            if my_leaves != exp_leaves {
                leaf_mismatches += 1;
            }
        }
        assert_eq!(
            leaf_mismatches, 0,
            "leaf path mismatches: {leaf_mismatches}"
        );
    }
}
