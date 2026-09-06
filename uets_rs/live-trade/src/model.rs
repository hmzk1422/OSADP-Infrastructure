//! Research model loading — finds and parses the LightGBM text export
//! (`uets_rs/models/lgb_model.txt`) so the strategy scores with the real
//! 15-feature model instead of the fallback logit proxy.

use engine::lgbm::LightGbm;

/// Candidate model paths, checked from the working directory upward.
/// The canonical home of the exported model is `uets_rs/models/` (the file no
/// longer lives under `uets_legacy/`); the flat `uets_rs/lgb_model.txt` variant
/// is a fallback for ad-hoc drops.
const MODEL_PATH_CANDIDATES: &[&str] = &["uets_rs/models/lgb_model.txt", "uets_rs/lgb_model.txt"];
const EXPECTED_FEATURE_COUNT: usize = 15;

/// Walk up from the current directory looking for any of `paths`.
pub fn find_model_path(paths: &[&str]) -> Option<std::path::PathBuf> {
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

/// Best-effort load of the research LightGBM text model. Returns `None`
/// (with a warning) when the file is absent or unparsable — the engine then
/// falls back to its deterministic logit proxy.
pub fn load_model() -> Option<LightGbm> {
    let path = find_model_path(MODEL_PATH_CANDIDATES)?;
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("[model] read error: {e} — running with proxy logit only");
            return None;
        }
    };
    match LightGbm::from_str(&text) {
        Ok(mdl) => {
            if mdl.num_features() != EXPECTED_FEATURE_COUNT {
                eprintln!(
                    "[model] rejected {}: expected {} features, found {} — running with proxy logit only",
                    path.display(),
                    EXPECTED_FEATURE_COUNT,
                    mdl.num_features()
                );
                return None;
            }
            println!(
                "[model] loaded {} ({} trees, {} features)",
                path.display(),
                mdl.num_trees(),
                mdl.num_features()
            );
            Some(mdl)
        }
        Err(e) => {
            eprintln!(
                "[model] failed to parse {}: {e} — running with proxy logit only",
                path.display()
            );
            None
        }
    }
}

/// An installed per-asset meta-labeller: the parsed model plus its take/avoid
/// threshold (both pinned by `build/meta_labeller_manifest.json`).
pub struct Labeller {
    pub model: LightGbm,
    pub threshold: f64,
}

// ============================================================================
// Regime-detection gate loader (research regime_hmm.py / regime_trees.py
// export → tools/export_regime_gate.py → build/regime_gate[_<sym>].json).
// The gate is the model that sits between the main executor and execution.
// ============================================================================

/// Load the regime-detection gate for `symbol`. Per-symbol export
/// (`build/regime_gate_<symbol>.json`) wins over the portfolio export
/// (`build/regime_gate.json`). Any absence/failure returns `None` and the
/// engine runs with the regime gate OPEN — pure addition, never blocks.
pub fn load_regime_gate(symbol: &str) -> Option<engine::RegimeGate> {
    let per_sym = format!("build/regime_gate_{symbol}.json");
    let path = find_model_path(&[per_sym.as_str(), "build/regime_gate.json"])?;
    let text = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let features: Vec<String> = v
        .get("features")?
        .as_array()?
        .iter()
        .map(|x| x.as_str().unwrap_or("").to_string())
        .collect();
        let nodes: Vec<engine::GateNode> = v
        .get("nodes")?
        .as_array()?
        .iter()
        .map(|x| {
            // Accept both object form ({"feature","threshold","left","right"})
            // and legacy 4-element array form ([feature, threshold, left, right]).
            if let Some(arr) = x.as_array() {
                if arr.len() != 4 {
                    return None;
                }
                Some(engine::GateNode {
                    feature: arr[0].as_i64().unwrap_or(0) as i32,
                    threshold: arr[1].as_f64().unwrap_or(0.0),
                    left: arr[2].as_i64().unwrap_or(-1) as i32,
                    right: arr[3].as_i64().unwrap_or(-1) as i32,
                })
            } else if let Some(obj) = x.as_object() {
                Some(engine::GateNode {
                    feature: obj.get("feature").and_then(|v| v.as_i64()).unwrap_or(-2) as i32,
                    threshold: obj.get("threshold").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    left: obj.get("left").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
                    right: obj.get("right").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
                })
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
    let favorable: Vec<bool> = v
        .get("favorable")?
        .as_array()?
        .iter()
        .map(|x| x.as_bool().unwrap_or(false))
        .collect();
    if features.is_empty() {
        eprintln!(
            "[regime] {symbol}: no features in export — gate OPEN",
        );
        return None;
    }
    if nodes.is_empty() || favorable.is_empty() {
        eprintln!("[regime] {symbol}: empty nodes/favorable in {} — gate OPEN", path.display());
        return None;
    }
    // favorable[] is indexed by node index (same as nodes[]).
    // Leaf nodes (left < 0 || right < 0) have meaningful favorable values;
    // non-leaf entries are ignored by is_tradeable (only called with leaf indices).
    if favorable.len() != nodes.len() {
        eprintln!(
            "[regime] {symbol}: favorable length {} != node count {} in {} — gate OPEN",
            favorable.len(),
            nodes.len(),
            path.display()
        );
        return None;
    }
    if !nodes_are_consistent(&nodes) {
        eprintln!("[regime] {symbol}: inconsistent nodes in {} — gate OPEN", path.display());
        return None;
    }
    println!(
        "[regime] {symbol}: regime gate loaded from {} ({} nodes, {} favorable)",
        path.display(),
        nodes.len(),
        favorable.iter().filter(|&&t| t).count()
    );
    Some(engine::RegimeGate {
        features,
        nodes,
        favorable,
    })
}
fn nodes_are_consistent(nodes: &[engine::GateNode]) -> bool {
    // Verify tree structure:
    // - Branch nodes must have valid children (left/right >= 0 and < nodes.len())
    // - Leaf nodes have left < 0 or right < 0
    // - Right child must have index > parent (prevents cycles)
    for (i, node) in nodes.iter().enumerate() {
        if node.left >= 0 && node.right >= 0 {
            // Branch node: children must be valid indices
            let max_child = node.left.max(node.right);
            if max_child as usize >= nodes.len() {
                return false;
            }
            // Right child must be after parent in array (tree ordering)
            if node.right as usize <= i {
                return false;
            }
        }
    }
    true
}

/// Load the per-asset meta-labeller gate (research/ofmr/meta_labeller.py).
///
/// `build/meta_labeller_manifest.json` — written by the research pipeline — is the
/// single source of truth: `gate_deployed` must be true, the entry names the
/// model file and the `meta_threshold`. Any absence/failure (no manifest, no
/// entry for this symbol, `gate_deployed=false`, unreadable/unparsable model,
/// feature-count mismatch) returns `None` and the engine runs with the gate
/// OPEN — the executor's behaviour is unchanged. Pure addition, never blocks.
pub fn load_labeller(symbol: &str, force: bool) -> Option<Labeller> {
    let manifest_path = find_model_path(&["build/meta_labeller_manifest.json"])?;
    let text = std::fs::read_to_string(&manifest_path).ok()?;
    let man: serde_json::Value = serde_json::from_str(&text).ok()?;
    // ONE portfolio labeller (executor-style: trained on the train symbol
    // only, no symbol-identity features) — the manifest's "portfolio" entry
    // is canonical. A per-symbol entry is accepted only for older manifests.
    let entry = man.get("portfolio").or_else(|| man.get(symbol))?;
    let deployed = entry
        .get("gate_deployed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !deployed && !force {
        println!("[meta] {symbol}: gate_deployed=false in manifest — gate OPEN");
        return None;
    }
    if force && !deployed {
        println!("[meta] {symbol}: --meta-gate on overrides gate_deployed=false");
    }
    let name = entry.get("model")?.as_str()?;
    let threshold = entry.get("meta_threshold")?.as_f64()?;
    let expected = entry
        .get("features")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(16);
    let cands: Vec<String> = vec![format!("uets_rs/models/{name}"), format!("uets_rs/{name}")];
    let cand_refs: Vec<&str> = cands.iter().map(|s| s.as_str()).collect();
    let path = find_model_path(&cand_refs)?;
    let mdl_text = std::fs::read_to_string(&path).ok()?;
    match LightGbm::from_str(&mdl_text) {
        Ok(mdl) => {
            if mdl.num_features() != expected {
                eprintln!(
                    "[meta] rejected {}: expected {expected} features, found {} — gate OPEN",
                    path.display(),
                    mdl.num_features()
                );
                return None;
            }
            println!(
                "[meta] loaded {} ({} trees, {expected} features, threshold {threshold:.2})",
                path.display(),
                mdl.num_trees()
            );
            Some(Labeller {
                model: mdl,
                threshold,
            })
        }
        Err(e) => {
            eprintln!("[meta] failed to parse {}: {e} — gate OPEN", path.display());
            None
        }
    }
}
