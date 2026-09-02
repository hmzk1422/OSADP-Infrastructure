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
