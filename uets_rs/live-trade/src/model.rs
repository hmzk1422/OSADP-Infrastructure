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
