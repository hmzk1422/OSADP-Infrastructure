//! Research model loading — finds and parses the LightGBM text export
//! (`main/models/lgb_model.txt`) so the strategy scores with the real
//! 15-feature model instead of the fallback logit proxy.

use engine::lgbm::LightGbm;

/// Candidate model paths, checked from the working directory upward.
const MODEL_PATH_CANDIDATES: &[&str] = &["main/models/lgb_model.txt", "models/lgb_model.txt"];

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
            println!(
                "[model] loaded {} ({} trees, {} features)",
                path.display(),
                mdl.num_trees(),
                mdl.num_features()
            );
            Some(mdl)
        }
        Err(e) => {
            eprintln!("[model] failed to parse {}: {e} — running with proxy logit only", path.display());
            None
        }
    }
}
