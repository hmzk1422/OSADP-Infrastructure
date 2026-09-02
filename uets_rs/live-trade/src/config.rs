//! Strategy config loading — one JSON file, read directly by every process.
//!
//! `--config` accepts EITHER a flat StrategyConfig JSON (same field names as
//! `engine::StrategyConfig`) OR the unified portfolio config
//! (`config/portfolio.json`): in the unified case the engine resolves its own
//! ticker — `_strategy_defaults` deep-merged with that asset's `strategy`
//! overrides — so no intermediate per-ticker config file is ever generated.
//! Unknown/missing keys are ignored (built-in defaults remain).

use engine::lgbm::LightGbm;
use engine::{FoldModel, StrategyConfig};
use std::path::Path;

/// Recursively merge `over` into `base` (override wins).
fn deep_merge(
    base: &mut serde_json::Map<String, serde_json::Value>,
    over: &serde_json::Map<String, serde_json::Value>,
) {
    for (k, v) in over {
        match (base.get_mut(k), v) {
            (Some(serde_json::Value::Object(b)), serde_json::Value::Object(o)) => {
                deep_merge(b, o)
            }
            _ => {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

/// Parse a flat JSON mapping onto [`StrategyConfig`]. Unknown/missing keys are
/// ignored (the built-in defaults remain). Returns `None` when the file cannot
/// be read or parsed — callers decide whether to fall back to defaults.
pub fn load_strategy_config(path: &Path, ticker: &str) -> Option<StrategyConfig> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let obj = if value.get("_strategy_defaults").is_some() || value.get("assets").is_some() {
        // Unified portfolio config: resolve `_strategy_defaults` + this
        // ticker's per-asset `strategy` overrides (deep merge, asset wins).
        let mut merged = value
            .get("_strategy_defaults")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        if let Some(assets) = value.get("assets").and_then(|v| v.as_array()) {
            let want = ticker.to_ascii_lowercase();
            for a in assets {
                let t = a
                    .get("ticker")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if t == want {
                    if let Some(s) = a.get("strategy").and_then(|v| v.as_object()) {
                        deep_merge(&mut merged, s);
                    }
                }
            }
        }
        serde_json::Value::Object(merged)
    } else {
        value
    };
    let obj = obj.as_object()?;
    let mut cfg = StrategyConfig::default();

    macro_rules! g_f64 {
        ($field:ident) => {
            if let Some(x) = obj.get(stringify!($field)).and_then(|v| v.as_f64()) {
                cfg.$field = x;
            }
        };
    }
    macro_rules! g_i64 {
        ($field:ident) => {
            if let Some(x) = obj.get(stringify!($field)).and_then(|v| v.as_i64()) {
                cfg.$field = x;
            }
        };
    }
    macro_rules! g_bool {
        ($field:ident) => {
            if let Some(x) = obj.get(stringify!($field)).and_then(|v| v.as_bool()) {
                cfg.$field = x;
            }
        };
    }

    g_i64!(micro_barrier_secs);
    g_i64!(timeframe_barrier_secs);
    g_i64!(lookahead_seconds);
    g_f64!(barrier_multiplier);
    g_f64!(rr_ratio);
    g_f64!(rsi_filter_threshold);
    g_bool!(enable_rsi_accel_filter);
    g_bool!(trade_direction_is_long);
    g_f64!(confidence_threshold);
    g_i64!(lockout_seconds);
    g_f64!(max_position_usd);
    g_f64!(min_order_size);
    g_f64!(fee_bps);
    g_bool!(enable_slippage);
    g_f64!(slippage_bps_fallback);
    g_f64!(min_barrier_pct);
    g_f64!(max_barrier_pct);
    g_f64!(risk_pct_per_trade);
    g_f64!(max_leverage);
    g_f64!(r_clip_cap);
    g_bool!(use_orderflow_features);
    g_f64!(hysteresis_buffer);
    g_bool!(score_barrier_as_rr);
    g_bool!(require_fold_coverage);
    g_bool!(one_candidate_per_bar);

    Some(cfg)
}
/// Load the walk-forward fold-model catalog exported by
/// `research/ofmr/ml_state_optimized.py` STEP 6a (`catalog.json` + `fold_XXX.txt`
/// LightGBM dumps inside `dir`). Models that fail to parse are skipped with a
/// warning so a partially exported catalog still runs. Returns folds sorted
/// by test-window start.
pub fn load_fold_catalog(dir: &Path) -> Result<Vec<FoldModel>, String> {
    let cat_path = dir.join("catalog.json");
    let text = std::fs::read_to_string(&cat_path)
        .map_err(|e| format!("cannot read {}: {e}", cat_path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("bad catalog JSON: {e}"))?;
    let mut out = Vec::new();
    if let Some(models) = value.get("models").and_then(|m| m.as_array()) {
        for m in models {
            let fold = m.get("fold").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let (Some(test_start_ms), Some(test_end_ms), Some(name)) = (
                m.get("test_start_ms").and_then(|v| v.as_i64()),
                m.get("test_end_ms").and_then(|v| v.as_i64()),
                m.get("model").and_then(|v| v.as_str()),
            ) else {
                continue;
            };
            let path = dir.join(name);
            match LightGbm::from_path(&path) {
                Ok(model) => out.push(FoldModel {
                    fold,
                    test_start_ms,
                    test_end_ms,
                    model,
                }),
                Err(e) => eprintln!("[wf] skipping {}: {e}", path.display()),
            }
        }
    }
    out.sort_by_key(|f| f.test_start_ms);
    Ok(out)
}
