//! Strategy config loading — single source of truth from `main/config.py`.
//!
//! `unified_runner.py live-test` exports `build/live_test_config.json` with
//! the same field names as `engine::StrategyConfig`. This deliberately small,
//! fully-typed loader applies only the keys present in the file on top of
//! `StrategyConfig::default()`, so a partial (or missing) file degrades to the
//! documented defaults instead of silently choosing an arbitrary config.

use engine::lgbm::LightGbm;
use engine::{FoldModel, StrategyConfig};
use std::path::Path;

/// Parse a flat JSON mapping onto [`StrategyConfig`]. Unknown/missing keys are
/// ignored (the built-in defaults remain). Returns `None` when the file cannot
/// be read or parsed — callers decide whether to fall back to defaults.
pub fn load_strategy_config(path: &Path) -> Option<StrategyConfig> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let obj = value.as_object()?;
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
