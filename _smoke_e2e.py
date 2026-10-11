"""Scratch end-to-end smoke test for the rewritten moe/strategy.py.

Runs features -> OOD -> (GP off / GP on) -> TB export against synthetic data
with the L2 schema, then verifies the exported bundle layout, ONNX parity,
manifest paths, and gp_features.json round-trip.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from strategy_system.strategy.moe.strategy import load_settings, strategy  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_raw(n=4000, seed=7):
    """Synthetic flat-schema frame matching the loader output.

    ``price`` (trade price) is OHLC-expanded by ``reconstruct`` into
    ``price_close`` for the macro group; ``best_bid_price`` / ``best_ask_price``
    similarly become ``*_close`` for the micro group.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="100ms", tz="UTC")
    shocks = rng.normal(0.0, 2e-4, n)
    price = 100.0 * np.exp(np.cumsum(shocks))
    half = 0.5e-3 * price
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "price": price,
            "best_bid_price": price - half,
            "best_ask_price": price + half,
            "best_bid_qty": rng.uniform(0.5, 5.0, n),
            "best_ask_qty": rng.uniform(0.5, 5.0, n),
            "quantity": rng.uniform(0.01, 1.0, n),
            "is_buyer_maker": rng.random(n) < 0.5,
        },
        index=idx,
    )
    for level in (1, 5, 10, 20):
        df[f"bid_qty_{level}"] = rng.uniform(0.1, 2.0, n)
        df[f"ask_qty_{level}"] = rng.uniform(0.1, 2.0, n)
    return df


settings = load_settings()
settings["ood"].update(
    encoder_type="pca",
    encoder_params={"embedding_dim": 8},
    sequence_window=8,
    normalization_window=100,
    normalization_min_periods=10,
    clip_z=8.0,
    segmenter_type="kmeans",
    segmenter_params={"n_clusters": 3},
    max_states=3,
    min_novel_observations=60,
    min_novel_episodes=2,
    candidate_episode_gap=8,
    min_dwell=3,
    mature_state_observations=60,
    learn_novel=True,
    fit_rows=0.7,
    timestamp_col=None,
)
settings["features"].update(htf_timeframe="1s", ltf_timeframe="100ms")
settings["gp"].update(enabled=False)
settings["outcome"]["forward_returns"].update(
    horizons=["1s", "2s"], timestamp_col=None,
)
settings["outcome"]["triple_barrier"].update(
    horizon=10, train_fraction=0.7, side=1, price_col="price_close",
)
settings["outcome"]["model"].update(min_state_rows=50)
settings["moe"].update(models_dir=str(ROOT / "_smoke_models"), min_latent_rows=100)

models_dir = Path(settings["moe"]["models_dir"])
if models_dir.exists():
    shutil.rmtree(models_dir)

raw = make_raw()
print("=== running pipeline (GP off) ===")
result = strategy(raw, settings, show_progress=False)

check("model frame built", len(result.features) > 0,
      f"{len(result.features)} rows, {len(result.features.columns)} cols")
check("targets built", not result.targets.empty)
check("regimes have latent_id", "latent_id" in result.regimes)
n_states = result.regimes["latent_id"].nunique()
print(f"  (states discovered: {n_states})")

check("regime bundle exported", (models_dir / "regime").is_dir(),
      str(models_dir / "regime"))
regime_meta = models_dir / "regime" / "regime_metadata.json"
check("regime_metadata.json exists", regime_meta.is_file())
if regime_meta.is_file():
    meta = json.loads(regime_meta.read_text(encoding="utf-8"))
    for key in ("prototypes", "routing_rule", "causal_scaling", "state_support"):
        check(f"regime metadata has {key}", key in meta)

check("linreg refit happened", result.linreg is not None,
      f"states fitted: {sorted(result.linreg.models) if result.linreg else None}")

latent_dirs = sorted(p for p in models_dir.glob("latent_*") if p.is_dir())
exported_any = (result.tb is not None and not result.tb.report.empty
                and result.tb.report["status"].eq("exported").any())
check("latent bundles exported", exported_any and len(latent_dirs) > 0,
      f"{len(latent_dirs)} dirs; report:\n"
      + (result.tb.report.to_string(index=False) if result.tb is not None else "tb=None"))
if exported_any:
    for d in latent_dirs:
        check(f"{d.name}/gp_features.json", (d / "gp_features.json").is_file())
        check(f"{d.name}/tb_regressor.onnx", (d / "tb_regressor.onnx").is_file())
        check(f"{d.name}/metadata.json", (d / "metadata.json").is_file())
    manifest = models_dir / "manifest.json"
    check("manifest.json exists", manifest.is_file())
    if manifest.is_file():
        mf = json.loads(manifest.read_text(encoding="utf-8"))
        check("manifest has regime + latents", "regime" in mf and "latents" in mf)
        ok_paths = True
        for lid, entry in mf["latents"].items():
            for key in ("tb_regressor_onnx", "gp_features", "metadata"):
                if not (models_dir / entry[key]).is_file():
                    ok_paths = False
                    print(f"    missing: {entry[key]}")
        check("manifest paths resolve", ok_paths)

        import onnx

        ok_order = True
        ok_parity = True
        for lid, entry in mf["latents"].items():
            m = json.loads((models_dir / entry["metadata"]).read_text(encoding="utf-8"))
            model = onnx.load(str(models_dir / entry["tb_regressor_onnx"]))
            onnx_inputs = [i.name for i in model.graph.input]
            if onnx_inputs != ["features"]:
                ok_order = False
                print(f"    {lid}: unexpected ONNX inputs {onnx_inputs}")
            if m["input_features"] != entry["input_features"]:
                ok_order = False
                print(f"    {lid}: metadata/manifest feature order mismatch")
            outputs = [o.name for o in model.graph.output]
            if outputs != ["tp_width", "sl_width"]:
                ok_order = False
                print(f"    {lid}: unexpected ONNX outputs {outputs}")
            for v in m["parity_max_abs_diff"].values():
                if not v < 1e-5:
                    ok_parity = False
                    print(f"    {lid}: parity diff {v}")
        check("metadata input order == manifest/ONNX contract", ok_order)
        check("recorded ONNX parity diffs < 1e-5", ok_parity)

print("\n=== running pipeline (GP on, tiny search) ===")
settings2 = json.loads(json.dumps({k: v for k, v in settings.items() if k != "features"}, default=str))
settings2["features"] = dict(settings["features"])
settings2["gp"].update(
    enabled=True,
    target_columns=["future_return__1s"],
    population_size=8, generations=1, max_height=3,
    evaluation_models=["ridge", "ols"],
    block_count=3, min_block_samples=10,
    min_state_samples=40, min_null_samples=40,
    temporal_windows=[2, 4],
    max_features_per_state_target=1,
    multiple_testing_alpha=0.5,
    min_train_ic=0.0, min_validation_ic=0.0,
    min_train_tstat=0.0, min_validation_tstat=0.0,
    validation_ic_retention=0.0, min_valid_blocks=1,
    train_fraction=0.7, purge_rows=0,
    workers=1,
)
settings2["moe"]["models_dir"] = str(ROOT / "_smoke_models_gp")
gp_dir = Path(settings2["moe"]["models_dir"])
if gp_dir.exists():
    shutil.rmtree(gp_dir)
result2 = strategy(raw, settings2, show_progress=False)
check("GP-on pipeline ran", result2.gp is not None)
gp_report = result2.gp.report if result2.gp is not None else pd.DataFrame()
print(f"  (GP selected {len(gp_report)} expressions)")
gp_dirs = sorted(p for p in gp_dir.glob("latent_*") if p.is_dir())
if len(gp_dirs) and len(gp_report):
    spec = json.loads((gp_dirs[0] / "gp_features.json").read_text(encoding="utf-8"))
    check("gp_features.json has expressions key", "expressions" in spec)
    ok_rt = True
    for item in spec.get("selected", []):
        expr_text = item["expression"]
        matched = [
            sel for sel in result2.gp.model.selected
            if str(sel.expression) == expr_text and sel.state_id == spec["latent_id"]
        ]
        if not matched:
            ok_rt = False
            print(f"    expression not found in fitted model: {expr_text}")
    check("exported expressions round-trip to the fitted model", ok_rt)

print()
print("=" * 72)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
print("=" * 72)
