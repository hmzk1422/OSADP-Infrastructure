"""Filesystem / serialization layer for the MoE bundle (the Rust contract).

Layout written under ``ops/models/``::

    regime/            encoder.onnx + regime_metadata.json   (written by export_moe)
    latent_<id>/       gp_features.json, tb_regressor.onnx, metadata.json
    manifest.json      index Rust reads ONCE at startup

Nothing here fits a model; it only serializes what the stages produced.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .paths import OPS_DIR

SCHEMA_VERSION = 1
GP_FEATURES_FILE = "gp_features.json"
TB_MODEL_FILE = "tb_regressor.onnx"
METADATA_FILE = "metadata.json"
REGIME_DIR = "regime"
REGIME_MODEL_FILE = "encoder.onnx"
REGIME_METADATA_FILE = "regime_metadata.json"
MANIFEST_FILE = "manifest.json"
DEFAULT_MODELS_DIR = OPS_DIR / "models"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def models_dir_from_cfg(cfg: dict | None = None) -> Path:
    """``moe.models_dir`` wins, else ``ops.models_dir``, else ``ops/models``."""
    from .paths import models_dir_from_cfg as _resolve

    return _resolve(cfg)


def latent_dir_name(latent_id: str) -> str:
    """``'2'`` -> ``latent_2``; ``'latent_2'`` stays; unsafe characters become ``_``."""
    clean = re.sub(r"[^0-9A-Za-z_\-]", "_", str(latent_id))
    return clean if clean.startswith("latent_") else f"latent_{clean}"


def _jsonable(value: Any) -> Any:
    """numpy / NaN / Path -> strict JSON (NaN and inf become null)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# gp_features.json  (feature formulas + horizon predictor, for Rust)
# --------------------------------------------------------------------------- #
def _terminal_spec(name: str, norm: dict) -> dict:
    """Decode a GP terminal name (``x__...``) into how Rust must compute it."""
    if name.startswith("x__log_return__"):
        source, transform = name[len("x__log_return__"):], "log_return"
    else:
        source, transform = name[len("x__"):], "identity"
    return {
        "name": name,
        "source_column": source,
        "pre_transform": transform,  # log_return: ln(v_t) - ln(v_{t-1}), NaN if v <= 0
        "normalization": "causal_rolling_zscore",
        "window": norm["normalization_window"],
        "min_periods": norm["normalization_min_periods"],
        "clip_z": norm["clip_z"],
    }


def build_gp_features_payload(
    latent_id: str,
    gp_model,
    terminal_names: list[str],
    horizon_model,
    horizons: list[str],
) -> dict:
    """Serializable description of one latent's GP features + horizon predictor.

    ``gp_model`` is the fitted ``GeneticFeatureModel``; ``horizon_model`` is the
    latent's ``StateLinregModel`` (predicts future_return__<h> from the gp features).
    """
    state = gp_model.to_dict()
    norm = {k: state[k] for k in ("normalization_window", "normalization_min_periods", "clip_z")}
    trees = {s["output_column"]: s for s in state["selected"]}
    order = {c: i for i, c in enumerate(horizon_model.feature_columns)}  # weight order
    items = sorted(
        (it for it in gp_model.selected
         if it.state_id == latent_id and it.output_column in order),
        key=lambda it: order[it.output_column],
    )

    features, terminals_used = [], set()
    by_length = sorted(terminal_names, key=len, reverse=True)
    for item in items:
        formula = str(item.expression)
        used = [t for t in by_length if t in formula]
        terminals_used.update(used)
        features.append({
            "output_column": item.output_column,
            "target_column": item.target_column,
            "formula": formula,
            "tree": trees[item.output_column]["expression"],  # node_to_dict(): machine-readable
            "terminals_used": used,
            "direction": item.direction,
            "train_ic": item.train_ic,
            "validation_ic": item.validation_ic,  # null when no holdout was used
            "n_train": item.n_train,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "latent_id": latent_id,
        "note": "Each feature is only defined while the regime router outputs this latent_id.",
        "input_terminals": [_terminal_spec(t, norm) for t in sorted(terminals_used)],
        "features": features,
        "horizon_model": {
            "type": "standardized_ridge_linreg",
            "formula": "pred[h] = sum_j ((x_j - mean_j) / scale_j) * coefficients[j][h] + intercept[h]",
            "feature_order": list(horizon_model.feature_columns),
            "target_order": list(horizon_model.target_columns),
            "horizons": list(horizons),
            "feature_mean": horizon_model.feature_mean,
            "feature_scale": horizon_model.feature_scale,
            "coefficients": horizon_model.coefficients,  # shape [n_features][n_horizons]
            "intercept": horizon_model.intercept,        # shape [n_horizons]
            "n_train": horizon_model.n_train,
        },
        "missing_value_policy": "non-finite feature -> row not tradeable (no prediction)",
    }


# --------------------------------------------------------------------------- #
# tb_regressor.onnx
# --------------------------------------------------------------------------- #
def export_tb_onnx(path: Path, tp_model, sl_model, n_features: int, parity_x: np.ndarray | None = None) -> dict:
    """Fuse tb_regressor's two pipelines into ONE graph: [MFE_R, MAE_R] = f(features).

    Both pipelines are ``StandardScaler -> LinearRegression`` fitted on the same X,
    so they share the scaler; the graph is Sub -> Div -> Gemm (no sklearn at runtime).
    Output ``excursion_r`` has shape [N, 2]: column 0 = MFE_R (TP model), 1 = MAE_R (SL model).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def parts(pipe):
        scaler, reg = pipe.steps[0][1], pipe.steps[-1][1]
        if scaler.n_features_in_ != n_features:
            raise ValueError(
                f"tb model expects {scaler.n_features_in_} features but {n_features} were exported; "
                "a feature column had no finite training value and was dropped inside tb_regressor")
        return scaler, np.ravel(reg.coef_), float(np.ravel(reg.intercept_)[0])

    sc_tp, w_tp, b_tp = parts(tp_model)
    sc_sl, w_sl, b_sl = parts(sl_model)
    if not (np.allclose(sc_tp.mean_, sc_sl.mean_) and np.allclose(sc_tp.scale_, sc_sl.scale_)):
        raise ValueError("tp/sl scalers differ; cannot fuse into one graph")

    f32 = np.float32
    inits = [
        numpy_helper.from_array(sc_tp.mean_.astype(f32), "mean"),
        numpy_helper.from_array(sc_tp.scale_.astype(f32), "scale"),
        numpy_helper.from_array(np.stack([w_tp, w_sl], axis=1).astype(f32), "W"),   # [F, 2]
        numpy_helper.from_array(np.array([b_tp, b_sl], dtype=f32), "B"),            # [2]
    ]
    nodes = [
        helper.make_node("Sub", ["features", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "scale"], ["scaled"]),
        helper.make_node("Gemm", ["scaled", "W", "B"], ["excursion_r"]),
    ]
    graph = helper.make_graph(
        nodes, "tb_regressor",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("excursion_r", TensorProto.FLOAT, [None, 2])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))

    parity = {"checked": False}
    if parity_x is not None and len(parity_x):
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(["excursion_r"], {"features": parity_x.astype(f32)})[0]
        want = np.column_stack([tp_model.predict(parity_x), sl_model.predict(parity_x)])
        err = float(np.max(np.abs(got - want) / (1.0 + np.abs(want))))
        if err > 1e-3:
            raise ValueError(f"ONNX parity failed for {path.name}: max rel err {err:.2e}")
        parity = {"checked": True, "rows": int(len(parity_x)), "max_rel_err": err}
    return parity


# --------------------------------------------------------------------------- #
# per-latent metadata.json + manifest.json
# --------------------------------------------------------------------------- #
def export_gp_features(models_dir: Path, latent_id: str, gp_model, horizon_model,
                       terminal_names: list[str], horizons: list[str]) -> Path:
    """Write ``latent_<id>/gp_features.json`` (feature formulas + horizon predictor)."""
    payload = build_gp_features_payload(latent_id, gp_model, terminal_names, horizon_model, horizons)
    return write_json(models_dir / latent_dir_name(latent_id) / GP_FEATURES_FILE, payload)


# --------------------------------------------------------------------------- #
# regime/ bundle  (encoder.onnx + regime_metadata.json, for the Rust router)
# --------------------------------------------------------------------------- #
def build_regime_metadata(detector) -> dict:
    """Serializable description of the fitted OOD detector (pure data)."""
    encoder = getattr(detector, "encoder", None)
    segmenter = getattr(detector, "segmenter", None)
    segmenter_state = segmenter.state_dict() if hasattr(segmenter, "state_dict") else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "encoder": {
            "type": type(encoder).__name__ if encoder is not None else None,
            "input_columns": list(getattr(encoder, "input_columns_", None) or []),
            "embedding_columns": list(getattr(encoder, "embedding_columns_", None) or []),
            "window": getattr(encoder, "window", getattr(encoder, "window_size", None)),
        },
        "segmenter": _jsonable(segmenter_state),
        "trade_gates": {
            "mature_state_observations": getattr(detector, "mature_state_observations", None),
            "require_model_for_trade": getattr(detector, "require_model_for_trade", None),
            "model_ready_states": sorted(getattr(segmenter, "model_ready_states", set()) or []),
        },
    }


def verify_regime_contract(detector, df, regimes) -> None:
    """Fail fast when the regime bundle would be unloadable or misaligned."""
    if regimes is None or len(regimes) == 0:
        raise ValueError("no regime labels to export; run_ood produced an empty frame")
    if "latent_id" not in regimes.columns:
        raise KeyError("regimes frame is missing the 'latent_id' column")
    encoder = getattr(detector, "encoder", None)
    if encoder is None or not getattr(encoder, "is_fitted_", False):
        raise RuntimeError("OOD encoder is not fitted; cannot export the regime bundle")
    segmenter = getattr(detector, "segmenter", None)
    if segmenter is None or not getattr(segmenter, "prototypes", {}):
        raise RuntimeError("OOD segmenter has no prototypes; cannot export the regime bundle")
    if len(regimes) != len(df):
        raise ValueError(
            f"regimes rows ({len(regimes)}) != input rows ({len(df)}); "
            "the regime bundle would misalign with the feature index"
        )


# --------------------------------------------------------------------------- #
# regime/ bundle writer: encoder.onnx + regime_metadata.json
# --------------------------------------------------------------------------- #
# Native LayerNorm lands at opset 17; torch's exporter may emit a higher opset
# (we record the ACTUAL opset it produced, see export_regime_bundle).
ENCODER_ONNX_OPSET = 17
# Collision-proof graph output name: torch's exporter re-uses the auto-name
# "embedding" for an internal activation, which then collides with the declared
# graph output and yields an invalid (non-SSA) model. A distinct name avoids it.
ENCODER_OUTPUT_NAME = "regime_embedding"


def export_regime_bundle(models_dir: Path, detector, parity_x: np.ndarray | None = None) -> Path:
    """Write ``regime/encoder.onnx`` + ``regime/regime_metadata.json`` (Rust router front-end).

    The ONNX graph is the fitted encoder's ``encode`` only: a *normalized* trailing
    window ``(B, window, n_features)`` -> embedding ``(B, embedding_width)``.  Causal
    z-score scaling and trailing-window assembly stay in Rust (pure data transforms,
    documented under ``input_pipeline``) -- exactly like the GP feature normalization.
    ``regime_metadata.json`` also carries the segmenter prototypes + scales +
    match_threshold so Rust reproduces the nearest-prototype assignment with the same
    RMS distance metric as ``HysteresisClusterRegimeSegmenter._distance``.

    Acceptance contract is onnxruntime *loading* the graph + numeric parity against
    torch (NOT ``onnx.checker``: the new torch exporter trips checker's version
    converter on valid graphs, and we verified parity is ~4e-8).
    """
    import torch
    import onnx

    encoder = getattr(detector, "encoder", None)
    net = getattr(encoder, "net_", None)
    if net is None or not getattr(encoder, "is_fitted_", False):
        raise RuntimeError(
            "regime encoder has no fitted torch network; ONNX export supports the "
            "torch encoders (vq-vae, ts2vec, tcn, transformer). Fit it first.")

    window = int(encoder.window)
    n_features = int(getattr(encoder, "_n_features", 0)
                     or len(getattr(encoder, "input_columns_", ()) or ()))
    if n_features <= 0:
        raise RuntimeError("encoder was fitted with zero input features; cannot size encoder.onnx")
    embedding_width = (int(len(encoder.embedding_columns_))
                       if getattr(encoder, "embedding_columns_", None)
                       else int(getattr(encoder, "_embedding_width", encoder.embedding_dim)))

    class _Encode(torch.nn.Module):
        """Wrap ``net.encode`` (VQVAENet/TS2VecNet/... define ``encode``/``loss``, no ``forward``)."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, window):
            return self.net.encode(window)

    out_dir = models_dir / REGIME_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / REGIME_MODEL_FILE
    net.eval()
    with torch.no_grad():
        torch.onnx.export(
            _Encode(net), (torch.zeros(1, window, n_features),), str(onnx_path),
            input_names=["window"], output_names=[ENCODER_OUTPUT_NAME],
            dynamic_axes={"window": {0: "batch"}, ENCODER_OUTPUT_NAME: {0: "batch"}},
            opset_version=ENCODER_ONNX_OPSET, do_constant_folding=True,
            verbose=False,   # suppress torch's ✅ trace log (crashes cp1252 consoles)
        )
    # Read back the ACTUAL opset torch produced (it may bump past what we requested).
    model_proto = onnx.load(str(onnx_path))
    opset = next((int(o.version) for o in model_proto.opset_import if o.domain in ("", "ai.onnx")),
                 ENCODER_ONNX_OPSET)

    parity = {"checked": False}
    if parity_x is not None and len(parity_x):
        import onnxruntime as ort
        arr = np.asarray(parity_x, dtype=np.float32)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # load == validity check
        got = sess.run([ENCODER_OUTPUT_NAME], {"window": arr})[0]
        want = net.encode(torch.from_numpy(arr)).detach().cpu().numpy()
        err = float(np.max(np.abs(got - want) / (1.0 + np.abs(want))))
        parity = {"checked": True, "rows": int(len(arr)), "max_rel_err": err}
        if err > 1e-3:
            raise ValueError(f"ONNX parity failed for {REGIME_MODEL_FILE}: max rel err {err:.2e}")

    meta = build_regime_metadata(detector)
    meta["model_file"] = REGIME_MODEL_FILE
    meta["onnx_opset"] = opset
    meta["model"] = {
        "kind": type(net).__name__, "format": "onnx",
        "input_name": "window", "input_dtype": "float32",
        "input_shape": [None, window, n_features],
        "output_name": ENCODER_OUTPUT_NAME, "output_dtype": "float32",
        "output_shape": [None, embedding_width],
        "output_columns": list(getattr(encoder, "embedding_columns_", []) or []),
        "input_note": ("input is a causal-z-scored trailing window; Rust applies "
                       "normalization + windowing first (see input_pipeline)"),
    }
    meta["input_pipeline"] = {
        "steps": [
            "select input_columns in the fixed fitted order (raw primitives)",
            "causal_rolling_zscore: shift(1) -> rolling(window, min_periods) mean/std (ddof=0) -> clip",
            "assemble the trailing `window` rows per anchor; leading rows stay NaN -> embedding NaN -> null",
        ],
        "normalization": {
            "window": getattr(encoder, "normalization_window", None),
            "min_periods": getattr(encoder, "normalization_min_periods", None),
            "clip_z": getattr(encoder, "clip_z", None),
        },
        "window": window,
        "input_columns": list(getattr(encoder, "input_columns_", []) or []),
    }
    seg_cfg = (meta.get("segmenter", {}).get("config", {}) or {})
    meta["assignment"] = {
        "metric": "rms((embedding - prototype) / max(scale, min_state_scale))",
        "decision": "argmin over prototypes; null if distance > match_threshold",
        "match_threshold": seg_cfg.get("match_threshold"),
        "min_state_scale": seg_cfg.get("min_state_scale"),
    }
    meta["onnx_parity"] = parity
    meta["sha256"] = {REGIME_MODEL_FILE: sha256_file(onnx_path)}
    write_json(out_dir / REGIME_METADATA_FILE, meta)
    return out_dir


def export_latent(
    models_dir: Path,
    latent_id: str,
    *,
    tp_model,
    sl_model,
    tb_feature_order: list[str],
    horizons: list[str],
    barrier: dict,
    err_rule: str,
    n_fit: int,
    parity_x: np.ndarray | None,
) -> Path:
    """Write tb_regressor.onnx + metadata.json (gp_features.json must already exist)."""
    out = models_dir / latent_dir_name(latent_id)
    gp_path = out / GP_FEATURES_FILE
    if not gp_path.exists():
        raise FileNotFoundError(f"{gp_path} missing: run_gp must export gp_features.json first")
    onnx_path = out / TB_MODEL_FILE
    parity = export_tb_onnx(onnx_path, tp_model, sl_model, len(tb_feature_order), parity_x)

    feature_defs = []
    for name in tb_feature_order:
        kind, h = name.split("__", 1)
        feature_defs.append({
            "name": name, "kind": kind, "horizon": h,
            "formula": (
                f"horizon_model.pred[{h}] from {GP_FEATURES_FILE} (inputs: its gp features)"
                if kind == "pred" else err_rule.replace("<h>", h)
            ),
        })
    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "latent_id": latent_id,
        "model_file": TB_MODEL_FILE,
        "gp_features_file": GP_FEATURES_FILE,
        "model": {
            "kind": "tb_regressor", "format": "onnx",
            "input_name": "features", "input_dtype": "float32",
            "input_shape": [None, len(tb_feature_order)],
            "output_name": "excursion_r", "output_dtype": "float32",
            "output_columns": ["MFE_R", "MAE_R"],
            "output_note": "excursions in R units (multiples of rr_unit); not price fractions",
        },
        "horizons": list(horizons),
        "input_features": feature_defs,  # ORDER IS THE ONNX INPUT ORDER
        "nan_policy": "non-finite input feature is replaced by 0.0 (same as tb_regressor training)",
        "barrier": barrier,
        "n_fit_rows": int(n_fit),
        "onnx_parity": parity,
        "sha256": {GP_FEATURES_FILE: sha256_file(gp_path), TB_MODEL_FILE: sha256_file(onnx_path)},
    }
    write_json(out / METADATA_FILE, meta)
    return out


def export_manifest(models_dir: Path, latent_ids: list[str], cfg_summary: dict | None = None) -> Path:
    """Build manifest.json from what is ON DISK (regime/ + metadata.json of ``latent_ids``).

    Rust reads this once, loads every model into memory keyed by latent_id and routes
    by dictionary lookup: no filesystem search at inference time.
    """
    regime_meta = models_dir / REGIME_DIR / REGIME_METADATA_FILE
    regime_model = models_dir / REGIME_DIR / REGIME_MODEL_FILE
    for needed in (regime_meta, regime_model):
        if not needed.exists():
            raise FileNotFoundError(f"{needed} missing: run_ood must export the regime bundle first")

    experts = {}
    for latent_id in latent_ids:  # only this run's latents; stale folders are ignored
        meta_path = models_dir / latent_dir_name(latent_id) / METADATA_FILE
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        folder = meta_path.parent
        experts[str(meta["latent_id"])] = {
            "dir": folder.name,
            "model": f"{folder.name}/{meta['model_file']}",
            "gp_features": f"{folder.name}/{meta['gp_features_file']}",
            "metadata": f"{folder.name}/{METADATA_FILE}",
            "horizons": meta["horizons"],
            "n_inputs": len(meta["input_features"]),
            "sha256": meta["sha256"],
        }
    if not experts:
        raise ValueError("no latent experts on disk; refusing to write a manifest")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "regime": {
            "model": f"{REGIME_DIR}/{REGIME_MODEL_FILE}",
            "metadata": f"{REGIME_DIR}/{REGIME_METADATA_FILE}",
            "sha256": {REGIME_MODEL_FILE: sha256_file(regime_model),
                       REGIME_METADATA_FILE: sha256_file(regime_meta)},
        },
        "router": {
            "key": "latent_id",
            "experts": experts,
            "unrouted_policy": "no_trade",  # null / latent without an expert -> flat
        },
        "config": cfg_summary or {},
    }
    return write_json(models_dir / MANIFEST_FILE, manifest)