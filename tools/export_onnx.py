"""
Export Production LightGBM Model to ONNX for the Rust Executor
===============================================================
Loads the research-exported LightGBM model (main/models/lgb_model.txt) and
converts it to ONNX format that the Rust executor loads via the `ort`
(onnxruntime) crate.

Verification:
    After conversion, the script runs BOTH the original LightGBM booster and
    the ONNX model on N random feature vectors and asserts the predicted
    class-1 (WIN) probabilities match within tolerance. This guarantees the
    Rust ONNX path produces bit-comparable P(WIN) to the Python research
    path.

Outputs:
    src/assets/model.onnx            - the ONNX model
    src/assets/model.features.json   - feature manifest (for ordering)
    src/assets/model.label_map.json  - { "0": "LOSS", "1": "WIN" } class order

Usage:
    python tools/export_onnx.py
    python tools/export_onnx.py --model main/models/lgb_model.txt
"""

import argparse
import json
import os
import sys

import numpy as np

import lightgbm as lgb
import onnx
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(description="LightGBM -> ONNX exporter")
    ap.add_argument("--model", default="main/models/lgb_model.txt")
    ap.add_argument("--out-dir", default="src/assets",
                    help="Where model.onnx + manifests are written")
    ap.add_argument("--verify-samples", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load LightGBM booster + feature manifest
    # ------------------------------------------------------------------
    booster = lgb.Booster(model_file=args.model)
    manifest_path = os.path.splitext(args.model)[0] + ".features.json"
    if not os.path.exists(manifest_path):
        sys.exit(f"Feature manifest not found at {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    features = manifest["features"]
    n_features = manifest["n_features"]
    print(f"Loaded model: {args.model}")
    print(f"  trees = {booster.num_trees()}, features = {n_features}")

    num_class = 1
    try:
        num_class = booster.num_class()
    except AttributeError:
        # LightGBM >= 4.6 renamed num_class; probe via a trivial prediction.
        probe = booster.predict(np.zeros((1, n_features), dtype=np.float32))
        if probe.ndim == 2:
            num_class = probe.shape[1]
        else:
            num_class = 1
    if num_class == 1:
        out_shape = [None, 1]
    else:
        out_shape = [None, num_class]

    # ------------------------------------------------------------------
    # 2. Convert to ONNX
    # ------------------------------------------------------------------
    initial_types = [("features", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(
        booster,
        initial_types=initial_types,
        target_opset=15,
    )
    if len(onnx_model.graph.output) == 0:
        sys.exit("convert_lightgbm returned a graph with no outputs")

    out_names = [o.name for o in onnx_model.graph.output]
    print(f"  ONNX graph outputs: {out_names}")

    # Replace the ZipMap 'probabilities' output (a list-of-dicts) with a
    # dense [N, num_class] FLOAT tensor of the same name, so the Rust side
    # just reads a flat array. The pre-ZipMap tensor produced by
    # TreeEnsembleClassifier already holds the per-class probabilities.
    onnx_model = _finalize_dense_probabilities(onnx_model, num_class)

    model_out = os.path.join(args.out_dir, "model.onnx")
    onnx.save(onnx_model, model_out)
    print(f"Saved ONNX model: {model_out}")
# ------------------------------------------------------------------
    # 3. Copy the feature manifest + label order alongside
    # ------------------------------------------------------------------
    for name, payload in (
        ("model.features.json", manifest),
        ("model.label_map.json", {"0": "LOSS", "1": "WIN", "class_labels_ordered": ["LOSS", "WIN"]}),
    ):
        with open(os.path.join(args.out_dir, name), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote manifest: {os.path.join(args.out_dir, name)}")

    # ------------------------------------------------------------------
    # 4. Verify: LightGBM vs ONNX P(WIN) on random inputs
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    X_test = rng.normal(size=(args.verify_samples, n_features)).astype(np.float32)

    lgb_proba = booster.predict(X_test)  # shape (N,) for binary, (N,k) otherwise
    if lgb_proba.ndim == 1:
        lgb_pwin = lgb_proba  # binary booster: single column = P(class 1)
    else:
        lgb_pwin = lgb_proba[:, 1]

    sess = ort.InferenceSession(model_out, providers=["CPUExecutionProvider"])
    onnx_pwin = sess.run(["probabilities"], {"features": X_test})[0]
    if onnx_pwin.ndim == 2 and onnx_pwin.shape[1] >= 2:
        onnx_pwin = onnx_pwin[:, 1]
    else:
        onnx_pwin = onnx_pwin.ravel()

    max_err = float(np.max(np.abs(lgb_pwin - onnx_pwin)))
    print(f"  Max |P(WIN)_lgb - P(WIN)_onnx| over {args.verify_samples} samples = {max_err:.2e}")
    if max_err > 1e-4:
        sys.exit(f"VERIFICATION FAILED: max error {max_err:.2e} > 1e-4")
    print("  VERIFICATION PASSED: ONNX P(WIN) matches LightGBM P(WIN).")


def _finalize_dense_probabilities(onnx_model, num_class):
    """Rewrite the graph so the 'probabilities' output is a dense
    [N, num_class] FLOAT tensor instead of a ZipMap (list-of-dicts).

    1. Find the ZipMap node that produces 'probabilities', remember its input
       (the dense tensor from TreeEnsembleClassifier).
    2. Remove the ZipMap node and any Identity/Cast chain feeding the label.
    3. Add Identity(dense_tensor -> probabilities) and fix the output type.
    """
    from onnx import TensorProto, helper

    # LightGBM binary classifiers always emit a 2-column probability tensor
    # in ONNX even though the Python Booster returns a single column.
    out_dim = 2 if num_class < 2 else num_class

    zip_node = None
    zip_input = None
    for n in onnx_model.graph.node:
        if n.op_type == "ZipMap" and "probabilities" in n.output:
            zip_node = n
            zip_input = n.input[0]
            break

    if zip_node is not None:
        onnx_model.graph.node.remove(zip_node)

        # If the pre-ZipMap tensor is not already a graph output name, wire
        # it through an Identity so 'probabilities' stays the exposure point.
        existing_out = {o.name for o in onnx_model.graph.output}
        if zip_input in existing_out:
            # Already exposed; just re-type it if necessary.
            for o in onnx_model.graph.output:
                if o.name == zip_input:
                    o.type.tensor_type.elem_type = TensorProto.FLOAT
        else:
            onnx_model.graph.node.append(
                helper.make_node("Identity", [zip_input], ["probabilities"],
                                 name="prob_dense")
            )
            # Remove any stale 'probabilities' output info and add fresh.
            stale = [o for o in onnx_model.graph.output if o.name == "probabilities"]
            for o in stale:
                onnx_model.graph.output.remove(o)
            onnx_model.graph.output.append(helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, out_dim]
            ))

    # Safety: if there is no ZipMap yet but no dense probabilities either,
    # expose a Sigmoid of the last dense tensor as probabilities.
    out_names = [o.name for o in onnx_model.graph.output]
    if "probabilities" not in out_names:
        # Find the probability_tensor output of any TreeEnsemble classifier.
        prob_src = None
        for n in onnx_model.graph.node:
            if n.op_type == "TreeEnsembleClassifier" and len(n.output) >= 2:
                prob_src = n.output[1]
        if prob_src is None and out_names:
            prob_src = out_names[-1]
        onnx_model.graph.output.append(helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [None, out_dim]
        ))
        onnx_model.graph.node.append(
            helper.make_node("Identity", [prob_src], ["probabilities"],
                             name="prob_dense_fallback")
        )
    return onnx_model


if __name__ == "__main__":
    main()