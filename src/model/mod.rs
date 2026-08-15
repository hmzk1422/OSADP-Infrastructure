/// ONNX model inference via the `ort` (ONNX Runtime) crate.
///
/// Loads `model.onnx` + `model.features.json` (exported by
/// `tools/export_onnx.py`) and exposes `predict_pwin(&[f32; 15]) -> f32`,
/// the probability of the WIN class (model class index 1).
use anyhow::Context;
use ndarray::Array2;
use ort::{inputs, session::Session, value::Tensor};

pub struct OnnxModel {
    session: Session,
    n_features: usize,
    feature_names: Vec<String>,
}

impl OnnxModel {
    /// Load the ONNX model and validate its input/output names.
    pub fn load(model_path: &str, manifest_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()
            .with_context(|| "ort session builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("loading ONNX model: {model_path}"))?;

        // Feature manifest: exact order + count the engine must reproduce.
        let manifest_src = std::fs::read_to_string(manifest_path)
            .with_context(|| format!("reading feature manifest: {manifest_path}"))?;
        let manifest: serde_json::Value =
            serde_json::from_str(&manifest_src).context("parsing feature manifest JSON")?;
        let feature_names: Vec<String> = manifest
            .get("features")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|f| f.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let n_features = feature_names.len();
        anyhow::ensure!(
            n_features > 0,
            "feature manifest at {manifest_path} has no 'features' array"
        );

                log::info!(
            "ONNX model loaded: {model_path} — {n_features} features; session ready"
        );

        Ok(Self { session, n_features, feature_names })
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Run inference on a single feature vector and return P(WIN) = class 1.
        pub fn predict_pwin(&mut self, features: &[f32]) -> anyhow::Result<f32> {
        anyhow::ensure!(
            features.len() == self.n_features,
            "feature count mismatch: engine produced {}, model expects {}",
            features.len(),
            self.n_features
        );

        let arr = Array2::from_shape_vec((1, self.n_features), features.to_vec())
            .context("building input tensor")?;
        let input = Tensor::from_array(arr)?.into_dyn();

        let outputs = self
            .session
            .run(inputs!["features" => input])
            .context("ONNX run")?;

        let probs = outputs["probabilities"]
            .try_extract_array::<f32>()
            .context("extracting probabilities tensor")?;
        let probs = probs
            .into_dimensionality::<ndarray::Ix2>()
            .context("probabilities tensor is not 2-D")?;
        // probabilities shape: (1, num_class). P(WIN) = class-1 = probs[[0, 1]].
        Ok(probs[[0, 1]])
    }
}