//! Model identity and selection.
//!
//! Selection is a separate trait from the decision rule so a strategy can be
//! pointed at a different model without the rule knowing: a model id is data,
//! and swapping it must not require editing the trading logic.

/// Identifies a trained model by name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Chooses which model serves a given (symbol, sequence) pair.
pub trait ModelSelector: Send + Sync {
    fn select(&self, symbol: &str, sequence: u64) -> ModelId;
}

/// Always returns the same model. The simple, predictable default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedModel(pub ModelId);

impl FixedModel {
    pub fn new(id: impl Into<ModelId>) -> Self {
        Self(id.into())
    }
}

impl ModelSelector for FixedModel {
    fn select(&self, _symbol: &str, _sequence: u64) -> ModelId {
        self.0.clone()
    }
}

/// Rotates models on a fixed cadence, e.g. to average several models' signals.
///
/// Rounds the sequence down to the cadence so the choice depends only on the
/// step, never on a clock: a backtest replaying the same rows then selects the
/// same models as the live run that produced them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RotatingModel {
    models: Vec<ModelId>,
    cadence: u64,
}

impl RotatingModel {
    /// Refuses an empty model list or a zero cadence, both of which would make
    /// `select` panic or silently divide by zero at the worst possible moment.
    pub fn new(models: Vec<ModelId>, cadence: u64) -> Result<Self, String> {
        if models.is_empty() {
            return Err("RotatingModel needs at least one model".to_string());
        }
        if cadence == 0 {
            return Err("RotatingModel cadence must be positive".to_string());
        }
        Ok(Self { models, cadence })
    }

    pub fn models(&self) -> &[ModelId] {
        &self.models
    }
}

impl ModelSelector for RotatingModel {
    fn select(&self, _symbol: &str, sequence: u64) -> ModelId {
        let step = (sequence / self.cadence) as usize % self.models.len();
        self.models[step].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_model_always_returns_itself() {
        let selector = FixedModel::new(ModelId::new("m1"));
        assert_eq!(selector.select("BTCUSDT", 0), ModelId::new("m1"));
        assert_eq!(selector.select("BTCUSDT", 999), ModelId::new("m1"));
    }

    #[test]
    fn rotation_depends_only_on_the_step_not_a_clock() {
        let selector =
            RotatingModel::new(vec![ModelId::new("a"), ModelId::new("b")], 10).expect("valid");
        assert_eq!(selector.select("BTCUSDT", 0), ModelId::new("a"));
        assert_eq!(selector.select("BTCUSDT", 9), ModelId::new("a"));
        assert_eq!(selector.select("BTCUSDT", 10), ModelId::new("b"));
        assert_eq!(selector.select("BTCUSDT", 20), ModelId::new("a"));
    }

    #[test]
    fn invalid_rotation_is_rejected_at_construction() {
        assert!(RotatingModel::new(Vec::new(), 10).is_err());
        assert!(RotatingModel::new(vec![ModelId::new("a")], 0).is_err());
    }
}
