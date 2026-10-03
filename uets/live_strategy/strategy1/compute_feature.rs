//! strategy1/compute_feature.rs — YOUR computation.
//!
//! Runs FIRST on every stream row: canonical `FeatureRow` in, computed
//! feature row out. Stateful: one instance per (strategy x symbol), so put
//! rolling windows / running sums here — they never leak to other strategies.
//!
//! Replace the body with your formula. Keep `Missing` semantics: emit
//! `insert_missing` while history is too short, never zero.

use crate::core::feature::FeatureRow;
use crate::live_strategy::ComputeFeatures;

/// Example: 3-row trailing mean of `price` -> feature `ma3`.
///
/// Delete this and write your own. The contract is only:
/// `compute(&mut self, row) -> feature row`.
pub struct Strategy1Features {
    window: Vec<f64>,
    size: usize,
}

impl Strategy1Features {
    pub fn new() -> Self {
        Self {
            window: Vec::new(),
            size: 3,
        }
    }
}

impl Default for Strategy1Features {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeFeatures for Strategy1Features {
    fn compute(&mut self, row: &FeatureRow) -> Result<FeatureRow, String> {
        let mut out = FeatureRow::new();
        match row.get("price") {
            Some(price) => {
                self.window.push(price);
                if self.window.len() > self.size {
                    self.window.remove(0);
                }
                let mean = self.window.iter().sum::<f64>() / self.window.len() as f64;
                out.insert("ma3", mean);
            }
            None => out.insert_missing("ma3"),
        }
        Ok(out)
    }

    fn output_names(&self) -> Vec<String> {
        vec!["ma3".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(price: Option<f64>) -> FeatureRow {
        let mut r = FeatureRow::new();
        match price {
            Some(p) => r.insert("price", p),
            None => r.insert_missing("price"),
        }
        r
    }

    #[test]
    fn trailing_mean_over_price() {
        let mut f = Strategy1Features::new();
        let r = f.compute(&row(Some(1.0))).unwrap();
        assert_eq!(r.get("ma3"), Some(1.0));
        f.compute(&row(Some(2.0))).unwrap();
        let r = f.compute(&row(Some(3.0))).unwrap();
        assert_eq!(r.get("ma3"), Some(2.0));
    }

    #[test]
    fn missing_input_stays_missing() {
        let mut f = Strategy1Features::new();
        let r = f.compute(&row(None)).unwrap();
        assert_eq!(r.get("ma3"), None);
    }
}
