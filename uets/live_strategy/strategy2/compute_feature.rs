//! strategy2/compute_feature.rs — YOUR computation for strategy2.
//!
//! Independent windows/state from strategy1: one instance per
//! (strategy x symbol) is owned by the runner.

use crate::core::feature::FeatureRow;
use crate::live_strategy::ComputeFeatures;

/// Example: z-score-ish distance of `price` from its 5-row mean -> `dz`.
pub struct Strategy2Features {
    window: Vec<f64>,
    size: usize,
}

impl Strategy2Features {
    pub fn new() -> Self {
        Self {
            window: Vec::new(),
            size: 5,
        }
    }
}

impl Default for Strategy2Features {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeFeatures for Strategy2Features {
    fn compute(&mut self, row: &FeatureRow) -> Result<FeatureRow, String> {
        let mut out = FeatureRow::new();
        match row.get("price") {
            Some(price) => {
                self.window.push(price);
                if self.window.len() > self.size {
                    self.window.remove(0);
                }
                let mean = self.window.iter().sum::<f64>() / self.window.len() as f64;
                out.insert("dz", price - mean);
            }
            None => out.insert_missing("dz"),
        }
        Ok(out)
    }

    fn output_names(&self) -> Vec<String> {
        vec!["dz".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deviation_from_mean() {
        let mut f = Strategy2Features::new();
        let mut row = FeatureRow::new();
        row.insert("price", 10.0);
        let out = f.compute(&row).unwrap();
        assert_eq!(out.get("dz"), Some(0.0));
    }
}
