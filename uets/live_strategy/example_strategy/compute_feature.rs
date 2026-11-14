//! A bounded five-price feature window for the reference strategy.
//!
//! Independent windows/state from strategy1: one instance per
//! (strategy x symbol) is owned by the runner.

use std::collections::VecDeque;

use crate::core::feature::FeatureRow;
use crate::live_strategy::ComputeFeatures;

pub const WINDOW_SIZE: usize = 5;

pub struct ExampleFeatures {
    window: VecDeque<f64>,
}

impl ExampleFeatures {
    pub fn new() -> Self {
        Self {
            window: VecDeque::with_capacity(WINDOW_SIZE),
        }
    }
}

impl Default for ExampleFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeFeatures for ExampleFeatures {
    fn compute(&mut self, row: &FeatureRow) -> Result<FeatureRow, String> {
        let mut out = FeatureRow::new();
        match row.get("price") {
            Some(price) => {
                self.window.push_back(price);
                if self.window.len() > WINDOW_SIZE {
                    self.window.pop_front();
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

    fn reset(&mut self) {
        self.window.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deviation_from_five_price_mean() {
        let mut f = ExampleFeatures::new();
        let mut row = FeatureRow::new();
        row.insert("price", 10.0);
        let out = f.compute(&row).unwrap();
        assert_eq!(out.get("dz"), Some(0.0));
    }

    #[test]
    fn price_window_stays_bounded() {
        let mut features = ExampleFeatures::new();
        for price in 0..10_000 {
            let mut row = FeatureRow::new();
            row.insert("price", price as f64);
            features.compute(&row).unwrap();
        }
        assert_eq!(features.window.len(), WINDOW_SIZE);
        assert_eq!(features.window.front(), Some(&9_995.0));
    }
}
