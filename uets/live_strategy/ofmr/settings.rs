//! OFMR runtime configuration.
//!
//! Loads `settings.yaml` instead of silently using hard-coded values. Search order:
//! `OFMR_SETTINGS_PATH`, then the known repository-relative locations below.

use serde::Deserialize;
use std::{fs, path::{Path, PathBuf}};

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[derive(Default)]
pub struct Settings {
    pub model: ModelSettings,
    pub feature: FeatureSettings,
    pub trade: TradeSettings,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSettings {
    pub path: String,
    pub threshold: f64,
    pub trade_direction: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FeatureSettings {
    /// Macro grid: seconds per higher-timeframe bar. Mirrors research
    /// `BAR_FREQ = f"{timeframe_barrier_secs}s"` (config/portfolio.json).
    pub htf_timeframe_secs: u64,
    /// Micro grid: seconds per lower-timeframe bar. Mirrors research
    /// `micro_barrier_secs` (the 1-second decision grid).
    pub ltf_timeframe_secs: u64,
    pub rsi_14_window: usize,
    pub rsi_2_window: usize,
    pub sma_14_window: usize,
    pub ma_slope_window: usize,
    pub price_roc_window_3: usize,
    pub price_roc_window_5: usize,
    pub rsi_z_window: usize,
    pub deeplob_z_window: usize,
    pub orderflow_z_window: usize,
    pub rsi_roc_window_3: usize,
    pub rsi_roc_window_5: usize,
    pub barrier_vol_bar_window: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TradeSettings {
    pub max_position_usd: f64,
    pub min_order_size: f64,
    pub max_hold_seconds: u64,
    pub lockout_seconds: f64,
    pub max_trades_per_hour: u32,
    /// Fixed stop-loss width, in basis points of entry price (mirrors
    /// `config.py` `STOP_LOSS_BPS` / `config/portfolio.json`).
    pub stop_loss_bps: f64,
    /// Fixed take-profit width, in basis points of entry price (mirrors
    /// `config.py` `TAKE_PROFIT_BPS` / `config/portfolio.json`).
    pub take_profit_bps: f64,
}

impl Default for ModelSettings {
    fn default() -> Self { Self { path: "uets/models/ofmr/trader.txt".into(), threshold: 0.50, trade_direction: "long".into() } }
}
impl Default for FeatureSettings {
    fn default() -> Self { Self {
        htf_timeframe_secs: 300, ltf_timeframe_secs: 1,
        rsi_14_window: 14, rsi_2_window: 2, sma_14_window: 14, ma_slope_window: 5,
        price_roc_window_3: 3, price_roc_window_5: 5, rsi_z_window: 20,
        deeplob_z_window: 10, orderflow_z_window: 10, rsi_roc_window_3: 3,
        rsi_roc_window_5: 5, barrier_vol_bar_window: 20,
    } }
}
impl Default for TradeSettings {
    fn default() -> Self { Self {
        max_position_usd: 10_000.0, min_order_size: 0.001,
        max_hold_seconds: 300, lockout_seconds: 1000.0,
        max_trades_per_hour: 300,
        stop_loss_bps: 5.0,
        take_profit_bps: 5.0,
    } }
}

impl Settings {
    /// Compatibility helper for old call sites. Prefer `try_load()` in runtime startup.
    /// This deliberately fails loudly instead of running a different strategy than configured.
    pub fn load() -> Self {
        Self::try_load().unwrap_or_else(|e| panic!("OFMR settings load failed: {e}"))
    }

    pub fn try_load() -> Result<Self, String> {
        let path = Self::find_path()?;
        let source = fs::read_to_string(&path)
            .map_err(|e| format!("read OFMR settings {}: {e}", path.display()))?;
        let settings: Self = if path.file_name().is_some_and(|name| name == "execution.yaml") {
            let document: serde_yaml::Value = serde_yaml::from_str(&source)
                .map_err(|e| format!("parse UETS settings {}: {e}", path.display()))?;
            match document.get("ofmr") {
                Some(value) => serde_yaml::from_value(value.clone())
                    .map_err(|e| format!("parse OFMR settings in {}: {e}", path.display()))?,
                None => serde_yaml::from_str(&source)
                    .map_err(|e| format!("parse OFMR settings {}: {e}", path.display()))?,
            }
        } else {
            serde_yaml::from_str(&source)
                .map_err(|e| format!("parse OFMR settings {}: {e}", path.display()))?
        };
        settings.validate().map_err(|e| format!("invalid OFMR settings {}: {e}", path.display()))?;
        eprintln!("[ofmr] loaded settings from {}", path.display());
        Ok(settings)
    }

    fn find_path() -> Result<PathBuf, String> {
        if let Some(value) = std::env::var_os("OFMR_SETTINGS_PATH") {
            let path = PathBuf::from(value);
            if path.is_file() { return Ok(path); }
            return Err(format!("OFMR_SETTINGS_PATH points to missing file {}", path.display()));
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            root.join("config/execution.yaml"),
        ];
        candidates.iter().find(|p| p.is_file()).cloned().ok_or_else(|| {
            format!("settings.yaml not found; checked: {} (or set OFMR_SETTINGS_PATH)",
                candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", "))
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.feature.htf_timeframe_secs == 0 { return Err("feature.htf_timeframe_secs must be > 0".into()); }
        if self.feature.ltf_timeframe_secs == 0 { return Err("feature.ltf_timeframe_secs must be > 0".into()); }
        if self.feature.ltf_timeframe_secs > self.feature.htf_timeframe_secs {
            return Err(format!(
                "feature.ltf_timeframe_secs ({}) must be <= feature.htf_timeframe_secs ({})",
                self.feature.ltf_timeframe_secs, self.feature.htf_timeframe_secs
            ));
        }
        let positive_windows = [
            ("feature.rsi_14_window", self.feature.rsi_14_window),
            ("feature.rsi_2_window", self.feature.rsi_2_window),
            ("feature.sma_14_window", self.feature.sma_14_window),
            ("feature.ma_slope_window", self.feature.ma_slope_window),
            ("feature.price_roc_window_3", self.feature.price_roc_window_3),
            ("feature.price_roc_window_5", self.feature.price_roc_window_5),
            ("feature.rsi_z_window", self.feature.rsi_z_window),
            ("feature.deeplob_z_window", self.feature.deeplob_z_window),
            ("feature.orderflow_z_window", self.feature.orderflow_z_window),
            ("feature.rsi_roc_window_3", self.feature.rsi_roc_window_3),
            ("feature.rsi_roc_window_5", self.feature.rsi_roc_window_5),
            ("feature.barrier_vol_bar_window", self.feature.barrier_vol_bar_window),
        ];
        for (name, value) in positive_windows { if value == 0 { return Err(format!("{name} must be > 0")); } }
        if self.model.path.trim().is_empty() { return Err("model.path must not be empty".into()); }
        if !self.model.threshold.is_finite() || !(0.0..=1.0).contains(&self.model.threshold) {
            return Err("model.threshold must be finite and between 0 and 1".into());
        }
        if !matches!(self.model.trade_direction.as_str(), "long" | "short" | "both") {
            return Err("model.trade_direction must be 'long', 'short', or 'both'".into());
        }
        if !self.trade.max_position_usd.is_finite() || self.trade.max_position_usd <= 0.0 {
            return Err("trade.max_position_usd must be positive and finite".into());
        }
        if !self.trade.min_order_size.is_finite() || self.trade.min_order_size <= 0.0 {
            return Err("trade.min_order_size must be positive and finite".into());
        }
        if !self.trade.lockout_seconds.is_finite() || self.trade.lockout_seconds < 0.0 {
            return Err("trade.lockout_seconds must be finite and >= 0".into());
        }
        if self.trade.max_trades_per_hour == 0 { return Err("trade.max_trades_per_hour must be > 0".into()); }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_execution_yaml_contains_loadable_ofmr_settings() {
        let document: serde_yaml::Value =
            serde_yaml::from_str(include_str!("../../config/execution.yaml")).unwrap();
        let settings: Settings = serde_yaml::from_value(document["ofmr"].clone()).unwrap();
        settings.validate().unwrap();
        assert_eq!(settings.trade.max_position_usd, 10_000.0);
        assert_eq!(settings.feature.ltf_timeframe_secs, 1);
    }

    #[test]
    fn parses_the_supplied_yaml_schema() {
        let yaml = r#"
model:
  path: "uets/models/ofmr/trader.txt"
  threshold: 0.50
  trade_direction: "long"
feature:
  htf_timeframe_secs: 300
  ltf_timeframe_secs: 1
  rsi_14_window: 14
  rsi_2_window: 2
  sma_14_window: 14
  ma_slope_window: 5
  price_roc_window_3: 3
  price_roc_window_5: 5
  rsi_z_window: 20
  deeplob_z_window: 10
  orderflow_z_window: 10
  rsi_roc_window_3: 3
  rsi_roc_window_5: 5
  barrier_vol_bar_window: 20
trade:
  max_position_usd: 10000.0
  min_order_size: 0.001
  max_hold_seconds: 300
  lockout_seconds: 1000.0
  max_trades_per_hour: 300
"#;
        let settings: Settings = serde_yaml::from_str(yaml).unwrap();
        settings.validate().unwrap();
        assert_eq!(settings.feature.rsi_2_window, 2);
        assert_eq!(settings.model.trade_direction, "long");
    }
}