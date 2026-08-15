/// Configuration for the Rust executor — mirrors `main/config.py`.
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    // ------------------------------------------------------------------
    // Exchange / data feed
    // ------------------------------------------------------------------
    #[serde(default = "default_symbol")]
    pub symbol: String,
    #[serde(default = "default_sidecar_host")]
    pub sidecar_host: String,
    #[serde(default = "default_sidecar_port")]
    pub sidecar_port: u16,

    // ------------------------------------------------------------------
    // Strategy parameters
    // ------------------------------------------------------------------
    #[serde(default = "default_trade_direction")]
    pub trade_direction: String,         // "short" or "long"
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
    #[serde(default = "default_rsi_filter_threshold")]
    pub rsi_filter_threshold: f32,
    #[serde(default = "default_enable_rsi_accel_filter")]
    pub enable_rsi_accel_filter: bool,
    #[serde(default = "default_lookahead_seconds")]
    pub lookahead_seconds: u64,
    #[serde(default = "default_barrier_multiplier")]
    pub barrier_multiplier: f32,
    #[serde(default = "default_rr_ratio")]
    pub rr_ratio: f32,
    #[serde(default = "default_min_barrier_pct")]
    pub min_barrier_pct: f32,
    #[serde(default = "default_max_barrier_pct")]
    pub max_barrier_pct: f32,
    #[serde(default = "default_timeframe_barrier")]
    pub timeframe_barrier: u64,          // seconds
    #[serde(default = "default_barrier_vol_bar_window")]
    pub barrier_vol_bar_window: usize,
    #[serde(default = "default_hysteresis_buffer")]
    pub hysteresis_buffer: f32,
    #[serde(default = "default_lockout_seconds")]
    pub lockout_seconds: u64,
    #[serde(default = "default_max_trades_per_hour")]
    pub max_trades_per_hour: u32,

    // ------------------------------------------------------------------
    // Risk / position sizing
    // ------------------------------------------------------------------
    #[serde(default = "default_stop_loss_bps")]
    pub stop_loss_bps: f32,
    #[serde(default = "default_take_profit_bps")]
    pub take_profit_bps: f32,
    #[serde(default = "default_max_position_usd")]
    pub max_position_usd: f32,
    #[serde(default = "default_min_order_size")]
    pub min_order_size: f32,
    #[serde(default = "default_fee_bps")]
    pub fee_bps: f32,
    #[serde(default = "default_enable_slippage")]
    pub enable_slippage: bool,

    // ------------------------------------------------------------------
    // Feature windows
    // ------------------------------------------------------------------
    #[serde(default = "default_use_orderflow")]
    pub use_orderflow: bool,
    #[serde(default = "default_rsi_14_window")]
    pub rsi_14_window: usize,
    #[serde(default = "default_rsi_2_window")]
    pub rsi_2_window: usize,
    #[serde(default = "default_sma_14_window")]
    pub sma_14_window: usize,
    #[serde(default = "default_ma_slope_window")]
    pub ma_slope_window: usize,
    #[serde(default = "default_price_roc_window_3")]
    pub price_roc_window_3: usize,
    #[serde(default = "default_deeplob_z_window")]
    pub deeplob_z_window: usize,
    #[serde(default = "default_rsi_z_window")]
    pub rsi_z_window: usize,
    #[serde(default = "default_orderflow_z_window")]
    pub orderflow_z_window: usize,
    #[serde(default = "default_retail_threshold")]
    pub retail_threshold: f32,

    // ------------------------------------------------------------------
    // VPIN (optional)
    // ------------------------------------------------------------------
    #[serde(default = "default_vpin_target_buckets")]
    pub vpin_target_buckets: u32,
    #[serde(default = "default_vpin_window_buckets")]
    pub vpin_window_buckets: u32,
    #[serde(default = "default_vpin_min_periods")]
    pub vpin_min_periods: u32,
    #[serde(default = "default_vpin_lookback_seconds")]
    pub vpin_lookback_seconds: u64,

    // ------------------------------------------------------------------
    // Model
    // ------------------------------------------------------------------
    #[serde(default = "default_model_path")]
    pub model_path: String,
    #[serde(default = "default_model_manifest_path")]
    pub model_manifest_path: String,

    // ------------------------------------------------------------------
    // Execution
    // ------------------------------------------------------------------
    #[serde(default = "default_dry_run")]
    pub dry_run: bool,
    #[serde(default = "default_seed_bars")]
    pub seed_bars: u32,
    #[serde(default = "default_seed_interval")]
    pub seed_interval: String,
    #[serde(default = "default_sidecar_python")]
    pub sidecar_python: String,
}

// ===========================================================================
// Defaults - mirror main/config.py
// ============================================================================

fn default_symbol() -> String { "BTC/USDT:USDT".into() }
fn default_sidecar_host() -> String { "127.0.0.1".into() }
fn default_sidecar_port() -> u16 { 9100 }
fn default_trade_direction() -> String { "short".into() }
fn default_confidence_threshold() -> f32 { 0.5 }
fn default_rsi_filter_threshold() -> f32 { 30.0 }
fn default_enable_rsi_accel_filter() -> bool { false }
fn default_lookahead_seconds() -> u64 { 6000 }
fn default_barrier_multiplier() -> f32 { 3.0 }
fn default_rr_ratio() -> f32 { 0.1 }
fn default_min_barrier_pct() -> f32 { 0.0 }
fn default_max_barrier_pct() -> f32 { 0.05 }
fn default_timeframe_barrier() -> u64 { 300 }
fn default_barrier_vol_bar_window() -> usize { 20 }
fn default_hysteresis_buffer() -> f32 { 0.0 }
fn default_lockout_seconds() -> u64 { 1000 }
fn default_max_trades_per_hour() -> u32 { 300 }
fn default_stop_loss_bps() -> f32 { 5.0 }
fn default_take_profit_bps() -> f32 { 5.0 }
fn default_max_position_usd() -> f32 { 100.0 }
fn default_min_order_size() -> f32 { 0.001 }
fn default_fee_bps() -> f32 { 0.0 }
fn default_enable_slippage() -> bool { false }
fn default_use_orderflow() -> bool { true }
fn default_rsi_14_window() -> usize { 14 }
fn default_rsi_2_window() -> usize { 2 }
fn default_sma_14_window() -> usize { 14 }
fn default_ma_slope_window() -> usize { 5 }
fn default_price_roc_window_3() -> usize { 3 }
fn default_deeplob_z_window() -> usize { 10 }
fn default_rsi_z_window() -> usize { 10 }
fn default_orderflow_z_window() -> usize { 10 }
fn default_retail_threshold() -> f32 { 0.05 }
fn default_vpin_target_buckets() -> u32 { 50 }
fn default_vpin_window_buckets() -> u32 { 10 }
fn default_vpin_min_periods() -> u32 { 5 }
fn default_vpin_lookback_seconds() -> u64 { 86400 }
fn default_model_path() -> String { "src/assets/model.onnx".into() }
fn default_model_manifest_path() -> String { "src/assets/model.features.json".into() }
fn default_dry_run() -> bool { true }
fn default_seed_bars() -> u32 { 500 }
fn default_seed_interval() -> String { "5m".into() }
fn default_sidecar_python() -> String { "python".into() }

impl Config {
    /// Load config from a TOML file, merging with built-in defaults.
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())?;
        Ok(toml::from_str(&s)?)
    }
}
