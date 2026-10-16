//! OFMR live-strategy settings.
//!
//! No `serde_yaml` dependency: the live values mirror
//! `config/portfolio.json` `_strategy_defaults` and
//! `strategy_system/strategy/ofmr/config.py`. `load()` returns the hardcoded
//! defaults; a future pass can overlay a YAML file with a hand-rolled parser.

/// LightGBM model configuration.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub path: String,
    pub threshold: f64,
    pub trade_direction: String,
}

/// Feature window configuration.
#[derive(Clone, Debug)]
pub struct FeatureConfig {
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

/// Trade execution configuration.
#[derive(Clone, Debug)]
pub struct TradeConfig {
    pub max_position_usd: f64,
    pub min_order_size: f64,
    pub stop_loss_bps: f64,
    pub take_profit_bps: f64,
    pub lockout_seconds: f64,
    pub max_trades_per_hour: u32,
}

/// All OFMR settings, hardcoded to match research defaults.
#[derive(Clone, Debug)]
pub struct Settings {
    pub model: ModelConfig,
    pub feature: FeatureConfig,
    pub trade: TradeConfig,
}

impl Settings {
    /// Hardcoded defaults matching `config/portfolio.json` and `config.py`.
    pub fn defaults() -> Self {
        Self {
            model: ModelConfig {
                path: "uets/models/ofmr/trader.txt".to_string(),
                threshold: 0.20,
                trade_direction: "long".to_string(),
            },
            feature: FeatureConfig {
                rsi_14_window: 14,
                rsi_2_window: 2,
                sma_14_window: 14,
                ma_slope_window: 5,
                price_roc_window_3: 3,
                price_roc_window_5: 5,
                rsi_z_window: 20,
                deeplob_z_window: 10,
                orderflow_z_window: 10,
                rsi_roc_window_3: 3,
                rsi_roc_window_5: 5,
                barrier_vol_bar_window: 20,
            },
            trade: TradeConfig {
                max_position_usd: 100.0,
                min_order_size: 0.001,
                stop_loss_bps: 5.0,
                take_profit_bps: 5.0,
                lockout_seconds: 1000.0,
                max_trades_per_hour: 300,
            },
        }
    }

    /// Load settings: defaults, optionally overlaid by a YAML file at
    /// `CARGO_MANIFEST_DIR/live_strategy/ofmr/settings.yaml` (parsed by a
    /// hand-rolled parser in a future pass). For now this returns defaults.
    pub fn load() -> Self {
        Self::defaults()
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::load()
    }
}
