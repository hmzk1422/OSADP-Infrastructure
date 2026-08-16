// Shared market event
pub struct Tick {
    pub price: f64,
    pub volume: f64,
}

// Shared trading strategy state
pub struct MomentumStrategy {
    pub symbol: String,
    // Add indicators, model weights, strategy state here
}

impl MomentumStrategy {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
        }
    }

    // Shared execution logic called by BOTH live-test and live-trade
    pub fn on_tick(&mut self, tick: &Tick) -> Option<String> {
        if tick.price > 100.0 {
            Some(format!("BUY signal for {}", self.symbol))
        } else {
            None
        }
    }
}