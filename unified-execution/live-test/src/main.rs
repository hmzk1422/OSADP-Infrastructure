use engine::{MomentumStrategy, Tick}; // <-- Change config to engine

fn main() {
    let mut strategy = MomentumStrategy::new("BTCUSDT");

    // Simulate market feed from CSV / Database / Parquet
    let historical_ticks = vec![
        Tick { price: 98.5, volume: 1.2 },
        Tick { price: 101.2, volume: 2.5 },
    ];

    for tick in historical_ticks {
        if let Some(signal) = strategy.on_tick(&tick) {
            println!("[SIMULATED EXECUTION] Generated: {}", signal);
        }
    }
}