mod bybit_ws;

use bybit_ws::{OrderBookWatcher, TradeWatcher};
use engine::{MomentumStrategy, Tick};
use std::sync::Arc;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

const SYMBOL: &str = "BTCUSDT";
const TRADES_CSV: &str = "trades.csv";
const ORDERBOOK_CSV: &str = "orderbook.csv";

async fn open_csv(path: &str, header: &str) -> std::io::Result<tokio::fs::File> {
    let is_new = !std::path::Path::new(path).exists();
    let mut file = OpenOptions::new().create(true).append(true).open(path).await?;
    if is_new {
        file.write_all(header.as_bytes()).await?;
    }
    Ok(file)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let strategy = Arc::new(Mutex::new(MomentumStrategy::new(SYMBOL)));

    let trade_strategy = Arc::clone(&strategy);
    let trades_task = tokio::spawn(async move {
        let mut trades_file = open_csv(TRADES_CSV, "timestamp,price,amount,side,id\n")
            .await
            .expect("failed to open trades.csv");

        loop {
            let mut watcher = match TradeWatcher::connect(SYMBOL).await {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("[trades] connect error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }
            };

            loop {
                match watcher.next_batch().await {
                    Ok(trades) => {
                        for trade in trades {
                            let row = format!("{trade}\n");
                            if let Err(e) = trades_file.write_all(row.as_bytes()).await {
                                eprintln!("[trades] csv write error: {e}");
                            }

                            let live_tick = Tick {
                                price: trade.price,
                                volume: trade.amount,
                            };

                            let mut strat = trade_strategy.lock().await;
                            if let Some(signal) = strat.on_tick(&live_tick) {
                                println!("[LIVE TRADES] Order Sent: {}", signal);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[trades] stream error: {e}");
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        break; // reconnect
                    }
                }
            }
        }
    });

    let ob_strategy = Arc::clone(&strategy);
    let ob_task = tokio::spawn(async move {
        let mut ob_file = open_csv(
            ORDERBOOK_CSV,
            "timestamp,bid_price,bid_amount,ask_price,ask_amount\n",
        )
        .await
        .expect("failed to open orderbook.csv");

        loop {
            let mut watcher = match OrderBookWatcher::connect(SYMBOL).await {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("[orderbook] connect error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }
            };

            loop {
                match watcher.next_top().await {
                    Ok(Some(top)) => {
                        let row = format!("{top}\n");
                        if let Err(e) = ob_file.write_all(row.as_bytes()).await {
                            eprintln!("[orderbook] csv write error: {e}");
                        }

                        let mid_price = (top.bid_price + top.ask_price) / 2.0;
                        let live_tick = Tick {
                            price: mid_price,
                            volume: 0.0,
                        };

                        let mut strat = ob_strategy.lock().await;
                        if let Some(signal) = strat.on_tick(&live_tick) {
                            println!("[LIVE LOB] Order Sent: {}", signal);
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("[orderbook] stream error: {e}");
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        break; // reconnect
                    }
                }
            }
        }
    });

    println!("Listening to Bybit Live Trades and LOB for {SYMBOL}...");

    let _ = tokio::join!(trades_task, ob_task);

    Ok(())
}