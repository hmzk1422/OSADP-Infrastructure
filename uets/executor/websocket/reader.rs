//! Standalone Binance websocket readers.
//!
//! Every public function here is a self-contained stream listener. They are
//! NOT meant to be run directly -- use the dispatcher binary instead, which
//! routes to these functions via `executor.rs`:
//!
//!     cargo run --bin dev_test -- help
//!
use binance::api::*;
use binance::userstream::*;
use binance::websockets::*;

use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

use tungstenite::Message;

fn now_us() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros()
}

// Manages a "user data stream" over REST (not a websocket by itself).
// A user data stream gives you a `listen_key` which you then feed into
// a websocket connection (see user_stream_websocket) to get live account
// updates. Needs a real API key -- it's tied to your account, not public market data.
pub fn user_stream() {
    let api_key_user = Some("YOUR_API_KEY".into());
    let user_stream: UserStream = Binance::new(api_key_user, None);

    if let Ok(answer) = user_stream.start() {
        println!("Data Stream Started ...");
        let listen_key = answer.listen_key;

        match user_stream.keep_alive(&listen_key) {
            Ok(msg) => println!("Keepalive user data stream: {:?}", msg),
            Err(e) => println!("Error: {}", e),
        }

        match user_stream.close(&listen_key) {
            Ok(msg) => println!("Close user data stream: {:?}", msg),
            Err(e) => println!("Error: {}", e),
        }
    } else {
        println!("Not able to start an User Stream (Check your API_KEY)");
    }
}

// Opens a websocket of your account's live stats using the listen_key from
// user_stream(): balances and your own order fills.
pub fn user_stream_websocket() {
    let keep_running = AtomicBool::new(true);
    let api_key_user = Some("YOUR_KEY".into());
    let user_stream: UserStream = Binance::new(api_key_user, None);

    if let Ok(answer) = user_stream.start() {
        let listen_key = answer.listen_key;

        let mut web_socket: WebSockets<'_> = WebSockets::new(|event: WebsocketEvent| {
            match event {
                WebsocketEvent::AccountUpdate(account_update) => {
                    for balance in &account_update.data.balances {
                        println!(
                            "Asset: {}, wallet_balance: {}, cross_wallet_balance: {}, balance: {}",
                            balance.asset,
                            balance.wallet_balance,
                            balance.cross_wallet_balance,
                            balance.balance_change
                        );
                    }
                }
                WebsocketEvent::OrderTrade(trade) => {
                    println!(
                        "Symbol: {}, Side: {}, Price: {}, Execution Type: {}",
                        trade.symbol, trade.side, trade.price, trade.execution_type
                    );
                }
                _ => (),
            };

            Ok(())
        });

        web_socket.connect(&listen_key).unwrap();
        if let Err(e) = web_socket.event_loop(&keep_running) {
            println!("Error: {}", e);
        }
        user_stream.close(&listen_key).unwrap();
        web_socket.disconnect().unwrap();
        println!("Userstrem closed and disconnected");
    } else {
        println!("Not able to start an User Stream (Check your API_KEY)");
    }
}

// Public market data websocket. No API key required.
// - market_websocket("btcusdt", "aggTrade")     --> aggregated trades
// - market_websocket("btcusdt", "depth@100ms")  --> diff depth stream (incremental)
// - market_websocket("btcusdt", "depth20@100ms")--> fixed top-20 depth snapshot
pub fn market_websocket(ticker: &str, datatype: &str) {
    let keep_running = AtomicBool::new(true);
    let stream = format!("{}@{}", ticker.to_lowercase(), datatype);

    let mut web_socket: WebSockets<'_> = WebSockets::new(|event: WebsocketEvent| {
        match event {
            // NOTE: "@aggTrade" streams emit WebsocketEvent::AggrTrades in
            // binance 0.21 -- WebsocketEvent::Trade is only the raw "@trade"
            // stream. (Both structs share symbol/price/qty fields but are
            // distinct types, hence two arms.)
            WebsocketEvent::AggrTrades(trade) => {
                println!(
                    "Symbol: {}, price: {}, qty: {}",
                    trade.symbol, trade.price, trade.qty
                );
            }
            WebsocketEvent::Trade(trade) => {
                println!(
                    "Symbol: {}, price: {}, qty: {}",
                    trade.symbol, trade.price, trade.qty
                );
            }
            WebsocketEvent::DepthOrderBook(depth) => {
                let recv_time_us = now_us();

                println!(
                    "symbol={} event_time={} recv_time_us={} latency_us={}",
                    depth.symbol,
                    depth.event_time,
                    recv_time_us,
                    recv_time_us.saturating_sub(depth.event_time as u128 * 1000),
                );
            }
            WebsocketEvent::OrderBook(order_book) => {
                println!(
                    "last_update_id: {}, Bids: {:?}, Ask: {:?}",
                    order_book.last_update_id, order_book.bids, order_book.asks
                );
            }
            _ => (),
        };
        Ok(())
    });

    web_socket.connect(&stream).unwrap();
    if let Err(e) = web_socket.event_loop(&keep_running) {
        println!("Error: {}", e);
    }
    web_socket.disconnect().unwrap();
    println!("disconnected");
}

// All-symbols mini-ticker stream ("!miniTicker@arr"). Every push is a JSON
// ARRAY of all symbols' 24h stats (close, high, low, quote volume).
//
// This is deliberately NOT going through the `binance` crate's WebSockets
// helper: its `Events` enum only matches JSON objects keyed on the "e" field,
// so it cannot represent array payloads at all (the old "!ticker@arr" version
// here silently dropped every message because of exactly that). The sibling
// "!ticker@arr" stream itself was also verified dead upstream (connects but
// never pushes, as of 2026-09-19), so this reader uses the working
// "!miniTicker@arr" stream and parses the raw text with serde_json.
pub fn all_trades_websocket() {
    let url = "wss://stream.binance.com:9443/ws/!miniTicker@arr";
    let (mut ws, _response) = tungstenite::connect(url).expect("websocket connect failed");
    println!("connected: {}", url);

    loop {
        match ws.read() {
            Ok(Message::Text(text)) => {
                let parsed: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let items = match parsed.as_array() {
                    Some(a) => a,
                    None => continue,
                };
                for item in items {
                    println!(
                        "Symbol: {}, close: {}, high: {}, low: {}, quote_vol: {}",
                        item["s"].as_str().unwrap_or("-"),
                        item["c"].as_str().unwrap_or("-"),
                        item["h"].as_str().unwrap_or("-"),
                        item["l"].as_str().unwrap_or("-"),
                        item["q"].as_str().unwrap_or("-")
                    );
                }
            }
            Ok(Message::Ping(_) | Message::Pong(_) | Message::Binary(_) | Message::Frame(_)) => (),
            Ok(Message::Close(frame)) => {
                println!("closed by server: {:?}", frame);
                break;
            }
            Err(e) => {
                println!("Error: {}", e);
                break;
            }
        }
    }
    println!("disconnected");
}

// 1-minute kline (candlestick) stream for `symbol`.
pub fn kline_websocket(symbol: &str) {
    let keep_running = AtomicBool::new(true);
    let kline = format!("{}@kline_1m", symbol.to_lowercase());
    let mut web_socket = WebSockets::new(|event: WebsocketEvent| {
        if let WebsocketEvent::Kline(kline_event) = event {
            println!(
                "Symbol: {}, high: {}, low: {}",
                kline_event.kline.symbol, kline_event.kline.high, kline_event.kline.low
            );
        }
        Ok(())
    });

    web_socket.connect(&kline).unwrap();
    if let Err(e) = web_socket.event_loop(&keep_running) {
        println!("Error: {}", e);
    }
    web_socket.disconnect().unwrap();
    println!("disconnected");
}

// 24h rolling ticker for `symbol`; prints average and close price on every
// push. Runs until interrupted (Ctrl+C).
pub fn ticker_websocket(symbol: &str) {
    let keep_running = AtomicBool::new(true);
    let stream = format!("{}@ticker", symbol.to_lowercase());

    let mut web_socket = WebSockets::new(|event: WebsocketEvent| {
        if let WebsocketEvent::DayTicker(ticker_event) = event {
            let average_price: f32 = ticker_event.average_price.parse().unwrap();
            let current_close: f32 = ticker_event.current_close.parse().unwrap();
            println!("{} - {}", average_price, current_close);
        }
        Ok(())
    });

    web_socket.connect(&stream).unwrap();
    if let Err(e) = web_socket.event_loop(&keep_running) {
        println!("Error: {}", e);
    }
    web_socket.disconnect().unwrap();
    println!("disconnected");
}

// Subscribes to multiple streams (diff-depth for the given symbols) over one
// connection. The same pattern works for mixing stream types too.
pub fn multiple_streams(symbols: &[&str]) {
    let endpoints: Vec<String> = symbols
        .iter()
        .map(|symbol| format!("{}@depth@100ms", symbol.to_lowercase()))
        .collect();

    let keep_running = AtomicBool::new(true);
    let mut web_socket = WebSockets::new(|event: WebsocketEvent| {
        if let WebsocketEvent::DepthOrderBook(depth_order_book) = event {
            let top_bid = depth_order_book.bids.first();
            let top_ask = depth_order_book.asks.first();
            println!(
                "Symbol: {}, top_bid: {:?}@{:?}, top_ask: {:?}@{:?} (levels: {} bid / {} ask)",
                depth_order_book.symbol,
                top_bid.map(|b| b.price),
                top_bid.map(|b| b.qty),
                top_ask.map(|a| a.price),
                top_ask.map(|a| a.qty),
                depth_order_book.bids.len(),
                depth_order_book.asks.len()
            );
        }
        Ok(())
    });

    web_socket.connect_multiple_streams(&endpoints).unwrap();
    if let Err(e) = web_socket.event_loop(&keep_running) {
        println!("Error: {}", e);
    }
    web_socket.disconnect().unwrap();
}
