//! Live market monitor for one symbol: aggTrades + partial book depth.
//!
//! Two stream threads feed a shared state; the main loop writes a JSON frame
//! every 100 ms to `out/live/<symbol>.json`, which `display.py` renders as a
//! self-updating orderbook + tape view:
//!
//!     # terminal 1
//!     cargo run --bin dev_test -- monitor btcusdt
//!     # terminal 2
//!     python display.py btcusdt
//!
//! Depth note: this monitor uses the PARTIAL BOOK DEPTH stream
//! (`@depth20@100ms`), which pushes COMPLETE snapshots of the top 20 bid and
//! ask levels every 100 ms -- 20 of each, verified on the wire. The
//! diff-depth stream (`@depth@100ms`) used by the `depth`/`multi` commands
//! only carries the CHANGED levels, which is why it never printed 20 rows.
//!
//! All parsing is done here with serde_json instead of the `binance` crate's
//! event enum (same rationale as `all_trades_websocket`): we control every
//! field, including trade direction ("m"), event times, and the raw
//! lastUpdateId.

use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tungstenite::{Message, connect};

/// Aggregated trade + book state for the monitored symbol.
#[derive(Default)]
struct State {
    // tape
    last_price: f64,
    prev_price: f64,
    first_price: f64, // session open (first trade observed)
    buy_vol: f64,
    sell_vol: f64,
    trade_count: u64,
    last_trade_ms: u64,
    recent: Vec<Value>, // last few trades, oldest first
    // partial book (top 20 each side, best-first)
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
    book_ms: u64, // local receive time of the last book frame
    book_update_id: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Handle one aggTrade push: `{"e":"aggTrade","E":ms,"p":"price","q":"qty",
/// "m":bool}` -- `m=true` means the buyer was the maker, i.e. the taker SOLD.
fn on_trade(state: &Arc<Mutex<State>>, text: &str) {
    let v = match serde_json::from_str::<Value>(text) {
        Ok(v) => v,
        Err(_) => return,
    };
    if v["e"].as_str() != Some("aggTrade") {
        return;
    }
    let price: f64 = v["p"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let qty: f64 = v["q"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let side = if v["m"].as_bool().unwrap_or(false) {
        "sell"
    } else {
        "buy"
    };
    let event_ms = v["E"].as_u64().unwrap_or(0);

    let mut st = state.lock().expect("state poisoned");
    st.prev_price = if st.trade_count > 0 {
        st.last_price
    } else {
        price
    };
    st.last_price = price;
    if st.first_price == 0.0 {
        st.first_price = price;
    }
    if side == "buy" {
        st.buy_vol += qty;
    } else {
        st.sell_vol += qty;
    }
    st.trade_count += 1;
    st.last_trade_ms = event_ms;
    st.recent
        .push(json!({"t": event_ms, "p": price, "q": qty, "side": side}));
    if st.recent.len() > 8 {
        st.recent.remove(0);
    }
}

/// Handle one partial-book snapshot: `{"lastUpdateId":N,"bids":[[p,q]..20],
/// "asks":[[p,q]..20]}`. bids are already price-desc, asks price-asc.
fn on_book(state: &Arc<Mutex<State>>, text: &str) {
    let v = match serde_json::from_str::<Value>(text) {
        Ok(v) => v,
        Err(_) => return,
    };
    let parse_side = |arr: Option<&Vec<Value>>| -> Vec<(f64, f64)> {
        arr.map(|levels| {
            levels
                .iter()
                .filter_map(|lv| {
                    let p: f64 = lv.get(0)?.as_str()?.parse().ok()?;
                    let q: f64 = lv.get(1)?.as_str()?.parse().ok()?;
                    Some((p, q))
                })
                .collect()
        })
        .unwrap_or_default()
    };
    let bids = parse_side(v["bids"].as_array());
    let asks = parse_side(v["asks"].as_array());
    if bids.is_empty() || asks.is_empty() {
        return;
    }

    let mut st = state.lock().expect("state poisoned");
    st.bids = bids;
    st.asks = asks;
    st.book_update_id = v["lastUpdateId"].as_u64().unwrap_or(0);
    st.book_ms = now_ms();
}

/// Fire-and-forget reconnecting stream thread.
fn spawn_stream(url: String, on_msg: impl FnMut(&str) + Send + 'static) {
    std::thread::spawn(move || {
        let mut on_msg = on_msg;
        loop {
            match connect(url.as_str()) {
                Ok((mut ws, _)) => loop {
                    match ws.read() {
                        Ok(Message::Text(text)) => on_msg(&text),
                        Ok(Message::Ping(_))
                        | Ok(Message::Pong(_))
                        | Ok(Message::Binary(_))
                        | Ok(Message::Frame(_)) => (),
                        Ok(Message::Close(frame)) => {
                            eprintln!("[{}] closed by server: {:?}", url, frame.map(|f| f.code));
                            break;
                        }
                        Err(e) => {
                            eprintln!("[{}] error: {}", url, e);
                            break;
                        }
                    }
                },
                Err(e) => eprintln!("[{}] connect failed: {}", url, e),
            }
            eprintln!("[{}] reconnecting in 3s...", url);
            std::thread::sleep(Duration::from_secs(3));
        }
    });
}

/// Run the monitor until interrupted. Writes one compact JSON frame every
/// 100 ms (temp-file + rename, so the display never sees a torn file).
pub fn monitor_websocket(symbol: &str) {
    let symbol = symbol.to_uppercase();
    let low = symbol.to_lowercase();
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("out")
        .join("live");
    let _ = fs::create_dir_all(&out_dir);
    let frame_path = out_dir.join(format!("{}.json", low));
    let tmp_path = frame_path.with_extension("json.tmp");

    let state = Arc::new(Mutex::new(State::default()));

    {
        let st = Arc::clone(&state);
        spawn_stream(
            format!("wss://stream.binance.com:9443/ws/{}@aggTrade", low),
            move |text| on_trade(&st, text),
        );
    }
    {
        let st = Arc::clone(&state);
        spawn_stream(
            format!("wss://stream.binance.com:9443/ws/{}@depth20@100ms", low),
            move |text| on_book(&st, text),
        );
    }

    println!("monitor: {}  (aggTrade + depth20@100ms)", symbol);
    println!("frames:  {}", frame_path.display());
    println!("render:  python display.py {}", low);
    println!("stop:    Ctrl+C  (one dot per second)");

    let mut last_beat = 0u64;
    loop {
        std::thread::sleep(Duration::from_millis(100));

        let (trades_frame, bids, asks) = {
            let st = state.lock().expect("state poisoned");
            let trades = json!({
                "last_price": st.last_price,
                "prev_price": st.prev_price,
                "first_price": st.first_price,
                "buy_vol": st.buy_vol,
                "sell_vol": st.sell_vol,
                "trade_count": st.trade_count,
                "last_trade_ms": st.last_trade_ms,
                "recent": st.recent.clone(),
            });
            (trades, st.bids.clone(), st.asks.clone())
        };
        let (book_ms, book_update_id) = {
            let st = state.lock().expect("state poisoned");
            (st.book_ms, st.book_update_id)
        };
        let frame = json!({
            "ts": now_ms(),
            "symbol": symbol,
            "trades": trades_frame,
            "bids": bids,
            "asks": asks,
            "book_ms": book_ms,
            "book_update_id": book_update_id,
        });

        if fs::write(&tmp_path, frame.to_string()).is_ok() {
            let _ = fs::remove_file(&frame_path);
            let _ = fs::rename(&tmp_path, &frame_path);
        }

        let now = now_ms();
        if now - last_beat >= 1000 {
            last_beat = now;
            print!(".");
            let _ = std::io::stdout().flush();
        }
    }
}
