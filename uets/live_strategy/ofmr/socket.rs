//! OFMR socket consumer — IPC Unix socket (`sockets/strategies/ofmr.sock`).
//!
//! Framing: `u32` LE length prefix + payload (max 1 MB). Each frame carries a
//! per-symbol tick row; we decode it into a `FeatureRow`, run `compute` +
//! `decide`, and emit trades through `execute()` (print-only here).
//!
//! On win32 (`cfg(not(unix))`) this is a compile-time stub so the crate
//! builds everywhere; the live loop only runs on Unix.

use std::path::Path;

use crate::core::feature::FeatureRow;
use crate::core::strategy::{Decision, Position};

const MAX_FRAME: usize = 1_048_576; // 1 MB

/// Outcome of claiming our socket path.
#[derive(Debug, PartialEq, Eq)]
enum Claim {
    Free,
    OwnedByLive(u32),
}

/// Decode one IPC frame into a `FeatureRow`. Frame body is expected to be a
/// simple `name:value` line protocol, one feature per line.
fn decode_frame(payload: &[u8]) -> Result<FeatureRow, String> {
    let text = std::str::from_utf8(payload)
        .map_err(|e| format!("frame payload is not UTF-8: {e}"))?;
    let mut row = FeatureRow::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((name, val)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        let val = val.trim();
        if let Ok(v) = val.parse::<f64>() {
            row.insert(name, v);
        } else {
            row.insert_missing(name);
        }
    }
    Ok(row)
}

/// Emit a trade decision (print-only in this phase).
fn execute(symbol: &str, decision: Decision, position: Position) {
    println!(
        "OFMR execute symbol={symbol} decision={decision:?} position={position:?}"
    );
}

/// Run the OFMR socket consumer loop (Unix only).
#[cfg(unix)]
pub fn run(socket_path: &str) -> Result<(), String> {
    use std::io::{Read, Write};
    use std::os::unix::net::{UnixListener, UnixStream};

    // Claim the socket path (reuse aggregator-style stale-socket cleanup).
    let claim = crate::aggregator::cleanup::claim(Path::new(socket_path))
        .map_err(|e| format!("claim socket {socket_path}: {e}"))?;
    match claim {
        Claim::OwnedByLive(_pid) => {
            return Err(format!("socket {socket_path} is owned by a live process"));
        }
        Claim::Free => {}
    }

    let listener = UnixListener::bind(socket_path)
        .map_err(|e| format!("bind {socket_path}: {e}"))?;

    let mut features = OfmrFeatures::new();
    let mut strategy = OfmrStrategy::new();
    let settings = Settings::load();

    let mut position = Position::Flat;

    for stream in listener.incoming() {
        let mut stream = stream.map_err(|e| format!("accept: {e}"))?;
        let mut len_buf = [0u8; 4];
        loop {
            // Read 4-byte LE length prefix.
            let mut got = 0usize;
            while got < 4 {
                let n = stream.read(&mut len_buf[got..])
                    .map_err(|e| format!("read length: {e}"))?;
                if n == 0 {
                    break; // peer closed
                }
                got += n;
            }
            if got < 4 {
                break;
            }
            let frame_len = u32::from_le_bytes(len_buf) as usize;
            if frame_len > MAX_FRAME {
                return Err(format!("frame too large: {frame_len} > {MAX_FRAME}"));
            }

            // Read payload.
            let mut payload = vec![0u8; frame_len];
            let mut got = 0usize;
            while got < frame_len {
                let n = stream.read(&mut payload[got..])
                    .map_err(|e| format!("read payload: {e}"))?;
                if n == 0 {
                    break;
                }
                got += n;
            }
            if got < frame_len {
                break;
            }

            // Decode -> compute -> decide -> execute.
            let row = decode_frame(&payload)?;
            let computed = features.compute(&row)
                .map_err(|e| format!("compute: {e}"))?;
            let state = crate::core::strategy::MarketState {
                features: computed,
                position,
                sequence: 0,
            };
            let decision = strategy.decide(&state);
            position = state.apply(decision);
            execute("UNKNOWN", decision, position);

            // Ack back to peer.
            let _ = stream.write_all(b"OK\n");
        }
    }

    Ok(())
}

/// Stub on non-Unix platforms: nothing to run.
#[cfg(not(unix))]
pub fn run(_socket_path: &str) -> Result<(), String> {
    Err("OFMR socket consumer requires a Unix platform".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_frame_parses_name_value_lines() {
        let payload = b"price:100.5\namount:1.25\nbid_price:100.49\n";
        let row = decode_frame(payload).unwrap();
        assert_eq!(row.get("price"), Some(100.5));
        assert_eq!(row.get("amount"), Some(1.25));
        assert_eq!(row.get("bid_price"), Some(100.49));
    }

    #[test]
    fn decode_frame_missing_on_bad_value() {
        let payload = b"price:not_a_number\n";
        let row = decode_frame(payload).unwrap();
        assert!(row.get("price").is_none());
        assert!(row.get_value("price").unwrap().is_missing());
    }
}