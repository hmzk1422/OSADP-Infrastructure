//! Fixed framing for Unix-domain datagrams: `u32` length prefix (little-endian)
//! + payload, payload cap [`MAX_FRAME`].
//!
//! Message boundaries come from the framing, not from the socket type, so the
//! code does not depend on `SOCK_SEQPACKET` semantics (payload limits and
//! availability differ across platforms -- see the user's correction).
//! Payloads are raw bytes (`Arc<[u8]>` at the fan-out point: one allocation
//! per frame, shared by every subscriber, zero copies).

/// Largest single frame accepted, in bytes.
pub const MAX_FRAME: usize = 1 << 20;

/// Encode a payload into a length-prefixed frame. `None` when over [`MAX_FRAME`].
pub fn encode(payload: &[u8]) -> Option<Vec<u8>> {
    if payload.len() > MAX_FRAME {
        return None;
    }
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
    Some(buf)
}

/// Decode one frame from the front of `buf`, consuming it. Returns `None`
/// when fewer than a full frame is buffered.
pub fn decode(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    if buf.len() < 4 {
        return None;
    }
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if len > MAX_FRAME || buf.len() < 4 + len {
        return None;
    }
    let payload = buf[4..4 + len].to_vec();
    buf.drain(..4 + len);
    Some(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut buf = encode(b"hello").expect("fits");
        buf.extend_from_slice(&encode(b"world").expect("fits"));
        assert_eq!(decode(&mut buf).expect("frame 1"), b"hello");
        assert_eq!(decode(&mut buf).expect("frame 2"), b"world");
        assert!(decode(&mut buf).is_none());
    }

    #[test]
    fn oversize_rejected() {
        assert!(encode(&vec![0u8; MAX_FRAME + 1]).is_none());
    }
}
