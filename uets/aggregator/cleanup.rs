//! Stale-socket cleanup with ownership checks.
//!
//! Never blindly `unlink`s at startup. A socket path is only claimed when it
//! is safe: if a live peer answers on it, the bind fails and the second
//! instance exits instead of stealing the socket. Lock file holds the owner's
//! pid (`<path>.lock`); a lock whose pid is dead may be reclaimed.

use std::path::Path;

/// Outcome of trying to claim a socket path.
#[derive(Debug, PartialEq, Eq)]
pub enum Claim {
    /// Safe to bind: no live owner.
    Free,
    /// Another live process owns this path; do not unlink.
    OwnedByLive(u32),
}

/// Check whether `sock` may be bound: a live peer or a live lock-holder
/// means hands off. Stale paths (dead pid, no listener) are unlinked here so
/// the caller can bind immediately after a `Free`.
#[cfg(unix)]
pub fn claim(sock: &Path) -> std::io::Result<Claim> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    // 1. A listener answers -> someone owns it. Do not touch.
    if UnixStream::connect(sock).is_ok() {
        return Ok(Claim::OwnedByLive(0));
    }
    // 2. Lock file with a live pid -> owned. Dead pid -> reclaim.
    let lock = sock.with_extension("lock");
    if lock.exists() {
        if let Ok(mut f) = std::fs::File::open(&lock) {
            let mut text = String::new();
            if f.read_to_string(&mut text).is_ok() {
                if let Ok(pid) = text.trim().parse::<u32>() {
                    if pid_alive(pid) {
                        return Ok(Claim::OwnedByLive(pid));
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&lock);
    }
    // 3. No live owner: remove the stale node (if any) and record our pid.
    if sock.exists() {
        std::fs::remove_file(sock)?;
    }
    if let Ok(mut f) = std::fs::File::create(&lock) {
        let _ = writeln!(f, "{}", std::process::id());
    }
    Ok(Claim::Free)
}

/// Fallback on non-unix: nothing to clean.
#[cfg(not(unix))]
pub fn claim(_sock: &Path) -> std::io::Result<Claim> {
    Ok(Claim::Free)
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    // Signal 0 performs error checking without delivering a signal.
    unsafe { libc_kill(pid, 0) == 0 }
}

#[cfg(unix)]
fn libc_kill(pid: u32, sig: i32) -> i32 {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    unsafe { kill(pid as i32, sig) }
}
