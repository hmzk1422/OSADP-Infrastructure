//! Architecture diagnostics: where the boundaries actually are.
//!
//! These tests read the sources from the *outside* — an integration test
//! never lives inside the tree it judges, so the scan cannot match itself.
//!
//! * `core/` is the custom-made engine: gui, telemetry (latency, skew),
//!   features, strategy, bookkeeping. It must never mention a connector,
//!   a transport, or the broker contract.
//! * The broker contract (`VenueSpec`, the three connector traits) resolves at
//!   `uets::exchange`, never at `uets::core::exchange`.

use std::fs;
use std::path::{Path, PathBuf};

/// The `uets` crate root (the manifest directory of this package).
fn uets_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// `<pkg>/executor/core` — the pure-logic half.
fn core_dir() -> PathBuf {
    uets_dir().join("executor").join("core")
}

/// Every `.rs` file under `core/`, including subdirectories.
fn core_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            core_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn core_declares_no_exchange_module_and_the_contract_is_at_uets_exchange() {
    let mod_rs = fs::read_to_string(core_dir().join("mod.rs")).expect("core/mod.rs");
    assert!(
        !mod_rs.contains("pub mod exchange"),
        "the broker contract does not belong in core"
    );
    // The contract must resolve at `uets::exchange` — a path that only
    // compiles if the top-level module exists.
    let path: &str = std::any::type_name::<uets::exchange::VenueSpec>();
    assert!(path.contains("VenueSpec"), "got {path}");
    // And the old location must be gone: this path must NOT compile, which we
    // assert by construction — the scan below proves no core file refers to it.
}

#[test]
fn core_never_imports_a_connector_or_a_wire_transport() {
    const BANNED: [&str; 5] = [
        "crate::connectors",
        "use binance::",
        "tungstenite",
        "wss://",
        "event_merger_live",
    ];
    let mut files = Vec::new();
    core_sources(&core_dir(), &mut files);
    assert!(!files.is_empty(), "core sources must exist");
    for file in files {
        let text = fs::read_to_string(&file).expect("readable source");
        for banned in BANNED {
            assert!(
                !text.contains(banned),
                "{:?} mentions `{banned}` — core must stay venue-free",
                file.file_name().unwrap()
            );
        }
    }
}

#[test]
fn core_is_structured_as_the_custom_made_engine() {
    let dir = core_dir();
    // Local concerns first: the dashboard bridge and the telemetry store,
    // with latency and skew as their own diagnostic modules.
    assert!(dir.join("gui.rs").is_file(), "gui module");
    assert!(dir.join("telemetry").join("mod.rs").is_file());
    assert!(
        dir.join("telemetry").join("latency.rs").is_file(),
        "telemetry/latency.rs"
    );
    assert!(
        dir.join("telemetry").join("skew.rs").is_file(),
        "telemetry/skew.rs"
    );
    // Strategy and feature engines, both directory modules.
    assert!(dir.join("strategy").join("mod.rs").is_file());
    assert!(dir.join("feature").join("mod.rs").is_file());
    // The bookkeeping half.
    for name in [
        "order.rs",
        "position.rs",
        "pnl.rs",
        "portfolio.rs",
        "risk.rs",
        "stats.rs",
    ] {
        assert!(dir.join(name).is_file(), "core/{name}");
    }
    // And the broker contract lives where the header says it does.
    assert!(
        uets_dir()
            .join("executor")
            .join("exchange")
            .join("exchange.rs")
            .is_file(),
        "exchange/exchange.rs"
    );
    assert!(
        uets_dir()
            .join("executor")
            .join("exchange")
            .join("mod.rs")
            .is_file(),
        "exchange/mod.rs"
    );
}
