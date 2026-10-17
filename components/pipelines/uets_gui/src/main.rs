use std::path::PathBuf;

#[cfg(unix)]
use uets_gui::publish_instance;
#[cfg(unix)]
use uets_gui::{InstanceSnapshot, Statistics};
use uets_gui::{Monitor, SocketSnapshotSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var_os("UETS_GUI_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sockets/gui.sock"));
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|value| value == "publish") {
        return publish_status(&path, &args[1..]);
    }
    let source = SocketSnapshotSource::bind(&path)?;
    eprintln!("UETS monitor listening on {}", path.display());
    Monitor::new(source).run()?;
    Ok(())
}

#[cfg(unix)]
fn publish_status(
    path: &std::path::Path,
    args: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() != 2 {
        return Err("usage: uets_gui publish <strategy>@<asset> <enabled|disabled|error>".into());
    }
    let Some((strategy_id, asset_id)) = args[0].split_once('@') else {
        return Err("strategy instance must use strategy@asset".into());
    };
    if !["enabled", "disabled", "error"].contains(&args[1].as_str()) {
        return Err("status must be enabled, disabled, or error".into());
    }
    let instance = InstanceSnapshot {
        strategy_id: strategy_id.to_string(),
        asset_id: asset_id.to_string(),
        statistics: Statistics {
            status: Some(args[1].clone()),
            ..Statistics::default()
        },
        equity_history: Vec::new(),
    };
    publish_instance(path, &instance)?;
    Ok(())
}

#[cfg(not(unix))]
fn publish_status(
    _path: &std::path::Path,
    _args: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    Err("status publishing requires Unix-domain sockets".into())
}
