#[cfg(unix)]
use std::collections::BTreeMap;
use std::io;
use std::path::Path;
#[cfg(unix)]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use crate::gui::InstanceSnapshot;
#[cfg(unix)]
use crate::gui::aggregate_instances;
use crate::gui::{DashboardSnapshot, SnapshotSource};

#[cfg(unix)]
pub fn publish_instance(path: impl AsRef<Path>, instance: &InstanceSnapshot) -> io::Result<()> {
    use std::os::unix::net::UnixDatagram;

    let socket = UnixDatagram::unbound()?;
    socket.connect(path)?;
    let payload = serde_json::to_vec(instance)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    socket.send(&payload)?;
    Ok(())
}

#[cfg(not(unix))]
pub fn publish_instance(
    _path: impl AsRef<Path>,
    _instance: &crate::gui::InstanceSnapshot,
) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "UETS instance publishing requires Unix-domain sockets",
    ))
}

#[cfg(unix)]
const INSTANCE_STALE_AFTER: Duration = Duration::from_secs(3);

#[cfg(unix)]
pub struct SocketSnapshotSource {
    socket: std::os::unix::net::UnixDatagram,
    instances: BTreeMap<String, (InstanceSnapshot, Instant)>,
    aggregate_equity_history: Vec<f64>,
    aggregate_starting_equity: f64,
}

#[cfg(unix)]
impl SocketSnapshotSource {
    pub fn bind(path: impl AsRef<Path>) -> io::Result<Self> {
        use std::os::unix::fs::FileTypeExt;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if path.exists() {
            let socket_type = std::fs::symlink_metadata(path)?.file_type().is_socket();
            if !socket_type {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("refusing to replace non-socket GUI path {}", path.display()),
                ));
            }
            let probe = std::os::unix::net::UnixDatagram::unbound()?;
            if probe.connect(path).is_ok() {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!("GUI socket is already active at {}", path.display()),
                ));
            }
            std::fs::remove_file(path)?;
        }
        let socket = std::os::unix::net::UnixDatagram::bind(path)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            instances: BTreeMap::new(),
            aggregate_equity_history: Vec::new(),
            aggregate_starting_equity: 0.0,
        })
    }

    pub fn forget_instance(&mut self, identity: &str) {
        self.instances.remove(identity);
    }
}

#[cfg(not(unix))]
impl SnapshotSource for SocketSnapshotSource {
    fn snapshot(&mut self) -> DashboardSnapshot {
        DashboardSnapshot::default()
    }
}

#[cfg(unix)]
impl SnapshotSource for SocketSnapshotSource {
    fn snapshot(&mut self) -> DashboardSnapshot {
        let mut buffer = [0u8; 64 * 1024];
        loop {
            match self.socket.recv(&mut buffer) {
                Ok(size) => {
                    if let Ok(instance) =
                        serde_json::from_slice::<InstanceSnapshot>(&buffer[..size])
                    {
                        let identity = instance.identity();
                        if !instance.strategy_id.is_empty() && !instance.asset_id.is_empty() {
                            self.instances.insert(identity, (instance, Instant::now()));
                        }
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }

        let now = Instant::now();
        let instances: Vec<_> = self
            .instances
            .values_mut()
            .map(|(instance, last_seen)| {
                if instance.statistics.status.as_deref() != Some("disabled")
                    && now.duration_since(*last_seen) > INSTANCE_STALE_AFTER
                {
                    instance.statistics.status = Some("error".to_string());
                    instance.statistics.last_error =
                        Some("No worker telemetry received for more than 3 seconds".to_string());
                }
                instance.clone()
            })
            .collect();
        let aggregate = aggregate_instances(
            &instances,
            &mut self.aggregate_equity_history,
            &mut self.aggregate_starting_equity,
        );
        DashboardSnapshot {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(|elapsed| format!("{} ms UTC", elapsed.as_millis())),
            instances,
            aggregate,
            aggregate_equity_history: self.aggregate_equity_history.clone(),
        }
    }
}

#[cfg(not(unix))]
pub struct SocketSnapshotSource;

#[cfg(not(unix))]
impl SocketSnapshotSource {
    pub fn bind(_path: impl AsRef<Path>) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the live UETS monitor requires Unix-domain sockets on a Unix host",
        ))
    }
}
