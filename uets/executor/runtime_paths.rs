use std::path::{Path, PathBuf};

fn has_runtime_config(root: &Path) -> bool {
    root.join("uets")
        .join("config")
        .join("execution.yaml")
        .is_file()
        || root.join("config").join("execution.yaml").is_file()
}

fn normalize_runtime_root(root: &Path) -> PathBuf {
    if root.join("config").join("execution.yaml").is_file() {
        if let Some(parent) = root.parent() {
            if parent
                .join("uets")
                .join("config")
                .join("execution.yaml")
                .is_file()
            {
                return parent.to_path_buf();
            }
        }
    }
    root.to_path_buf()
}

fn resolve_workspace_root(
    explicit_home: Option<&Path>,
    current_dir: Option<&Path>,
    executable: Option<&Path>,
    source_workspace: &Path,
) -> PathBuf {
    if let Some(home) = explicit_home {
        return if home.is_absolute() {
            home.to_path_buf()
        } else if let Some(current_dir) = current_dir {
            current_dir.join(home)
        } else {
            home.to_path_buf()
        };
    }

    if let Some(current_dir) = current_dir {
        if has_runtime_config(current_dir) {
            return normalize_runtime_root(current_dir);
        }
    }

    if let Some(executable) = executable {
        if let Some(parent) = executable.parent() {
            if has_runtime_config(parent) {
                return normalize_runtime_root(parent);
            }
            if let Some(root) = parent.parent() {
                if has_runtime_config(root) {
                    return normalize_runtime_root(root);
                }
            }
        }
    }

    source_workspace.to_path_buf()
}

/// Resolve the runtime data root without depending on the compile-time checkout.
///
/// `UETS_HOME` takes precedence; otherwise, a runtime config in the current
/// directory or next to the executable is preferred before the source tree.
pub fn workspace_root() -> PathBuf {
    let explicit_home = std::env::var_os("UETS_HOME").map(PathBuf::from);
    let current_dir = std::env::current_dir().ok();
    let executable = std::env::current_exe().ok();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_workspace = manifest_dir.parent().unwrap_or(&manifest_dir);

    resolve_workspace_root(
        explicit_home.as_deref(),
        current_dir.as_deref(),
        executable.as_deref(),
        source_workspace,
    )
}

/// Resolve the UETS package directory in either a checkout or a deployment.
pub fn package_dir() -> PathBuf {
    let root = workspace_root();
    let workspace_package = root.join("uets");
    if workspace_package.is_dir() {
        workspace_package
    } else {
        root
    }
}

pub fn config_dir() -> PathBuf {
    package_dir().join("config")
}

/// `UETS_EXECUTION_CONFIG` can point at an alternate live execution YAML.
pub fn execution_config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("UETS_EXECUTION_CONFIG").map(PathBuf::from) {
        return if path.is_absolute() {
            path
        } else {
            workspace_root().join(path)
        };
    }
    config_dir().join("execution.yaml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_home_precedes_runtime_and_source_paths() {
        let temp = tempfile::tempdir().unwrap();
        let explicit = temp.path().join("deployment");
        let current = temp.path().join("checkout");
        std::fs::create_dir_all(current.join("uets/config")).unwrap();
        std::fs::write(
            current.join("uets/config/execution.yaml"),
            "mode: live-test\n",
        )
        .unwrap();

        assert_eq!(
            resolve_workspace_root(Some(&explicit), Some(&current), None, Path::new("source"),),
            explicit
        );
    }

    #[test]
    fn runtime_config_is_found_next_to_the_binary() {
        let temp = tempfile::tempdir().unwrap();
        let deployment = temp.path().join("deployment");
        let bin = deployment.join("bin");
        std::fs::create_dir_all(bin.join("../uets/config")).unwrap();
        std::fs::write(
            deployment.join("uets/config/execution.yaml"),
            "mode: live-test\n",
        )
        .unwrap();

        assert_eq!(
            resolve_workspace_root(None, None, Some(&bin.join("uets")), Path::new("source"),),
            deployment
        );
    }

    #[test]
    fn package_directory_uses_the_workspace_root_when_running_cargo_tests() {
        let temp = tempfile::tempdir().unwrap();
        let workspace = temp.path();
        let package = workspace.join("uets");
        std::fs::create_dir_all(package.join("config")).unwrap();
        std::fs::write(package.join("config/execution.yaml"), "mode: live-test\n").unwrap();

        assert_eq!(
            resolve_workspace_root(None, Some(&package), None, Path::new("source")),
            workspace
        );
    }
}
