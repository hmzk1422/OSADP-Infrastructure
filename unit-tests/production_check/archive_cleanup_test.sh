#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
state_dir="$(mktemp -d)"
trap 'rm -rf -- "${state_dir}"' EXIT
fake_bin="${state_dir}/bin"
mkdir -p "${fake_bin}"

cat >"${fake_bin}/df" <<'STUB'
#!/usr/bin/env bash
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/mock 100 95 5 95%% /\n'
STUB

cat >"${fake_bin}/flock" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB

cat >"${fake_bin}/tmux" <<'STUB'
#!/usr/bin/env bash
case "$1" in
    has-session|set-window-option)
        exit 0
        ;;
    display-message)
        cat "${TMUX_STATE}"
        ;;
    send-keys)
        printf '1\n' >"${TMUX_STATE}"
        ;;
    respawn-pane)
        printf '0\n' >"${TMUX_STATE}"
        ;;
    *)
        echo "unexpected tmux command: $*" >&2
        exit 90
        ;;
esac
STUB

cat >"${fake_bin}/zstd" <<'STUB'
#!/usr/bin/env bash
case "$1" in
    -T0)
        shift 2
        cat >"$2"
        ;;
    -t)
        [[ -s "$2" ]]
        ;;
    -dc)
        cat "$2"
        ;;
    *)
        echo "unexpected zstd command: $*" >&2
        exit 91
        ;;
esac
STUB

cat >"${fake_bin}/tar" <<'STUB'
#!/usr/bin/env bash
case "$1" in
    -cf)
        cat "$4/rocksdb_backup/checkpoint" "$4/manifest.txt"
        ;;
    -tf)
        cat >/dev/null
        ;;
    -xf)
        while [[ "$1" != "-C" ]]; do shift; done
        destination="$2"
        cat >/dev/null
        mkdir -p "${destination}/rocksdb_backup"
        printf 'checkpoint data\n' >"${destination}/rocksdb_backup/checkpoint"
        ;;
    *)
        echo "unexpected tar command: $*" >&2
        exit 92
        ;;
esac
STUB

cat >"${fake_bin}/aws" <<'STUB'
#!/usr/bin/env bash
if [[ "${AWS_FAIL_UPLOAD:-0}" == "1" && "$1 $2" == "s3 cp" &&
      "$3" != s3://* ]]; then
    echo "simulated S3 upload failure" >&2
    exit 93
fi
if [[ "$1 $2" == "s3 cp" ]]; then
    if [[ "$3" == s3://* ]]; then
        cp "${AWS_OBJECT}" "$4"
    else
        cp "$3" "${AWS_OBJECT}"
    fi
elif [[ "$1 $2" == "s3api head-object" ]]; then
    printf 'GLACIER_IR\n'
else
    echo "unexpected aws command: $*" >&2
    exit 94
fi
STUB

cat >"${state_dir}/uets" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "db checkpoint" ]]; then
    mkdir -p "$4"
    printf 'checkpoint data\n' >"$4/checkpoint"
elif [[ "$1 $2" == "db verify-checkpoint" ]]; then
    if [[ "${UETS_FAIL_RESTORE_VERIFY:-0}" == "1" && "$4" == */restore/rocksdb_backup ]]; then
        echo "simulated restored database verification failure" >&2
        exit 96
    fi
    [[ -s "$4/checkpoint" ]]
else
    echo "unexpected UETS command: $*" >&2
    exit 95
fi
STUB

chmod +x "${fake_bin}/df" "${fake_bin}/flock" "${fake_bin}/tmux" "${fake_bin}/zstd" \
    "${fake_bin}/tar" "${fake_bin}/aws" "${state_dir}/uets"

make_fixture() {
    local name="$1"
    local root="${state_dir}/${name}/live_db"
    mkdir -p "${root}/telemetry/session-1" "${root}/portfolio/session-1" \
        "${root}/trades/session-1" "${root}/control"
    printf 'session-1\n' >"${root}/current_session"
    printf 'database bytes\n' >"${root}/telemetry/session-1/data"
    printf 'account state\n' >"${root}/portfolio/session-1/data"
    printf 'trade history\n' >"${root}/trades/session-1/data"
    printf '0\n' >"${state_dir}/${name}/tmux.dead"
    printf '%s\n' "${root}"
}

run_archive() {
    local root="$1"
    local state="$2"
    PATH="${fake_bin}:${PATH}" \
        UETS_LIVE_DB="${root}" \
        UETS_HOME="${state}/home" \
        UETS_BIN="${state_dir}/uets" \
        UETS_TMUX_SESSION=uets \
        UETS_TMUX_PANE=uets:0.0 \
        AWS_ARCHIVE_URI=s3://test-bucket/uets \
        AWS_OBJECT="${state_dir}/uploaded-archive" \
        ARCHIVE_TMP_DIR="${state_dir}/tmp" \
        TMUX_STATE="${state}/tmux.dead" \
        DISK_THRESHOLD_PERCENT=90 \
        bash "${ROOT}/cloud.sh"
}

mkdir -p "${state_dir}/tmp"
success_root="$(make_fixture success)"
success_state="${state_dir}/success"
if ! run_archive "${success_root}" "${success_state}"; then
    echo "archive should succeed with a verified upload" >&2
    exit 1
fi
[[ ! -e "${success_root}/telemetry/session-1" ]] || {
    echo "verified local telemetry database was not deleted" >&2
    exit 1
}
[[ -f "${success_root}/portfolio/session-1/data" &&
   -f "${success_root}/trades/session-1/data" ]] || {
    echo "portfolio or trade database was incorrectly deleted" >&2
    exit 1
}
[[ "$(<"${success_root}/current_session")" == "session-1" ]] || {
    echo "session marker was not preserved for restart recovery" >&2
    exit 1
}
[[ "$(<"${success_state}/tmux.dead")" == "0" ]] || {
    echo "UETS was not restarted after the archive" >&2
    exit 1
}

failure_root="$(make_fixture failure)"
failure_state="${state_dir}/failure"
if AWS_FAIL_UPLOAD=1 run_archive "${failure_root}" "${failure_state}"; then
    echo "archive unexpectedly succeeded despite the simulated S3 failure" >&2
    exit 1
fi
[[ -d "${failure_root}/telemetry/session-1" ]] || {
    echo "local telemetry database was deleted after failed upload" >&2
    exit 1
}
[[ "$(<"${failure_state}/tmux.dead")" == "0" ]] || {
    echo "UETS was not restarted after failed upload" >&2
    exit 1
}

restore_failure_root="$(make_fixture restore-failure)"
restore_failure_state="${state_dir}/restore-failure"
if UETS_FAIL_RESTORE_VERIFY=1 run_archive "${restore_failure_root}" "${restore_failure_state}"; then
    echo "archive unexpectedly succeeded despite failed restore verification" >&2
    exit 1
fi
[[ -d "${restore_failure_root}/telemetry/session-1" ]] || {
    echo "local telemetry database was deleted after failed restore verification" >&2
    exit 1
}
[[ "$(<"${restore_failure_state}/tmux.dead")" == "0" ]] || {
    echo "UETS was not restarted after failed restore verification" >&2
    exit 1
}

echo "Verified archive deletes local telemetry only after restore verification and resumes UETS; failures retain local DB."
