#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
state_dir="$(mktemp -d)"
trap 'rm -rf -- "${state_dir}"' EXIT
fake_bin="${state_dir}/bin"
mkdir -p "${fake_bin}"

cat >"${fake_bin}/df" <<'STUB'
#!/usr/bin/env bash
if [[ -e "${STATE_DIR}/disk_high" ]]; then
    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/mock 100 95 5 95%% /\n'
else
    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/mock 100 80 20 80%% /\n'
fi
STUB

cat >"${fake_bin}/flock" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB

cat >"${fake_bin}/tmux" <<'STUB'
#!/usr/bin/env bash
TMUX_STATE_FILE="${STATE_DIR}/tmux.state"
case "$1" in
    has-session) exit 0 ;;
    display-message)
        if [[ -f "${TMUX_STATE_FILE}" ]]; then cat "${TMUX_STATE_FILE}"; else echo 0; fi
        ;;
    send-keys) echo 1 >"${TMUX_STATE_FILE}" ;;
    respawn-pane) echo 0 >"${TMUX_STATE_FILE}" ;;
    set-window-option) ;;
    *) echo "unexpected tmux command: $*" >&2; exit 90 ;;
esac
STUB

cat >"${fake_bin}/zstd" <<'STUB'
#!/usr/bin/env bash
case "$1" in
    -T0) shift 2; cat >"$2" ;;
    -t) [[ -s "$2" ]] ;;
    -dc) cat "$2" ;;
    *) echo "unexpected zstd command: $*" >&2; exit 91 ;;
esac
STUB

cat >"${fake_bin}/tar" <<'STUB'
#!/usr/bin/env bash
case "$1" in
    -cf) cat "$4/rocksdb_backup/checkpoint" "$4/manifest.txt" ;;
    -tf) cat >/dev/null ;;
    -xf) while [[ "$1" != "-C" ]]; do shift; done; destination="$2"; mkdir -p "${destination}/rocksdb_backup"; printf 'checkpoint data\n' >"${destination}/rocksdb_backup/checkpoint" ;;
    *) echo "unexpected tar command: $*" >&2; exit 92 ;;
esac
STUB

cat >"${fake_bin}/sha256sum" <<'STUB'
#!/usr/bin/env bash
printf 'deadbeef  %s\n' "$1"
STUB

cat >"${fake_bin}/aws" <<'STUB'
#!/usr/bin/env bash
if [[ "$1 $2" == "s3 cp" ]]; then
    if [[ "$3" == s3://* ]]; then cp "${AWS_OBJECT}" "$4"; else cp "$3" "${AWS_OBJECT}"; fi
elif [[ "$1 $2" == "s3api head-object" ]]; then
    printf 'GLACIER_IR\n'
else
    echo "unexpected aws command: $*" >&2; exit 94
fi
STUB

cat >"${state_dir}/uets" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "db checkpoint" ]]; then
    mkdir -p "$4"
    printf 'checkpoint data\n' >"$4/checkpoint"
    if [[ "$3" == "2026-11-12" ]]; then rm -f -- "${STATE_DIR}/disk_high"; fi
elif [[ "$1 $2" == "db verify-checkpoint" ]]; then
    [[ -s "$4/checkpoint" ]]
else
    echo "unexpected UETS command: $*" >&2; exit 95
fi
STUB

chmod +x "${fake_bin}/df" "${fake_bin}/flock" "${fake_bin}/tmux" "${fake_bin}/zstd" \
    "${fake_bin}/tar" "${fake_bin}/sha256sum" "${fake_bin}/aws" "${state_dir}/uets"

make_fixture() {
    local name="$1"
    local root="${state_dir}/${name}/live_db"
    for day in 2026-11-12 2026-11-13 2026-11-14; do
        mkdir -p "${root}/${day}/telemetry/session-1" \
                 "${root}/${day}/portfolio/session-1" \
                 "${root}/${day}/trades/session-1"
        printf 'telemetry-%s\n' "${day}" >"${root}/${day}/telemetry/session-1/data"
        printf 'portfolio-%s\n' "${day}" >"${root}/${day}/portfolio/session-1/data"
        printf 'trades-%s\n' "${day}" >"${root}/${day}/trades/session-1/data"
    done
    printf 'session-1\n' >"${root}/current_session"
    printf '%s\n' "${root}"
}

run_archive() {
    local root="$1"
    local state="$2"
    echo 0 >"${state}/tmux.state"
    touch "${state}/disk_high"
    PATH="${fake_bin}:${PATH}" \
        STATE_DIR="${state}" \
        UETS_LIVE_DB="${root}" \
        UETS_HOME="${state}/home" \
        UETS_BIN="${state_dir}/uets" \
        UETS_TMUX_SESSION=uets \
        UETS_TMUX_PANE=uets:0.0 \
        AWS_ARCHIVE_URI=s3://test-bucket/uets \
        AWS_OBJECT="${state}/uploaded-archive" \
        ARCHIVE_TMP_DIR="${state}/tmp" \
        DISK_THRESHOLD_PERCENT=90 \
        bash "${ROOT}/cloud.sh"
}

mkdir -p "${state_dir}/tmp"
root="$(make_fixture success)"
state="${state_dir}/success"

if ! run_archive "${root}" "${state}"; then
    echo "FIFO daily-partition archive should succeed" >&2
    exit 1
fi

if [[ -e "${root}/2026-11-12" ]]; then
    echo "Oldest daily partition 2026-11-12 was not deleted after verified upload" >&2
    exit 1
fi

for day in 2026-11-13 2026-11-14; do
    if [[ ! -d "${root}/${day}/telemetry" ]] ||
       [[ ! -d "${root}/${day}/portfolio" ]] ||
       [[ ! -d "${root}/${day}/trades" ]]; then
        echo "Daily partition ${day} is missing a subdirectory after archive" >&2
        exit 1
    fi
done

if [[ "$(<"${root}/current_session")" != "session-1" ]]; then
    echo "current_session was not preserved for TUI restart" >&2
    exit 1
fi

echo "FIFO daily-partition archive: oldest day exported and deleted locally; newer days intact; TUI session preserved."