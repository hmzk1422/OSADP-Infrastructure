#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DB_ROOT="${UETS_LIVE_DB:-${ROOT}/market_data/live_db}"
if (( $# > 1 )); then
    echo "Usage: cloud.sh [session-id]" >&2
    exit 2
fi
SESSION_ID="${1:-}"
if [[ -z "${SESSION_ID}" && -f "${DB_ROOT}/current_session" ]]; then
    SESSION_ID="$(<"${DB_ROOT}/current_session")"
fi
if [[ -z "${SESSION_ID}" ]]; then
    echo "Pass a session ID or start a UETS session first." >&2
    exit 2
fi
DB_PATH="${DB_ROOT}/telemetry/${SESSION_ID}"
THRESHOLD_PERCENT="${DISK_THRESHOLD_PERCENT:-90}"
ARCHIVE_TMP_DIR="${ARCHIVE_TMP_DIR:-${TMPDIR:-/tmp}}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
AWS_ARCHIVE_URI="${AWS_ARCHIVE_URI:-}"
UETS_BIN="${UETS_BIN:-${ROOT}/bin/uets}"
UETS_HOME="${UETS_HOME:-${ROOT}}"
UETS_TMUX_SESSION="${UETS_TMUX_SESSION:-uets}"
UETS_TMUX_PANE="${UETS_TMUX_PANE:-${UETS_TMUX_SESSION}:0.0}"
UETS_STOP_TIMEOUT_SECONDS="${UETS_STOP_TIMEOUT_SECONDS:-60}"

if [[ ! "${SESSION_ID}" =~ ^[A-Za-z0-9._-]+$ ]] ||
   [[ "${SESSION_ID}" == "." || "${SESSION_ID}" == ".." ]]; then
    echo "Invalid session ID." >&2
    exit 2
fi
if [[ ! "${THRESHOLD_PERCENT}" =~ ^[0-9]+$ ]] ||
   (( THRESHOLD_PERCENT < 1 || THRESHOLD_PERCENT > 100 )); then
    echo "DISK_THRESHOLD_PERCENT must be an integer from 1 to 100." >&2
    exit 2
fi
if [[ ! "${UETS_STOP_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] ||
   (( UETS_STOP_TIMEOUT_SECONDS < 1 )); then
    echo "UETS_STOP_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 2
fi
if [[ ! -d "${DB_PATH}" ]]; then
    echo "Telemetry RocksDB session not found: ${DB_PATH}" >&2
    exit 1
fi
if [[ -L "${DB_PATH}" || -L "${DB_ROOT}/telemetry" ]]; then
    echo "Refusing to archive or remove a symlinked telemetry database path." >&2
    exit 1
fi
if [[ ! -x "${UETS_BIN}" ]]; then
    echo "UETS executable is missing or not executable: ${UETS_BIN}" >&2
    exit 1
fi
if [[ ! "${AWS_ARCHIVE_URI}" =~ ^s3://[^/]+(/.*)?$ ]]; then
    echo "Set AWS_ARCHIVE_URI to an S3 bucket or prefix, for example s3://my-bucket/uets." >&2
    exit 2
fi

for command_name in aws zstd tar sha256sum flock df awk tmux realpath; do
    command -v "${command_name}" >/dev/null || {
        echo "Missing dependency: ${command_name}" >&2
        exit 1
    }
done

mkdir -p "${ARCHIVE_TMP_DIR}"
exec 9>"${ARCHIVE_TMP_DIR}/uets-cloud-archive.lock"
flock -n 9 || {
    echo "Another UETS archive job is running." >&2
    exit 1
}

used_percent="$(df -P "${DB_PATH}" | awk 'END { gsub(/%/, "", $5); print $5 }')"
if [[ ! "${used_percent}" =~ ^[0-9]+$ ]]; then
    echo "Could not determine disk utilization for ${DB_PATH}." >&2
    exit 1
fi
if (( used_percent < THRESHOLD_PERCENT )); then
    echo "Disk usage: ${used_percent}%. Below ${THRESHOLD_PERCENT}% threshold."
    exit 0
fi

if ! tmux has-session -t "${UETS_TMUX_SESSION}" 2>/dev/null; then
    echo "UETS tmux session '${UETS_TMUX_SESSION}' is not running; refusing local deletion." >&2
    exit 1
fi
pane_state="$(tmux display-message -p -t "${UETS_TMUX_PANE}" '#{pane_dead}')"
if [[ "${pane_state}" != "0" && "${pane_state}" != "1" ]]; then
    echo "Could not inspect UETS tmux pane '${UETS_TMUX_PANE}'." >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive_name="telemetry-${SESSION_ID}-${timestamp}.tar.zst"
tmp_dir="$(mktemp -d "${ARCHIVE_TMP_DIR%/}/uets-archive.XXXXXX")"
uets_stopped=false
session_marker="${DB_ROOT}/current_session"
active_marker="${DB_ROOT}/control/${SESSION_ID}.active"
archive_marker="${UETS_HOME}/run/archive.pending"

restore_session_marker() {
    local marker_tmp="${session_marker}.archive.$$"
    printf '%s\n' "${SESSION_ID}" >"${marker_tmp}"
    mv -f -- "${marker_tmp}" "${session_marker}"
}

resume_uets() {
    restore_session_marker
    pane_state="$(tmux display-message -p -t "${UETS_TMUX_PANE}" '#{pane_dead}')"
    if [[ "${pane_state}" == "1" ]]; then
        tmux respawn-pane -k -t "${UETS_TMUX_PANE}"
    elif [[ "${pane_state}" != "0" ]]; then
        echo "Cannot restart UETS: tmux pane state is unknown." >&2
        return 1
    fi
    rm -f -- "${archive_marker}"
    tmux set-window-option -t "${UETS_TMUX_PANE}" remain-on-exit off
    uets_stopped=false
}

finish() {
    local status=$?
    trap - EXIT
    if [[ "${uets_stopped}" == true ]]; then
        echo "Restarting UETS with the same database session after archive attempt." >&2
        if ! resume_uets; then
            echo "CRITICAL: could not restart UETS; start it manually in tmux session '${UETS_TMUX_SESSION}'." >&2
            status=1
        fi
    fi
    if [[ -n "${tmp_dir:-}" && -d "${tmp_dir}" ]]; then
        rm -rf -- "${tmp_dir}"
    fi
    exit "${status}"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

uets_stopped=true
mkdir -p "$(dirname -- "${archive_marker}")"
printf '%s %s\n' "$$" "${SESSION_ID}" >"${archive_marker}"
tmux set-window-option -t "${UETS_TMUX_PANE}" remain-on-exit on
if [[ "${pane_state}" == "0" ]]; then
    echo "Stopping UETS cleanly for a consistent archive."
    tmux send-keys -t "${UETS_TMUX_PANE}" Escape q
    deadline=$((SECONDS + UETS_STOP_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        pane_state="$(tmux display-message -p -t "${UETS_TMUX_PANE}" '#{pane_dead}')"
        if [[ "${pane_state}" == "1" ]]; then
            break
        fi
        sleep 1
    done
    if [[ "${pane_state}" != "1" ]]; then
        echo "UETS did not stop within ${UETS_STOP_TIMEOUT_SECONDS}s; local database retained." >&2
        exit 1
    fi
fi

if [[ -f "${active_marker}" ]]; then
    active_pid="$(<"${active_marker}")"
    if [[ "${active_pid}" =~ ^[0-9]+$ ]] && kill -0 "${active_pid}" 2>/dev/null; then
        echo "UETS process ${active_pid} is still active; refusing to remove its database." >&2
        exit 1
    fi
    rm -f -- "${active_marker}"
fi

backup_dir="${tmp_dir}/rocksdb_backup"
archive_path="${tmp_dir}/${archive_name}"

export UETS_LIVE_DB="${DB_ROOT}"
"${UETS_BIN}" db checkpoint "${SESSION_ID}" "${backup_dir}"
"${UETS_BIN}" db verify-checkpoint "${SESSION_ID}" "${backup_dir}"

cat >"${tmp_dir}/manifest.txt" <<EOF
UETS telemetry RocksDB checkpoint
Session: ${SESSION_ID}
Created UTC: ${timestamp}
Original DB: ${DB_PATH}
Snapshot: RocksDB Checkpoint created while UETS was stopped
Compression: Zstandard
EOF

tar -cf - -C "${tmp_dir}" rocksdb_backup manifest.txt |
    zstd -T0 -6 -o "${archive_path}"
zstd -t "${archive_path}"
zstd -dc "${archive_path}" | tar -tf - >/dev/null
checksum="$(sha256sum "${archive_path}" | awk '{print $1}')"

s3_location="${AWS_ARCHIVE_URI#s3://}"
bucket="${s3_location%%/*}"
if [[ "${s3_location}" == */* ]]; then
    prefix="${s3_location#*/}"
    prefix="${prefix%/}"
else
    prefix=""
fi
object_key="${prefix:+${prefix}/}telemetry/${SESSION_ID}/${archive_name}"
s3_uri="s3://${bucket}/${object_key}"

echo "Uploading ${archive_name} (${checksum}) to ${s3_uri}"
aws s3 cp "${archive_path}" "${s3_uri}" \
    --region "${AWS_REGION}" \
    --storage-class GLACIER_IR \
    --metadata "sha256=${checksum},session-id=${SESSION_ID}"

uploaded_class="$(aws s3api head-object \
    --bucket "${bucket}" \
    --key "${object_key}" \
    --region "${AWS_REGION}" \
    --query 'StorageClass' \
    --output text)"
if [[ "${uploaded_class}" != "GLACIER_IR" ]]; then
    echo "S3 storage-class verification failed: expected GLACIER_IR, got ${uploaded_class}." >&2
    exit 1
fi

downloaded_archive="${tmp_dir}/downloaded-${archive_name}"
aws s3 cp "${s3_uri}" "${downloaded_archive}" --region "${AWS_REGION}" --only-show-errors
downloaded_checksum="$(sha256sum "${downloaded_archive}" | awk '{print $1}')"
if [[ "${downloaded_checksum}" != "${checksum}" ]]; then
    echo "Downloaded S3 object checksum does not match the uploaded archive." >&2
    exit 1
fi
mkdir -p "${tmp_dir}/restore"
zstd -dc "${downloaded_archive}" | tar -xf - -C "${tmp_dir}/restore"
"${UETS_BIN}" db verify-checkpoint \
    "${SESSION_ID}" "${tmp_dir}/restore/rocksdb_backup"

db_root_real="$(realpath -e -- "${DB_ROOT}")"
telemetry_root_real="$(realpath -e -- "${DB_ROOT}/telemetry")"
db_path_real="$(realpath -e -- "${DB_PATH}")"
expected_db_path="${db_root_real}/telemetry/${SESSION_ID}"
if [[ "${telemetry_root_real}" != "${db_root_real}/telemetry" ||
      "${db_path_real}" != "${expected_db_path}" ||
      ! -d "${db_path_real}" || -L "${DB_PATH}" ]]; then
    echo "Telemetry database path changed or escaped its expected location; refusing deletion." >&2
    exit 1
fi

echo "S3 upload and restore verification passed; deleting local telemetry database ${db_path_real}."
retired_db_path="${telemetry_root_real}/.archived-${SESSION_ID}-${timestamp}"
if [[ -e "${retired_db_path}" ]]; then
    echo "Refusing to overwrite existing retired database path ${retired_db_path}." >&2
    exit 1
fi
mv -- "${db_path_real}" "${retired_db_path}"
rm -rf -- "${retired_db_path}"
if [[ -e "${retired_db_path}" || -e "${db_path_real}" ]]; then
    echo "Local telemetry database cleanup did not complete." >&2
    exit 1
fi

resume_uets

printf '%s\n' \
    "UETS archive uploaded and restore-verified." \
    "Storage class: ${uploaded_class}" \
    "Session: ${SESSION_ID}" \
    "S3 location: ${s3_uri}" \
    "SHA-256: ${checksum}" \
    "Local telemetry RocksDB was deleted after verified upload; trade and portfolio databases were retained."
