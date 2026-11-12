#!/usr/bin/env bash
set -euo pipefail

# UETS daily-partition Glacier archiver.
#
# RocksDB data is stored under $UETS_LIVE_DB as daily UTC partitions:
#   ${DB_ROOT}/YYYY-MM-DD/{telemetry,portfolio,trades}
#
# The TUI still displays per-session via current_session, but the on-disk
# RocksDB layout is partitioned by day. When disk usage exceeds
# DISK_THRESHOLD_PERCENT, cloud.sh acts as a FIFO queue: the oldest daily
# partition is checkpointed, uploaded to S3 Glacier Instant Retrieval,
# download-verified, restore-verified, then deleted from local disk. The loop
# repeats until disk usage is back below threshold. Only completed (non-current)
# days are ever archived, so the active trading day and the TUI are never
# reset or blocked by a swap.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DB_ROOT="${UETS_LIVE_DB:-${ROOT}/market_data/live_db}"
if (( $# > 1 )); then
    echo "Usage: cloud.sh [YYYY-MM-DD]" >&2
    exit 2
fi
TARGET_DATE="${1:-}"
THRESHOLD_PERCENT="${DISK_THRESHOLD_PERCENT:-90}"
ARCHIVE_TMP_DIR="${ARCHIVE_TMP_DIR:-${TMPDIR:-/tmp}}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
AWS_ARCHIVE_URI="${AWS_ARCHIVE_URI:-}"
UETS_BIN="${UETS_BIN:-${ROOT}/bin/uets}"
UETS_HOME="${UETS_HOME:-${ROOT}}"
UETS_TMUX_SESSION="${UETS_TMUX_SESSION:-uets}"
UETS_TMUX_PANE="${UETS_TMUX_PANE:-${UETS_TMUX_SESSION}:0.0}"
UETS_STOP_TIMEOUT_SECONDS="${UETS_STOP_TIMEOUT_SECONDS:-60}"

if [[ -n "${TARGET_DATE}" ]] && [[ ! "${TARGET_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Invalid date. Use YYYY-MM-DD." >&2
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
if [[ ! -d "${DB_ROOT}" ]]; then
    echo "Live DB root not found: ${DB_ROOT}" >&2
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

for command_name in aws zstd tar sha256sum flock df awk tmux realpath sort head; do
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

list_daily_partitions() {
    local entry name
    for entry in "${DB_ROOT}"/*/; do
        [[ -d "${entry}" ]] || continue
        entry="${entry%/}"
        name="$(basename -- "${entry}")"
        if [[ "${name}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] &&
           { [[ -d "${entry}/telemetry" ]] || [[ -d "${entry}/portfolio" ]] || [[ -d "${entry}/trades" ]]; }; then
            printf '%s\n' "${name}"
        fi
    done | sort
}

stop_uets() {
    local pane_state deadline
    tmux set-window-option -t "${UETS_TMUX_PANE}" remain-on-exit on >/dev/null 2>&1 || true
    pane_state="$(tmux display-message -p -t "${UETS_TMUX_PANE}" '#{pane_dead}' 2>/dev/null || echo 1)"
    if [[ "${pane_state}" == "0" ]]; then
        echo "Stopping UETS cleanly for a consistent archive."
        tmux send-keys -t "${UETS_TMUX_PANE}" Escape q
        deadline=$((SECONDS + UETS_STOP_TIMEOUT_SECONDS))
        while (( SECONDS < deadline )); do
            pane_state="$(tmux display-message -p -t "${UETS_TMUX_PANE}" '#{pane_dead}')"
            if [[ "${pane_state}" == "1" ]]; then
                return 0
            fi
            sleep 1
        done
        echo "UETS did not stop within ${UETS_STOP_TIMEOUT_SECONDS}s; local database retained." >&2
        return 1
    fi
    return 0
}

resume_uets() {
    local pane_state
    pane_state="$(tmux display-message -p -t "${UETS_TMUX_PANE}" '#{pane_dead}' 2>/dev/null || echo 1)"
    if [[ "${pane_state}" == "1" ]]; then
        tmux respawn-pane -k -t "${UETS_TMUX_PANE}"
    elif [[ "${pane_state}" != "0" ]]; then
        echo "Cannot restart UETS: tmux pane state is unknown." >&2
        return 1
    fi
    tmux set-window-option -t "${UETS_TMUX_PANE}" remain-on-exit off >/dev/null 2>&1 || true
    return 0
}

archive_day() {
    local day="$1"
    local day_dir="${DB_ROOT}/${day}"
    local tmp_dir timestamp archive_name checksum
    local bucket prefix object_key s3_uri uploaded_class
    local downloaded_archive downloaded_checksum

    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    archive_name="telemetry-${day}-${timestamp}.tar.zst"
    tmp_dir="$(mktemp -d "${ARCHIVE_TMP_DIR%/}/uets-archive.XXXXXX")"

    export UETS_LIVE_DB="${DB_ROOT}"
    "${UETS_BIN}" db checkpoint "${day}" "${tmp_dir}/rocksdb_backup"
    "${UETS_BIN}" db verify-checkpoint "${day}" "${tmp_dir}/rocksdb_backup"

    cat >"${tmp_dir}/manifest.txt" <<EOF
UETS telemetry RocksDB daily partition
Date: ${day}
Created UTC: ${timestamp}
Original DB: ${day_dir}
Snapshot: RocksDB MVCC daily checkpoint created while UETS was stopped
Compression: Zstandard
EOF

    tar -cf - -C "${tmp_dir}" rocksdb_backup manifest.txt |
        zstd -T0 -6 -o "${tmp_dir}/${archive_name}"
    zstd -t "${tmp_dir}/${archive_name}"
    zstd -dc "${tmp_dir}/${archive_name}" | tar -tf - >/dev/null
    checksum="$(sha256sum "${tmp_dir}/${archive_name}" | awk '{print $1}')"

    s3_location="${AWS_ARCHIVE_URI#s3://}"
    bucket="${s3_location%%/*}"
    if [[ "${s3_location}" == */* ]]; then
        prefix="${s3_location#*/}"
        prefix="${prefix%/}"
    else
        prefix=""
    fi
    object_key="${prefix:+${prefix}/}telemetry/${day}/${archive_name}"
    s3_uri="s3://${bucket}/${object_key}"

    echo "Uploading ${archive_name} (${checksum}) to ${s3_uri}"
    aws s3 cp "${tmp_dir}/${archive_name}" "${s3_uri}" \
        --region "${AWS_REGION}" \
        --storage-class GLACIER_IR \
        --metadata "sha256=${checksum},date=${day}"

    uploaded_class="$(aws s3api head-object \
        --bucket "${bucket}" \
        --key "${object_key}" \
        --region "${AWS_REGION}" \
        --query 'StorageClass' \
        --output text)"
    if [[ "${uploaded_class}" != "GLACIER_IR" ]]; then
        echo "S3 storage-class verification failed: expected GLACIER_IR, got ${uploaded_class}." >&2
        rm -rf -- "${tmp_dir}"
        return 1
    fi

    downloaded_archive="${tmp_dir}/downloaded-${archive_name}"
    aws s3 cp "${s3_uri}" "${downloaded_archive}" --region "${AWS_REGION}" --only-show-errors
    downloaded_checksum="$(sha256sum "${downloaded_archive}" | awk '{print $1}')"
    if [[ "${downloaded_checksum}" != "${checksum}" ]]; then
        echo "Downloaded S3 object checksum does not match the uploaded archive." >&2
        rm -rf -- "${tmp_dir}"
        return 1
    fi

    mkdir -p "${tmp_dir}/restore"
    zstd -dc "${downloaded_archive}" | tar -xf - -C "${tmp_dir}/restore"
    "${UETS_BIN}" db verify-checkpoint \
        "${day}" "${tmp_dir}/restore/rocksdb_backup"

    echo "S3 upload and restore verification passed; deleting local daily partition ${day_dir}."
    rm -rf -- "${day_dir}"
    if [[ -e "${day_dir}" ]]; then
        echo "Local daily partition cleanup did not complete: ${day_dir}" >&2
        rm -rf -- "${tmp_dir}"
        return 1
    fi

    rm -rf -- "${tmp_dir}"
    echo "Daily partition ${day} archived to Glacier and removed from local disk."
    return 0
}

while true; do
    used_percent="$(df -P "${DB_ROOT}" | awk 'END { gsub(/%/, "", $5); print $5 }')"
    if [[ ! "${used_percent}" =~ ^[0-9]+$ ]]; then
        echo "Could not determine disk utilization for ${DB_ROOT}." >&2
        exit 1
    fi
    if (( used_percent < THRESHOLD_PERCENT )); then
        echo "Disk usage: ${used_percent}%. Below ${THRESHOLD_PERCENT}% threshold."
        exit 0
    fi

    oldest="$(list_daily_partitions | head -n1)"
    if [[ -z "${oldest}" ]]; then
        echo "Disk usage ${used_percent}% is above threshold but no daily partitions to archive." >&2
        exit 1
    fi

    if [[ -n "${TARGET_DATE}" ]]; then
        if [[ "${oldest}" != "${TARGET_DATE}" ]]; then
            echo "Target date ${TARGET_DATE} is not the oldest partition (${oldest}). Nothing to archive." >&2
            exit 0
        fi
    fi

    if ! stop_uets; then
        echo "Failed to stop UETS; aborting archive of ${oldest}." >&2
        exit 1
    fi

    if ! archive_day "${oldest}"; then
        echo "Archive of ${oldest} failed; restarting UETS." >&2
        resume_uets || true
        exit 1
    fi

    if ! resume_uets; then
        echo "CRITICAL: could not restart UETS after archiving ${oldest}; start it manually in tmux session '${UETS_TMUX_SESSION}'." >&2
        exit 1
    fi
done