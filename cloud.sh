#!/usr/bin/env bash
set -euo pipefail

# UETS daily-partition Glacier archiver (FIFO).
#
# RocksDB is stored under $UETS_LIVE_DB as daily UTC partitions:
#   ${DB_ROOT}/YYYY-MM-DD/{telemetry,portfolio,trades}
#
# The writer seals a completed day by writing ${DB_ROOT}/YYYY-MM-DD/.sealed
# after rolling over to the next day. The archiver never touches the live day
# and never stops UETS. Sealed days are checkpointed read-only with
# `db checkpoint`, uploaded to S3 Glacier Instant Retrieval, download-verified,
# restore-verified, then deleted from local disk to keep usage below threshold.
#
# Only days where name < today's UTC date AND a .sealed marker exists are
# eligible. If disk is above threshold and no eligible day exists, the script
# exits non-zero without deleting anything.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DB_ROOT="${UETS_LIVE_DB:-${ROOT}/market_data/live_db}"
UETS_TODAY="${UETS_TODAY:-$(date -u +%F)}"
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

if [[ -n "${TARGET_DATE}" ]] && [[ ! "${TARGET_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Invalid date. Use YYYY-MM-DD." >&2
    exit 2
fi
if [[ ! "${THRESHOLD_PERCENT}" =~ ^[0-9]+$ ]] ||
   (( THRESHOLD_PERCENT < 1 || THRESHOLD_PERCENT > 100 )); then
    echo "DISK_THRESHOLD_PERCENT must be an integer from 1 to 100." >&2
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

for command_name in aws zstd tar sha256sum flock df awk realpath sort head date; do
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

tmp_dir=""
trap '[[ -n "${tmp_dir}" && -d "${tmp_dir}" ]] && rm -rf -- "${tmp_dir}"' EXIT

is_archiveable() {
    local d="$1"
    [[ "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || return 1
    [[ "$d" < "${UETS_TODAY}" ]] || return 1
    [[ -f "${DB_ROOT}/${d}/.sealed" ]] || return 1
    [[ -d "${DB_ROOT}/${d}/telemetry" ]] || [[ -d "${DB_ROOT}/${d}/portfolio" ]] || [[ -d "${DB_ROOT}/${d}/trades" ]] || return 1
    return 0
}

list_archiveable_days() {
    local entry name
    shopt -s nullglob
    for entry in "${DB_ROOT}"/*/; do
        shopt -u nullglob
        entry="${entry%/}"
        name="$(basename -- "${entry}")"
        is_archiveable "${name}" && printf '%s\n' "${name}"
    done | sort
}

archive_day() {
    local day="$1"
    local day_dir="${DB_ROOT}/${day}"
    local timestamp archive_name checksum
    local bucket prefix object_key s3_uri uploaded_class
    local downloaded_archive downloaded_checksum
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    archive_name="telemetry-${day}-${timestamp}.tar.zst"
    tmp_dir="$(mktemp -d "${ARCHIVE_TMP_DIR%/}/uets-archive.XXXXXX")"

    export UETS_LIVE_DB="${DB_ROOT}"
    "${UETS_BIN}" db checkpoint "${day}" "${tmp_dir}/rocksdb_backup" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    "${UETS_BIN}" db verify-checkpoint "${day}" "${tmp_dir}/rocksdb_backup" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }

    cat >"${tmp_dir}/manifest.txt" <<EOF
UETS telemetry RocksDB daily partition
Date: ${day}
Created UTC: ${timestamp}
Original DB: ${day_dir}
Snapshot: RocksDB Checkpoint of sealed daily partition (live UETS untouched)
Compression: Zstandard
EOF

    tar -cf - -C "${tmp_dir}" rocksdb_backup manifest.txt |
        zstd -T0 -6 -o "${tmp_dir}/${archive_name}" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    zstd -t "${tmp_dir}/${archive_name}" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    zstd -dc "${tmp_dir}/${archive_name}" | tar -tf - >/dev/null || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    checksum="$(sha256sum "${tmp_dir}/${archive_name}" | awk '{print $1}')" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }

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
        --metadata "sha256=${checksum},date=${day}" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }

    uploaded_class="$(aws s3api head-object \
        --bucket "${bucket}" \
        --key "${object_key}" \
        --region "${AWS_REGION}" \
        --query 'StorageClass' \
        --output text)" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    if [[ "${uploaded_class}" != "GLACIER_IR" ]]; then
        echo "S3 storage-class verification failed: expected GLACIER_IR, got ${uploaded_class}." >&2
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    fi

    downloaded_archive="${tmp_dir}/downloaded-${archive_name}"
    aws s3 cp "${s3_uri}" "${downloaded_archive}" --region "${AWS_REGION}" --only-show-errors || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    downloaded_checksum="$(sha256sum "${downloaded_archive}" | awk '{print $1}')" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    if [[ "${downloaded_checksum}" != "${checksum}" ]]; then
        echo "Downloaded S3 object checksum does not match the uploaded archive." >&2
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    fi

    mkdir -p "${tmp_dir}/restore" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    zstd -dc "${downloaded_archive}" | tar -xf - -C "${tmp_dir}/restore" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    "${UETS_BIN}" db verify-checkpoint "${day}" "${tmp_dir}/restore/rocksdb_backup" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }

    echo "S3 upload and restore verification passed; deleting local daily partition ${day_dir}."
    rm -rf -- "${day_dir}" || {
        rm -rf -- "${tmp_dir}" 2>/dev/null || true
        tmp_dir=""
        return 1
    }
    rm -rf -- "${tmp_dir}"
    tmp_dir=""
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

    if [[ -n "${TARGET_DATE}" ]]; then
        if ! is_archiveable "${TARGET_DATE}"; then
            echo "Target date ${TARGET_DATE} is not a sealed, archiveable day." >&2
            exit 2
        fi
        archive_day "${TARGET_DATE}" || exit 1
        exit 0
    fi

    oldest=""
    for d in $(list_archiveable_days); do
        oldest="${d}"
        break
    done
    if [[ -z "${oldest}" ]]; then
        echo "Disk usage ${used_percent}% is above threshold but no sealed, completed days to archive." >&2
        exit 1
    fi

    archive_day "${oldest}" || exit 1
done