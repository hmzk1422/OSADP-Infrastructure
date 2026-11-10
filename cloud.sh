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
if [[ ! -d "${DB_PATH}" ]]; then
    echo "Telemetry RocksDB session not found: ${DB_PATH}" >&2
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

for command_name in aws zstd tar sha256sum flock df awk; do
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

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive_name="telemetry-${SESSION_ID}-${timestamp}.tar.zst"
tmp_dir="$(mktemp -d "${ARCHIVE_TMP_DIR%/}/uets-archive.XXXXXX")"
trap 'rm -rf -- "${tmp_dir}"' EXIT
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
Snapshot: RocksDB Checkpoint from the owning UETS DB handle when running
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

printf '%s\n' \
    "UETS archive uploaded and restore-verified." \
    "Storage class: ${uploaded_class}" \
    "Session: ${SESSION_ID}" \
    "S3 location: ${s3_uri}" \
    "SHA-256: ${checksum}" \
    "Original RocksDB remains untouched."
