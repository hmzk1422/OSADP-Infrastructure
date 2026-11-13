#!/usr/bin/env bash
#
# cloud.sh - disk-pressure FIFO archive of sealed UTC days to S3 (Glacier IR).
#
#   cloud.sh                 archive the oldest eligible days until disk usage is
#                            below DISK_THRESHOLD_PERCENT
#   cloud.sh YYYY-MM-DD      archive that one day regardless of disk usage
#
# A day is eligible only when it is a completed UTC day (< today), carries the
# `.sealed` marker (no writer holds its databases) and is not one of the newest
# UETS_KEEP_RECENT_DAYS sealed days (kept so a restarted worker can recover).
#
# Per day: db checkpoint -> tar|zstd -> S3 upload -> storage-class check ->
# download -> SHA-256 compare -> restore -> db verify-checkpoint -> rename to
# .archived-<day>.<pid> -> rm -rf. The local day is only touched after EVERY
# verification step passed. Nothing here ever talks to tmux or the running UETS.
#
# Every step is checked explicitly: `set -e` is ignored inside functions that
# are called from `if` / `||`, so relying on it would delete data after a failure.
#
# Environment:
#   UETS_LIVE_DB (or DB_ROOT)  live database root         (default: <repo>/market_data/live_db)
#   UETS_BIN                   UETS binary running `db ...` (default: dev_test)
#   AWS_ARCHIVE_URI            s3://bucket/prefix           (required)
#   ARCHIVE_STORAGE_CLASS      S3 storage class             (default: GLACIER_IR)
#   ARCHIVE_TMP_DIR            scratch space, ideally a separate volume
#                                                           (default: /var/tmp/uets-archive-tmp)
#   ARCHIVE_TMP_MARGIN_KB      extra free space required    (default: 65536)
#   DISK_THRESHOLD_PERCENT     archive while usage >= this  (default: 90)
#   UETS_KEEP_RECENT_DAYS      newest sealed days to keep   (default: 1)
#   UETS_TODAY                 override "today" (UTC, tests only)
#
# Exit codes: 0 ok / below threshold, 1 archive failure or nothing eligible
# while over threshold, 2 bad usage or configuration, 3 another run is active.

set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DB_ROOT="${UETS_LIVE_DB:-${DB_ROOT:-${SCRIPT_DIR}/market_data/live_db}}"
ARCHIVE_TMP_DIR="${ARCHIVE_TMP_DIR:-/var/tmp/uets-archive-tmp}"
DISK_THRESHOLD_PERCENT="${DISK_THRESHOLD_PERCENT:-90}"
KEEP_RECENT_DAYS="${UETS_KEEP_RECENT_DAYS:-1}"
UETS_BIN="${UETS_BIN:-dev_test}"
AWS_ARCHIVE_URI="${AWS_ARCHIVE_URI:-}"
STORAGE_CLASS="${ARCHIVE_STORAGE_CLASS:-GLACIER_IR}"
TMP_MARGIN_KB="${ARCHIVE_TMP_MARGIN_KB:-65536}"
TODAY="${UETS_TODAY:-$(date -u +%F)}"

die_config() {
    echo "ERROR: $*" >&2
    exit 2
}

fail() {
    echo "ERROR: $*" >&2
}

is_day_name() {
    [[ "$1" =~ ^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$ ]]
}

# ---------------------------------------------------------------- arguments
if (( $# > 1 )); then
    die_config "usage: cloud.sh [YYYY-MM-DD]"
fi
TARGET_DATE="${1:-}"
if [[ -n "${TARGET_DATE}" ]] && ! is_day_name "${TARGET_DATE}"; then
    die_config "invalid date '${TARGET_DATE}'; use YYYY-MM-DD"
fi
is_day_name "${TODAY}" || die_config "invalid UETS_TODAY '${TODAY}'"
[[ "${DISK_THRESHOLD_PERCENT}" =~ ^[0-9]+$ ]] || die_config "DISK_THRESHOLD_PERCENT must be a number"
[[ "${KEEP_RECENT_DAYS}" =~ ^[0-9]+$ ]] || die_config "UETS_KEEP_RECENT_DAYS must be a number"
[[ "${TMP_MARGIN_KB}" =~ ^[0-9]+$ ]] || die_config "ARCHIVE_TMP_MARGIN_KB must be a number"
[[ "${AWS_ARCHIVE_URI}" == s3://* ]] || die_config "AWS_ARCHIVE_URI must be set to s3://bucket/prefix"
[[ -d "${DB_ROOT}" ]] || die_config "database root not found: ${DB_ROOT}"

for tool in aws zstd tar sha256sum flock df du awk mktemp; do
    command -v "${tool}" >/dev/null 2>&1 || die_config "required tool '${tool}' not found in PATH"
done
command -v "${UETS_BIN}" >/dev/null 2>&1 || die_config "UETS binary not found: ${UETS_BIN} (set UETS_BIN)"

mkdir -p -- "${ARCHIVE_TMP_DIR}"

# ------------------------------------------------------------ cleanup traps
tmp_dir=""

cleanup_on_exit() {
    local rc=$?
    trap - EXIT
    if [[ -n "${tmp_dir:-}" && -d "${tmp_dir}" ]]; then
        rm -rf -- "${tmp_dir}"
    fi
    exit "${rc}"
}

trap cleanup_on_exit EXIT
trap 'echo "Terminated by SIGTERM" >&2; exit 143' TERM
trap 'echo "Interrupted by SIGINT" >&2; exit 130' INT

# --------------------------------------------------------- single instance
exec 9>"${ARCHIVE_TMP_DIR}/.cloud.lock"
if ! flock -n 9; then
    echo "Another cloud.sh run is active (lock ${ARCHIVE_TMP_DIR}/.cloud.lock); exiting." >&2
    exit 3
fi

# Scratch space on the same filesystem as the live data competes with it for
# exactly the space we are trying to free.
db_dev="$(stat -c %d "${DB_ROOT}" 2>/dev/null || true)"
tmp_dev="$(stat -c %d "${ARCHIVE_TMP_DIR}" 2>/dev/null || true)"
if [[ -n "${db_dev}" && "${db_dev}" == "${tmp_dev}" ]]; then
    echo "WARNING: ARCHIVE_TMP_DIR (${ARCHIVE_TMP_DIR}) is on the same filesystem as ${DB_ROOT}; use a separate volume." >&2
fi

# Leftovers of interrupted runs (SIGKILL between rename and rm, killed mid-archive).
# Safe: we hold the lock, so nothing here belongs to a live run.
for stale in "${DB_ROOT}"/.archived-* "${ARCHIVE_TMP_DIR}"/archive_????-??-??_*; do
    if [[ -e "${stale}" ]]; then
        echo "Removing leftover from an interrupted run: ${stale}"
        rm -rf -- "${stale}" || echo "WARNING: could not remove ${stale}" >&2
    fi
done

# ----------------------------------------------------------------- helpers
skipped_days=" "
skipped_count=0

is_skipped() { [[ "${skipped_days}" == *" $1 "* ]]; }

mark_skipped() {
    skipped_days+="$1 "
    skipped_count=$(( skipped_count + 1 ))
}

disk_usage_percent() {
    local out pct
    out="$(df -P "${DB_ROOT}")" || return 1
    pct="$(awk 'NR==2 {gsub(/%/, "", $5); print $5}' <<<"${out}")"
    [[ "${pct}" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "${pct}"
}

# Sealed, completed days, oldest first, in the global array sealed_days.
sealed_days=()
collect_sealed_days() {
    local d name
    sealed_days=()
    for d in "${DB_ROOT}"/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]; do
        [[ -d "${d}" ]] || continue
        name="${d##*/}"
        is_day_name "${name}" || continue
        [[ "${name}" < "${TODAY}" ]] || continue
        [[ -f "${d}/.sealed" ]] || continue
        sealed_days+=("${name}")
    done
}

# Oldest sealed day that is not among the newest KEEP_RECENT_DAYS and has not
# failed this run. Sets `oldest`; returns 1 when there is none.
oldest=""
pick_oldest_eligible() {
    local usable i
    collect_sealed_days
    usable=$(( ${#sealed_days[@]} - KEEP_RECENT_DAYS ))
    oldest=""
    for (( i = 0; i < usable; i++ )); do
        if is_skipped "${sealed_days[i]}"; then
            continue
        fi
        oldest="${sealed_days[i]}"
        return 0
    done
    return 1
}

# Never `return` early without leaving the local day untouched: all failure
# paths below return 1 BEFORE the rename/rm at the very end.
_archive_day() {
    local day="$1"
    local day_dir="${DB_ROOT}/${day}"
    local day_kb avail_kb need_kb
    local ckpt archive downloaded restore local_sum remote_sum
    local uri rest bucket key actual_class retired

    echo "Archiving sealed day ${day}"

    # 1. Enough scratch space for checkpoint + archive, then download + restore.
    day_kb="$(du -sk "${day_dir}" | awk 'NR==1 {print $1}')" || { fail "cannot size ${day_dir}"; return 1; }
    [[ "${day_kb}" =~ ^[0-9]+$ ]] || { fail "cannot size ${day_dir}"; return 1; }
    avail_kb="$(df -Pk "${ARCHIVE_TMP_DIR}" | awk 'NR==2 {print $4}')" || { fail "cannot read free space of ${ARCHIVE_TMP_DIR}"; return 1; }
    [[ "${avail_kb}" =~ ^[0-9]+$ ]] || { fail "cannot read free space of ${ARCHIVE_TMP_DIR}"; return 1; }
    need_kb=$(( day_kb * 2 + TMP_MARGIN_KB ))
    if (( avail_kb < need_kb )); then
        fail "insufficient temporary space on ${ARCHIVE_TMP_DIR} for ${day}: need ${need_kb} KB, have ${avail_kb} KB"
        return 1
    fi

    tmp_dir="$(mktemp -d "${ARCHIVE_TMP_DIR}/archive_${day}_XXXXXX")" || { fail "cannot create a work directory in ${ARCHIVE_TMP_DIR}"; return 1; }
    ckpt="${tmp_dir}/rocksdb_backup"
    archive="${tmp_dir}/uets_${day}.tar.zst"
    downloaded="${tmp_dir}/downloaded.tar.zst"
    restore="${tmp_dir}/restore"

    # 2. Consistent checkpoint of every database of the day (never the live day).
    if ! "${UETS_BIN}" db checkpoint "${day}" "${ckpt}"; then
        fail "db checkpoint failed for ${day}"
        return 1
    fi

    # 3. Compress; remember the checksum of exactly what is uploaded.
    if ! tar -cf - -C "${tmp_dir}" rocksdb_backup | zstd -T0 -q -o "${archive}"; then
        fail "could not build the archive for ${day}"
        return 1
    fi
    if ! zstd -t "${archive}"; then
        fail "archive of ${day} failed its own integrity test"
        return 1
    fi
    local_sum="$(sha256sum "${archive}" | awk '{print $1}')" || { fail "cannot checksum the archive of ${day}"; return 1; }
    [[ -n "${local_sum}" ]] || { fail "cannot checksum the archive of ${day}"; return 1; }
    rm -rf -- "${ckpt}"

    # 4. Upload and confirm the storage class.
    uri="${AWS_ARCHIVE_URI%/}/daily/${day}/uets_${day}.tar.zst"
    if ! aws s3 cp "${archive}" "${uri}" --storage-class "${STORAGE_CLASS}" --only-show-errors; then
        fail "S3 upload failed for ${day} (${uri})"
        return 1
    fi
    rest="${uri#s3://}"
    bucket="${rest%%/*}"
    key="${rest#*/}"
    actual_class="$(aws s3api head-object --bucket "${bucket}" --key "${key}" --query StorageClass --output text)" \
        || { fail "cannot read back the uploaded object ${uri}"; return 1; }
    if [[ "${actual_class}" != "${STORAGE_CLASS}" ]]; then
        fail "uploaded object has storage class '${actual_class}', expected '${STORAGE_CLASS}'"
        return 1
    fi
    rm -f -- "${archive}"

    # 5. Download it again and prove it is byte-identical.
    if ! aws s3 cp "${uri}" "${downloaded}" --only-show-errors; then
        fail "S3 download failed for ${day} (${uri})"
        return 1
    fi
    remote_sum="$(sha256sum "${downloaded}" | awk '{print $1}')" || { fail "cannot checksum the downloaded archive of ${day}"; return 1; }
    if [[ "${remote_sum}" != "${local_sum}" ]]; then
        fail "SHA-256 of the downloaded archive does not match the uploaded archive for ${day}"
        return 1
    fi
    if ! zstd -t "${downloaded}"; then
        fail "downloaded archive of ${day} is not a valid zstd stream"
        return 1
    fi

    # 6. Restore it and open every database from the restored copy.
    mkdir -p -- "${restore}" || { fail "cannot create ${restore}"; return 1; }
    if ! zstd -dc "${downloaded}" | tar -xf - -C "${restore}"; then
        fail "could not extract the downloaded archive of ${day}"
        return 1
    fi
    if ! "${UETS_BIN}" db verify-checkpoint "${day}" "${restore}/rocksdb_backup"; then
        fail "restored database verification failed for ${day}"
        return 1
    fi
    rm -rf -- "${downloaded}" "${restore}"

    # 7. Only now touch the local day. Rename first (atomic): a partially
    #    deleted day with a .sealed marker can never be left behind.
    if [[ ! -f "${day_dir}/.sealed" ]]; then
        fail "${day} lost its .sealed marker during the archive; keeping it"
        return 1
    fi
    retired="${DB_ROOT}/.archived-${day}.$$"
    if ! mv -- "${day_dir}" "${retired}"; then
        fail "could not retire local day ${day}; keeping it"
        return 1
    fi
    if ! rm -rf -- "${retired}"; then
        # The day is safely in S3 and already out of the day listing; the next
        # run sweeps ${retired}.
        echo "WARNING: archived ${day} but could not fully remove ${retired}; it will be swept on the next run." >&2
    fi
    echo "Archived and removed local day ${day} (${uri})"
    return 0
}

archive_day() {
    local rc=0
    _archive_day "$1" || rc=$?
    if [[ -n "${tmp_dir:-}" && -d "${tmp_dir}" ]]; then
        rm -rf -- "${tmp_dir}"
    fi
    tmp_dir=""
    return "${rc}"
}

# ------------------------------------------------------------- seal days
# Completed days whose databases are closed get their `.sealed` marker. Also
# migrates days written before sealing existed. Never seals today.
if ! "${UETS_BIN}" db seal-days "${TODAY}"; then
    echo "WARNING: 'db seal-days ${TODAY}' failed; only already-sealed days can be archived." >&2
fi

# ---------------------------------------------------------- explicit date
if [[ -n "${TARGET_DATE}" ]]; then
    collect_sealed_days
    eligible=0
    usable=$(( ${#sealed_days[@]} - KEEP_RECENT_DAYS ))
    for (( i = 0; i < ${#sealed_days[@]}; i++ )); do
        if [[ "${sealed_days[i]}" == "${TARGET_DATE}" ]]; then
            eligible=1
            if (( i >= usable )); then
                eligible=2
            fi
        fi
    done
    if (( eligible == 0 )); then
        echo "ERROR: ${TARGET_DATE} is not a sealed, completed day (today is ${TODAY}); refusing to archive it." >&2
        exit 2
    fi
    if (( eligible == 2 )); then
        echo "ERROR: ${TARGET_DATE} is among the newest ${KEEP_RECENT_DAYS} sealed day(s) kept for worker recovery; set UETS_KEEP_RECENT_DAYS=0 to archive it." >&2
        exit 2
    fi
    if ! archive_day "${TARGET_DATE}"; then
        echo "ERROR: Archive of ${TARGET_DATE} failed; its local copy was kept." >&2
        exit 1
    fi
    exit 0
fi

# ---------------------------------------------------------- FIFO under pressure
# One failing day (corrupt, half-deleted, S3 hiccup) must not block the newer
# days: it is skipped for the rest of this run, kept locally, and the exit code
# stays non-zero so it is noticed.
while true; do
    usage="$(disk_usage_percent)" || { echo "ERROR: cannot read disk usage of ${DB_ROOT}" >&2; exit 1; }

    if (( usage < DISK_THRESHOLD_PERCENT )); then
        echo "Disk usage: ${usage}%. Below ${DISK_THRESHOLD_PERCENT}% threshold."
        if (( skipped_count > 0 )); then
            echo "WARNING: ${skipped_count} skipped after failure (kept locally); see the errors above." >&2
            exit 1
        fi
        exit 0
    fi

    if ! pick_oldest_eligible; then
        echo "ERROR: Disk usage is ${usage}% (threshold ${DISK_THRESHOLD_PERCENT}%), but no sealed, completed day is eligible (newest ${KEEP_RECENT_DAYS} kept for worker recovery, ${skipped_count} skipped after failure)." >&2
        exit 1
    fi

    echo "Disk usage is ${usage}%, at or above the ${DISK_THRESHOLD_PERCENT}% threshold; archiving the oldest eligible day ${oldest}."
    if ! archive_day "${oldest}"; then
        echo "ERROR: Archive of ${oldest} failed; its local copy was kept and it is skipped for this run." >&2
        mark_skipped "${oldest}"
    fi
done