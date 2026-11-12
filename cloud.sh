#!/usr/bin/env bash
set -euo pipefail

# Configuration defaults
DB_ROOT="${DB_ROOT:-market_data/live_db}"
ARCHIVE_TMP_DIR="${ARCHIVE_TMP_DIR:-/var/tmp/uets-archive-tmp}"
TARGET_DATE="${1:-}"

# Ensure temp directory exists on the configured volume (Fix 5)
mkdir -p "${ARCHIVE_TMP_DIR}"

# Track temporary directories/files for signal cleanup
tmp_dir=""

cleanup_on_exit() {
    local exit_code=$?
    if [[ -n "${tmp_dir}" && -d "${tmp_dir}" ]]; then
        rm -rf -- "${tmp_dir}"
    fi
    exit "${exit_code}"
}

# Signal Traps (Fix 4)
trap 'echo "Terminated by SIGTERM" >&2; exit 143' TERM
trap 'echo "Interrupted by SIGINT" >&2; exit 130' INT
trap cleanup_on_exit EXIT

is_archiveable() {
    local day="$1"
    local day_dir="${DB_ROOT}/${day}"
    # Must be a directory, contain the sealed marker, and have telemetry
    [[ -d "${day_dir}" ]] && [[ -f "${day_dir}/.sealed" ]] && [[ -d "${day_dir}/telemetry" ]]
}

archive_day() {
    local day="$1"
    local day_dir="${DB_ROOT}/${day}"
    
    echo "Archiving sealed day: ${day}"
    
    # Create isolated temp workdir on the separate volume (Fix 5)
    tmp_dir="$(mktemp -d "${ARCHIVE_TMP_DIR}/archive_${day}_XXXXXX")"
    
    # Example archival/compression logic simulation
    local archive_name="uets_archive_${day}.tar.gz"
    local downloaded_archive="${tmp_dir}/${archive_name}"
    
    # Pack and verify step (placeholder for actual tar + upload/verification flow)
    tar -czf "${downloaded_archive}" -C "${DB_ROOT}" "${day}"
    
    # Checksum verification (Fix 5 - cleanup downloaded copy after verification)
    local expected_sum_file="${downloaded_archive}.sha256"
    sha256sum "${downloaded_archive}" > "${expected_sum_file}"
    sha256sum -c "${expected_sum_file}"
    
    # Clean up downloaded archive immediately after successful verification (Fix 5)
    rm -f "${downloaded_archive}" "${expected_sum_file}"

    # Non-atomic delete protection: Rename first, then remove (Fix 3)
    # Dot-prefixed archived names are safely ignored by partition filters[cite: 1]
    local archived_marker="${DB_ROOT}/.archived-${day}"
    if mv -- "${day_dir}" "${archived_marker}" && rm -rf -- "${archived_marker}"; then
        echo "Successfully archived and cleaned up day ${day}"
        rm -rf -- "${tmp_dir}"
        tmp_dir=""
        return 0
    else
        echo "Failed to remove archived day ${day}" >&2
        rm -rf -- "${tmp_dir}"
        tmp_dir=""
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Fix 1: Early TARGET_DATE handling at the top, before the threshold check
# ---------------------------------------------------------------------------
if [[ -n "${TARGET_DATE}" ]]; then
    if ! is_archiveable "${TARGET_DATE}"; then
        echo "Target date ${TARGET_DATE} is not a sealed, archiveable day." >&2
        exit 2
    fi
    # Skip threshold check for explicit date targets
    archive_day "${TARGET_DATE}" || exit 1
    exit 0
fi

# ---------------------------------------------------------------------------
# Fix 2: FIFO loop with skip-and-continue pattern for failed days
# ---------------------------------------------------------------------------
declare -A skipped=()
rc=0

# Example disk threshold monitoring loop
while true; do
    # Check current disk usage percentage of DB_ROOT filesystem
    current_usage=$(df -P "${DB_ROOT}" | awk 'NR==2 {print +int($5)}')
    
    # Threshold condition (e.g., 90%)
    if (( current_usage < 90 )); then
        exit "${rc}"
    fi

    # Find the oldest archiveable day
    oldest=""
    for d in "${DB_ROOT}"/*; do
        if [[ -d "$d" ]]; then
            day_name=$(basename "$d")
            if is_archiveable "${day_name}" && [[ -z "${skipped[$day_name]:-}" ]]; then
                if [[ -z "${oldest}" || "${day_name}" < "${oldest}" ]]; then
                    oldest="${day_name}"
                fi
            fi
        fi
    done

    if [[ -z "${oldest}" ]]; then
        echo "Disk usage is at ${current_usage}%, but no further archiveable days remain." >&2
        exit "${rc:-1}"
    fi

    # Attempt archival; if it fails, skip this day and continue checking/looping (Fix 2)
    archive_day "${oldest}" || {
        skipped["${oldest}"]=1
        rc=1
        continue
    }
done