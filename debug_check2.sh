#!/usr/bin/env bash
set -euo pipefail
root="$(mktemp -d)"
rm -rf "$root"
mkdir -p "$root/2026-11-09/telemetry" "$root/2026-11-09/portfolio" "$root/2026-11-09/trades"
touch "$root/2026-11-09/.sealed"
mkdir -p "$root/2026-11-10/telemetry" "$root/2026-11-10/portfolio" "$root/2026-11-10/trades"
touch "$root/2026-11-10/.sealed"
mkdir -p "$root/2026-11-11/telemetry" "$root/2026-11-11/portfolio" "$root/2026-11-11/trades"
touch "$root/2026-11-11/.sealed"
mkdir -p "$root/2026-11-12/telemetry" "$root/2026-11-12/portfolio" "$root/2026-11-12/trades"
echo "session-1" >"$root/current_session"

export UETS_LIVE_DB="$root"
export UETS_BIN="/bin/false"
export AWS_ARCHIVE_URI=s3://test-bucket/uets
export DISK_THRESHOLD_PERCENT=90
export ARCHIVE_TMP_DIR=/tmp

echo "=== Before ==="
ls -la "$root/"
#!/usr/bin/env bash
set -euo pipefail
export UETS_TODAY=2026-11-12
root="$(pwd)/debug_live_db"
rm -rf "$root"
mkdir -p "$root/2026-11-09/telemetry" "$root/2026-11-09/portfolio" "$root/2026-11-09/trades"
touch "$root/2026-11-09/.sealed"
mkdir -p "$root/2026-11-10/telemetry" "$root/2026-11-10/portfolio" "$root/2026-11-10/trades"
touch "$root/2026-11-10/.sealed"
mkdir -p "$root/2026-11-11/telemetry" "$root/2026-11-11/portfolio" "$root/2026-11-11/trades"
touch "$root/2026-11-11/.sealed"
mkdir -p "$root/2026-11-12/telemetry" "$root/2026-11-12/portfolio" "$root/2026-11-12/trades"
echo "session-1" >"$root/current_session"

export UETS_LIVE_DB="$root"
export UETS_BIN="/bin/false"
export AWS_ARCHIVE_URI=s3://test-bucket/uets
export DISK_THRESHOLD_PERCENT=90
export ARCHIVE_TMP_DIR=/tmp

# Inline list_archiveable_days function
list_archiveable_days() {
    local entry name
    shopt -s nullglob
    for entry in "${DB_ROOT}"/*/; do
        shopt -u nullglob
        entry="${entry%/}"
        name="$(basename -- "${entry}")"
        if [[ -n "${skipped[$name]:-}" ]]; then
            continue
        fi
        is_archiveable "${name}" && printf '%s\n' "${name}"
    done | sort
}

# Inline is_archiveable function
is_archiveable() {
    local d="$1"
    [[ "$d" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || return 1
    [[ "$d" < "${UETS_TODAY}" ]] || return 1
    [[ -f "${DB_ROOT}/${d}/.sealed" ]] || return 1
    [[ -d "${DB_ROOT}/${d}/telemetry" ]] || [[ -d "${DB_ROOT}/${d}/portfolio" ]] || [[ -d "${DB_ROOT}/${d}/trades" ]] || return 1
    return 0
}

declare -A skipped=()
rc=0

echo "=== Before ==="
ls -la "$root/"

echo "=== list_archiveable_days ==="
for d in $(list_archiveable_days); do echo "archiveable: $d"; done

echo "=== main loop ==="
while true; do
    used_percent="$(df -P "$DB_ROOT" | awk 'END { gsub(/%/, "", $5); print $5 }')"
    echo "DEBUG used_percent=$used_percent"
    if [[ ! "$used_percent" =~ ^[0-9]+$ ]]; then
        echo "Could not determine disk utilization"
        exit 1
    fi
    if (( used_percent < THRESHOLD_PERCENT )); then
        echo "Below threshold"
        exit 0
    fi

    oldest=""
    for d in $(list_archiveable_days); do
        oldest="$d"
        break
    done
    echo "DEBUG oldest=$oldest"
    if [[ -z "$oldest" ]]; then
        echo "no sealed, completed days to archive"
        exit 1
    fi

    archive_day "$oldest" || { echo "archive failed for $oldest"; break; }
done

echo "=== After ==="
ls -la "$root/"
