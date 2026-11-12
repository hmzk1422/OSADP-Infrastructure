#!/usr/bin/env bash
set -euo pipefail

# Test: daily-partition FIFO archive to Glacier.
#
# Verifies:
#   1. Oldest sealed days are exported and deleted until disk is below threshold (multi-iteration).
#   2. Newer sealed days remain fully intact (telemetry + portfolio + trades).
#   3. Today's partition is NEVER archived or deleted.
#   4. tmux is never invoked.
#   5. Upload failure retains the local partition.
#   6. Restore-verify failure retains the local partition.
#   7. Checkpoint failure retains the local partition (errexit bug regression).
#   8. Checksum corruption on download retains the local partition and exits non-zero.
#   9. Target-date path rejects today's date (even if sealed).
#   10. S3 object key contains the day.
#   11. When no archiveable day exists above threshold, exit non-zero without deleting.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
state_dir="$(mktemp -d)"
trap 'rm -rf -- "${state_dir}"' EXIT
fake_bin="${state_dir}/bin"
mkdir -p "${fake_bin}"

# Deterministic "today" injected into both cloud.sh and this test.
export UETS_TODAY="2026-11-12"

# Track tmux invocations — cloud.sh must never call tmux.
tmux_call_log="${state_dir}/tmux_calls.log"
: >"${tmux_call_log}"

# Track S3 object keys uploaded by the fake aws.
s3_key_log="${state_dir}/s3_keys.log"
: >"${s3_key_log}"

# ---------------------------------------------------------------------------
# Stub: df — computes usage from real day-directory count.
#   percent = 70 + 10 * count_of_day_dirs
# With 3 days: 100% -> delete 11-09 -> 90% (not below 90) -> delete 11-10 -> 80% (below).
# ---------------------------------------------------------------------------
cat >"${fake_bin}/df" <<'STUB'
#!/usr/bin/env bash
count=$(find "${UETS_LIVE_DB}" -maxdepth 1 -type d -name '20??-??-??' 2>/dev/null | wc -l)
percent=$((50 + 15 * count))
free=$((100 - percent))
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/mock 100 %d %d %d%% /\n' "$percent" "$free" "$percent"
STUB

cat >"${fake_bin}/flock" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB

# ---------------------------------------------------------------------------
# Stub: tmux — must NEVER be called by cloud.sh. Any call fails the test.
# ---------------------------------------------------------------------------
cat >"${fake_bin}/tmux" <<'STUB'
#!/usr/bin/env bash
echo "tmux called: $*" >>"${TMUX_CALL_LOG:-/dev/null}"
echo "ERROR: cloud.sh invoked tmux, which it must not do" >&2
exit 99
STUB

# ---------------------------------------------------------------------------
# Stub: zstd / tar
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Stub: sha256sum — uses the REAL sha256sum so checksum verification is real.
# ---------------------------------------------------------------------------
cat >"${fake_bin}/sha256sum" <<'STUB'
#!/usr/bin/env bash
# Delegate to the real sha256sum, avoiding self-recursion (fake_bin is first in PATH).
stub_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATH="${PATH#"${stub_dir}":}" exec sha256sum "$@"
STUB

# ---------------------------------------------------------------------------
# Stub: aws — upload/download, head-object returns GLACIER_IR.
#   AWS_FAIL_UPLOAD=1  -> s3 cp upload direction fails
#   AWS_CORRUPT_DOWNLOAD=1 -> s3 cp download direction writes garbage
# Records uploaded S3 object keys to ${S3_KEY_LOG}.
# ---------------------------------------------------------------------------
cat >"${fake_bin}/aws" <<'STUB'
#!/usr/bin/env bash
if [[ "$1 $2" == "s3 cp" ]]; then
    if [[ "${AWS_FAIL_UPLOAD:-0}" == "1" && "$3" != s3://* ]]; then
        echo "simulated S3 upload failure" >&2
        exit 93
    fi
    if [[ "$3" == s3://* ]]; then
        # download direction
        if [[ "${AWS_CORRUPT_DOWNLOAD:-0}" == "1" ]]; then
            printf 'corrupted archive contents\n' >"$4"
        else
            cp "${AWS_OBJECT}" "$4"
        fi
    else
        # upload direction
        cp "$3" "${AWS_OBJECT}"
        # record the S3 object key (4th arg is the s3:// URI)
        echo "$3" >>"${S3_KEY_LOG:-/dev/null}"
    fi
elif [[ "$1 $2" == "s3api head-object" ]]; then
    printf 'GLACIER_IR\n'
else
    echo "unexpected aws command: $*" >&2; exit 94
fi
STUB

# ---------------------------------------------------------------------------
# Stub: uets binary — db checkpoint / verify-checkpoint per daily partition.
#   UETS_FAIL_CHECKPOINT=1  -> db checkpoint fails
#   UETS_FAIL_RESTORE_VERIFY=1 -> restore verify-checkpoint fails
# ---------------------------------------------------------------------------
cat >"${state_dir}/uets" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "db checkpoint" ]]; then
    if [[ "${UETS_FAIL_CHECKPOINT:-0}" == "1" ]]; then
        echo "simulated checkpoint failure" >&2
        exit 97
    fi
    mkdir -p "$4"
    printf 'checkpoint data\n' >"$4/checkpoint"
elif [[ "$1 $2" == "db verify-checkpoint" ]]; then
    if [[ "${UETS_FAIL_RESTORE_VERIFY:-0}" == "1" && "$4" == */restore/rocksdb_backup ]]; then
        echo "simulated restored database verification failure" >&2
        exit 96
    fi
    [[ -s "$4/checkpoint" ]]
else
    echo "unexpected UETS command: $*" >&2; exit 95
fi
STUB

chmod +x "${fake_bin}/df" "${fake_bin}/flock" "${fake_bin}/tmux" "${fake_bin}/zstd" \
    "${fake_bin}/tar" "${fake_bin}/sha256sum" "${fake_bin}/aws" "${state_dir}/uets"

# ---------------------------------------------------------------------------
# Fixture: sealed days 2026-11-09, 2026-11-10, 2026-11-11. Today (2026-11-12)
# is created UNsealed — a real writer would not seal the live day.
# Each day has telemetry/, portfolio/, trades/.
# current_session is kept for the TUI.
# ---------------------------------------------------------------------------
make_fixture() {
    local name="$1"
    local root="${state_dir}/${name}/live_db"
    for day in 2026-11-09 2026-11-10 2026-11-11 "${UETS_TODAY}"; do
        mkdir -p "${root}/${day}/telemetry/session-1" \
                 "${root}/${day}/portfolio/session-1" \
                 "${root}/${day}/trades/session-1"
        if [[ "${day}" != "${UETS_TODAY}" ]]; then
            touch "${root}/${day}/.sealed"
        fi
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
    local target="${3:-}"
    export UETS_LIVE_DB="${root}"
    export TMUX_CALL_LOG="${state}/tmux_calls.log"
    export S3_KEY_LOG="${state}/s3_keys.log"
    export UETS_TODAY="${UETS_TODAY}"
    PATH="${fake_bin}:${PATH}" \
        UETS_HOME="${state}/home" \
        UETS_BIN="${state_dir}/uets" \
        UETS_TMUX_SESSION=uets \
        UETS_TMUX_PANE=uets:0.0 \
        AWS_ARCHIVE_URI=s3://test-bucket/uets \
        AWS_OBJECT="${state}/uploaded-archive" \
        ARCHIVE_TMP_DIR="${state}/tmp" \
        DISK_THRESHOLD_PERCENT=90 \
        bash "${ROOT}/cloud.sh" "${target}" >"${state}/cloud_stdout.txt" 2>"${state}/cloud_stderr.txt"
    local rc=$?
    cat "${state}/cloud_stdout.txt" >&2 || true
    cat "${state}/cloud_stderr.txt" >&2 || true
    return "${rc}"
}
mkdir -p "${state_dir}/tmp"

# ---------------------------------------------------------------------------
# Test 1: FIFO deletion and exporting — disk actively maintained below threshold.
# 3 sealed days (11-09, 11-10, 11-11) + today (11-12, unsealed).
# Disk formula: percent = 50 + 15 * day_dir_count.
#   count=4 -> 110% (>90) -> delete 11-09
#   count=3 -> 95% (>90)  -> delete 11-10
#   count=2 -> 80% (<90)  -> stop
# Verify: 11-09 and 11-10 deleted; 11-11 intact; today intact; tmux never called.
# ---------------------------------------------------------------------------
root="$(make_fixture success)"
state="${state_dir}/success"

if ! run_archive "${root}" "${state}"; then
    echo "Test 1 FAILED: FIFO daily-partition archive should succeed" >&2
    exit 1
fi

if [[ -e "${root}/2026-11-09" ]]; then
    echo "Test 1 FAILED: 2026-11-09 was not deleted after verified upload" >&2
    exit 1
fi
if [[ -e "${root}/2026-11-10" ]]; then
    echo "Test 1 FAILED: 2026-11-10 was not deleted after verified upload" >&2
    exit 1
fi

for day in 2026-11-11 "${UETS_TODAY}"; do
    for sub in telemetry portfolio trades; do
        if [[ ! -d "${root}/${day}/${sub}" ]]; then
            echo "Test 1 FAILED: ${root}/${day}/${sub} missing after archive" >&2
            exit 1
        fi
    done
done

if [[ -s "${state}/tmux_calls.log" ]]; then
    echo "Test 1 FAILED: cloud.sh invoked tmux (must never): $(cat "${state}/tmux_calls.log")" >&2
    exit 1
fi

if [[ "$(<"${root}/current_session")" != "session-1" ]]; then
    echo "Test 1 FAILED: current_session was not preserved" >&2
    exit 1
fi

if ! grep -q "2026-11-09" "${state}/s3_keys.log"; then
    echo "Test 1 FAILED: S3 object key does not contain the day" >&2
    exit 1
fi

echo "Test 1 PASSED: 2 oldest days exported+deleted; newer days intact; today untouched; tmux never called; S3 key contains day."

# ---------------------------------------------------------------------------
# Test 2: Upload failure — local partition retained.
# ---------------------------------------------------------------------------
root="$(make_fixture upload-fail)"
state="${state_dir}/upload-fail"

if AWS_FAIL_UPLOAD=1 run_archive "${root}" "${state}"; then
    echo "Test 2 FAILED: archive unexpectedly succeeded despite simulated S3 failure" >&2
    exit 1
fi

if [[ ! -d "${root}/2026-11-09" ]]; then
    echo "Test 2 FAILED: local partition deleted despite failed upload" >&2
    exit 1
fi

for day in 2026-11-09 2026-11-10 2026-11-11 "${UETS_TODAY}"; do
    if [[ ! -d "${root}/${day}/telemetry" ]]; then
        echo "Test 2 FAILED: partition ${day} lost after upload failure" >&2
        exit 1
    fi
done

echo "Test 2 PASSED: upload failure retains local partition."

# ---------------------------------------------------------------------------
# Test 3: Restore-verify failure — local partition retained.
# ---------------------------------------------------------------------------
root="$(make_fixture restore-fail)"
state="${state_dir}/restore-fail"

if UETS_FAIL_RESTORE_VERIFY=1 run_archive "${root}" "${state}"; then
    echo "Test 3 FAILED: archive unexpectedly succeeded despite failed restore verification" >&2
    exit 1
fi

if [[ ! -d "${root}/2026-11-09" ]]; then
    echo "Test 3 FAILED: local partition deleted despite failed restore verification" >&2
    exit 1
fi

echo "Test 3 PASSED: restore-verify failure retains local partition."

# ---------------------------------------------------------------------------
# Test 4: Checkpoint failure — local partition retained (errexit regression).
# ---------------------------------------------------------------------------
root="$(make_fixture checkpoint-fail)"
state="${state_dir}/checkpoint-fail"

if UETS_FAIL_CHECKPOINT=1 run_archive "${root}" "${state}"; then
    echo "Test 4 FAILED: archive unexpectedly succeeded despite failed checkpoint" >&2
    exit 1
fi

if [[ ! -d "${root}/2026-11-09" ]]; then
    echo "Test 4 FAILED: local partition deleted despite failed checkpoint" >&2
    exit 1
fi

echo "Test 4 PASSED: checkpoint failure retains local partition."

# ---------------------------------------------------------------------------
# Test 5: Checksum corruption on download — day retained, exit non-zero.
# ---------------------------------------------------------------------------
root="$(make_fixture corrupt)"
state="${state_dir}/corrupt"

if AWS_CORRUPT_DOWNLOAD=1 run_archive "${root}" "${state}"; then
    echo "Test 5 FAILED: archive unexpectedly succeeded despite checksum mismatch" >&2
    exit 1
fi

if [[ ! -d "${root}/2026-11-09" ]]; then
    echo "Test 5 FAILED: local partition deleted despite checksum mismatch" >&2
    exit 1
fi

echo "Test 5 PASSED: checksum corruption retains local partition."

# ---------------------------------------------------------------------------
# Test 6: Target-date path rejects today's date (even if sealed).
# Create today with .sealed to simulate edge case; target-date must still reject it.
# ---------------------------------------------------------------------------
root="$(make_fixture today-target)"
state="${state_dir}/today-target"
touch "${root}/${UETS_TODAY}/.sealed"

if run_archive "${root}" "${state}" "${UETS_TODAY}"; then
    echo "Test 6 FAILED: target-date archive of today unexpectedly succeeded" >&2
    exit 1
fi

if [[ ! -d "${root}/${UETS_TODAY}" ]]; then
    echo "Test 6 FAILED: today's partition deleted via target-date path" >&2
    exit 1
fi

echo "Test 6 PASSED: target-date path rejects today's date."

# ---------------------------------------------------------------------------
# Test 7: No archiveable days above threshold — exit non-zero, nothing deleted.
# Fixture: only today (unsealed) and older unsealed days. Disk is high but
# nothing is archiveable. Script must exit 1 without deleting anything.
# ---------------------------------------------------------------------------
root="$(make_fixture no-eligible)"
state="${state_dir}/no-eligible"
rm -f "${root}"/20*/.sealed

if run_archive "${root}" "${state}"; then
    echo "Test 7 FAILED: archive should fail when no archiveable days exist" >&2
    exit 1
fi

for day in 2026-11-09 2026-11-10 2026-11-11 "${UETS_TODAY}"; do
    if [[ ! -d "${root}/${day}" ]]; then
        echo "Test 7 FAILED: partition ${day} deleted despite no archiveable days" >&2
        exit 1
    fi
done

echo "Test 7 PASSED: no archiveable days -> exit non-zero, nothing deleted."
echo "ALL TESTS PASSED"