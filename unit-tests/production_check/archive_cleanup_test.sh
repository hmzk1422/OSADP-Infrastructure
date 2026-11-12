#!/usr/bin/env bash
set -euo pipefail

# Test: daily-partition FIFO archive to Glacier (cloud.sh).
#
# Proves, against stubs that mirror the real `uets db ...` contract:
#   1. Deletion + export keep disk below the threshold: oldest sealed days are
#      checkpointed, uploaded and deleted one by one until usage drops, and the
#      loop stops as soon as it is below (df usage is derived from the real
#      number of day directories, so deletion is what lowers it).
#   2. Days are cleanly separated and every kept day still has telemetry/,
#      portfolio/ and trades/. Each uploaded archive contains all three.
#   3. The live day is never archived or deleted, and neither is a day a writer
#      still holds open (unsealed). The newest sealed day is kept by default.
#   4. Trading/TUI are never interrupted: tmux is never invoked and the only
#      UETS commands issued are `db seal-days`, `db checkpoint`,
#      `db verify-checkpoint` (no stop, no signal, no restart).
#   5. Any failure (checkpoint, upload, corrupted download, restore verify, low
#      temp space) keeps the local day and exits non-zero; later days are still
#      attempted after one fails.
#   6. Existing unsealed days are migrated by `db seal-days`; an explicit date
#      refuses today; interrupted deletes are cleaned up on the next run.
#
# Rollover itself (00:00 UTC swap, carry-forward, no producer stall) is tested
# in uets_db/src/partition.rs and persistence.rs (cargo test).

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
state_dir="$(mktemp -d)"
trap 'rm -rf -- "${state_dir}"' EXIT
fake_bin="${state_dir}/bin"
mkdir -p "${fake_bin}" "${state_dir}/tmp"

export UETS_TODAY="2026-11-12"
SEALED_DAYS=(2026-11-09 2026-11-10 2026-11-11)

die() {
    echo "FAILED: $*" >&2
    exit 1
}

# --- df: usage is derived from the real day-directory count -----------------
#   percent = 50 + 15 * count   (4 days -> 110, 3 -> 95, 2 -> 80, 1 -> 65)
cat >"${fake_bin}/df" <<'STUB'
#!/usr/bin/env bash
if [[ "$1" == "-Pk" ]]; then
    avail="${DF_TMP_AVAIL_KB:-999999999}"
    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/tmp 999999999 1 %s 1%% /tmp\n' "${avail}"
    exit 0
fi
if [[ -n "${DF_FORCE_PERCENT:-}" ]]; then
    percent="${DF_FORCE_PERCENT}"
else
    count=$(find "${UETS_LIVE_DB}" -maxdepth 1 -type d -name '20??-??-??' | wc -l)
    percent=$((50 + 15 * count))
fi
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/mock 100 %d %d %d%% /\n' "${percent}" "$((100 - percent))" "${percent}"
STUB

cat >"${fake_bin}/flock" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB

# --- tmux: tripwire. cloud.sh must never touch the running UETS ------------
cat >"${fake_bin}/tmux" <<'STUB'
#!/usr/bin/env bash
echo "tmux $*" >>"${TMUX_CALL_LOG:-/dev/null}"
echo "ERROR: cloud.sh invoked tmux, which it must never do" >&2
exit 99
STUB

# --- zstd: passthrough so zstd is not required; tar/sha256sum are real -------
cat >"${fake_bin}/zstd" <<'STUB'
#!/usr/bin/env bash
case "$1" in
    -T0) shift 2; cat >"$2" ;;
    -t) [[ -s "$2" ]] ;;
    -dc) cat "$2" ;;
    *) echo "unexpected zstd command: $*" >&2; exit 91 ;;
esac
STUB

# --- aws: records uploaded S3 URIs; can fail upload or corrupt the download --
cat >"${fake_bin}/aws" <<'STUB'
#!/usr/bin/env bash
if [[ "$1 $2" == "s3 cp" ]]; then
    if [[ "$3" == s3://* ]]; then
        if [[ "${AWS_CORRUPT_DOWNLOAD:-0}" == "1" ]]; then
            printf 'corrupted archive contents\n' >"$4"
        else
            cp "${AWS_OBJECT}" "$4"
        fi
    else
        if [[ "${AWS_FAIL_UPLOAD:-0}" == "1" ]]; then
            echo "simulated S3 upload failure" >&2
            exit 93
        fi
        cp "$3" "${AWS_OBJECT}"
        echo "$4" >>"${S3_KEY_LOG}"
    fi
elif [[ "$1 $2" == "s3api head-object" ]]; then
    printf 'GLACIER_IR\n'
else
    echo "unexpected aws command: $*" >&2; exit 94
fi
STUB

# --- uets: mirrors the real `db seal-days|checkpoint|verify-checkpoint` ------
cat >"${state_dir}/uets" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
echo "$*" >>"${UETS_CALL_LOG}"
[[ "$1" == "db" ]] || { echo "unexpected UETS command: $*" >&2; exit 95; }
case "$2" in
    seal-days)
        today="$3"
        for dir in "${UETS_LIVE_DB}"/20??-??-??; do
            [[ -d "${dir}" ]] || continue
            name="$(basename "${dir}")"
            [[ "${name}" < "${today}" ]] || continue
            [[ -f "${dir}/.sealed" ]] && continue
            # A writer still holds the day open (simulated by a .locked file).
            [[ -f "${dir}/.locked" ]] && continue
            touch "${dir}/.sealed"
            echo "sealed ${name}"
        done
        ;;
    checkpoint)
        day="$3"; dest="$4"
        [[ "${UETS_FAIL_CHECKPOINT:-0}" == "1" || "${UETS_FAIL_CHECKPOINT_DAY:-}" == "${day}" ]] && {
            echo "simulated checkpoint failure" >&2; exit 97; }
        [[ "${day}" < "${UETS_TODAY}" ]] || { echo "refusing to checkpoint ${day}: not a completed day" >&2; exit 98; }
        [[ -f "${UETS_LIVE_DB}/${day}/.sealed" ]] || { echo "day ${day} is not sealed" >&2; exit 98; }
        mkdir -p "${dest}"
        : >"${dest}/MANIFEST"
        for kind in telemetry portfolio trades; do
            for src in "${UETS_LIVE_DB}/${day}/${kind}"/*/; do
                [[ -d "${src}" ]] || continue
                session="$(basename "${src}")"
                mkdir -p "${dest}/${kind}"
                cp -r "${src%/}" "${dest}/${kind}/${session}"
                echo "${kind}/${session}" >>"${dest}/MANIFEST"
            done
        done
        ;;
    verify-checkpoint)
        dir="$4"
        if [[ "${UETS_FAIL_RESTORE_VERIFY:-0}" == "1" && "${dir}" == */restore/* ]]; then
            echo "simulated restored database verification failure" >&2; exit 96
        fi
        [[ -s "${dir}/MANIFEST" ]] || { echo "checkpoint has no MANIFEST" >&2; exit 96; }
        while IFS= read -r entry; do
            [[ -s "${dir}/${entry}/data" ]] || { echo "checkpoint is missing ${entry}" >&2; exit 96; }
        done <"${dir}/MANIFEST"
        ;;
    *) echo "unexpected UETS command: $*" >&2; exit 95 ;;
esac
STUB

chmod +x "${fake_bin}/df" "${fake_bin}/flock" "${fake_bin}/tmux" "${fake_bin}/zstd" \
    "${fake_bin}/aws" "${state_dir}/uets"

# --- fixtures -----------------------------------------------------------------
# Sealed days 11-09..11-11, plus today (11-12) which is NOT sealed, as a real
# writer would leave it. Every day has telemetry/, portfolio/ and trades/.
make_fixture() {
    local name="$1" day kind root="${state_dir}/${1}/live_db"
    for day in "${SEALED_DAYS[@]}" "${UETS_TODAY}"; do
        for kind in telemetry portfolio trades; do
            mkdir -p "${root}/${day}/${kind}/session-1"
            printf '%s-%s\n' "${kind}" "${day}" >"${root}/${day}/${kind}/session-1/data"
        done
        if [[ "${day}" != "${UETS_TODAY}" ]]; then
            touch "${root}/${day}/.sealed"
        fi
    done
    printf 'session-1\n' >"${root}/current_session"
    mkdir -p "${state_dir}/${name}/tmp"
    printf '%s\n' "${root}"
}

# Run cloud.sh; exports the result to RC, STDOUT_FILE and STDERR_FILE.
run_archive() {
    local root="$1" state="$2" target="${3:-}"
    export UETS_LIVE_DB="${root}"
    export TMUX_CALL_LOG="${state}/tmux_calls.log"
    export S3_KEY_LOG="${state}/s3_keys.log"
    export UETS_CALL_LOG="${state}/uets_calls.log"
    : >"${TMUX_CALL_LOG}" >"${S3_KEY_LOG}" >"${UETS_CALL_LOG}"
    STDOUT_FILE="${state}/stdout.txt"
    STDERR_FILE="${state}/stderr.txt"
    RC=0
    PATH="${fake_bin}:${PATH}" \
        UETS_BIN="${state_dir}/uets" \
        AWS_ARCHIVE_URI=s3://test-bucket/uets \
        AWS_OBJECT="${state}/uploaded-archive" \
        ARCHIVE_TMP_DIR="${state}/tmp" \
        DISK_THRESHOLD_PERCENT="${DISK_THRESHOLD_PERCENT:-90}" \
        bash "${ROOT}/cloud.sh" ${target:+"${target}"} >"${STDOUT_FILE}" 2>"${STDERR_FILE}" || RC=$?
    cat "${STDOUT_FILE}" "${STDERR_FILE}" >&2
}

assert_rc() { [[ "${RC}" == "$1" ]] || die "$2: expected exit $1, got ${RC}"; }
assert_rc_nonzero() { [[ "${RC}" != "0" ]] || die "$1: expected a non-zero exit"; }
assert_stderr_has() { grep -q -- "$1" "${STDERR_FILE}" || die "$2: stderr lacks '$1'"; }
assert_deleted() { [[ ! -e "${1}/${2}" ]] || die "$3: ${2} should have been deleted"; }
assert_all_kinds() {
    local root="$1" day="$2" kind
    for kind in telemetry portfolio trades; do
        [[ -s "${root}/${day}/${kind}/session-1/data" ]] || die "${3}: ${day}/${kind} missing or emptied"
    done
}
assert_retained() {
    local root="$1" label="$2" day
    shift 2
    for day in "$@"; do assert_all_kinds "${root}" "${day}" "${label}"; done
}
assert_no_tmux() { [[ ! -s "${1}/tmux_calls.log" ]] || die "$2: cloud.sh invoked tmux: $(cat "${1}/tmux_calls.log")"; }
assert_no_leftovers() {
    local retired
    for retired in "${1}"/.archived-*; do
        [[ ! -e "${retired}" ]] || die "$2: leftover ${retired}"
    done
}
upload_count() { wc -l <"${1}/s3_keys.log" | tr -d ' '; }
assert_uets_commands_only_archive_ops() {
    local line
    while IFS= read -r line; do
        case "${line}" in
            "db seal-days "*|"db checkpoint "*|"db verify-checkpoint "*) ;;
            *) die "$2: unexpected UETS command (anything else would disturb trading): ${line}" ;;
        esac
    done <"${1}/uets_calls.log"
}
assert_never_checkpointed() {
    ! grep -q "db checkpoint $2 " "${1}/uets_calls.log" || die "$3: $2 must never be checkpointed"
}

# ---------------------------------------------------------------------------
# Test 1: FIFO export + deletion keeps disk below the threshold.
#   count=4 -> 110% -> archive+delete 11-09
#   count=3 ->  95% -> archive+delete 11-10
#   count=2 ->  80% -> stop (11-11 is the newest sealed day: kept)
# ---------------------------------------------------------------------------
root="$(make_fixture t1)"; state="${state_dir}/t1"
run_archive "${root}" "${state}"
assert_rc 0 "Test 1"
assert_deleted "${root}" 2026-11-09 "Test 1"
assert_deleted "${root}" 2026-11-10 "Test 1"
assert_retained "${root}" "Test 1" 2026-11-11 "${UETS_TODAY}"
assert_stdout_ok=$(grep -c "Disk usage: 80%. Below 90% threshold." "${state}/stdout.txt" || true)
[[ "${assert_stdout_ok}" == "1" ]] || die "Test 1: run must end below the threshold (80%)"
[[ "$(upload_count "${state}")" == "2" ]] || die "Test 1: expected exactly 2 uploads"
grep -q "/daily/2026-11-09/" "${state}/s3_keys.log" || die "Test 1: 11-09 not uploaded under daily/2026-11-09/"
grep -q "/daily/2026-11-10/" "${state}/s3_keys.log" || die "Test 1: 11-10 not uploaded under daily/2026-11-10/"
! grep -Eq "2026-11-1[12]" "${state}/s3_keys.log" || die "Test 1: a kept day or today was uploaded"
# The last uploaded archive is a real tar: it must hold all three databases + MANIFEST.
listing="$(tar -tf "${state}/uploaded-archive")"
for kind in telemetry portfolio trades; do
    grep -q "rocksdb_backup/${kind}/session-1/data" <<<"${listing}" || die "Test 1: archive lacks ${kind}"
done
grep -q "rocksdb_backup/MANIFEST" <<<"${listing}" || die "Test 1: archive lacks MANIFEST"
assert_no_tmux "${state}" "Test 1"
assert_uets_commands_only_archive_ops "${state}" "Test 1"
assert_never_checkpointed "${state}" 2026-11-11 "Test 1"
assert_never_checkpointed "${state}" "${UETS_TODAY}" "Test 1"
assert_no_leftovers "${root}" "Test 1"
[[ "$(<"${root}/current_session")" == "session-1" ]] || die "Test 1: current_session must be preserved"
echo "Test 1 PASSED: 2 oldest days exported + deleted until 80% < 90%; newer days and today intact with all 3 folders; no tmux, only archive commands."

# ---------------------------------------------------------------------------
# Test 2: upload failure keeps every day.
# ---------------------------------------------------------------------------
root="$(make_fixture t2)"; state="${state_dir}/t2"
AWS_FAIL_UPLOAD=1 run_archive "${root}" "${state}"
assert_rc_nonzero "Test 2"
assert_stderr_has "S3 upload failed" "Test 2"
assert_retained "${root}" "Test 2" "${SEALED_DAYS[@]}" "${UETS_TODAY}"
assert_no_tmux "${state}" "Test 2"
echo "Test 2 PASSED: upload failure retains all local days, exits non-zero."

# ---------------------------------------------------------------------------
# Test 3: restore verification failure keeps every day (upload did happen).
# ---------------------------------------------------------------------------
root="$(make_fixture t3)"; state="${state_dir}/t3"
UETS_FAIL_RESTORE_VERIFY=1 run_archive "${root}" "${state}"
assert_rc_nonzero "Test 3"
assert_stderr_has "restored database verification failed" "Test 3"
[[ "$(upload_count "${state}")" -ge 1 ]] || die "Test 3: failure must happen after the upload"
assert_retained "${root}" "Test 3" "${SEALED_DAYS[@]}" "${UETS_TODAY}"
echo "Test 3 PASSED: restore-verify failure retains all local days."

# ---------------------------------------------------------------------------
# Test 4: checkpoint failure keeps every day and uploads nothing
# (regression for the errexit-inside-condition data-loss bug).
# ---------------------------------------------------------------------------
root="$(make_fixture t4)"; state="${state_dir}/t4"
UETS_FAIL_CHECKPOINT=1 run_archive "${root}" "${state}"
assert_rc_nonzero "Test 4"
assert_stderr_has "db checkpoint failed" "Test 4"
[[ "$(upload_count "${state}")" == "0" ]] || die "Test 4: nothing may be uploaded after a failed checkpoint"
assert_retained "${root}" "Test 4" "${SEALED_DAYS[@]}" "${UETS_TODAY}"
echo "Test 4 PASSED: checkpoint failure retains all local days and uploads nothing."

# ---------------------------------------------------------------------------
# Test 5: corrupted download (real sha256sum) keeps every day.
# ---------------------------------------------------------------------------
root="$(make_fixture t5)"; state="${state_dir}/t5"
AWS_CORRUPT_DOWNLOAD=1 run_archive "${root}" "${state}"
assert_rc_nonzero "Test 5"
assert_stderr_has "does not match the uploaded archive" "Test 5"
assert_retained "${root}" "Test 5" "${SEALED_DAYS[@]}" "${UETS_TODAY}"
echo "Test 5 PASSED: checksum mismatch retains all local days."

# ---------------------------------------------------------------------------
# Test 6: an explicit date can never archive today, even if it is sealed.
# ---------------------------------------------------------------------------
root="$(make_fixture t6)"; state="${state_dir}/t6"
touch "${root}/${UETS_TODAY}/.sealed"
run_archive "${root}" "${state}" "${UETS_TODAY}"
assert_rc 2 "Test 6"
assert_stderr_has "not a sealed, completed day" "Test 6"
assert_retained "${root}" "Test 6" "${SEALED_DAYS[@]}" "${UETS_TODAY}"
assert_never_checkpointed "${state}" "${UETS_TODAY}" "Test 6"
echo "Test 6 PASSED: explicit date refuses today."

# ---------------------------------------------------------------------------
# Test 7: days a writer still holds open are never sealed or archived.
# ---------------------------------------------------------------------------
root="$(make_fixture t7)"; state="${state_dir}/t7"
for day in "${SEALED_DAYS[@]}"; do rm -f "${root}/${day}/.sealed"; touch "${root}/${day}/.locked"; done
run_archive "${root}" "${state}"
assert_rc 1 "Test 7"
assert_stderr_has "no sealed, completed day is eligible" "Test 7"
assert_retained "${root}" "Test 7" "${SEALED_DAYS[@]}" "${UETS_TODAY}"
grep -q "db seal-days ${UETS_TODAY}" "${state}/uets_calls.log" || die "Test 7: seal-days was not run with today's date"
[[ "$(upload_count "${state}")" == "0" ]] || die "Test 7: nothing may be uploaded"
echo "Test 7 PASSED: locked (still open) days are never archived."

# ---------------------------------------------------------------------------
# Test 8: migration. Days written before sealing existed (no .sealed, no lock)
# are sealed by `db seal-days` and then archived normally.
# ---------------------------------------------------------------------------
root="$(make_fixture t8)"; state="${state_dir}/t8"
for day in "${SEALED_DAYS[@]}"; do rm -f "${root}/${day}/.sealed"; done
run_archive "${root}" "${state}"
assert_rc 0 "Test 8"
assert_deleted "${root}" 2026-11-09 "Test 8"
assert_deleted "${root}" 2026-11-10 "Test 8"
assert_retained "${root}" "Test 8" 2026-11-11 "${UETS_TODAY}"
[[ ! -f "${root}/${UETS_TODAY}/.sealed" ]] || die "Test 8: today must never be sealed"
echo "Test 8 PASSED: pre-existing unsealed days are sealed then archived; today is never sealed."

# ---------------------------------------------------------------------------
# Test 9: one failing day does not block newer days; exit is non-zero.
# ---------------------------------------------------------------------------
root="$(make_fixture t9)"; state="${state_dir}/t9"
UETS_FAIL_CHECKPOINT_DAY=2026-11-09 run_archive "${root}" "${state}"
assert_rc 1 "Test 9"
assert_retained "${root}" "Test 9" 2026-11-09 2026-11-11 "${UETS_TODAY}"
assert_deleted "${root}" 2026-11-10 "Test 9"
assert_stderr_has "Archive of 2026-11-09 failed" "Test 9"
assert_stderr_has "1 skipped after failure" "Test 9"
echo "Test 9 PASSED: failed day kept and skipped; the next day was still archived; exit non-zero."

# ---------------------------------------------------------------------------
# Test 10: the newest sealed day is kept by default (worker recovery), and
# UETS_KEEP_RECENT_DAYS=0 lets disk pressure consume it.
# ---------------------------------------------------------------------------
root="$(make_fixture t10a)"; state="${state_dir}/t10a"
DISK_THRESHOLD_PERCENT=70 run_archive "${root}" "${state}"
assert_rc 1 "Test 10 (default keep)"
assert_stderr_has "newest 1 kept" "Test 10 (default keep)"
assert_retained "${root}" "Test 10 (default keep)" 2026-11-11 "${UETS_TODAY}"
root="$(make_fixture t10b)"; state="${state_dir}/t10b"
UETS_KEEP_RECENT_DAYS=0 DISK_THRESHOLD_PERCENT=70 run_archive "${root}" "${state}"
assert_rc 0 "Test 10 (keep 0)"
for day in "${SEALED_DAYS[@]}"; do assert_deleted "${root}" "${day}" "Test 10 (keep 0)"; done
assert_retained "${root}" "Test 10 (keep 0)" "${UETS_TODAY}"
echo "Test 10 PASSED: newest sealed day kept by default; keep=0 archives it under pressure; today always kept."

# ---------------------------------------------------------------------------
# Test 11: below threshold is a no-op; an explicit date archives regardless of
# the threshold; interrupted deletes are cleaned up.
# ---------------------------------------------------------------------------
root="$(make_fixture t11)"; state="${state_dir}/t11"
mkdir -p "${root}/.archived-2026-11-01.123/trades"
touch "${root}/.archived-2026-11-01.123/trades/data"
DF_FORCE_PERCENT=10 run_archive "${root}" "${state}"
assert_rc 0 "Test 11 (below threshold)"
assert_retained "${root}" "Test 11 (below threshold)" "${SEALED_DAYS[@]}" "${UETS_TODAY}"
assert_no_leftovers "${root}" "Test 11 (cleanup of interrupted delete)"
DF_FORCE_PERCENT=10 run_archive "${root}" "${state}" 2026-11-09
assert_rc 0 "Test 11 (explicit date)"
assert_deleted "${root}" 2026-11-09 "Test 11 (explicit date)"
assert_retained "${root}" "Test 11 (explicit date)" 2026-11-10 2026-11-11 "${UETS_TODAY}"
echo "Test 11 PASSED: below threshold is a no-op; explicit date archives; stale retired dirs removed."

# ---------------------------------------------------------------------------
# Test 12: not enough temp space fails the day before touching anything.
# ---------------------------------------------------------------------------
root="$(make_fixture t12)"; state="${state_dir}/t12"
DF_TMP_AVAIL_KB=1 run_archive "${root}" "${state}"
assert_rc_nonzero "Test 12"
assert_stderr_has "insufficient temporary space" "Test 12"
assert_retained "${root}" "Test 12" "${SEALED_DAYS[@]}" "${UETS_TODAY}"
[[ "$(upload_count "${state}")" == "0" ]] || die "Test 12: nothing may be uploaded"
echo "Test 12 PASSED: insufficient temp space retains all days."

echo "ALL TESTS PASSED"