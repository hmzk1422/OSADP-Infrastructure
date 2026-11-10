#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${UETS_BIN:-}" ]]; then
    UETS_BIN="${UETS_BIN}"
elif [[ -x "${ROOT}/bin/uets" ]]; then
    UETS_BIN="${ROOT}/bin/uets"
elif [[ -x "${ROOT}/target/release/dev_test" ]]; then
    UETS_BIN="${ROOT}/target/release/dev_test"
else
    echo "UETS executable not found; set UETS_BIN or build target/release/dev_test" >&2
    exit 127
fi
if [[ "${UETS_BIN}" != /* ]]; then
    UETS_BIN="${ROOT}/${UETS_BIN}"
fi
if [[ ! -x "${UETS_BIN}" ]]; then
    echo "UETS executable is missing or not executable: ${UETS_BIN}" >&2
    exit 127
fi

export UETS_HOME="${UETS_HOME:-${ROOT}}"
LOG_DIR="${UETS_HOME}/logs/uets"
RUN_DIR="${UETS_HOME}/run"
mkdir -p "${LOG_DIR}" "${RUN_DIR}"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
RUN_LOG="${LOG_DIR}/watchdog-run.log"
RUN_PID_FILE="${RUN_DIR}/uets.pid"
LOCK_FILE="${RUN_DIR}/watchdog.lock"
MAX_CONSECUTIVE_FAILURES="${UETS_MAX_RESTARTS:-5}"
RESTART_DELAY="${UETS_RESTART_DELAY_SECONDS:-5}"
STABLE_SECONDS="${UETS_STABLE_SECONDS:-3600}"
consecutive_failures=0
current_pid=""

for setting in MAX_CONSECUTIVE_FAILURES RESTART_DELAY STABLE_SECONDS; do
    value="${!setting}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "${setting} must be a positive integer" >&2
        exit 2
    fi
done

command -v flock >/dev/null || {
    echo "flock is required to prevent duplicate UETS watchdogs." >&2
    exit 1
}
exec 8>"${LOCK_FILE}"
flock -n 8 || {
    echo "Another UETS watchdog already owns ${LOCK_FILE}." >&2
    exit 1
}

log() {
    local message="$1"
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${message}" | tee -a "${WATCHDOG_LOG}" >&2
}

oom_kill_count() {
    local events_file="${UETS_CGROUP_MEMORY_EVENTS:-/sys/fs/cgroup/memory.events}"
    if [[ -r "${events_file}" ]]; then
        awk '$1 == "oom_kill" { print $2 }' "${events_file}"
    else
        printf '0\n'
    fi
}

cleanup() {
    if [[ -n "${current_pid}" ]] && kill -0 "${current_pid}" 2>/dev/null; then
        kill -TERM "${current_pid}" 2>/dev/null || true
        wait "${current_pid}" 2>/dev/null || true
    fi
    rm -f -- "${RUN_PID_FILE}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while true; do
    started_at="$(date +%s)"
    oom_before="$(oom_kill_count)"
    log "starting ${UETS_BIN} run --reuse-rocksdb $*"

    "${UETS_BIN}" run --reuse-rocksdb "$@" \
        <&0 2> >(tee -a "${RUN_LOG}" >&2) &
    current_pid=$!
    printf '%s\n' "${current_pid}" >"${RUN_PID_FILE}"
    if wait "${current_pid}"; then
        status=0
    else
        status=$?
    fi
    current_pid=""
    rm -f -- "${RUN_PID_FILE}"

    finished_at="$(date +%s)"
    runtime_seconds=$((finished_at - started_at))
    oom_after="$(oom_kill_count)"
    if (( runtime_seconds >= STABLE_SECONDS )); then
        consecutive_failures=0
    fi

    if (( status == 0 )); then
        log "UETS exited normally (exit=0); treating this as an intentional quit"
        exit 0
    fi

    consecutive_failures=$((consecutive_failures + 1))
    cause="exit=${status}"
    if (( status == 137 )); then
        cause+=" (SIGKILL; possible OOM)"
    fi
    if [[ "${oom_before}" =~ ^[0-9]+$ && "${oom_after}" =~ ^[0-9]+$ ]] &&
       (( oom_after > oom_before )); then
        cause+="; cgroup oom_kill count increased ${oom_before}->${oom_after}"
    fi
    log "unexpected UETS exit ${cause}; runtime_seconds=${runtime_seconds} consecutive_failures=${consecutive_failures}; restarting with the same RocksDB session"

    if (( consecutive_failures >= MAX_CONSECUTIVE_FAILURES )); then
        log "restart limit reached; stopping to avoid an uncontrolled crash loop"
        exit "${status}"
    fi
    sleep "${RESTART_DELAY}"
done
