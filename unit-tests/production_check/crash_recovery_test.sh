#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
command -v docker >/dev/null || {
    echo "Docker is required for the Linux OOM crash-recovery test." >&2
    exit 1
}

image="${UETS_CRASH_TEST_IMAGE:-ubuntu:24.04}"
memory_limit="${UETS_CRASH_TEST_MEMORY:-256m}"
container_name="uets-crash-check-${RANDOM}-$$"
state_dir="$(mktemp -d)"
cleanup() {
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    rm -rf -- "${state_dir}"
}
trap cleanup EXIT

cat >"${state_dir}/crash-fixture.sh" <<'FIXTURE'
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == "run" ]] || exit 64
original_args="$*"
shift
[[ "${1:-}" == "--reuse-rocksdb" ]] || {
    echo "restart did not request RocksDB reuse" >&2
    exit 65
}
shift

db_root="${UETS_LIVE_DB:?UETS_LIVE_DB is required}"
mkdir -p "${db_root}"
attempt_file="${db_root}/fixture-attempts"
attempt=0
if [[ -f "${attempt_file}" ]]; then
    attempt="$(<"${attempt_file}")"
fi
attempt=$((attempt + 1))
printf '%s\n' "${attempt}" >"${attempt_file}"
printf '%s\n' "${original_args}" >>"${db_root}/launch-args.log"

if (( attempt == 1 )); then
    printf 'fixture-session\n' >"${db_root}/current_session"
    echo "fixture simulating an out-of-memory crash"
    payload="$(head -c 536870912 /dev/zero | tr '\000' 'x')"
    printf '%s\n' "${#payload}" >/dev/null
    exit 66
fi

[[ "$(<"${db_root}/current_session")" == "fixture-session" ]] || {
    echo "previous session marker was not reused" >&2
    exit 67
}
grep -q -- '--reuse-rocksdb' "${db_root}/launch-args.log" || {
    echo "restart arguments did not contain --reuse-rocksdb" >&2
    exit 68
}
printf 'reseed complete; reused fixture-session\n' >"${db_root}/recovered"
echo "fixture reseed complete"
FIXTURE
chmod +x "${state_dir}/crash-fixture.sh"

docker run --detach --name "${container_name}" \
    --memory="${memory_limit}" --memory-swap="${memory_limit}" \
    --volume "${ROOT}:/workspace:ro" \
    --volume "${state_dir}:/state" \
    --workdir /workspace \
    "${image}" \
    bash -c 'UETS_BIN=/state/crash-fixture.sh UETS_HOME=/state/home UETS_LIVE_DB=/state/live_db UETS_MAX_RESTARTS=3 UETS_RESTART_DELAY_SECONDS=1 UETS_STABLE_SECONDS=60 bash /workspace/restart_uets.sh ofmr@btcusdt'

deadline=$((SECONDS + 90))
while (( SECONDS < deadline )); do
    state="$(docker inspect --format '{{.State.Status}}' "${container_name}")"
    if [[ "${state}" == "exited" ]]; then
        break
    fi
    sleep 1
done

state="$(docker inspect --format '{{.State.Status}}' "${container_name}")"
if [[ "${state}" != "exited" ]]; then
    echo "Watchdog did not complete the simulated OOM/restart test within 90 seconds." >&2
    docker logs "${container_name}" >&2 || true
    exit 1
fi

exit_code="$(docker inspect --format '{{.State.ExitCode}}' "${container_name}")"
if [[ "${exit_code}" != "0" ]]; then
    echo "Crash-recovery container exited with status ${exit_code}." >&2
    docker logs "${container_name}" >&2 || true
    exit 1
fi
if [[ ! -f "${state_dir}/live_db/recovered" ]] ||
   ! grep -q "reseed complete" "${state_dir}/live_db/recovered" ||
   ! grep -Eq "SIGKILL; possible OOM|cgroup oom_kill count increased" \
       "${state_dir}/home/logs/uets/watchdog.log"; then
    echo "The watchdog did not detect the simulated OOM and complete its recovery checks." >&2
    docker logs "${container_name}" >&2 || true
    exit 1
fi

echo "OOM restart, --reuse-rocksdb propagation, and prior-session reseed fixture passed."
