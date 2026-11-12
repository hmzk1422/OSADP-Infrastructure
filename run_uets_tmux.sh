#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${UETS_TMUX_SESSION:-uets}"
UETS_HOME="${UETS_HOME:-${HOME}/uets-data}"
UETS_LIVE_DB="${UETS_LIVE_DB:-${UETS_HOME}/live_db}"
UETS_BIN="${UETS_BIN:-${ROOT}/bin/uets}"
PANE="${SESSION}:0.0"

export UETS_HOME UETS_LIVE_DB UETS_BIN

command -v tmux >/dev/null || {
    echo "tmux is required to provide the interactive UETS terminal." >&2
    exit 1
}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "UETS tmux session '${SESSION}' already exists." >&2
    exit 1
fi

printf -v launch_command '%q ' "${ROOT}/restart_uets.sh" "$@"
tmux new-session -d -s "${SESSION}" -c "${ROOT}" "sleep infinity"
tmux set-window-option -t "${PANE}" remain-on-exit on
tmux respawn-pane -k -t "${PANE}" "exec ${launch_command}"

stop_session() {
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux send-keys -t "${PANE}" C-c
        local deadline=$((SECONDS + 30))
        while (( SECONDS < deadline )); do
            local pane_state
            pane_state="$(tmux display-message -p -t "${PANE}" '#{pane_dead}')"
            if [[ "${pane_state}" == "1" ]]; then
                break
            fi
            sleep 1
        done
        tmux kill-session -t "${SESSION}" 2>/dev/null || true
    fi
}
trap stop_session EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while tmux has-session -t "${SESSION}" 2>/dev/null; do
    pane_state="$(tmux display-message -p -t "${PANE}" '#{pane_dead}')"
    archive_marker="${UETS_HOME}/run/archive.pending"
    if [[ "${pane_state}" == "0" && -f "${archive_marker}" ]]; then
        read -r archive_pid _ <"${archive_marker}" || true
        if [[ ! "${archive_pid:-}" =~ ^[0-9]+$ ]] ||
           ! kill -0 "${archive_pid}" 2>/dev/null; then
            rm -f -- "${archive_marker}"
        fi
    fi
    if [[ "${pane_state}" == "1" ]]; then
        if [[ -f "${archive_marker}" ]]; then
            read -r archive_pid archive_session <"${archive_marker}" || true
            if [[ "${archive_pid}" =~ ^[0-9]+$ ]] && kill -0 "${archive_pid}" 2>/dev/null; then
                sleep 1
                continue
            fi
            if [[ "${archive_session:-}" =~ ^[A-Za-z0-9._-]+$ &&
                  "${archive_session}" != "." && "${archive_session}" != ".." ]]; then
                session_tmp="${UETS_LIVE_DB}/current_session.archive.$$"
                printf '%s\n' "${archive_session}" >"${session_tmp}"
                mv -f -- "${session_tmp}" "${UETS_LIVE_DB}/current_session"
                rm -f -- "${archive_marker}"
                tmux respawn-pane -k -t "${PANE}"
                continue
            fi
            rm -f -- "${archive_marker}"
        fi
        pane_status="$(tmux display-message -p -t "${PANE}" '#{pane_dead_status}')"
        tmux kill-session -t "${SESSION}" 2>/dev/null || true
        exit "${pane_status:-1}"
    fi
    sleep 1
done
