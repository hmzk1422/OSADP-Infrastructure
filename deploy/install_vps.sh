#!/usr/bin/env bash
set -Eeuo pipefail

readonly APP_NAME="uets-linux-x86_64"
readonly APP_ROOT="/opt/uets"
readonly APP_DIR="${APP_ROOT}/${APP_NAME}"
readonly CONFIG_DIR="/etc/uets"
readonly STATE_DIR="/var/lib/uets"
readonly SERVICE_NAME="uets-tmux"
readonly RELEASE_BUNDLE_URL="https://github.com/hmzk1422/osadp-core/releases/latest/download/uets-linux-x86_64.tar.gz"

bundle=""
strategy="ofmr@btcusdt"
archive_uri=""
aws_region="ap-southeast-2"
update_only=0

usage() {
    cat <<'USAGE'
Install UETS on Ubuntu/Debian or Amazon Linux 2023.

Usage:
  sudo bash install_vps.sh [options]

Options:
  --bundle VALUE       Use a local or HTTPS UETS bundle instead of latest release
  --strategy VALUE     Strategy@asset to start (default: ofmr@btcusdt)
  --archive-uri VALUE  Optional s3://bucket/prefix for verified archives
  --region VALUE       AWS region for S3 archives (default: ap-southeast-2)
  --update             Install the latest release if a newer version is available
  -h, --help           Show this help

With no --bundle, downloads the latest release automatically. On first install
it configures the system and starts UETS. Installed updates are checked daily.
Runtime data remains under /var/lib/uets across UETS restarts and updates.
USAGE
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --bundle|--strategy|--archive-uri|--region)
            (($# >= 2)) || die "missing value for $1"
            case "$1" in
                --bundle) bundle="$2" ;;
                --strategy) strategy="$2" ;;
                --archive-uri) archive_uri="$2" ;;
                --region) aws_region="$2" ;;
            esac
            shift 2
            ;;
        --update)
            update_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if ((update_only)) && [[ -n "${bundle}" || -n "${archive_uri}" ]]; then
    die "--update cannot be combined with --bundle or --archive-uri"
fi
[[ "${strategy}" =~ ^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$ ]] ||
    die "invalid strategy selector '${strategy}'"
[[ "${aws_region}" =~ ^[A-Za-z0-9-]+$ ]] || die "invalid AWS region"
if [[ -n "${archive_uri}" && ! "${archive_uri}" =~ ^s3://[A-Za-z0-9._-]+(/[A-Za-z0-9._/-]+)?$ ]]; then
    die "archive URI must use s3://bucket/prefix"
fi

[[ "${EUID}" -eq 0 ]] || die "run this installer with sudo or as root"
[[ "$(uname -m)" == "x86_64" ]] || die "the supplied UETS bundle requires x86_64"
command -v systemctl >/dev/null 2>&1 || die "systemd is required"
[[ -r /etc/os-release ]] || die "cannot identify the Linux distribution"

# shellcheck disable=SC1091
source /etc/os-release
case "${ID:-}" in
    ubuntu|debian)
        if ((!update_only)); then
            export DEBIAN_FRONTEND=noninteractive
            apt-get update
            packages=(ca-certificates curl tar tmux util-linux zstd)
            if [[ -n "${archive_uri}" || -f "${CONFIG_DIR}/archive.env" ]]; then
                packages+=(awscli)
            fi
            apt-get install -y "${packages[@]}"
        fi
        ;;
    amzn)
        if ((!update_only)); then
            packages=(ca-certificates curl tar tmux util-linux zstd)
            if [[ -n "${archive_uri}" || -f "${CONFIG_DIR}/archive.env" ]]; then
                packages+=(awscli-2)
            fi
            dnf install -y "${packages[@]}"
        fi
        ;;
    *)
        die "unsupported distribution '${ID:-unknown}'; use Ubuntu, Debian, or Amazon Linux 2023"
        ;;
esac

if [[ -n "${bundle}" && "${bundle}" != https://* && (! -r "${bundle}" || ! -f "${bundle}") ]]; then
    die "bundle file is not readable: ${bundle}"
fi

stage="$(mktemp -d)"
old_app=""
service_was_active=0
cleanup() {
    rm -rf -- "${stage}"
}
trap cleanup EXIT

if [[ -z "${bundle}" || "${bundle}" == https://* ]]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to download the bundle"
    if [[ -z "${bundle}" ]]; then
        bundle="${RELEASE_BUNDLE_URL}"
    fi
    curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
        "${bundle}" --output "${stage}/${APP_NAME}.tar.gz"
    if [[ "${bundle}" == "${RELEASE_BUNDLE_URL}" ]]; then
        curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
            "${RELEASE_BUNDLE_URL}.sha256" --output "${stage}/${APP_NAME}.tar.gz.sha256"
        (cd "${stage}" && sha256sum --check "${APP_NAME}.tar.gz.sha256") ||
            die "release bundle SHA-256 verification failed"
    fi
    bundle="${stage}/${APP_NAME}.tar.gz"
fi

tar -tzf "${bundle}" >/dev/null || die "bundle is not a valid .tar.gz archive"
tar -xzf "${bundle}" -C "${stage}"
new_app="${stage}/${APP_NAME}"
[[ -x "${new_app}/bin/uets" ]] || die "bundle must contain ${APP_NAME}/bin/uets"
[[ -s "${new_app}/VERSION" ]] || die "bundle must contain a VERSION file"
for required in restart_uets.sh run_uets_tmux.sh cloud.sh \
    deploy/systemd/uets-tmux.service \
    deploy/systemd/uets-cloud-archive.service \
    deploy/systemd/uets-cloud-archive.timer \
    deploy/systemd/uets-update.service \
    deploy/systemd/uets-update.timer; do
    [[ -f "${new_app}/${required}" ]] || die "bundle is missing ${required}"
done
chmod 0755 "${new_app}/restart_uets.sh" "${new_app}/run_uets_tmux.sh" "${new_app}/cloud.sh"
"${new_app}/bin/uets" help >/dev/null || die "UETS binary failed its help check"
new_version="$(<"${new_app}/VERSION")"
[[ "${new_version}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid bundle version '${new_version}'"

if ((update_only)); then
    [[ -d "${APP_DIR}" ]] || die "UETS is not installed; run install_vps.sh without --update first"
    current_version=""
    if [[ -r "${APP_DIR}/VERSION" ]]; then
        current_version="$(<"${APP_DIR}/VERSION")"
    fi
    if [[ "${current_version}" == "${new_version}" ]]; then
        printf 'UETS is already up to date (%s).\n' "${new_version}"
        exit 0
    fi
    printf 'Updating UETS: %s -> %s\n' "${current_version:-unknown}" "${new_version}"
fi

if ! getent group uets >/dev/null; then
    groupadd --system uets
fi
if ! id uets >/dev/null 2>&1; then
    useradd --system --gid uets --home-dir "${STATE_DIR}" --create-home --shell /bin/bash uets
fi

install -d -o uets -g uets -m 0750 \
    "${STATE_DIR}" "${STATE_DIR}/live_db" "${STATE_DIR}/logs/uets" \
    "${STATE_DIR}/run" "${APP_ROOT}"
install -d -o uets -g uets -m 0750 /var/tmp/uets-archive-tmp
install -d -o root -g root -m 0755 "${CONFIG_DIR}"

if [[ -d "${APP_DIR}/uets/config" ]]; then
    for config_file in execution.yaml ofmr.yaml; do
        if [[ -f "${APP_DIR}/uets/config/${config_file}" ]]; then
            cp -p -- "${APP_DIR}/uets/config/${config_file}" \
                "${new_app}/uets/config/${config_file}"
        fi
    done
fi

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    service_was_active=1
    systemctl stop "${SERVICE_NAME}"
fi

if [[ -d "${APP_DIR}" ]]; then
    old_app="${APP_DIR}.previous.$(date -u +%Y%m%d%H%M%S).$$"
    mv -- "${APP_DIR}" "${old_app}"
fi
if ! mv -- "${new_app}" "${APP_DIR}"; then
    if [[ -n "${old_app}" ]]; then
        mv -- "${old_app}" "${APP_DIR}"
    fi
    die "could not install the UETS bundle"
fi
chown -R uets:uets "${APP_DIR}"

if ((!update_only)); then
    printf 'UETS_STRATEGY=%s\n' "${strategy}" >"${CONFIG_DIR}/uets.env"
    chmod 0644 "${CONFIG_DIR}/uets.env"
fi
install -o root -g root -m 0644 \
    "${APP_DIR}/deploy/systemd/uets-tmux.service" \
    /etc/systemd/system/uets-tmux.service
install -o root -g root -m 0644 \
    "${APP_DIR}/deploy/systemd/uets-cloud-archive.service" \
    /etc/systemd/system/uets-cloud-archive.service
if [[ -n "${archive_uri}" ]]; then
    cat >"${CONFIG_DIR}/archive.env" <<EOF
AWS_ARCHIVE_URI=${archive_uri}
AWS_REGION=${aws_region}
DISK_THRESHOLD_PERCENT=90
ARCHIVE_TMP_DIR=/var/tmp/uets-archive-tmp
UETS_HOME=${STATE_DIR}
UETS_LIVE_DB=${STATE_DIR}/live_db
UETS_BIN=${APP_DIR}/bin/uets
UETS_TMUX_SESSION=uets
EOF
    chmod 0600 "${CONFIG_DIR}/archive.env"
fi

systemctl daemon-reload
install -o root -g root -m 0644 \
    "${APP_DIR}/deploy/systemd/uets-cloud-archive.timer" \
    /etc/systemd/system/uets-cloud-archive.timer
install -o root -g root -m 0644 \
    "${APP_DIR}/deploy/systemd/uets-update.service" \
    /etc/systemd/system/uets-update.service
install -o root -g root -m 0644 \
    "${APP_DIR}/deploy/systemd/uets-update.timer" \
    /etc/systemd/system/uets-update.timer
systemctl daemon-reload

if ((!update_only)) && [[ -f "${CONFIG_DIR}/archive.env" ]]; then
    systemctl enable --now uets-cloud-archive.timer
fi
if ((!update_only)); then
    systemctl enable --now uets-update.timer
    start_service=1
else
    start_service="${service_was_active}"
fi
if ((start_service)); then
    if ((!update_only)); then
        service_start_command=(systemctl enable --now "${SERVICE_NAME}")
    else
        service_start_command=(systemctl start "${SERVICE_NAME}")
    fi
fi
if ((start_service)) && ! "${service_start_command[@]}"; then
    if [[ -n "${old_app}" ]]; then
        systemctl stop "${SERVICE_NAME}" >/dev/null 2>&1 || true
        failed_app="${APP_DIR}.failed.$(date -u +%Y%m%d%H%M%S)"
        mv -- "${APP_DIR}" "${failed_app}"
        mv -- "${old_app}" "${APP_DIR}"
        rm -rf -- "${failed_app}"
        for unit in uets-tmux uets-cloud-archive uets-update; do
            previous_unit="${APP_DIR}/deploy/systemd/${unit}.service"
            if [[ -f "${previous_unit}" ]]; then
                install -o root -g root -m 0644 "${previous_unit}" \
                    "/etc/systemd/system/${unit}.service"
            fi
        done
        if [[ -f "${APP_DIR}/deploy/systemd/uets-update.timer" ]]; then
            install -o root -g root -m 0644 \
                "${APP_DIR}/deploy/systemd/uets-update.timer" \
                /etc/systemd/system/uets-update.timer
        fi
        if [[ -f "${APP_DIR}/deploy/systemd/uets-cloud-archive.timer" ]]; then
            install -o root -g root -m 0644 \
                "${APP_DIR}/deploy/systemd/uets-cloud-archive.timer" \
                /etc/systemd/system/uets-cloud-archive.timer
        fi
        systemctl daemon-reload
        if ((service_was_active)); then
            systemctl start "${SERVICE_NAME}" || true
        fi
    fi
    die "UETS service failed to start; inspect: journalctl -u ${SERVICE_NAME} -n 100 --no-pager"
fi
if [[ -n "${old_app}" ]]; then
    rm -rf -- "${old_app}"
fi

printf 'UETS %s installed.\n' "${new_version}"
if ((start_service)); then
    printf 'UETS service is running.\n'
else
    printf 'UETS service is installed but was not started.\n'
fi
printf 'Attach to the TUI: sudo -u uets tmux attach -t uets\n'
printf 'Service status:     systemctl status %s --no-pager\n' "${SERVICE_NAME}"
printf 'Automatic updates:  systemctl status uets-update.timer --no-pager\n'
if [[ -f "${CONFIG_DIR}/archive.env" ]]; then
    printf 'Archive timer:      systemctl status uets-cloud-archive.timer --no-pager\n'
else
    printf 'S3 archives are not configured. Re-run with --archive-uri s3://bucket/prefix to enable them.\n'
fi
