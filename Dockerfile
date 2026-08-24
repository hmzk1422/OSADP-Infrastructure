# =============================================================================
# OSADP-Core — Runtime container (Oracle Cloud / x86 & ARM edge servers)
# =============================================================================
# Consumes the pre-built manylinux wheels produced by the CI `wheels` job,
# downloaded into dist/ before this build. Multi-arch: linux/amd64 + arm64.
#
# NOTE: containers cannot run on Android/Termux (kernel restrictions). For
# on-device installs see docs/TERMUX.md — proot-distro + pip wheel directly.
# The image supports live-trade / live-test / doctor; `backtest` additionally
# needs the heavy research stack (see requirements.txt) and is intentionally
# kept out of this slim runtime image.
# =============================================================================
FROM python:3.11-slim

LABEL org.opencontainers.image.title="OSADP-Core" \
      org.opencontainers.image.description="Hybrid Python/Rust quant framework — unified_runner runtime" \
      org.opencontainers.image.source="https://github.com/hmzak1422/intraday-momentum-composite"

WORKDIR /app

# Copy the pre-compiled wheels provided by GitHub Actions
COPY dist/ ./dist/

# Install the correct wheel based on the target architecture
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      pip install --no-cache-dir dist/*manylinux*x86_64.whl; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
      pip install --no-cache-dir dist/*aarch64.whl; \
    else \
      echo "Unsupported architecture: $(uname -m)" && exit 1; \
    fi && rm -rf dist/

# The runner CLI itself (not shipped inside the wheel)
COPY unified_runner.py ./

# Non-root runtime; RocksDB store lives on a writable volume
RUN useradd --create-home --shell /usr/sbin/nologin trader \
    && mkdir -p /app/data \
    && chown -R trader:trader /app
USER trader
VOLUME ["/app/data"]

# Make the container act as your CLI
ENTRYPOINT ["python", "unified_runner.py"]