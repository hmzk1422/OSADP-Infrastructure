# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy the pre-compiled wheels provided by GitHub Actions
COPY dist/ ./dist/

# Install the correct wheel based on the target architecture
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      pip install dist/*x86_64.whl; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
      pip install dist/*aarch64.whl; \
    else \
      echo "Unsupported architecture" && exit 1; \
    fi

# Copy your strategy CLI script and config
COPY unified_runner.py ./
# COPY config.json ./ (If you have other files needed)

# Make the container act as your CLI
ENTRYPOINT ["python", "unified_runner.py"]