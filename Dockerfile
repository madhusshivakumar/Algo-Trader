FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for numpy/pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install supercronic (cron replacement for containers — no syslog dependency)
# Update the URL and SHA if a newer version is available:
# https://github.com/aptible/supercronic/releases
ARG TARGETARCH
RUN ARCH=$(case "$TARGETARCH" in arm64) echo "arm64" ;; *) echo "amd64" ;; esac) \
    && curl -fsSL -o /usr/local/bin/supercronic \
       "https://github.com/aptible/supercronic/releases/download/v0.2.33/supercronic-linux-${ARCH}" \
    && chmod +x /usr/local/bin/supercronic

# Python dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create non-root user for security
RUN groupadd -r trader && useradd -r -g trader -d /app -s /sbin/nologin trader \
    && mkdir -p logs data db \
    && chown -R trader:trader /app

USER trader

# Default: run the trading engine
CMD ["python", "main.py"]
