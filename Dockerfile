FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for numpy/pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install supercronic (cron replacement for containers — no syslog dependency)
# Update the URL and SHA if a newer version is available:
# https://github.com/aptible/supercronic/releases
ARG SUPERCRONIC_URL=https://github.com/aptible/supercronic/releases/download/v0.2.33/supercronic-linux-amd64
RUN curl -fsSLO "$SUPERCRONIC_URL" \
    && chmod +x supercronic-linux-amd64 \
    && mv supercronic-linux-amd64 /usr/local/bin/supercronic

# Python dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create directories that may not exist
RUN mkdir -p logs data

# Default: run the trading engine
CMD ["python", "main.py"]
