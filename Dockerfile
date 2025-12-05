# ============================================================================
# ZEONE Node - Production Docker Image
# ============================================================================
#
# [BUILD]
#   docker build -t zeone:latest .
#
# [RUN]
#   docker run -d --name zeone \
#     -p 8468:8468 \
#     -p 8080:8080 \
#     -v zeone_data:/app/data \
#     zeone:latest
#
# ============================================================================

# Stage 1: Builder
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/core.txt /build/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Runtime
FROM python:3.10-slim-bookworm AS runtime

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ZEONE_DATA_DIR=/app/data \
    ZEONE_LOG_DIR=/app/logs

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 zeone

# Create data directories
RUN mkdir -p /app/data /app/logs && \
    chown -R zeone:zeone /app

# Copy application code
COPY --chown=zeone:zeone . /app/

# Switch to non-root user
USER zeone

# Volumes for data persistence
VOLUME ["/app/data", "/app/logs"]

# Expose ports
# 8468: P2P Network
# 8080: WebUI
# 1080: SOCKS5 Proxy
EXPOSE 8468 8080 1080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost',8468)); s.close()" || exit 1

# Labels
LABEL org.opencontainers.image.title="ZEONE Node" \
      org.opencontainers.image.description="Decentralized P2P Network Node" \
      org.opencontainers.image.vendor="Siberium" \
      org.opencontainers.image.source="https://github.com/siberium-net/zeone"

# Entry point
ENTRYPOINT ["python", "main.py"]
CMD ["--port", "8468"]
