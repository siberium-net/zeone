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
# [MULTI-ARCH BUILD]
#   docker buildx build --platform linux/amd64,linux/arm64 -t zeone:latest .
#
# ============================================================================

# ============================================================================
# Stage 1: Builder - Compile dependencies with native extensions
# ============================================================================
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

# [CRITICAL] Install build dependencies for C extensions
# Required for: aiohttp (yarl, frozenlist), pynacl (libsodium)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libsodium-dev \
    && rm -rf /var/lib/apt/lists/*

# [CRITICAL] Upgrade pip, setuptools, wheel FIRST
# This prevents "SetuptoolsDeprecationWarning" and build failures
RUN python -m pip install --no-cache-dir --upgrade \
    pip>=24.0 \
    setuptools>=68.0 \
    wheel>=0.42.0

# Copy only requirements first (for caching)
COPY requirements/core.txt /build/requirements.txt

# Install Python dependencies to /install prefix
# Using --prefix allows clean copy to runtime stage
RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.10-slim-bookworm AS runtime

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    ZEONE_DATA_DIR=/app/data \
    ZEONE_LOG_DIR=/app/logs \
    # Disable pip version check (not needed in container)
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy pre-compiled packages from builder
COPY --from=builder /install /usr/local

# Install ONLY runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libraries for native extensions
    libssl3 \
    libffi8 \
    libsodium23 \
    # Networking tools (useful for debugging)
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 zeone

# Create data directories with proper permissions
RUN mkdir -p /app/data /app/logs && \
    chown -R zeone:zeone /app

# Copy application code
COPY --chown=zeone:zeone . /app/

# Remove unnecessary files from image
RUN rm -rf \
    /app/.git \
    /app/.github \
    /app/tests \
    /app/docs \
    /app/contracts/*.sol \
    /app/build \
    /app/dist \
    /app/*.spec \
    /app/.venv \
    /app/venv \
    /app/__pycache__ \
    /app/*/__pycache__ \
    /app/*/*/__pycache__

# Switch to non-root user
USER zeone

# Volumes for data persistence
VOLUME ["/app/data", "/app/logs"]

# Expose ports
# 8468: P2P Network (TCP/UDP)
# 8080: WebUI (HTTP)
# 1080: SOCKS5 Proxy
EXPOSE 8468 8080 1080

# Health check - verify P2P port is listening
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost',8468)); s.close()" || exit 1

# OpenContainer Image labels (OCI standard)
LABEL org.opencontainers.image.title="ZEONE Node" \
      org.opencontainers.image.description="Decentralized P2P Network Node with AI, VPN, Storage, Economy" \
      org.opencontainers.image.vendor="Siberium" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/siberium-net/zeone" \
      org.opencontainers.image.documentation="https://docs.zeone.network"

# Entry point
ENTRYPOINT ["python", "main.py"]
CMD ["--port", "8468"]


# ============================================================================
# Stage 3: Development Image (optional)
# ============================================================================
FROM runtime AS development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    ipython \
    && apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

USER zeone

# Override for development - don't auto-start
CMD ["--help"]
