# ─────────────────────────────────────────────────────────────────────────────
# SutraDB v3.0 Industrial Hardened Multi-Stage Dockerfile
# Adheres to strict supply-chain hardening standards:
# - Multi-stage isolated build (no compilers in production runtime)
# - Unprivileged non-root user (UID 10001:10001)
# - Minimal attack surface (python:3.12-slim runtime)
# ─────────────────────────────────────────────────────────────────────────────

# STAGE 1: Compiler & Builder
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

RUN pip install --no-cache-dir --upgrade pip maturin

WORKDIR /build

COPY Cargo.toml Cargo.lock pyproject.toml README.md LICENSE ./
COPY src ./src
COPY sutradb ./sutradb

# Build optimized abi3 release wheel
RUN maturin build --release --out /wheels

# STAGE 2: Hardened Production Runtime
FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SUTRA_DATA_DIR=/data

# Create unprivileged non-root user (UID 10001)
RUN groupadd -g 10001 nonroot && \
    useradd -u 10001 -g nonroot -s /bin/false -m nonroot

WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /wheels /wheels

# Install production dependencies and compiled wheel
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Setup persistent data directory with nonroot ownership
RUN mkdir -p /data /app && chown -R nonroot:nonroot /data /app

USER 10001:10001

EXPOSE 8765

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health').read()" || exit 1

ENTRYPOINT ["python3", "-m", "sutradb.server"]
CMD ["8765", "/data"]
