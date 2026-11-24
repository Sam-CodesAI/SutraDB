# ─────────────────────────────────────────────────────────────────────────────
# SutraDB v5.0 Industrial Hardened Multi-Stage Production Dockerfile
# Conforms strictly to supply-chain hardening standards:
# - Multi-stage isolated compilation (no compilers or protoc in runner image)
# - Unprivileged non-root service execution (UID 10001:10001)
# - Dual binary gRPC (50051) + REST/Bento Web UI (8765) runtime
# - Minimal attack surface based on python:3.12-slim
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
    protobuf-compiler \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

RUN pip install --no-cache-dir --upgrade pip maturin

WORKDIR /build

COPY Cargo.toml Cargo.lock pyproject.toml README.md LICENSE build.rs ./
COPY proto ./proto
COPY src ./src
COPY sutradb ./sutradb

# Build standalone high-throughput binary gRPC server with release profile
RUN cargo build --release --bin sutradb-server && \
    mkdir -p /binaries && \
    cp target/release/sutradb-server /binaries/sutradb-server && \
    maturin build --release --out /wheels && \
    cargo clean

# STAGE 2: Hardened Production Runtime
FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SUTRA_MODE=all \
    SUTRA_HOST=0.0.0.0 \
    SUTRA_GRPC_PORT=50051 \
    SUTRA_REST_PORT=8765 \
    SUTRA_DATA_DIR=/data \
    PATH="/usr/local/bin:${PATH}"

# Create unprivileged non-root user (UID 10001)
RUN groupadd -g 10001 nonroot && \
    useradd -u 10001 -g nonroot -s /bin/bash -m nonroot

WORKDIR /app

# Copy compiled binary gRPC server from builder stage
COPY --from=builder /binaries/sutradb-server /usr/local/bin/sutradb-server

# Copy built wheel from builder stage and install runtime dependencies
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Copy operational scripts
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY sutra-healthcheck.py /usr/local/bin/sutra-healthcheck.py
RUN chmod +x /usr/local/bin/sutradb-server /usr/local/bin/docker-entrypoint.sh /usr/local/bin/sutra-healthcheck.py

# Setup persistent storage directory with unprivileged ownership
RUN mkdir -p /data /app && chown -R nonroot:nonroot /data /app

# Switch to unprivileged non-root user
USER 10001:10001

# Expose gRPC port (50051) and REST/UI port (8765)
EXPOSE 50051 8765

VOLUME ["/data"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 /usr/local/bin/sutra-healthcheck.py || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["all"]
