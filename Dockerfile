# ─────────────────────────────────────────────────────────────────────────────
# SutraDB v5.5.0 Industrial Hardened Production Dockerfile
# Multi-stage build: Rust builder compiles the gRPC binary + Python wheel,
# then copies only artifacts into the production runner.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Rust builder ────────────────────────────────────────────────────
FROM rust:1-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages maturin

WORKDIR /build
COPY Cargo.toml Cargo.lock build.rs ./
COPY src/ src/
COPY proto/ proto/

# Build the standalone gRPC server binary (release, without Python/PyO3 feature)
RUN cargo build --release --no-default-features --bin sutradb-server

# Build the Python extension wheel
COPY pyproject.toml README.md ./
COPY sutradb/ sutradb/
RUN maturin build --release --out /wheels

# ── Stage 2: Production runner ───────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runner

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

# Copy binary from builder
COPY --from=builder /build/target/release/sutradb-server /usr/local/bin/sutradb-server

# Copy and install wheel from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/sutradb_core*.whl && \
    rm -rf /wheels

# Copy operational scripts
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY sutra-healthcheck.py /usr/local/bin/sutra-healthcheck.py
RUN chmod +x /usr/local/bin/sutradb-server /usr/local/bin/docker-entrypoint.sh /usr/local/bin/sutra-healthcheck.py

# Setup persistent storage directory with unprivileged ownership
RUN mkdir -p /data /app && chown -R nonroot:nonroot /data /app

# Switch to unprivileged non-root user
USER 10001:10001

# Expose gRPC port (50051) and REST/HTTP Bridge port (8765)
EXPOSE 50051 8765

VOLUME ["/data"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 /usr/local/bin/sutra-healthcheck.py || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["all"]
