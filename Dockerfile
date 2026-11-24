# ─────────────────────────────────────────────────────────────────────────────
# SutraDB v5.1.0 Industrial Hardened Production Dockerfile
# Conforms strictly to supply-chain hardening standards:
# - Unprivileged non-root service execution (UID 10001:10001)
# - No compilers or build toolchains in production runtime
# - Dual binary gRPC (50051) + REST/Workbench (8765) runtime
# - Minimal attack surface based on python:3.12-slim
# ─────────────────────────────────────────────────────────────────────────────

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

# Copy pre-compiled standalone binary gRPC server
COPY dist/bin/sutradb-server /usr/local/bin/sutradb-server

# Copy pre-compiled release wheel and install runtime dependencies
COPY dist/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/sutradb_core*.whl && \
    rm -rf /wheels

# Copy operational scripts
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY sutra-healthcheck.py /usr/local/bin/sutra-healthcheck.py
RUN chmod +x /usr/local/bin/sutradb-server /usr/local/bin/docker-entrypoint.sh /usr/local/bin/sutra-healthcheck.py

# Setup persistent storage directory with unprivileged ownership
RUN mkdir -p /data /app && chown -R nonroot:nonroot /data /app

# Switch to unprivileged non-root user
USER 10001:10001

# Expose gRPC port (50051) and REST/Workbench port (8765)
EXPOSE 50051 8765

VOLUME ["/data"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 /usr/local/bin/sutra-healthcheck.py || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["all"]
