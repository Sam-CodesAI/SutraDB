#!/bin/bash
set -eo pipefail

# SutraDB Production Entrypoint Supervisor
# Supports multi-mode orchestration (all, grpc, rest) with graceful signal trapping.

export SUTRA_MODE="${SUTRA_MODE:-all}"
export SUTRA_HOST="${SUTRA_HOST:-0.0.0.0}"
export SUTRA_GRPC_PORT="${SUTRA_GRPC_PORT:-50051}"
export SUTRA_REST_PORT="${SUTRA_REST_PORT:-8765}"
export SUTRA_DATA_DIR="${SUTRA_DATA_DIR:-/data}"

mkdir -p "$SUTRA_DATA_DIR" 2>/dev/null || true

# Handle passed CLI arguments
if [ "$#" -gt 0 ]; then
    if [ "$1" = "all" ] || [ "$1" = "grpc" ] || [ "$1" = "rest" ]; then
        SUTRA_MODE="$1"
        shift
    elif [ "$1" != "sutradb" ] && [ "$1" != "sutradb-server" ]; then
        exec "$@"
    fi
fi

echo "=========================================================="
echo "  SutraDB v5.0 Production Daemon Manager"
echo "  Mode       : ${SUTRA_MODE}"
echo "  Storage    : ${SUTRA_DATA_DIR}"
echo "  UID / GID  : $(id -u):$(id -g)"
echo "=========================================================="

if [ "$SUTRA_MODE" = "grpc" ]; then
    echo "[SutraDB] Launching standalone gRPC Engine on ${SUTRA_HOST}:${SUTRA_GRPC_PORT}..."
    exec sutradb-server --host "$SUTRA_HOST" --port "$SUTRA_GRPC_PORT" --data-dir "$SUTRA_DATA_DIR" "$@"
fi

if [ "$SUTRA_MODE" = "rest" ]; then
    echo "[SutraDB] Launching REST API & Bento UI on ${SUTRA_HOST}:${SUTRA_REST_PORT}..."
    exec python3 -m sutradb.server "$SUTRA_REST_PORT" "$SUTRA_DATA_DIR" "$@"
fi

if [ "$SUTRA_MODE" = "all" ]; then
    echo "[SutraDB] Launching Dual Daemons: gRPC (${SUTRA_GRPC_PORT}) & REST (${SUTRA_REST_PORT})..."
    
    sutradb-server --host "$SUTRA_HOST" --port "$SUTRA_GRPC_PORT" --data-dir "$SUTRA_DATA_DIR" &
    GRPC_PID=$!
    
    python3 -m sutradb.server "$SUTRA_REST_PORT" "$SUTRA_DATA_DIR" &
    REST_PID=$!
    
    cleanup() {
        echo ""
        echo "[SutraDB] Termination signal caught. Initiating graceful shutdown..."
        kill -TERM "$GRPC_PID" 2>/dev/null || true
        kill -TERM "$REST_PID" 2>/dev/null || true
        wait "$GRPC_PID" 2>/dev/null || true
        wait "$REST_PID" 2>/dev/null || true
        echo "[SutraDB] All daemons halted cleanly."
        exit 0
    }
    
    trap cleanup SIGTERM SIGINT SIGHUP
    
    wait -n "$GRPC_PID" "$REST_PID"
    EXIT_STATUS=$?
    echo "[SutraDB] A daemon process exited unexpectedly with code $EXIT_STATUS."
    cleanup
fi
