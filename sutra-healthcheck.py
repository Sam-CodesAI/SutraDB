#!/usr/bin/env python3
"""
SutraDB Production Health Check Probe.
Inspects SUTRA_MODE and verifies active service endpoints:
- In 'rest' or 'all' mode: verifies HTTP GET http://127.0.0.1:<PORT>/health returns 200 OK.
- In 'grpc' or 'all' mode: verifies TCP socket connection to 127.0.0.1:<GRPC_PORT>.
"""
import os
import socket
import sys
import urllib.request

mode = os.getenv("SUTRA_MODE", "all").lower()
grpc_port = int(os.getenv("SUTRA_GRPC_PORT", "50051"))
rest_port = int(os.getenv("SUTRA_REST_PORT", "8765"))

def check_rest(port: int) -> bool:
    try:
        url = f"http://127.0.0.1:{port}/health"
        req = urllib.request.Request(url, headers={"User-Agent": "SutraDB-HealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception as e:
        sys.stderr.write(f"[HealthCheck] REST failure: {e}\n")
        return False

def check_grpc(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2.0):
            return True
    except Exception as e:
        sys.stderr.write(f"[HealthCheck] gRPC socket failure: {e}\n")
        return False

ok = True
if mode in ("rest", "all"):
    if not check_rest(rest_port):
        ok = False

if mode in ("grpc", "all"):
    if not check_grpc(grpc_port):
        ok = False

if not ok:
    sys.exit(1)

sys.exit(0)
