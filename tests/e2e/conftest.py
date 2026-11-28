"""
Shared pytest fixtures and isolation helpers for SutraDB v5.5.0 E2E test suite.
All tests run opaque-box against public Python SDK, FastAPI REST/UI, Playground,
and sutradb-server (gRPC + loopback HTTP bridge) interfaces with tmp_path and
ephemeral port isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from http.server import HTTPServer
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from fastapi.testclient import TestClient
import httpx
import pytest
import uvicorn

from playground.app import PlaygroundHandler, seed_database
from sutradb import NativeCollection, SutraDB
from sutradb.server import create_app


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_free_port() -> int:
    """Allocate an ephemeral unused TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_sutradb_server_bin() -> Path:
    """Locate the sutradb-server binary without silently skipping missing builds."""
    candidates: List[Path] = []
    env_bin = os.getenv("SUTRA_SERVER_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    which_bin = shutil.which("sutradb-server")
    if which_bin:
        candidates.append(Path(which_bin))
    candidates.extend([
        REPO_ROOT / "target" / "release" / "sutradb-server",
        REPO_ROOT / "target" / "debug" / "sutradb-server",
    ])
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    raise AssertionError(
        "sutradb-server binary not found in SUTRA_SERVER_BIN, PATH, "
        "target/release/sutradb-server, or target/debug/sutradb-server"
    )


def _wait_for_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


@dataclass
class ServerHandle:
    """Managed sutradb-server subprocess handle with gRPC + HTTP bridge metadata."""

    proc: subprocess.Popen[str]
    binary: Path
    grpc_port: int
    bridge_port: int
    data_dir: Path
    _stopped: bool = False
    _stdout_buf: str = ""
    _stderr_buf: str = ""

    @property
    def grpc_addr(self) -> str:
        return f"127.0.0.1:{self.grpc_port}"

    @property
    def bridge_url(self) -> str:
        return f"http://127.0.0.1:{self.bridge_port}"

    def stop(self, timeout: float = 5.0) -> Tuple[str, str]:
        if not self._stopped:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    out, err = self.proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    out, err = self.proc.communicate(timeout=timeout)
            else:
                out, err = self.proc.communicate()
            self._stdout_buf = out or ""
            self._stderr_buf = err or ""
            self._stopped = True
        return self._stdout_buf, self._stderr_buf


def _close_db_wals(db: SutraDB) -> None:
    for col in list(db.collections.values()):
        wal = getattr(col, "wal", None)
        if wal is not None:
            try:
                wal.close()
            except Exception:
                pass


@pytest.fixture
def py_db(tmp_path: Path) -> Generator[SutraDB, None, None]:
    """Isolated SutraDB instance backed by tmp_path."""
    db_dir = tmp_path / "py_db"
    db = SutraDB(persist_directory=str(db_dir))
    try:
        yield db
    finally:
        _close_db_wals(db)


@pytest.fixture
def py_collection_factory(tmp_path: Path) -> Generator[Callable[..., Any], None, None]:
    """Factory creating isolated pure-Python Collection instances inside tmp_path."""
    from sutradb import Collection

    created: List[Collection] = []
    counter = 0

    def _create(
        name: str = "test_col",
        dimension: int = 4,
        metric: str = "cosine",
        enable_wal: bool = False,
        analyzer: Optional[Any] = None,
        embedding_function: Optional[Any] = None,
        storage_filename: Optional[str] = None,
    ) -> Collection:
        nonlocal counter
        counter += 1
        fname = storage_filename or f"{name}_{counter}.sutra"
        col = Collection(
            name=name,
            dimension=dimension,
            metric=metric,
            storage_path=str(tmp_path / fname),
            enable_wal=enable_wal,
            analyzer=analyzer,
            embedding_function=embedding_function,
        )
        created.append(col)
        return col

    try:
        yield _create
    finally:
        for col in created:
            wal = getattr(col, "wal", None)
            if wal is not None:
                try:
                    wal.close()
                except Exception:
                    pass


@pytest.fixture
def native_collection_factory(tmp_path: Path) -> Callable[..., Any]:
    """Factory creating isolated Rust NativeCollection instances inside tmp_path."""
    if NativeCollection is None:
        pytest.fail("NativeCollection (sutradb._sutradb) is not available")

    counter = 0

    def _create(
        name: str = "native_col",
        dimension: int = 4,
        metric: str = "cosine",
        enable_wal: bool = False,
        analyzer: Optional[Dict[str, Any]] = None,
        storage_filename: Optional[str] = None,
        use_storage_path: bool = True,
    ) -> Any:
        nonlocal counter
        counter += 1
        storage_path: Optional[str] = None
        if use_storage_path:
            fname = storage_filename or f"{name}_{counter}.sutra"
            storage_path = str(tmp_path / fname)
        return NativeCollection(
            name=name,
            dimension=dimension,
            metric=metric,
            storage_path=storage_path,
            enable_wal=enable_wal,
            analyzer=analyzer,
        )

    return _create


@pytest.fixture
def rest_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Isolated FastAPI TestClient backed by a fresh tmp_path SutraDB."""
    rest_dir = tmp_path / "rest_db"
    db = SutraDB(persist_directory=str(rest_dir))
    app = create_app(data_dir=str(rest_dir), db=db)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        _close_db_wals(db)


@pytest.fixture
def uvicorn_server_factory(tmp_path: Path) -> Generator[Callable[..., Tuple[str, SutraDB]], None, None]:
    """Factory starting a live ephemeral uvicorn ASGI server for real TCP SutraClient tests."""
    running: List[Tuple[uvicorn.Server, threading.Thread, SutraDB]] = []
    idx = 0

    def _start(backend_url: Optional[str] = None) -> Tuple[str, SutraDB]:
        nonlocal idx
        idx += 1
        data_dir = tmp_path / f"uvicorn_db_{idx}"
        data_dir.mkdir(parents=True, exist_ok=True)
        db = SutraDB(persist_directory=str(data_dir))
        prev_backend = os.environ.get("SUTRA_BACKEND_URL")
        if backend_url is not None:
            os.environ["SUTRA_BACKEND_URL"] = backend_url
        try:
            app = create_app(data_dir=str(data_dir), db=db if backend_url is None else None)
        finally:
            if backend_url is not None:
                if prev_backend is None:
                    os.environ.pop("SUTRA_BACKEND_URL", None)
                else:
                    os.environ["SUTRA_BACKEND_URL"] = prev_backend

        port = get_free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        deadline = time.monotonic() + 8.0
        base_url = f"http://127.0.0.1:{port}"
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{base_url}/health", timeout=0.5)
                if resp.status_code == 200:
                    break
            except Exception:
                time.sleep(0.05)
        else:
            server.should_exit = True
            thread.join(timeout=3.0)
            raise AssertionError(f"uvicorn server failed to start on {base_url}")

        running.append((server, thread, db))
        return base_url, db

    try:
        yield _start
    finally:
        for server, thread, db in reversed(running):
            server.should_exit = True
            thread.join(timeout=5.0)
            _close_db_wals(db)


@pytest.fixture
def playground_server(tmp_path: Path) -> Generator[str, None, None]:
    """Isolated ephemeral Playground HTTPServer using a per-fixture handler subclass."""
    pg_dir = tmp_path / "playground_db"
    db = SutraDB(persist_directory=str(pg_dir))
    col = seed_database(db)

    scoped_handler = type(
        "_ScopedPlaygroundHandler",
        (PlaygroundHandler,),
        {
            "db": db,
            "collection": col,
            "log_message": lambda self, format, *args: None,
        },
    )

    httpd = HTTPServer(("127.0.0.1", 0), scoped_handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    assert _wait_for_tcp("127.0.0.1", port, timeout=5.0), "Playground server did not bind"
    try:
        yield base_url
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3.0)
        _close_db_wals(db)


@pytest.fixture
def sutradb_server_factory(tmp_path: Path) -> Generator[Callable[..., ServerHandle], None, None]:
    """
    Factory spawning sutradb-server with --http-bridge-addr and bounded readiness polling.
    Explicitly asserts --http-bridge-addr support rather than skipping pre-M3 binaries.
    """
    handles: List[ServerHandle] = []
    idx = 0

    def _spawn(data_dir: Optional[Path] = None, extra_args: Optional[List[str]] = None) -> ServerHandle:
        nonlocal idx
        idx += 1
        binary = resolve_sutradb_server_bin()

        help_proc = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        help_out = (help_proc.stdout or "") + (help_proc.stderr or "")
        assert "--http-bridge-addr" in help_out, (
            f"sutradb-server binary at {binary} does not support --http-bridge-addr "
            f"(required by M3 SUTRA_MODE=all contract). --help output:\n{help_out}"
        )

        target_dir = data_dir if data_dir is not None else (tmp_path / f"server_data_{idx}")
        target_dir.mkdir(parents=True, exist_ok=True)
        grpc_port = get_free_port()
        bridge_port = get_free_port()
        while bridge_port == grpc_port:
            bridge_port = get_free_port()

        cmd = [
            str(binary),
            "--host", "127.0.0.1",
            "--port", str(grpc_port),
            "--data-dir", str(target_dir),
            "--http-bridge-addr", f"127.0.0.1:{bridge_port}",
        ]
        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        handle = ServerHandle(
            proc=proc,
            binary=binary,
            grpc_port=grpc_port,
            bridge_port=bridge_port,
            data_dir=target_dir,
        )
        handles.append(handle)

        deadline = time.monotonic() + 10.0
        bridge_ready = False
        grpc_ready = False
        while time.monotonic() < deadline:
            rc = proc.poll()
            if rc is not None:
                out, err = handle.stop()
                raise AssertionError(
                    f"sutradb-server exited prematurely (code={rc}).\nstdout:\n{out}\nstderr:\n{err}"
                )
            if not bridge_ready:
                try:
                    resp = httpx.get(f"{handle.bridge_url}/health", timeout=0.5)
                    if resp.status_code == 200:
                        bridge_ready = True
                except Exception:
                    pass
            if not grpc_ready:
                try:
                    with socket.create_connection(("127.0.0.1", grpc_port), timeout=0.3):
                        grpc_ready = True
                except OSError:
                    pass
            if bridge_ready and grpc_ready:
                return handle
            time.sleep(0.05)

        out, err = handle.stop()
        raise AssertionError(
            f"Timed out waiting for sutradb-server readiness "
            f"(bridge_ready={bridge_ready}, grpc_ready={grpc_ready}).\n"
            f"stdout:\n{out}\nstderr:\n{err}"
        )

    try:
        yield _spawn
    finally:
        for h in reversed(handles):
            h.stop()


@pytest.fixture
def sutra_all_mode_stack(
    sutradb_server_factory: Callable[..., ServerHandle],
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Tuple[ServerHandle, TestClient], None, None]:
    """
    Unified SUTRA_MODE=all stack: sutradb-server owning data_dir + FastAPI REST
    proxying through SUTRA_BACKEND_URL to sutradb-server's HTTP bridge.
    """
    handle = sutradb_server_factory()
    monkeypatch.setenv("SUTRA_BACKEND_URL", handle.bridge_url)
    monkeypatch.setenv("SUTRA_DATA_DIR", str(handle.data_dir))
    app = create_app(data_dir=str(handle.data_dir))
    with TestClient(app) as client:
        yield handle, client
