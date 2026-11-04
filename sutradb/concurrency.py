"""
Thread-safety primitives for SutraDB.

Provides a ReadWriteLock that allows multiple concurrent readers but exclusive
writer access, critical for multi-user/concurrent-write workloads.
"""

import threading
from contextlib import contextmanager
from typing import Generator


class ReadWriteLock:
    """
    A read-write lock (multiple-reader / single-writer lock) with writer preference.

    Multiple threads can hold the read lock simultaneously, but only one
    thread can hold the write lock, and no readers are allowed while writing.
    Incoming readers are blocked if writers are waiting, preventing writer starvation.

    Usage:
        lock = ReadWriteLock()

        # Reading (shared access)
        with lock.read():
            data = collection.query(...)

        # Writing (exclusive access)
        with lock.write():
            collection.insert(...)
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers: int = 0
        self._writer: bool = False
        self._writer_thread: threading.Thread | None = None
        self._waiting_writers: int = 0
        self._write_count: int = 0

    @contextmanager
    def read(self) -> Generator[None, None, None]:
        """Context manager for acquiring/releasing a read lock."""
        self._acquire_read()
        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write(self) -> Generator[None, None, None]:
        """Context manager for acquiring/releasing a write lock."""
        self._acquire_write()
        try:
            yield
        finally:
            self._release_write()

    def _acquire_read(self) -> None:
        with self._cond:
            # Allow re-entrant read from writer thread
            if self._writer and self._writer_thread == threading.current_thread():
                self._readers += 1
                return
            # Writer-preference: block incoming readers when a writer is active OR waiting
            while self._writer or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1

    def _release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def _acquire_write(self) -> None:
        with self._cond:
            if self._writer and self._writer_thread == threading.current_thread():
                self._write_count += 1
                return
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
                self._writer_thread = threading.current_thread()
                self._write_count = 1
            finally:
                self._waiting_writers -= 1

    def _release_write(self) -> None:
        with self._cond:
            self._write_count -= 1
            if self._write_count == 0:
                self._writer = False
                self._writer_thread = None
                self._cond.notify_all()
