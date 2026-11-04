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
        self._reader_threads: dict[int, int] = {}

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
            ident = threading.get_ident()
            # Allow re-entrant read from writer thread
            if self._writer and self._writer_thread == threading.current_thread():
                self._readers += 1
                self._reader_threads[ident] = self._reader_threads.get(ident, 0) + 1
                return
            # Allow re-entrant read from thread that already holds read lock (avoids deadlock)
            if ident in self._reader_threads:
                self._readers += 1
                self._reader_threads[ident] += 1
                return
            # Writer-preference: block new incoming readers when a writer is active OR waiting
            while self._writer or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1
            self._reader_threads[ident] = 1

    def _release_read(self) -> None:
        with self._cond:
            ident = threading.get_ident()
            is_writer_holding = self._writer and self._writer_thread == threading.current_thread()
            if ident not in self._reader_threads and not is_writer_holding:
                raise RuntimeError("Cannot release read lock: lock not acquired by current thread")
            self._readers -= 1
            if ident in self._reader_threads:
                self._reader_threads[ident] -= 1
                if self._reader_threads[ident] == 0:
                    del self._reader_threads[ident]
            if self._readers == 0:
                self._cond.notify_all()

    def _acquire_write(self) -> None:
        with self._cond:
            ident = threading.get_ident()
            # Allow re-entrant write from writer thread
            if self._writer and self._writer_thread == threading.current_thread():
                self._write_count += 1
                return
            # Prohibit lock upgrades from reader to writer to avoid instant self-deadlock
            if ident in self._reader_threads:
                raise RuntimeError("Cannot upgrade read lock to write lock (deadlock prevention)")
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
            if not self._writer or self._writer_thread != threading.current_thread():
                raise RuntimeError("Cannot release write lock: lock not held by current thread")
            self._write_count -= 1
            if self._write_count == 0:
                self._writer = False
                self._writer_thread = None
                self._cond.notify_all()
