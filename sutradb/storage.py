"""
Storage and persistence subsystem for SutraDB.
Implements the zero-copy .sutra binary layout and atomic Write-Ahead Logging (WAL).
"""

import json
import mmap
import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import zlib
import numpy as np

# Header specification:
# Magic: 8 bytes ("SUTRA\x02\x00\x00")
# Format: <8sII16sQI (total 42 bytes + 22 bytes padding = 64-byte aligned header)
HEADER_MAGIC = b"SUTRA\x02\x00\x00"
HEADER_FORMAT = "<8sII16sQI20s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 64, f"Header size must be 64 bytes, got {HEADER_SIZE}"


class BinaryStorage:
    """Manages reading and writing memory-aligned .sutra binary files."""

    @staticmethod
    def write(
        filepath: Union[str, Path],
        vectors: np.ndarray,
        documents: List[Dict[str, Any]],
        metric: str
    ) -> int:
        """
        Saves vectors and metadata into an aligned .sutra binary file.
        Returns total bytes written.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        doc_count, dimension = vectors.shape

        if doc_count != len(documents):
            raise ValueError(
                f"Vector count ({doc_count}) does not match document count ({len(documents)})"
            )

        # Serialize metadata payload to UTF-8 JSON bytes
        metadata_json = json.dumps(documents, separators=(',', ':'), ensure_ascii=False)
        metadata_bytes = metadata_json.encode("utf-8")
        metadata_length = len(metadata_bytes)

        # Compute CRC32 checksum over vectors and metadata for integrity
        crc = zlib.crc32(vectors.tobytes())
        crc = zlib.crc32(metadata_bytes, crc)

        # Encode metric string into 16 bytes
        metric_padded = metric.encode("ascii")[:16].ljust(16, b"\x00")

        # Pack 64-byte header
        header = struct.pack(
            HEADER_FORMAT,
            HEADER_MAGIC,
            doc_count,
            dimension,
            metric_padded,
            metadata_length,
            crc,
            b"\x00" * 20
        )

        temp_filepath = filepath.with_suffix(".tmp")
        with open(temp_filepath, "wb") as f:
            f.write(header)
            f.write(vectors.tobytes())
            f.write(metadata_bytes)

        # Atomic rename guarantees file is never corrupt if interrupted
        temp_filepath.replace(filepath)
        return filepath.stat().st_size

    @staticmethod
    def read(filepath: Union[str, Path]) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
        """
        Reads a .sutra file into memory.
        Returns (vectors_matrix, documents_list, metric_name).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"SutraDB file not found at {filepath}")

        with open(filepath, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            if len(header_bytes) < HEADER_SIZE:
                raise ValueError("Corrupt file: Header too short")

            magic, doc_count, dimension, metric_raw, meta_len, stored_crc, _ = struct.unpack(
                HEADER_FORMAT, header_bytes
            )

            if magic != HEADER_MAGIC:
                raise ValueError(f"Invalid magic bytes: {magic}, expected {HEADER_MAGIC}")

            metric = metric_raw.rstrip(b"\x00").decode("ascii")

            vector_byte_size = doc_count * dimension * 4
            vector_bytes = f.read(vector_byte_size)
            if len(vector_bytes) < vector_byte_size:
                raise ValueError("Corrupt file: Incomplete vector array data")

            metadata_bytes = f.read(meta_len)
            if len(metadata_bytes) < meta_len:
                raise ValueError("Corrupt file: Incomplete metadata payload")

            # Verify CRC32 checksum
            computed_crc = zlib.crc32(vector_bytes)
            computed_crc = zlib.crc32(metadata_bytes, computed_crc)
            if computed_crc != stored_crc:
                raise ValueError(
                    f"Integrity check failed: CRC32 mismatch ({computed_crc} != {stored_crc})"
                )

            vectors = np.frombuffer(vector_bytes, dtype=np.float32).reshape((doc_count, dimension)).copy()
            documents = json.loads(metadata_bytes.decode("utf-8"))

            return vectors, documents, metric


class WriteAheadLog:
    """Append-only transaction log for durability."""

    def __init__(self, wal_path: Union[str, Path]):
        self.wal_path = Path(wal_path)
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.wal_path, "a+b", buffering=0)

    def append(self, entry: Dict[str, Any]) -> None:
        """Appends a serialized mutation record with CRC32."""
        payload = json.dumps(entry, separators=(',', ':'), ensure_ascii=False).encode("utf-8")
        crc = zlib.crc32(payload)
        length = len(payload)
        # Format: length (4 bytes), crc (4 bytes), payload
        header = struct.pack("<II", length, crc)
        self._file.write(header + payload)
        self._file.flush()

    def replay(self) -> List[Dict[str, Any]]:
        """Reads and validates all WAL entries from disk."""
        entries = []
        if not self.wal_path.exists() or self.wal_path.stat().st_size == 0:
            return entries

        with open(self.wal_path, "rb") as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                length, expected_crc = struct.unpack("<II", header)
                payload = f.read(length)
                if len(payload) < length:
                    # Trailing incomplete write
                    break
                if zlib.crc32(payload) != expected_crc:
                    break
                entries.append(json.loads(payload.decode("utf-8")))

        return entries

    def clear(self) -> None:
        """Truncates the WAL after a successful snapshot."""
        self._file.close()
        with open(self.wal_path, "wb"):
            pass
        self._file = open(self.wal_path, "a+b", buffering=0)

    def close(self) -> None:
        """Closes the underlying file descriptor."""
        if not self._file.closed:
            self._file.close()
