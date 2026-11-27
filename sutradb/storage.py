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
# Magic: 8 bytes ("SUTRA\x02\x00\x00", "SUTRA\x03\x00\x00", "SUTRA\x04\x00\x00")
# Format: <8sII16sQI20s (total 64-byte aligned header)
HEADER_MAGIC_V2 = b"SUTRA\x02\x00\x00"
HEADER_MAGIC_V3 = b"SUTRA\x03\x00\x00"
HEADER_MAGIC_V4 = b"SUTRA\x04\x00\x00"
HEADER_MAGIC = HEADER_MAGIC_V2
HEADER_FORMAT = "<8sII16sQI20s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 64, f"Header size must be 64 bytes, got {HEADER_SIZE}"


def _decode_wal_binary(payload: bytes) -> Optional[Dict[str, Any]]:
    """Decodes SutraDB binary WAL record (`SWB1`) into a dict matching JSON WAL entries."""
    if not payload.startswith(b"SWB1"):
        return None
    off = 4
    n = len(payload)
    if off + 2 > n:
        return None
    (op_len,) = struct.unpack_from("<H", payload, off)
    off += 2
    if off + op_len + 4 > n:
        return None
    op = payload[off : off + op_len].decode("utf-8")
    off += op_len
    (doc_count,) = struct.unpack_from("<I", payload, off)
    off += 4
    docs = []
    for _ in range(doc_count):
        if off + 5 > n:
            return None
        flag = payload[off]
        off += 1
        (json_len,) = struct.unpack_from("<I", payload, off)
        off += 4
        if off + json_len > n:
            return None
        doc_obj = json.loads(payload[off : off + json_len].decode("utf-8"))
        off += json_len
        if flag == 1:
            if off + 4 > n:
                return None
            (v_len,) = struct.unpack_from("<I", payload, off)
            off += 4
            byte_len = v_len * 4
            if off + byte_len > n:
                return None
            doc_obj["vector"] = list(struct.unpack_from(f"<{v_len}f", payload, off))
            off += byte_len
        docs.append(doc_obj)
    if off + 4 > n:
        return None
    (id_count,) = struct.unpack_from("<I", payload, off)
    off += 4
    ids = []
    for _ in range(id_count):
        if off + 4 > n:
            return None
        (id_len,) = struct.unpack_from("<I", payload, off)
        off += 4
        if off + id_len > n:
            return None
        ids.append(payload[off : off + id_len].decode("utf-8"))
        off += id_len
    if off != n:
        return None
    return {"op": op, "docs": docs, "ids": ids}


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

        vectors = np.ascontiguousarray(vectors, dtype="<f4")
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be a 2D array, got shape {vectors.shape}")
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
        vector_bytes = vectors.tobytes()
        crc = zlib.crc32(vector_bytes)
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
            f.write(vector_bytes)
            f.write(metadata_bytes)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename guarantees file is never corrupt if interrupted
        temp_filepath.replace(filepath)
        return filepath.stat().st_size

    @staticmethod
    def write_sq8_v4(
        filepath: Union[str, Path],
        sq8_params: List[Tuple[float, float]],
        sq8_data: np.ndarray,
        documents: List[Dict[str, Any]],
        metric: str
    ) -> int:
        """
        Saves V4 SQ8-quantized vectors with per-document (min_val, scale) parameters
        and metadata into an aligned .sutra binary file.
        Layout: [64B header][sq8_params: doc_count * 8B][sq8_data: doc_count * dim B][metadata_json]
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        sq8_data = np.ascontiguousarray(sq8_data, dtype=np.uint8)
        if sq8_data.ndim != 2:
            raise ValueError(f"SQ8 data must be a 2D array, got shape {sq8_data.shape}")
        doc_count, dimension = sq8_data.shape

        if doc_count != len(documents):
            raise ValueError(
                f"SQ8 vector count ({doc_count}) does not match document count ({len(documents)})"
            )
        if doc_count != len(sq8_params):
            raise ValueError(
                f"SQ8 parameter count ({len(sq8_params)}) does not match vector count ({doc_count})"
            )

        if doc_count == 0:
            params_bytes = b""
        else:
            params_arr = np.asarray(sq8_params, dtype="<f4").reshape((doc_count, 2))
            params_bytes = np.ascontiguousarray(params_arr, dtype="<f4").tobytes()

        sq8_bytes = sq8_data.tobytes()
        metadata_json = json.dumps(documents, separators=(',', ':'), ensure_ascii=False)
        metadata_bytes = metadata_json.encode("utf-8")
        metadata_length = len(metadata_bytes)

        crc = zlib.crc32(params_bytes)
        crc = zlib.crc32(sq8_bytes, crc)
        crc = zlib.crc32(metadata_bytes, crc)

        metric_padded = metric.encode("ascii")[:16].ljust(16, b"\x00")
        extra = b"\x01" + b"\x00" * 19  # quant_mode = 1 (SQ8)

        header = struct.pack(
            HEADER_FORMAT,
            HEADER_MAGIC_V4,
            doc_count,
            dimension,
            metric_padded,
            metadata_length,
            crc,
            extra
        )

        temp_filepath = filepath.with_suffix(".tmp")
        with open(temp_filepath, "wb") as f:
            f.write(header)
            f.write(params_bytes)
            f.write(sq8_bytes)
            f.write(metadata_bytes)
            f.flush()
            os.fsync(f.fileno())

        temp_filepath.replace(filepath)
        return filepath.stat().st_size

    @staticmethod
    def read_detailed(
        filepath: Union[str, Path]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], str, int, Optional[Tuple[np.ndarray, List[Tuple[float, float]]]]]:
        """
        Reads a .sutra V2, V3, or V4 file into memory with quantization metadata.
        Returns (vectors_matrix, documents_list, metric_name, quant_mode, sq8_info),
        where sq8_info is (u8_matrix, [(min_val, scale), ...]) for V4 SQ8 or None otherwise.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"SutraDB file not found at {filepath}")

        with open(filepath, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            if len(header_bytes) < HEADER_SIZE:
                raise ValueError("Corrupt file: Header too short")

            magic, doc_count, dimension, metric_raw, meta_len, stored_crc, extra = struct.unpack(
                HEADER_FORMAT, header_bytes
            )

            if magic not in (HEADER_MAGIC_V2, HEADER_MAGIC_V3, HEADER_MAGIC_V4):
                raise ValueError(
                    f"Invalid magic bytes: {magic}, expected {HEADER_MAGIC_V2}, {HEADER_MAGIC_V3}, or {HEADER_MAGIC_V4}"
                )

            metric = metric_raw.rstrip(b"\x00").decode("ascii")
            quant_mode = 0 if magic == HEADER_MAGIC_V2 else (extra[0] if len(extra) > 0 else 0)
            if quant_mode not in (0, 1):
                raise ValueError(f"Unsupported quant_mode: {quant_mode}")

            file_size = os.fstat(f.fileno()).st_size
            remaining = max(0, file_size - HEADER_SIZE)
            params_size = (doc_count * 8) if (magic == HEADER_MAGIC_V4 and quant_mode == 1) else 0
            vec_size = doc_count * dimension * (1 if quant_mode == 1 else 4)

            if params_size + vec_size + meta_len > remaining:
                raise ValueError("Corrupt file: Truncated payload or invalid header dimensions")

            sq8_info: Optional[Tuple[np.ndarray, List[Tuple[float, float]]]] = None

            if magic == HEADER_MAGIC_V4 and quant_mode == 1:
                params_bytes = f.read(params_size)
                if len(params_bytes) < params_size:
                    raise ValueError("Corrupt file: Incomplete SQ8 parameter payload")

                vector_bytes = f.read(vec_size)
                if len(vector_bytes) < vec_size:
                    raise ValueError("Corrupt file: Incomplete vector array data")

                metadata_bytes = f.read(meta_len)
                if len(metadata_bytes) < meta_len:
                    raise ValueError("Corrupt file: Incomplete metadata payload")

                computed_crc = zlib.crc32(params_bytes)
                computed_crc = zlib.crc32(vector_bytes, computed_crc)
                computed_crc = zlib.crc32(metadata_bytes, computed_crc)
                if computed_crc != stored_crc:
                    raise ValueError(
                        f"Integrity check failed: CRC32 mismatch ({computed_crc} != {stored_crc})"
                    )

                if doc_count == 0:
                    params_arr = np.empty((0, 2), dtype=np.float32)
                    u8_arr = np.empty((0, dimension), dtype=np.uint8)
                    vectors = np.empty((0, dimension), dtype=np.float32)
                else:
                    params_arr = np.frombuffer(params_bytes, dtype="<f4").reshape((doc_count, 2)).astype(np.float32, copy=True)
                    u8_arr = np.frombuffer(vector_bytes, dtype=np.uint8).reshape((doc_count, dimension)).copy()
                    mins = params_arr[:, 0:1]
                    scales = params_arr[:, 1:2]
                    vectors = (u8_arr.astype(np.float32) * scales + mins).astype(np.float32, copy=False)

                sq8_info = (u8_arr, [(float(m), float(s)) for m, s in params_arr])
            elif quant_mode == 1:
                # Legacy V3 SQ8 fallback (scaled to [0, 1])
                vector_bytes = f.read(vec_size)
                if len(vector_bytes) < vec_size:
                    raise ValueError("Corrupt file: Incomplete vector array data")

                metadata_bytes = f.read(meta_len)
                if len(metadata_bytes) < meta_len:
                    raise ValueError("Corrupt file: Incomplete metadata payload")

                computed_crc = zlib.crc32(vector_bytes)
                computed_crc = zlib.crc32(metadata_bytes, computed_crc)
                if computed_crc != stored_crc:
                    raise ValueError(
                        f"Integrity check failed: CRC32 mismatch ({computed_crc} != {stored_crc})"
                    )

                u8_arr = np.frombuffer(vector_bytes, dtype=np.uint8).reshape((doc_count, dimension))
                vectors = (u8_arr.astype(np.float32) / 255.0).copy()
            else:
                vector_bytes = f.read(vec_size)
                if len(vector_bytes) < vec_size:
                    raise ValueError("Corrupt file: Incomplete vector array data")

                metadata_bytes = f.read(meta_len)
                if len(metadata_bytes) < meta_len:
                    raise ValueError("Corrupt file: Incomplete metadata payload")

                computed_crc = zlib.crc32(vector_bytes)
                computed_crc = zlib.crc32(metadata_bytes, computed_crc)
                if computed_crc != stored_crc:
                    raise ValueError(
                        f"Integrity check failed: CRC32 mismatch ({computed_crc} != {stored_crc})"
                    )

                vectors = np.frombuffer(vector_bytes, dtype="<f4").reshape((doc_count, dimension)).astype(np.float32, copy=True)

            documents = json.loads(metadata_bytes.decode("utf-8"))
            return vectors, documents, metric, quant_mode, sq8_info

    @staticmethod
    def read(filepath: Union[str, Path]) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
        """
        Reads a .sutra file into memory.
        Returns (vectors_matrix, documents_list, metric_name).
        """
        vectors, documents, metric, _, _ = BinaryStorage.read_detailed(filepath)
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
        try:
            if hasattr(os, "fdatasync"):
                os.fdatasync(self._file.fileno())
            else:
                os.fsync(self._file.fileno())
        except (AttributeError, OSError):
            try:
                os.fsync(self._file.fileno())
            except (AttributeError, OSError):
                pass

    def flush(self) -> None:
        """Flushes buffers and forces sync to disk for true durability."""
        self._file.flush()
        try:
            if hasattr(os, "fdatasync"):
                os.fdatasync(self._file.fileno())
            else:
                os.fsync(self._file.fileno())
        except (AttributeError, OSError):
            try:
                os.fsync(self._file.fileno())
            except (AttributeError, OSError):
                pass

    def replay(self) -> List[Dict[str, Any]]:
        """Reads and validates all WAL entries from disk."""
        entries = []
        if not self.wal_path.exists() or self.wal_path.stat().st_size == 0:
            return entries

        with open(self.wal_path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                length, expected_crc = struct.unpack("<II", header)
                remaining = file_size - f.tell()
                if length > remaining:
                    # Truncated or corrupt record length
                    break
                payload = f.read(length)
                if len(payload) < length:
                    # Trailing incomplete write
                    break
                if zlib.crc32(payload) != expected_crc:
                    break
                if payload.startswith(b"SWB1"):
                    rec = _decode_wal_binary(payload)
                    if rec is None:
                        break
                    entries.append(rec)
                else:
                    entries.append(json.loads(payload.decode("utf-8")))

        return entries

    def clear(self) -> None:
        """Truncates the WAL after a successful snapshot."""
        self._file.close()
        with open(self.wal_path, "wb") as f:
            f.flush()
            try:
                os.fsync(f.fileno())
            except (AttributeError, OSError):
                pass
        self._file = open(self.wal_path, "a+b", buffering=0)

    def close(self) -> None:
        """Closes the underlying file descriptor."""
        if not self._file.closed:
            self._file.close()
