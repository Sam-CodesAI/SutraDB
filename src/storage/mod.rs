pub mod header;
pub mod index_file;
pub mod wal;
pub mod mmap;

pub use header::{SutraHeader, HEADER_MAGIC_V2, HEADER_MAGIC_V3, HEADER_MAGIC_V4, HEADER_SIZE};
pub use index_file::{index_sidecar_path, IndexStorage, PersistedIndexes, RestoredIndexes};
pub use wal::{WriteAheadLog, WalRecord};
pub use mmap::MmapVectorStorage;

use std::fs::File;
use std::io::Write;
use std::path::Path;
use crc32fast::Hasher;
use crate::error::SutraError;

/// Validates collection names against `^[a-zA-Z0-9_-]{1,128}$`.
/// Rejects path traversal (`..`, `/`, `\`), whitespace, empty strings, and NUL bytes.
pub fn validate_collection_name(name: &str) -> Result<(), SutraError> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(SutraError::InvalidQuery(format!(
            "Invalid collection name '{}': must match ^[a-zA-Z0-9_-]{{1,128}}$",
            name
        )));
    }
    Ok(())
}

/// High-level binary storage helper matching SutraDB .sutra layout.
pub struct BinaryStorage;

impl BinaryStorage {
    pub fn write_f32<P: AsRef<Path>>(
        path: P,
        vectors: &[f32],
        doc_count: u32,
        dimension: u32,
        metric: &str,
        documents_json: &[u8],
    ) -> Result<u64, SutraError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let expected = (doc_count as usize)
            .checked_mul(dimension as usize)
            .ok_or_else(|| SutraError::InvalidHeader("Vector count * dimension overflows usize".into()))?;
        if vectors.len() != expected {
            return Err(SutraError::DimensionMismatch {
                expected,
                got: vectors.len(),
            });
        }

        let vec_byte_len = vectors
            .len()
            .checked_mul(4)
            .ok_or_else(|| SutraError::InvalidHeader("Vector byte length overflows usize".into()))?;
        let vec_bytes = unsafe {
            std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vec_byte_len)
        };

        let mut hasher = Hasher::new();
        hasher.update(vec_bytes);
        hasher.update(documents_json);
        let crc = hasher.finalize();

        let header = SutraHeader::new(
            doc_count,
            dimension,
            metric,
            documents_json.len() as u64,
            crc,
            0, // Float32
        );

        let temp_path = path.with_extension("tmp");
        {
            let mut file = File::create(&temp_path)?;
            file.write_all(&header.to_bytes())?;
            file.write_all(vec_bytes)?;
            file.write_all(documents_json)?;
            file.flush()?;
            file.sync_data()?;
        }

        std::fs::rename(&temp_path, path)?;
        Ok(std::fs::metadata(path)?.len())
    }

    pub fn write_sq8<P: AsRef<Path>>(
        path: P,
        vectors: &[u8],
        doc_count: u32,
        dimension: u32,
        metric: &str,
        documents_json: &[u8],
    ) -> Result<u64, SutraError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let expected = (doc_count as usize)
            .checked_mul(dimension as usize)
            .ok_or_else(|| SutraError::InvalidHeader("Vector count * dimension overflows usize".into()))?;
        if vectors.len() != expected {
            return Err(SutraError::DimensionMismatch {
                expected,
                got: vectors.len(),
            });
        }

        let mut hasher = Hasher::new();
        hasher.update(vectors);
        hasher.update(documents_json);
        let crc = hasher.finalize();

        let header = SutraHeader::new(
            doc_count,
            dimension,
            metric,
            documents_json.len() as u64,
            crc,
            1, // SQ8
        );

        let temp_path = path.with_extension("tmp");
        {
            let mut file = File::create(&temp_path)?;
            file.write_all(&header.to_bytes())?;
            file.write_all(vectors)?;
            file.write_all(documents_json)?;
            file.flush()?;
            file.sync_data()?;
        }

        std::fs::rename(&temp_path, path)?;
        Ok(std::fs::metadata(path)?.len())
    }

    pub fn write_sq8_v4<P: AsRef<Path>>(
        path: P,
        sq8_params: &[(f32, f32)],
        vectors: &[u8],
        doc_count: u32,
        dimension: u32,
        metric: &str,
        documents_json: &[u8],
    ) -> Result<u64, SutraError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let count_usize = doc_count as usize;
        if sq8_params.len() != count_usize {
            return Err(SutraError::DimensionMismatch {
                expected: count_usize,
                got: sq8_params.len(),
            });
        }

        let expected_vec_bytes = count_usize
            .checked_mul(dimension as usize)
            .ok_or_else(|| SutraError::InvalidHeader("Vector count * dimension overflows usize".into()))?;
        if vectors.len() != expected_vec_bytes {
            return Err(SutraError::DimensionMismatch {
                expected: expected_vec_bytes,
                got: vectors.len(),
            });
        }

        let params_byte_len = count_usize
            .checked_mul(8)
            .ok_or_else(|| SutraError::InvalidHeader("SQ8 params length overflows usize".into()))?;
        let mut params_bytes = Vec::with_capacity(params_byte_len);
        for &(min_val, scale) in sq8_params {
            params_bytes.extend_from_slice(&min_val.to_le_bytes());
            params_bytes.extend_from_slice(&scale.to_le_bytes());
        }

        let mut hasher = Hasher::new();
        hasher.update(&params_bytes);
        hasher.update(vectors);
        hasher.update(documents_json);
        let crc = hasher.finalize();

        let header = SutraHeader::new_v4(
            doc_count,
            dimension,
            metric,
            documents_json.len() as u64,
            crc,
            1, // SQ8
        );

        let temp_path = path.with_extension("tmp");
        {
            let mut file = File::create(&temp_path)?;
            file.write_all(&header.to_bytes())?;
            file.write_all(&params_bytes)?;
            file.write_all(vectors)?;
            file.write_all(documents_json)?;
            file.flush()?;
            file.sync_data()?;
        }

        std::fs::rename(&temp_path, path)?;
        Ok(std::fs::metadata(path)?.len())
    }
}
