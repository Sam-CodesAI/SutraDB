pub mod header;
pub mod wal;
pub mod mmap;

pub use header::{SutraHeader, HEADER_MAGIC_V2, HEADER_MAGIC_V3, HEADER_SIZE};
pub use wal::{WriteAheadLog, WalRecord};
pub use mmap::MmapVectorStorage;

use std::fs::File;
use std::io::Write;
use std::path::Path;
use crc32fast::Hasher;
use crate::error::SutraError;

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

        assert_eq!(vectors.len(), (doc_count as usize) * (dimension as usize));

        let vec_bytes = unsafe {
            std::slice::from_raw_parts(vectors.as_ptr() as *const u8, vectors.len() * 4)
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

        assert_eq!(vectors.len(), (doc_count as usize) * (dimension as usize));

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
        }

        std::fs::rename(&temp_path, path)?;
        Ok(std::fs::metadata(path)?.len())
    }
}
