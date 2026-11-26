use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;
use memmap2::Mmap;
use crc32fast::Hasher;
use crate::error::SutraError;
use super::header::{SutraHeader, HEADER_MAGIC_V4, HEADER_SIZE};

/// Zero-copy memory-mapped file reader for .sutra binary files.
pub struct MmapVectorStorage {
    mmap: Mmap,
    header: SutraHeader,
    sq8_params_offset: usize,
    has_sq8_params: bool,
    vector_offset: usize,
    _vector_bytes_len: usize,
    metadata_offset: usize,
    payload_end: usize,
    checksum_verified: OnceLock<Result<(), String>>,
}

impl MmapVectorStorage {
    /// Opens a `.sutra` file with eager full-payload CRC32 verification.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SutraError> {
        let storage = Self::open_lazy(path)?;
        storage.verify_checksum()?;
        Ok(storage)
    }

    /// Opens a `.sutra` file with O(1) header verification and lazy payload CRC32 checking.
    /// Avoids touching data pages until they are explicitly read or `verify_checksum()` is called.
    pub fn open_lazy<P: AsRef<Path>>(path: P) -> Result<Self, SutraError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(SutraError::InvalidHeader("File size smaller than header".into()));
        }

        let header = SutraHeader::from_bytes(&mmap[0..HEADER_SIZE])?;
        if header.quant_mode != 0 && header.quant_mode != 1 {
            return Err(SutraError::InvalidHeader(format!(
                "Unsupported quant_mode: {}",
                header.quant_mode
            )));
        }

        let doc_count = usize::try_from(header.doc_count)
            .map_err(|_| SutraError::InvalidHeader("doc_count overflows usize".into()))?;
        let dimension = usize::try_from(header.dimension)
            .map_err(|_| SutraError::InvalidHeader("dimension overflows usize".into()))?;
        let metadata_length = usize::try_from(header.metadata_length)
            .map_err(|_| SutraError::InvalidHeader("metadata_length overflows usize".into()))?;

        let has_sq8_params = header.magic == *HEADER_MAGIC_V4 && header.quant_mode == 1;
        let sq8_params_len = if has_sq8_params {
            doc_count
                .checked_mul(8)
                .ok_or_else(|| SutraError::InvalidHeader("SQ8 params length overflow".into()))?
        } else {
            0
        };

        let bytes_per_dim: usize = if header.quant_mode == 1 { 1 } else { 4 };
        let vector_bytes_len = doc_count
            .checked_mul(dimension)
            .and_then(|n| n.checked_mul(bytes_per_dim))
            .ok_or_else(|| SutraError::InvalidHeader("Vector payload size overflow".into()))?;

        let sq8_params_offset = HEADER_SIZE;
        let vector_offset = sq8_params_offset
            .checked_add(sq8_params_len)
            .ok_or_else(|| SutraError::InvalidHeader("vector_offset overflow".into()))?;
        let metadata_offset = vector_offset
            .checked_add(vector_bytes_len)
            .ok_or_else(|| SutraError::InvalidHeader("metadata_offset overflow".into()))?;
        let payload_end = metadata_offset
            .checked_add(metadata_length)
            .ok_or_else(|| SutraError::InvalidHeader("payload_end overflow".into()))?;

        if mmap.len() < payload_end {
            return Err(SutraError::InvalidHeader("File truncated".into()));
        }

        Ok(Self {
            mmap,
            header,
            sq8_params_offset,
            has_sq8_params,
            vector_offset,
            _vector_bytes_len: vector_bytes_len,
            metadata_offset,
            payload_end,
            checksum_verified: OnceLock::new(),
        })
    }

    /// Verifies the full payload CRC32 checksum once and caches the result.
    pub fn verify_checksum(&self) -> Result<(), SutraError> {
        let res = self.checksum_verified.get_or_init(|| {
            let mut hasher = Hasher::new();
            hasher.update(&self.mmap[HEADER_SIZE..self.payload_end]);
            let computed_crc = hasher.finalize();
            if computed_crc != self.header.crc32 {
                Err(format!(
                    "CRC32 checksum mismatch: expected {}, got {}",
                    self.header.crc32, computed_crc
                ))
            } else {
                Ok(())
            }
        });
        match res {
            Ok(()) => Ok(()),
            Err(msg) => Err(SutraError::InvalidHeader(msg.clone())),
        }
    }

    pub fn header(&self) -> &SutraHeader {
        &self.header
    }

    /// Returns a zero-copy slice of float32 for doc at index `doc_idx`.
    /// Returns an error if quant_mode != 0 or index out of range.
    #[inline]
    pub fn try_get_vector_f32(&self, doc_idx: usize) -> Result<&[f32], SutraError> {
        if self.header.quant_mode != 0 {
            return Err(SutraError::InvalidHeader("Not a float32 collection".into()));
        }
        let doc_count = self.header.doc_count as usize;
        if doc_idx >= doc_count {
            return Err(SutraError::InvalidHeader(format!(
                "Document index {} out of bounds (doc_count={})",
                doc_idx, doc_count
            )));
        }

        let dim = self.header.dimension as usize;
        if dim == 0 {
            return Ok(&[]);
        }
        let stride = dim
            .checked_mul(4)
            .ok_or_else(|| SutraError::InvalidHeader("Vector stride overflow".into()))?;
        let offset = doc_idx
            .checked_mul(stride)
            .and_then(|off| self.vector_offset.checked_add(off))
            .ok_or_else(|| SutraError::InvalidHeader("Vector offset overflow".into()))?;
        let end = offset
            .checked_add(stride)
            .ok_or_else(|| SutraError::InvalidHeader("Vector end offset overflow".into()))?;
        if end > self.metadata_offset {
            return Err(SutraError::InvalidHeader("Vector slice out of bounds".into()));
        }
        let slice = &self.mmap[offset..end];

        Ok(unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const f32, dim)
        })
    }

    /// Returns a zero-copy slice of float32 for doc at index `doc_idx`.
    /// Panics if quant_mode != 0 or index out of range.
    #[inline]
    pub fn get_vector_f32(&self, doc_idx: usize) -> &[f32] {
        self.try_get_vector_f32(doc_idx)
            .expect("get_vector_f32 failed")
    }

    /// Returns a zero-copy slice of u8 for quantized doc at index `doc_idx`.
    #[inline]
    pub fn try_get_vector_sq8(&self, doc_idx: usize) -> Result<&[u8], SutraError> {
        if self.header.quant_mode != 1 {
            return Err(SutraError::InvalidHeader("Not a SQ8 collection".into()));
        }
        let doc_count = self.header.doc_count as usize;
        if doc_idx >= doc_count {
            return Err(SutraError::InvalidHeader(format!(
                "Document index {} out of bounds (doc_count={})",
                doc_idx, doc_count
            )));
        }

        let dim = self.header.dimension as usize;
        let offset = doc_idx
            .checked_mul(dim)
            .and_then(|off| self.vector_offset.checked_add(off))
            .ok_or_else(|| SutraError::InvalidHeader("SQ8 vector offset overflow".into()))?;
        let end = offset
            .checked_add(dim)
            .ok_or_else(|| SutraError::InvalidHeader("SQ8 vector end overflow".into()))?;
        if end > self.metadata_offset {
            return Err(SutraError::InvalidHeader("SQ8 vector slice out of bounds".into()));
        }
        Ok(&self.mmap[offset..end])
    }

    /// Returns a zero-copy slice of u8 for quantized doc at index `doc_idx`.
    #[inline]
    pub fn get_vector_sq8(&self, doc_idx: usize) -> &[u8] {
        self.try_get_vector_sq8(doc_idx)
            .expect("get_vector_sq8 failed")
    }

    /// Returns `(min_val, scale)` for a V4 SQ8 document at index `doc_idx`.
    #[inline]
    pub fn get_sq8_params(&self, doc_idx: usize) -> Result<(f32, f32), SutraError> {
        if !self.has_sq8_params {
            return Err(SutraError::InvalidHeader(
                "SQ8 params not present (requires V4 SQ8 snapshot)".into(),
            ));
        }
        let doc_count = self.header.doc_count as usize;
        if doc_idx >= doc_count {
            return Err(SutraError::InvalidHeader(format!(
                "Document index {} out of bounds (doc_count={})",
                doc_idx, doc_count
            )));
        }
        let start = doc_idx
            .checked_mul(8)
            .and_then(|off| self.sq8_params_offset.checked_add(off))
            .ok_or_else(|| SutraError::InvalidHeader("SQ8 params offset overflow".into()))?;
        let end = start
            .checked_add(8)
            .ok_or_else(|| SutraError::InvalidHeader("SQ8 params end overflow".into()))?;
        if end > self.vector_offset {
            return Err(SutraError::InvalidHeader("SQ8 params slice out of bounds".into()));
        }
        let min_val = f32::from_le_bytes(self.mmap[start..start + 4].try_into().unwrap());
        let scale = f32::from_le_bytes(self.mmap[start + 4..end].try_into().unwrap());
        Ok((min_val, scale))
    }

    /// Returns metadata JSON bytes.
    pub fn metadata_bytes(&self) -> &[u8] {
        let end = self.metadata_offset + self.header.metadata_length as usize;
        &self.mmap[self.metadata_offset..end]
    }
}
