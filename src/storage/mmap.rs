use std::fs::File;
use std::path::Path;
use memmap2::Mmap;
use crate::error::SutraError;
use super::header::{SutraHeader, HEADER_SIZE};

/// Zero-copy memory-mapped file reader for .sutra binary files.
pub struct MmapVectorStorage {
    mmap: Mmap,
    header: SutraHeader,
    vector_offset: usize,
    _vector_bytes_len: usize,
    metadata_offset: usize,
}

impl MmapVectorStorage {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SutraError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(SutraError::InvalidHeader("File size smaller than header".into()));
        }

        let header = SutraHeader::from_bytes(&mmap[0..HEADER_SIZE])?;
        let bytes_per_dim = if header.quant_mode == 1 { 1 } else { 4 };
        let vector_bytes_len = (header.doc_count as usize) * (header.dimension as usize) * bytes_per_dim;

        let vector_offset = HEADER_SIZE;
        let metadata_offset = vector_offset + vector_bytes_len;

        if mmap.len() < metadata_offset + (header.metadata_length as usize) {
            return Err(SutraError::InvalidHeader("File truncated".into()));
        }

        Ok(Self {
            mmap,
            header,
            vector_offset,
            _vector_bytes_len: vector_bytes_len,
            metadata_offset,
        })
    }

    pub fn header(&self) -> &SutraHeader {
        &self.header
    }

    /// Returns a zero-copy slice of float32 for doc at index `doc_idx`.
    /// Panics if quant_mode != 0 or index out of range.
    #[inline]
    pub fn get_vector_f32(&self, doc_idx: usize) -> &[f32] {
        assert_eq!(self.header.quant_mode, 0, "Not a float32 collection");
        assert!(doc_idx < self.header.doc_count as usize);

        let dim = self.header.dimension as usize;
        let start = self.vector_offset + doc_idx * dim * 4;
        let end = start + dim * 4;
        let slice = &self.mmap[start..end];

        unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const f32, dim)
        }
    }

    /// Returns a zero-copy slice of u8 for quantized doc at index `doc_idx`.
    #[inline]
    pub fn get_vector_sq8(&self, doc_idx: usize) -> &[u8] {
        assert_eq!(self.header.quant_mode, 1, "Not a SQ8 collection");
        assert!(doc_idx < self.header.doc_count as usize);

        let dim = self.header.dimension as usize;
        let start = self.vector_offset + doc_idx * dim;
        let end = start + dim;
        &self.mmap[start..end]
    }

    /// Returns metadata JSON bytes.
    pub fn metadata_bytes(&self) -> &[u8] {
        let end = self.metadata_offset + self.header.metadata_length as usize;
        &self.mmap[self.metadata_offset..end]
    }
}
