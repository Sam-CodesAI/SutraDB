use std::ops::Index;
use rayon::prelude::*;

use super::bq::{quantize_bq_into_words, try_hamming_distance_words};
use super::sq8::{quantize_sq8_into, sq8_score_slice};
use super::Metric;
use crate::error::SutraError;

/// Abstract read-only view over either a legacy `Vec<Vec<f32>>` or a contiguous `VectorArena`.
pub trait VectorStore: Sync {
    fn vec_len(&self) -> usize;
    fn vec_slice(&self, idx: usize) -> &[f32];
}

impl VectorStore for [Vec<f32>] {
    #[inline]
    fn vec_len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn vec_slice(&self, idx: usize) -> &[f32] {
        &self[idx]
    }
}

impl VectorStore for Vec<Vec<f32>> {
    #[inline]
    fn vec_len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn vec_slice(&self, idx: usize) -> &[f32] {
        &self[idx]
    }
}

/// Contiguous single-allocation float32 vector arena (`id * dim..(id + 1) * dim`).
#[derive(Debug, Clone, Default)]
pub struct VectorArena {
    pub data: Vec<f32>,
    pub dim: usize,
}

impl VectorArena {
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Option<&[f32]> {
        if self.dim == 0 {
            return None;
        }
        let start = idx.checked_mul(self.dim)?;
        let end = start.checked_add(self.dim)?;
        self.data.get(start..end)
    }

    #[inline]
    pub fn push_slice(&mut self, v: &[f32]) {
        self.data.extend_from_slice(v);
    }

    #[inline]
    pub fn set_slice(&mut self, idx: usize, v: &[f32]) {
        let start = idx * self.dim;
        let end = start + self.dim;
        self.data[start..end].copy_from_slice(v);
    }

    #[inline]
    pub fn as_flat_slice(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn iter(&self) -> std::slice::ChunksExact<'_, f32> {
        debug_assert!(self.dim > 0);
        self.data.chunks_exact(self.dim.max(1))
    }

    #[inline]
    pub fn par_iter(&self) -> rayon::slice::ChunksExact<'_, f32> {
        debug_assert!(self.dim > 0);
        self.data.par_chunks_exact(self.dim.max(1))
    }
}

impl Index<usize> for VectorArena {
    type Output = [f32];

    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        let start = idx * self.dim;
        &self.data[start..start + self.dim]
    }
}

impl VectorStore for VectorArena {
    #[inline]
    fn vec_len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn vec_slice(&self, idx: usize) -> &[f32] {
        &self[idx]
    }
}

/// Contiguous single-slab 8-bit Scalar Quantization arena (`u8` bytes + per-vector `(min_val, scale)`).
#[derive(Debug, Clone, Default)]
pub struct SQ8Arena {
    pub data: Vec<u8>,
    pub params: Vec<(f32, f32)>,
    pub dim: usize,
}

impl SQ8Arena {
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            params: Vec::new(),
            dim,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.params.clear();
    }

    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.dim);
        self.params.reserve(additional);
    }

    #[inline]
    pub fn push_vec(&mut self, v: &[f32]) {
        let (min_val, scale) = quantize_sq8_into(v, &mut self.data);
        self.params.push((min_val, scale));
    }

    #[inline]
    pub fn push_raw(&mut self, bytes: &[u8], min_val: f32, scale: f32) {
        self.data.extend_from_slice(bytes);
        self.params.push((min_val, scale));
    }

    #[inline]
    pub fn set_vec(&mut self, idx: usize, v: &[f32]) {
        let mut scratch = Vec::with_capacity(self.dim);
        let (min_val, scale) = quantize_sq8_into(v, &mut scratch);
        let start = idx * self.dim;
        self.data[start..start + self.dim].copy_from_slice(&scratch);
        self.params[idx] = (min_val, scale);
    }

    #[inline]
    pub fn data_slice(&self, idx: usize) -> &[u8] {
        let start = idx * self.dim;
        &self.data[start..start + self.dim]
    }

    #[inline]
    pub fn dequantize(&self, idx: usize) -> Vec<f32> {
        let (min_val, scale) = self.params[idx];
        self.data_slice(idx)
            .iter()
            .map(|&b| min_val + (b as f32) * scale)
            .collect()
    }

    #[inline]
    pub fn try_score_idx(
        &self,
        idx: usize,
        query: &[f32],
        query_sum: f32,
        metric: Metric,
    ) -> Result<f32, SutraError> {
        let (min_val, scale) = self.params[idx];
        sq8_score_slice(self.data_slice(idx), min_val, scale, query, query_sum, metric)
    }

    /// Quantizes all rows of a contiguous FP32 arena directly into the flat SQ8 slab.
    pub fn from_flat_f32(flat_f32: &[f32], dim: usize) -> Self {
        if dim == 0 || flat_f32.is_empty() {
            return Self::new(dim);
        }
        let n = flat_f32.len() / dim;
        let mut arena = Self {
            data: Vec::with_capacity(n * dim),
            params: Vec::with_capacity(n),
            dim,
        };
        for row in flat_f32.chunks_exact(dim) {
            arena.push_vec(row);
        }
        arena
    }
}

/// Contiguous single-slab 1-bit Binary Quantization arena (`u64` words packed consecutively).
#[derive(Debug, Clone, Default)]
pub struct BQArena {
    pub bits: Vec<u64>,
    pub dim: usize,
    pub words_per_vec: usize,
}

impl BQArena {
    pub fn new(dim: usize) -> Self {
        Self {
            bits: Vec::new(),
            dim,
            words_per_vec: dim.div_ceil(64),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        if self.words_per_vec == 0 {
            0
        } else {
            self.bits.len() / self.words_per_vec
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    pub fn clear(&mut self) {
        self.bits.clear();
    }

    #[inline]
    pub fn push_vec(&mut self, v: &[f32], centroid: Option<&[f32]>) {
        let start = self.bits.len();
        self.bits.resize(start + self.words_per_vec, 0u64);
        quantize_bq_into_words(v, centroid, &mut self.bits[start..start + self.words_per_vec]);
    }

    #[inline]
    pub fn set_vec(&mut self, idx: usize, v: &[f32], centroid: Option<&[f32]>) {
        let start = idx * self.words_per_vec;
        let end = start + self.words_per_vec;
        quantize_bq_into_words(v, centroid, &mut self.bits[start..end]);
    }

    #[inline]
    pub fn word_slice(&self, idx: usize) -> &[u64] {
        let start = idx * self.words_per_vec;
        &self.bits[start..start + self.words_per_vec]
    }

    #[inline]
    pub fn try_hamming_idx(&self, idx: usize, query_words: &[u64]) -> Result<u32, SutraError> {
        try_hamming_distance_words(self.word_slice(idx), query_words)
    }

    pub fn from_flat_f32(flat_f32: &[f32], dim: usize, centroid: Option<&[f32]>) -> Self {
        let words_per_vec = dim.div_ceil(64);
        if dim == 0 || flat_f32.is_empty() {
            return Self::new(dim);
        }
        let n = flat_f32.len() / dim;
        let mut bits = vec![0u64; n * words_per_vec];
        bits.par_chunks_exact_mut(words_per_vec)
            .zip(flat_f32.par_chunks_exact(dim))
            .for_each(|(out_words, vec_slice)| {
                quantize_bq_into_words(vec_slice, centroid, out_words);
            });
        Self {
            bits,
            dim,
            words_per_vec,
        }
    }
}

/// Contiguous single-slab Product Quantization arena (`u8` codes packed `m` bytes per vector).
#[derive(Debug, Clone, Default)]
pub struct PQArena {
    pub codes: Vec<u8>,
    pub m: usize,
}

impl PQArena {
    pub fn new(m: usize) -> Self {
        Self {
            codes: Vec::new(),
            m,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        if self.m == 0 {
            0
        } else {
            self.codes.len() / self.m
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    pub fn clear(&mut self) {
        self.codes.clear();
    }

    #[inline]
    pub fn push_codes(&mut self, c: &[u8]) {
        self.codes.extend_from_slice(c);
    }

    #[inline]
    pub fn set_codes(&mut self, idx: usize, c: &[u8]) {
        let start = idx * self.m;
        self.codes[start..start + self.m].copy_from_slice(c);
    }

    #[inline]
    pub fn code_slice(&self, idx: usize) -> &[u8] {
        let start = idx * self.m;
        &self.codes[start..start + self.m]
    }
}
