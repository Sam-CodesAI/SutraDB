use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use crc32fast::Hasher;
use roaring::RoaringBitmap;

use crate::colbert::ColbertCentroids;
use crate::distance::pq::PQCodebook;
use crate::distance::{BQArena, Metric, PQArena};
use crate::error::SutraError;
use crate::index::hnsw::{HNSWConfig, HNSWIndex};
use crate::index::ivf::{IVFConfig, IVFIndex};

pub const IDX_MAGIC: &[u8; 8] = b"SUTRAIDX";
pub const IDX_VERSION: u32 = 1;
pub const IDX_HEADER_SIZE: usize = 64;

const SECTION_HNSW: u8 = 1;
const SECTION_IVF: u8 = 2;
const SECTION_BQ: u8 = 3;
const SECTION_PQ: u8 = 4;
const SECTION_COLBERT_CENTROIDS: u8 = 5;

/// Returns the canonical index sidecar path `<primary>.idx` (e.g. `foo.sutra.idx`).
pub fn index_sidecar_path<P: AsRef<Path>>(primary_path: P) -> PathBuf {
    let mut os = primary_path.as_ref().as_os_str().to_os_string();
    os.push(".idx");
    PathBuf::from(os)
}

fn temp_sidecar_path(idx_path: &Path) -> PathBuf {
    let mut os = idx_path.as_os_str().to_os_string();
    os.push(".tmp");
    PathBuf::from(os)
}

/// Borrowed references to all optional built/trained indexes for persistence.
pub struct PersistedIndexes<'a> {
    pub hnsw: Option<&'a HNSWIndex>,
    pub ivf: Option<&'a IVFIndex>,
    pub bq: Option<(&'a BQArena, Option<&'a [f32]>)>,
    pub pq: Option<(&'a PQArena, &'a PQCodebook)>,
    pub colbert_centroids: Option<(usize, &'a ColbertCentroids)>,
    pub dirty_index: bool,
}

impl<'a> PersistedIndexes<'a> {
    pub fn is_empty(&self) -> bool {
        self.hnsw.is_none()
            && self.ivf.is_none()
            && self.bq.is_none()
            && self.pq.is_none()
            && self.colbert_centroids.is_none()
    }
}

/// Owned index structures restored directly from `.sutra.idx`.
pub struct RestoredIndexes {
    pub hnsw: Option<HNSWIndex>,
    pub ivf: Option<IVFIndex>,
    pub bq: Option<(BQArena, Option<Vec<f32>>)>,
    pub pq: Option<(PQArena, PQCodebook)>,
    pub colbert_centroids: Option<(usize, ColbertCentroids)>,
    pub dirty_index: bool,
}

struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], SutraError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| SutraError::InvalidHeader("Index section length overflow".into()))?;
        if end > self.buf.len() {
            return Err(SutraError::InvalidHeader(
                "Truncated index section payload".into(),
            ));
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, SutraError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, SutraError> {
        let b = self.read_exact(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, SutraError> {
        let b = self.read_exact(4)?;
        Ok(i32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, SutraError> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, SutraError> {
        let b = self.read_exact(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_usize_checked(&mut self) -> Result<usize, SutraError> {
        usize::try_from(self.read_u32()?)
            .map_err(|_| SutraError::InvalidHeader("u32 length overflows usize".into()))
    }

    fn read_f32_vec(&mut self, count: usize) -> Result<Vec<f32>, SutraError> {
        let byte_len = count
            .checked_mul(4)
            .ok_or_else(|| SutraError::InvalidHeader("f32 vector byte length overflow".into()))?;
        let raw = self.read_exact(byte_len)?;
        let (chunks, _) = raw.as_chunks::<4>();
        let mut out = Vec::with_capacity(count);
        for &chunk in chunks {
            out.push(f32::from_le_bytes(chunk));
        }
        Ok(out)
    }

    fn read_u32_vec(&mut self, count: usize) -> Result<Vec<u32>, SutraError> {
        let byte_len = count
            .checked_mul(4)
            .ok_or_else(|| SutraError::InvalidHeader("u32 vector byte length overflow".into()))?;
        let raw = self.read_exact(byte_len)?;
        let (chunks, _) = raw.as_chunks::<4>();
        let mut out = Vec::with_capacity(count);
        for &chunk in chunks {
            out.push(u32::from_le_bytes(chunk));
        }
        Ok(out)
    }
}

fn push_u32_slice(out: &mut Vec<u8>, slice: &[u32]) {
    out.reserve(slice.len() * 4);
    for &v in slice {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_f32_slice(out: &mut Vec<u8>, slice: &[f32]) {
    out.reserve(slice.len() * 4);
    for &v in slice {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn encode_hnsw(hnsw: &HNSWIndex) -> Result<Vec<u8>, SutraError> {
    let mut out = Vec::with_capacity(1024);
    out.extend_from_slice(&(hnsw.config.m as u32).to_le_bytes());
    out.extend_from_slice(&(hnsw.config.m0 as u32).to_le_bytes());
    out.extend_from_slice(&(hnsw.config.ef_construction as u32).to_le_bytes());
    out.extend_from_slice(&(hnsw.config.ef_search as u32).to_le_bytes());
    out.extend_from_slice(&hnsw.config.ml.to_le_bytes());
    out.extend_from_slice(&hnsw.config.seed.to_le_bytes());

    match hnsw.entry_point {
        Some(ep) => {
            out.push(1);
            out.extend_from_slice(&ep.to_le_bytes());
        }
        None => {
            out.push(0);
            out.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    out.extend_from_slice(&(hnsw.max_level as u32).to_le_bytes());

    out.extend_from_slice(&(hnsw.node_levels.len() as u32).to_le_bytes());
    for lvl_opt in &hnsw.node_levels {
        let encoded: i32 = match lvl_opt {
            Some(l) => i32::try_from(*l)
                .map_err(|_| SutraError::InvalidHeader("HNSW node level overflows i32".into()))?,
            None => -1,
        };
        out.extend_from_slice(&encoded.to_le_bytes());
    }

    out.extend_from_slice(&(hnsw.layers.len() as u32).to_le_bytes());
    for layer in &hnsw.layers {
        out.extend_from_slice(&(layer.len() as u32).to_le_bytes());
        for nbrs in layer {
            out.extend_from_slice(&(nbrs.len() as u32).to_le_bytes());
            push_u32_slice(&mut out, nbrs);
        }
    }

    let mut bitmap_bytes = Vec::with_capacity(hnsw.tombstones.serialized_size());
    hnsw.tombstones
        .serialize_into(&mut bitmap_bytes)
        .map_err(|e| SutraError::Serialization(e.to_string()))?;
    out.extend_from_slice(&(bitmap_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bitmap_bytes);
    Ok(out)
}

fn decode_hnsw(payload: &[u8]) -> Result<HNSWIndex, SutraError> {
    let mut r = ByteReader::new(payload);
    let m = r.read_usize_checked()?;
    let m0 = r.read_usize_checked()?;
    let ef_construction = r.read_usize_checked()?;
    let ef_search = r.read_usize_checked()?;
    let ml = r.read_f64()?;
    let seed = r.read_u64()?;
    let config = HNSWConfig {
        m,
        m0,
        ef_construction,
        ef_search,
        ml,
        seed,
    };

    let has_ep = r.read_u8()?;
    let ep_val = r.read_u32()?;
    let entry_point = if has_ep != 0 { Some(ep_val) } else { None };
    let max_level = r.read_usize_checked()?;

    let node_levels_len = r.read_usize_checked()?;
    let mut node_levels = Vec::with_capacity(node_levels_len);
    for _ in 0..node_levels_len {
        let v = r.read_i32()?;
        if v < -1 {
            return Err(SutraError::InvalidHeader(format!(
                "Invalid HNSW node level encoding: {}",
                v
            )));
        }
        node_levels.push(if v < 0 { None } else { Some(v as usize) });
    }

    let num_layers = r.read_usize_checked()?;
    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let num_nodes = r.read_usize_checked()?;
        let mut layer = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let deg = r.read_usize_checked()?;
            layer.push(r.read_u32_vec(deg)?);
        }
        layers.push(layer);
    }

    let tomb_len = r.read_usize_checked()?;
    let tomb_bytes = r.read_exact(tomb_len)?;
    let tombstones = RoaringBitmap::deserialize_from(tomb_bytes)
        .map_err(|e| SutraError::InvalidHeader(format!("Corrupt HNSW tombstones: {}", e)))?;

    if r.remaining() != 0 {
        return Err(SutraError::InvalidHeader(
            "Trailing bytes in HNSW index section".into(),
        ));
    }

    if let Some(ep) = entry_point {
        if (ep as usize) >= node_levels.len() || layers.is_empty() || max_level >= layers.len() {
            return Err(SutraError::InvalidHeader(
                "HNSW entry_point or max_level out of bounds".into(),
            ));
        }
    }
    for layer in &layers {
        if layer.len() != node_levels.len() {
            return Err(SutraError::InvalidHeader(
                "HNSW layer node count mismatch".into(),
            ));
        }
        for nbrs in layer {
            for &nbr in nbrs {
                if (nbr as usize) >= node_levels.len() {
                    return Err(SutraError::InvalidHeader(
                        "HNSW neighbor ID out of bounds".into(),
                    ));
                }
            }
        }
    }

    Ok(HNSWIndex::from_persisted(
        config,
        entry_point,
        max_level,
        layers,
        node_levels,
        tombstones,
    ))
}

fn encode_ivf(ivf: &IVFIndex) -> Vec<u8> {
    let mut out = Vec::with_capacity(512);
    out.extend_from_slice(&(ivf.config.n_lists as u32).to_le_bytes());
    out.extend_from_slice(&(ivf.config.nprobe as u32).to_le_bytes());
    out.extend_from_slice(&(ivf.config.kmeans_iters as u32).to_le_bytes());
    out.extend_from_slice(&ivf.config.seed.to_le_bytes());
    out.push(ivf.metric.to_u8());
    out.push(u8::from(ivf.is_trained));
    out.extend_from_slice(&(ivf.n_vectors as u32).to_le_bytes());

    let num_centroids = ivf.centroids.len() as u32;
    let centroid_dim = ivf.centroids.first().map_or(0, Vec::len) as u32;
    out.extend_from_slice(&num_centroids.to_le_bytes());
    out.extend_from_slice(&centroid_dim.to_le_bytes());
    for c in &ivf.centroids {
        push_f32_slice(&mut out, c);
    }

    let mut list_entries: Vec<(&usize, &Vec<u32>)> = ivf.inverted_lists.iter().collect();
    list_entries.sort_by_key(|(&k, _)| k);
    out.extend_from_slice(&(list_entries.len() as u32).to_le_bytes());
    for (&list_id, docs) in list_entries {
        out.extend_from_slice(&(list_id as u32).to_le_bytes());
        out.extend_from_slice(&(docs.len() as u32).to_le_bytes());
        push_u32_slice(&mut out, docs);
    }
    out
}

fn decode_ivf(payload: &[u8]) -> Result<IVFIndex, SutraError> {
    let mut r = ByteReader::new(payload);
    let config = IVFConfig {
        n_lists: r.read_usize_checked()?,
        nprobe: r.read_usize_checked()?,
        kmeans_iters: r.read_usize_checked()?,
        seed: r.read_u64()?,
    };
    let metric = Metric::from_u8(r.read_u8()?)?;
    let is_trained = r.read_u8()? != 0;
    let n_vectors = r.read_usize_checked()?;

    let num_centroids = r.read_usize_checked()?;
    let centroid_dim = r.read_usize_checked()?;
    let mut centroids = Vec::with_capacity(num_centroids);
    for _ in 0..num_centroids {
        centroids.push(r.read_f32_vec(centroid_dim)?);
    }

    let num_lists = r.read_usize_checked()?;
    let mut inverted_lists = HashMap::with_capacity(num_lists);
    for _ in 0..num_lists {
        let list_id = r.read_usize_checked()?;
        let len = r.read_usize_checked()?;
        inverted_lists.insert(list_id, r.read_u32_vec(len)?);
    }

    if r.remaining() != 0 {
        return Err(SutraError::InvalidHeader(
            "Trailing bytes in IVF index section".into(),
        ));
    }

    Ok(IVFIndex {
        config,
        metric,
        centroids,
        inverted_lists,
        is_trained,
        n_vectors,
    })
}

fn encode_bq(arena: &BQArena, centroid: Option<&[f32]>) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 + arena.bits.len() * 8);
    out.extend_from_slice(&(arena.dim as u32).to_le_bytes());
    out.extend_from_slice(&(arena.words_per_vec as u32).to_le_bytes());
    out.extend_from_slice(&(arena.bits.len() as u64).to_le_bytes());
    for &w in &arena.bits {
        out.extend_from_slice(&w.to_le_bytes());
    }
    match centroid {
        Some(c) => {
            out.push(1);
            out.extend_from_slice(&(c.len() as u32).to_le_bytes());
            push_f32_slice(&mut out, c);
        }
        None => {
            out.push(0);
        }
    }
    out
}

fn decode_bq(payload: &[u8]) -> Result<(BQArena, Option<Vec<f32>>), SutraError> {
    let mut r = ByteReader::new(payload);
    let dim = r.read_usize_checked()?;
    let words_per_vec = r.read_usize_checked()?;
    let num_words = usize::try_from(r.read_u64()?)
        .map_err(|_| SutraError::InvalidHeader("BQ word count overflows usize".into()))?;
    let raw_words_bytes = num_words
        .checked_mul(8)
        .ok_or_else(|| SutraError::InvalidHeader("BQ byte length overflow".into()))?;
    let words_slice = r.read_exact(raw_words_bytes)?;
    let (word_chunks, _) = words_slice.as_chunks::<8>();
    let mut bits = Vec::with_capacity(num_words);
    for &chunk in word_chunks {
        bits.push(u64::from_le_bytes(chunk));
    }

    let has_centroid = r.read_u8()?;
    let centroid = if has_centroid != 0 {
        let len = r.read_usize_checked()?;
        Some(r.read_f32_vec(len)?)
    } else {
        None
    };

    if r.remaining() != 0 {
        return Err(SutraError::InvalidHeader(
            "Trailing bytes in BQ index section".into(),
        ));
    }

    Ok((
        BQArena {
            bits,
            dim,
            words_per_vec,
        },
        centroid,
    ))
}

fn encode_pq(arena: &PQArena, cb: &PQCodebook) -> Vec<u8> {
    let codebook_floats: usize = cb.centroids.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(32 + codebook_floats * 4 + arena.codes.len());
    out.extend_from_slice(&(cb.dimension as u32).to_le_bytes());
    out.extend_from_slice(&(cb.m as u32).to_le_bytes());
    out.extend_from_slice(&(cb.k as u32).to_le_bytes());
    out.extend_from_slice(&(cb.sub_dim as u32).to_le_bytes());
    out.extend_from_slice(&(cb.centroids.len() as u32).to_le_bytes());
    for sub in &cb.centroids {
        out.extend_from_slice(&(sub.len() as u32).to_le_bytes());
        push_f32_slice(&mut out, sub);
    }
    out.extend_from_slice(&(arena.m as u32).to_le_bytes());
    out.extend_from_slice(&(arena.codes.len() as u64).to_le_bytes());
    out.extend_from_slice(&arena.codes);
    out
}

fn decode_pq(payload: &[u8]) -> Result<(PQArena, PQCodebook), SutraError> {
    let mut r = ByteReader::new(payload);
    let dimension = r.read_usize_checked()?;
    let m = r.read_usize_checked()?;
    let k = r.read_usize_checked()?;
    let sub_dim = r.read_usize_checked()?;
    let num_subs = r.read_usize_checked()?;
    let mut centroids = Vec::with_capacity(num_subs);
    for _ in 0..num_subs {
        let len = r.read_usize_checked()?;
        centroids.push(r.read_f32_vec(len)?);
    }
    let arena_m = r.read_usize_checked()?;
    let codes_len = usize::try_from(r.read_u64()?)
        .map_err(|_| SutraError::InvalidHeader("PQ codes length overflows usize".into()))?;
    let codes = r.read_exact(codes_len)?.to_vec();

    if r.remaining() != 0 {
        return Err(SutraError::InvalidHeader(
            "Trailing bytes in PQ index section".into(),
        ));
    }

    if m == 0 || k == 0 || k > 256 || sub_dim == 0 || num_subs != m || arena_m != m || dimension != m.saturating_mul(sub_dim) {
        return Err(SutraError::InvalidHeader(
            "Invalid PQ codebook geometry".into(),
        ));
    }
    let expected_sub_len = k.checked_mul(sub_dim).ok_or_else(|| {
        SutraError::InvalidHeader("PQ codebook subvector table overflow".into())
    })?;
    for sub in &centroids {
        if sub.len() != expected_sub_len {
            return Err(SutraError::InvalidHeader(
                "PQ codebook subvector length mismatch".into(),
            ));
        }
    }
    if !codes.len().is_multiple_of(arena_m) {
        return Err(SutraError::InvalidHeader(
            "PQ codes length is not a multiple of m".into(),
        ));
    }

    Ok((
        PQArena { codes, m: arena_m },
        PQCodebook {
            dimension,
            m,
            k,
            sub_dim,
            centroids,
        },
    ))
}

fn encode_colbert_centroids(token_dim: usize, cc: &ColbertCentroids) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(&(token_dim as u32).to_le_bytes());
    let num_centroids = cc.centroids.len() as u32;
    let centroid_dim = cc.centroids.first().map_or(0, Vec::len) as u32;
    out.extend_from_slice(&num_centroids.to_le_bytes());
    out.extend_from_slice(&centroid_dim.to_le_bytes());
    for c in &cc.centroids {
        push_f32_slice(&mut out, c);
    }
    let mut postings_entries: Vec<(&usize, &Vec<u32>)> = cc.postings.iter().collect();
    postings_entries.sort_by_key(|(&k, _)| k);
    out.extend_from_slice(&(postings_entries.len() as u32).to_le_bytes());
    for (&c_idx, docs) in postings_entries {
        out.extend_from_slice(&(c_idx as u32).to_le_bytes());
        out.extend_from_slice(&(docs.len() as u32).to_le_bytes());
        push_u32_slice(&mut out, docs);
    }
    out
}

fn decode_colbert_centroids(payload: &[u8]) -> Result<(usize, ColbertCentroids), SutraError> {
    let mut r = ByteReader::new(payload);
    let token_dim = r.read_usize_checked()?;
    let num_centroids = r.read_usize_checked()?;
    let centroid_dim = r.read_usize_checked()?;
    let mut centroids = Vec::with_capacity(num_centroids);
    for _ in 0..num_centroids {
        centroids.push(r.read_f32_vec(centroid_dim)?);
    }
    let num_postings = r.read_usize_checked()?;
    let mut postings = HashMap::with_capacity(num_postings);
    for _ in 0..num_postings {
        let c_idx = r.read_usize_checked()?;
        let len = r.read_usize_checked()?;
        postings.insert(c_idx, r.read_u32_vec(len)?);
    }

    if r.remaining() != 0 {
        return Err(SutraError::InvalidHeader(
            "Trailing bytes in ColBERT centroids section".into(),
        ));
    }

    Ok((
        token_dim,
        ColbertCentroids {
            centroids,
            postings,
        },
    ))
}

fn write_section(file: &mut File, tag: u8, payload: &[u8]) -> Result<(), SutraError> {
    let mut hasher = Hasher::new();
    hasher.update(payload);
    let payload_crc = hasher.finalize();

    let mut framing = [0u8; 16];
    framing[0] = tag;
    // bytes 1..4 reserved
    framing[4..12].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    framing[12..16].copy_from_slice(&payload_crc.to_le_bytes());

    file.write_all(&framing)?;
    file.write_all(payload)?;
    Ok(())
}

/// Atomic writer/reader for versioned `.sutra.idx` sidecars with header + per-section CRC32.
pub struct IndexStorage;

impl IndexStorage {
    /// Atomically persists built/trained indexes alongside `primary_path` (or removes any stale sidecar if empty).
    pub fn write_for_parent<P: AsRef<Path>>(
        primary_path: P,
        doc_count: u32,
        dimension: u32,
        parent_crc32: u32,
        indexes: &PersistedIndexes<'_>,
    ) -> Result<u64, SutraError> {
        let idx_path = index_sidecar_path(&primary_path);
        if indexes.is_empty() {
            if idx_path.exists() {
                std::fs::remove_file(&idx_path)?;
            }
            return Ok(0);
        }

        let mut sections: Vec<(u8, Vec<u8>)> = Vec::new();
        if let Some(hnsw) = indexes.hnsw {
            sections.push((SECTION_HNSW, encode_hnsw(hnsw)?));
        }
        if let Some(ivf) = indexes.ivf {
            sections.push((SECTION_IVF, encode_ivf(ivf)));
        }
        if let Some((arena, centroid)) = indexes.bq {
            sections.push((SECTION_BQ, encode_bq(arena, centroid)));
        }
        if let Some((arena, cb)) = indexes.pq {
            sections.push((SECTION_PQ, encode_pq(arena, cb)));
        }
        if let Some((token_dim, cc)) = indexes.colbert_centroids {
            sections.push((SECTION_COLBERT_CENTROIDS, encode_colbert_centroids(token_dim, cc)));
        }

        let mut header = [0u8; IDX_HEADER_SIZE];
        header[0..8].copy_from_slice(IDX_MAGIC);
        header[8..12].copy_from_slice(&IDX_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&doc_count.to_le_bytes());
        header[16..20].copy_from_slice(&dimension.to_le_bytes());
        header[20..24].copy_from_slice(&parent_crc32.to_le_bytes());
        header[24..28].copy_from_slice(&(sections.len() as u32).to_le_bytes());
        header[28] = u8::from(indexes.dirty_index);

        let mut h_hasher = Hasher::new();
        h_hasher.update(&header[0..60]);
        let h_crc = h_hasher.finalize();
        header[60..64].copy_from_slice(&h_crc.to_le_bytes());

        if let Some(parent) = idx_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temp_path = temp_sidecar_path(&idx_path);
        {
            let mut file = File::create(&temp_path)?;
            file.write_all(&header)?;
            for (tag, payload) in &sections {
                write_section(&mut file, *tag, payload)?;
            }
            file.flush()?;
            file.sync_data()?;
        }

        std::fs::rename(&temp_path, &idx_path)?;
        Ok(std::fs::metadata(&idx_path)?.len())
    }

    /// Reads and verifies `.sutra.idx` if present and matching `primary_path`'s header.
    /// Returns `Ok(None)` when no sidecar exists or when a valid sidecar belongs to an older/external
    /// `.sutra` rewrite (graceful V2/V3/V4 fallback). Returns `Err` on corrupted header/section CRC32.
    pub fn read_for_parent<P: AsRef<Path>>(
        primary_path: P,
        expected_doc_count: u32,
        expected_dimension: u32,
        expected_parent_crc32: u32,
    ) -> Result<Option<RestoredIndexes>, SutraError> {
        let idx_path = index_sidecar_path(&primary_path);
        if !idx_path.exists() {
            return Ok(None);
        }

        let file = File::open(&idx_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        if mmap.len() < IDX_HEADER_SIZE {
            return Err(SutraError::InvalidHeader(
                "Index sidecar smaller than 64-byte header".into(),
            ));
        }

        if &mmap[0..8] != IDX_MAGIC {
            return Err(SutraError::InvalidHeader(
                "Invalid index sidecar magic".into(),
            ));
        }

        let stored_hcrc = u32::from_le_bytes(mmap[60..64].try_into().unwrap());
        let mut h_hasher = Hasher::new();
        h_hasher.update(&mmap[0..60]);
        let computed_hcrc = h_hasher.finalize();
        if computed_hcrc != stored_hcrc {
            return Err(SutraError::InvalidHeader(format!(
                "Index sidecar header CRC32 mismatch: expected {}, got {}",
                stored_hcrc, computed_hcrc
            )));
        }

        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != IDX_VERSION {
            return Err(SutraError::InvalidHeader(format!(
                "Unsupported index sidecar version: {}",
                version
            )));
        }

        let doc_count = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
        let dimension = u32::from_le_bytes(mmap[16..20].try_into().unwrap());
        let parent_crc32 = u32::from_le_bytes(mmap[20..24].try_into().unwrap());
        let section_count = u32::from_le_bytes(mmap[24..28].try_into().unwrap()) as usize;
        let dirty_index = mmap[28] != 0;

        // If the parent .sutra file was rewritten externally without updating .sutra.idx, ignore stale sidecar cleanly.
        if doc_count != expected_doc_count
            || dimension != expected_dimension
            || parent_crc32 != expected_parent_crc32
        {
            return Ok(None);
        }

        let mut restored = RestoredIndexes {
            hnsw: None,
            ivf: None,
            bq: None,
            pq: None,
            colbert_centroids: None,
            dirty_index,
        };

        let mut offset = IDX_HEADER_SIZE;
        for _ in 0..section_count {
            let hdr_end = offset
                .checked_add(16)
                .ok_or_else(|| SutraError::InvalidHeader("Section offset overflow".into()))?;
            if hdr_end > mmap.len() {
                return Err(SutraError::InvalidHeader(
                    "Truncated index sidecar section header".into(),
                ));
            }
            let tag = mmap[offset];
            let payload_len = usize::try_from(u64::from_le_bytes(
                mmap[offset + 4..offset + 12].try_into().unwrap(),
            ))
            .map_err(|_| SutraError::InvalidHeader("Section payload length overflow".into()))?;
            let stored_crc = u32::from_le_bytes(mmap[offset + 12..offset + 16].try_into().unwrap());

            let payload_end = hdr_end
                .checked_add(payload_len)
                .ok_or_else(|| SutraError::InvalidHeader("Section payload end overflow".into()))?;
            if payload_end > mmap.len() {
                return Err(SutraError::InvalidHeader(
                    "Truncated index sidecar section payload".into(),
                ));
            }

            let payload = &mmap[hdr_end..payload_end];
            let mut hasher = Hasher::new();
            hasher.update(payload);
            let computed_crc = hasher.finalize();
            if computed_crc != stored_crc {
                return Err(SutraError::InvalidHeader(format!(
                    "Index sidecar section {} CRC32 mismatch: expected {}, got {}",
                    tag, stored_crc, computed_crc
                )));
            }

            match tag {
                SECTION_HNSW => restored.hnsw = Some(decode_hnsw(payload)?),
                SECTION_IVF => restored.ivf = Some(decode_ivf(payload)?),
                SECTION_BQ => restored.bq = Some(decode_bq(payload)?),
                SECTION_PQ => restored.pq = Some(decode_pq(payload)?),
                SECTION_COLBERT_CENTROIDS => {
                    restored.colbert_centroids = Some(decode_colbert_centroids(payload)?)
                }
                unknown => {
                    return Err(SutraError::InvalidHeader(format!(
                        "Unknown index sidecar section tag: {}",
                        unknown
                    )));
                }
            }

            offset = payload_end;
        }

        if offset != mmap.len() {
            return Err(SutraError::InvalidHeader(
                "Trailing bytes in index sidecar file".into(),
            ));
        }

        Ok(Some(restored))
    }
}
