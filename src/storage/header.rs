use crate::error::SutraError;

pub const HEADER_MAGIC_V2: &[u8; 8] = b"SUTRA\x02\x00\x00";
pub const HEADER_MAGIC_V3: &[u8; 8] = b"SUTRA\x03\x00\x00";
pub const HEADER_SIZE: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SutraHeader {
    pub magic: [u8; 8],
    pub doc_count: u32,
    pub dimension: u32,
    pub metric: String,
    pub metadata_length: u64,
    pub crc32: u32,
    pub quant_mode: u8, // 0 = Float32, 1 = SQ8
}

impl SutraHeader {
    pub fn new(doc_count: u32, dimension: u32, metric: &str, metadata_length: u64, crc32: u32, quant_mode: u8) -> Self {
        Self {
            magic: *HEADER_MAGIC_V3,
            doc_count,
            dimension,
            metric: metric.to_string(),
            metadata_length,
            crc32,
            quant_mode,
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.doc_count.to_le_bytes());
        buf[12..16].copy_from_slice(&self.dimension.to_le_bytes());

        let mut metric_bytes = [0u8; 16];
        let raw = self.metric.as_bytes();
        let copy_len = raw.len().min(16);
        metric_bytes[0..copy_len].copy_from_slice(&raw[0..copy_len]);
        buf[16..32].copy_from_slice(&metric_bytes);

        buf[32..40].copy_from_slice(&self.metadata_length.to_le_bytes());
        buf[40..44].copy_from_slice(&self.crc32.to_le_bytes());
        buf[44] = self.quant_mode;
        // Remaining bytes 45..64 are 0 padding
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, SutraError> {
        if buf.len() < HEADER_SIZE {
            return Err(SutraError::InvalidHeader("Header too short".into()));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);
        if magic != *HEADER_MAGIC_V2 && magic != *HEADER_MAGIC_V3 {
            return Err(SutraError::InvalidHeader(format!(
                "Invalid magic bytes: {:?}",
                magic
            )));
        }

        let doc_count = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let dimension = u32::from_le_bytes(buf[12..16].try_into().unwrap());

        let metric_raw = &buf[16..32];
        let end = metric_raw.iter().position(|&b| b == 0).unwrap_or(16);
        let metric = String::from_utf8_lossy(&metric_raw[0..end]).to_string();

        let metadata_length = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        let crc32 = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        let quant_mode = if magic == *HEADER_MAGIC_V2 { 0 } else { buf[44] };

        Ok(Self {
            magic,
            doc_count,
            dimension,
            metric,
            metadata_length,
            crc32,
            quant_mode,
        })
    }
}
