use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use crate::error::SutraError;

const WAL_BIN_MAGIC: [u8; 4] = *b"SWB1";

#[derive(Deserialize)]
struct WalRecordRaw {
    op: String,
    #[serde(default)]
    docs: Vec<serde_json::Value>,
    #[serde(default)]
    doc: Option<serde_json::Value>,
    #[serde(default)]
    ids: Vec<String>,
    #[serde(default)]
    id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "WalRecordRaw")]
pub struct WalRecord {
    pub op: String, // "insert", "update", "delete"
    #[serde(default)]
    pub docs: Vec<serde_json::Value>,
    #[serde(default)]
    pub ids: Vec<String>,
}

impl From<WalRecordRaw> for WalRecord {
    fn from(raw: WalRecordRaw) -> Self {
        let mut docs = raw.docs;
        if docs.is_empty() {
            if let Some(single) = raw.doc {
                docs.push(single);
            }
        }
        let mut ids = raw.ids;
        if ids.is_empty() {
            if let Some(single) = raw.id {
                ids.push(single);
            }
        }
        Self {
            op: raw.op,
            docs,
            ids,
        }
    }
}

/// Encodes a `WalRecord` into compact binary format (`SWB1`), storing dense `f32` vectors
/// as raw little-endian bytes instead of ASCII decimals.
fn encode_wal_record_binary(record: &WalRecord) -> Result<Vec<u8>, SutraError> {
    let mut buf = Vec::with_capacity(256);
    buf.extend_from_slice(&WAL_BIN_MAGIC);

    let op_bytes = record.op.as_bytes();
    let op_len = u16::try_from(op_bytes.len())
        .map_err(|_| SutraError::Serialization("WAL op name too long".into()))?;
    buf.extend_from_slice(&op_len.to_le_bytes());
    buf.extend_from_slice(op_bytes);

    let doc_count = u32::try_from(record.docs.len())
        .map_err(|_| SutraError::Serialization("WAL docs count overflow".into()))?;
    buf.extend_from_slice(&doc_count.to_le_bytes());
    for d in &record.docs {
        // Extract vector slice if present as JSON array of numbers for compact LE f32 encoding
        if let Some(obj) = d.as_object() {
            if let Some(vec_arr) = obj.get("vector").and_then(|v| v.as_array()) {
                let mut rest = obj.clone();
                rest.remove("vector");
                let json_bytes = serde_json::to_vec(&serde_json::Value::Object(rest))
                    .map_err(|e| SutraError::Serialization(e.to_string()))?;
                buf.push(1u8); // flag: has binary vector
                buf.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(&json_bytes);
                buf.extend_from_slice(&(vec_arr.len() as u32).to_le_bytes());
                for x in vec_arr {
                    let f = x.as_f64().unwrap_or(0.0) as f32;
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                continue;
            }
        }
        let json_bytes = serde_json::to_vec(d)
            .map_err(|e| SutraError::Serialization(e.to_string()))?;
        buf.push(0u8); // flag: pure JSON doc
        buf.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json_bytes);
    }

    let id_count = u32::try_from(record.ids.len())
        .map_err(|_| SutraError::Serialization("WAL ids count overflow".into()))?;
    buf.extend_from_slice(&id_count.to_le_bytes());
    for id in &record.ids {
        let id_bytes = id.as_bytes();
        let id_len = u32::try_from(id_bytes.len())
            .map_err(|_| SutraError::Serialization("WAL id too long".into()))?;
        buf.extend_from_slice(&id_len.to_le_bytes());
        buf.extend_from_slice(id_bytes);
    }

    Ok(buf)
}

fn read_u16_le(slice: &[u8], off: &mut usize) -> Option<u16> {
    let bytes: [u8; 2] = slice.get(*off..*off + 2)?.try_into().ok()?;
    *off += 2;
    Some(u16::from_le_bytes(bytes))
}

fn read_u32_le(slice: &[u8], off: &mut usize) -> Option<u32> {
    let bytes: [u8; 4] = slice.get(*off..*off + 4)?.try_into().ok()?;
    *off += 4;
    Some(u32::from_le_bytes(bytes))
}

fn decode_wal_record_binary(payload: &[u8]) -> Option<WalRecord> {
    if payload.len() < 4 || payload[0..4] != WAL_BIN_MAGIC {
        return None;
    }
    let mut off = 4usize;

    let op_len = read_u16_le(payload, &mut off)? as usize;
    let op_bytes = payload.get(off..off.checked_add(op_len)?)?;
    off += op_len;
    let op = String::from_utf8(op_bytes.to_vec()).ok()?;

    let doc_count = read_u32_le(payload, &mut off)? as usize;
    let mut docs = Vec::with_capacity(doc_count.min(65536));
    for _ in 0..doc_count {
        let flag = *payload.get(off)?;
        off += 1;
        let json_len = read_u32_le(payload, &mut off)? as usize;
        let json_slice = payload.get(off..off.checked_add(json_len)?)?;
        off += json_len;
        let mut val: serde_json::Value = serde_json::from_slice(json_slice).ok()?;
        if flag == 1 {
            let v_len = read_u32_le(payload, &mut off)? as usize;
            let byte_len = v_len.checked_mul(4)?;
            let f_slice = payload.get(off..off.checked_add(byte_len)?)?;
            off += byte_len;
            let mut vec_vals = Vec::with_capacity(v_len);
            let (chunks, _) = f_slice.as_chunks::<4>();
            for &chunk in chunks {
                let f = f32::from_le_bytes(chunk);
                vec_vals.push(serde_json::Value::from(f as f64));
            }
            if let Some(map) = val.as_object_mut() {
                map.insert("vector".to_string(), serde_json::Value::Array(vec_vals));
            }
        }
        docs.push(val);
    }

    let id_count = read_u32_le(payload, &mut off)? as usize;
    let mut ids = Vec::with_capacity(id_count.min(65536));
    for _ in 0..id_count {
        let id_len = read_u32_le(payload, &mut off)? as usize;
        let id_bytes = payload.get(off..off.checked_add(id_len)?)?;
        off += id_len;
        ids.push(String::from_utf8(id_bytes.to_vec()).ok()?);
    }

    if off != payload.len() {
        return None;
    }

    Some(WalRecord { op, docs, ids })
}

/// Append-only Write-Ahead Log with buffered writes (`BufWriter`), batched group commit,
/// compact binary framing (`SWB1`) with JSON replay backward compatibility, CRC32 verification,
/// and corrupted-tail recovery.
pub struct WriteAheadLog {
    path: PathBuf,
    writer: BufWriter<File>,
    pub sync_on_append: bool,
}

impl WriteAheadLog {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SutraError> {
        Self::open_with_sync(path, true)
    }

    pub fn open_with_sync<P: AsRef<Path>>(path: P, sync_on_append: bool) -> Result<Self, SutraError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::with_capacity(64 * 1024, file),
            sync_on_append,
        })
    }

    pub fn set_sync_on_append(&mut self, sync: bool) {
        self.sync_on_append = sync;
    }

    /// Appends a mutation entry using compact binary framing, syncing immediately if `sync_on_append` is enabled.
    pub fn append(&mut self, record: &WalRecord) -> Result<(), SutraError> {
        let payload = encode_wal_record_binary(record)?;
        self.write_framed(&payload, self.sync_on_append)
    }

    /// Buffers a mutation entry in userspace without issuing `fdatasync` until `flush()`.
    pub fn append_buffered(&mut self, record: &WalRecord) -> Result<(), SutraError> {
        let payload = encode_wal_record_binary(record)?;
        self.write_framed(&payload, false)
    }

    /// Writes a batch of WAL records into the buffered writer and commits with a single group `flush` + `sync_data`.
    pub fn append_batch(&mut self, records: &[WalRecord]) -> Result<(), SutraError> {
        if records.is_empty() {
            return Ok(());
        }
        for rec in records {
            let payload = encode_wal_record_binary(rec)?;
            self.write_framed(&payload, false)?;
        }
        self.flush()
    }

    /// Appends an arbitrary serializable JSON WAL record (e.g. `"config"` metadata entry).
    pub fn append_value(&mut self, value: &serde_json::Value) -> Result<(), SutraError> {
        let payload = serde_json::to_vec(value)
            .map_err(|e| SutraError::Serialization(e.to_string()))?;
        self.write_framed(&payload, self.sync_on_append)
    }

    fn write_framed(&mut self, payload: &[u8], sync: bool) -> Result<(), SutraError> {
        let len = u32::try_from(payload.len())
            .map_err(|_| SutraError::Serialization("WAL record exceeds u32::MAX".into()))?;

        let mut hasher = Hasher::new();
        hasher.update(payload);
        let crc = hasher.finalize();

        let mut header = [0u8; 8];
        header[0..4].copy_from_slice(&len.to_le_bytes());
        header[4..8].copy_from_slice(&crc.to_le_bytes());

        self.writer.write_all(&header)?;
        self.writer.write_all(payload)?;
        if sync {
            self.flush()?;
        }
        Ok(())
    }

    /// Flushes userspace `BufWriter` buffers and issues `sync_data` for durable group commit.
    pub fn flush(&mut self) -> Result<(), SutraError> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    /// Decodes a validated payload buffer supporting both `SWB1` compact binary and JSON frames.
    pub fn decode_payload(payload: &[u8]) -> Option<WalRecord> {
        if payload.starts_with(&WAL_BIN_MAGIC) {
            decode_wal_record_binary(payload)
        } else {
            serde_json::from_slice::<WalRecord>(payload).ok()
        }
    }

    /// Counts valid WAL frames read-only without truncating or opening a writer.
    pub fn count_records_at<P: AsRef<Path>>(wal_path: P) -> u64 {
        let mut reader = match File::open(wal_path) {
            Ok(f) => f,
            Err(_) => return 0,
        };
        let file_len = match reader.metadata() {
            Ok(m) => m.len(),
            Err(_) => return 0,
        };
        if file_len == 0 {
            return 0;
        }
        let mut count: u64 = 0;
        let mut offset: u64 = 0;
        loop {
            let mut header = [0u8; 8];
            if reader.read_exact(&mut header).is_err() {
                break;
            }
            let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
            let expected_crc = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

            if offset.saturating_add(8).saturating_add(len as u64) > file_len {
                break;
            }
            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() {
                break;
            }
            let mut hasher = Hasher::new();
            hasher.update(&payload);
            if hasher.finalize() != expected_crc {
                break;
            }
            if Self::decode_payload(&payload).is_none() {
                break;
            }
            count += 1;
            offset += 8 + len as u64;
        }
        count
    }

    /// Replays all valid binary (`SWB1`) and legacy JSON records from the WAL.
    /// In case of torn writes or trailing corruption, truncates file to last valid boundary.
    pub fn replay(&mut self) -> Result<Vec<WalRecord>, SutraError> {
        self.writer.flush()?;
        let mut reader = File::open(&self.path)?;
        let file_len = reader.metadata()?.len();
        let mut records = Vec::new();
        let mut valid_offset: u64 = 0;

        loop {
            let mut header = [0u8; 8];
            match reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(e) => return Err(SutraError::Io(e)),
            }

            let len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let expected_crc = u32::from_le_bytes(header[4..8].try_into().unwrap());

            if valid_offset.saturating_add(8).saturating_add(len as u64) > file_len {
                break;
            }

            let mut payload = vec![0u8; len];
            match reader.read_exact(&mut payload) {
                Ok(()) => {}
                Err(_) => break,
            }

            let mut hasher = Hasher::new();
            hasher.update(&payload);
            let computed_crc = hasher.finalize();

            if computed_crc != expected_crc {
                break;
            }

            match Self::decode_payload(&payload) {
                Some(rec) => {
                    records.push(rec);
                    valid_offset += 8 + len as u64;
                }
                None => break,
            }
        }

        if file_len > valid_offset {
            let file = OpenOptions::new().write(true).open(&self.path)?;
            file.set_len(valid_offset)?;
        }

        Ok(records)
    }

    /// Truncates the WAL to 0 bytes after a full snapshot is persisted,
    /// then reopens the handle in buffered append mode for subsequent writes.
    pub fn clear(&mut self) -> Result<(), SutraError> {
        self.writer.flush()?;
        {
            let truncated = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&self.path)?;
            truncated.sync_data()?;
        }
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&self.path)?;
        self.writer = BufWriter::with_capacity(64 * 1024, file);
        Ok(())
    }
}

impl Drop for WriteAheadLog {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}
