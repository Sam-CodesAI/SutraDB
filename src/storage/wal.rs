use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use crate::error::SutraError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub op: String, // "insert", "update", "delete"
    #[serde(default)]
    pub docs: Vec<serde_json::Value>,
    #[serde(default)]
    pub ids: Vec<String>,
}

/// Append-only Write-Ahead Log with CRC32 integrity verification
/// and corrupted-tail automatic recovery.
pub struct WriteAheadLog {
    path: PathBuf,
    file: File,
}

impl WriteAheadLog {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SutraError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self { path, file })
    }

    /// Appends a mutation entry with [u32 length, u32 crc32, payload_bytes].
    pub fn append(&mut self, record: &WalRecord) -> Result<(), SutraError> {
        let payload = serde_json::to_vec(record)
            .map_err(|e| SutraError::Serialization(e.to_string()))?;
        let len = payload.len() as u32;

        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let crc = hasher.finalize();

        let mut header = [0u8; 8];
        header[0..4].copy_from_slice(&len.to_le_bytes());
        header[4..8].copy_from_slice(&crc.to_le_bytes());

        self.file.write_all(&header)?;
        self.file.write_all(&payload)?;
        self.file.flush()?;
        Ok(())
    }

    /// Replays all valid records from the WAL.
    /// In case of torn writes or trailing corruption, truncates file to last valid boundary.
    pub fn replay(&mut self) -> Result<Vec<WalRecord>, SutraError> {
        let mut reader = File::open(&self.path)?;
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

            let mut payload = vec![0u8; len];
            match reader.read_exact(&mut payload) {
                Ok(()) => {}
                Err(_) => {
                    // Torn write: truncate back to last valid offset
                    break;
                }
            }

            let mut hasher = Hasher::new();
            hasher.update(&payload);
            let computed_crc = hasher.finalize();

            if computed_crc != expected_crc {
                // CRC mismatch: torn write detected
                break;
            }

            match serde_json::from_slice::<WalRecord>(&payload) {
                Ok(rec) => {
                    records.push(rec);
                    valid_offset += 8 + len as u64;
                }
                Err(_) => break,
            }
        }

        // If corruption was detected, heal the WAL file by truncating to valid_offset
        let file_len = reader.metadata()?.len();
        if file_len > valid_offset {
            let file = OpenOptions::new().write(true).open(&self.path)?;
            file.set_len(valid_offset)?;
        }

        Ok(records)
    }

    /// Truncates the WAL to 0 bytes after a full snapshot is persisted.
    pub fn clear(&mut self) -> Result<(), SutraError> {
        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        Ok(())
    }
}
