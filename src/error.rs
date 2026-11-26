use thiserror::Error;
#[cfg(feature = "python")]
use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::PyErr;

#[derive(Error, Debug)]
pub enum SutraError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    #[error("Integrity check failed: CRC32 mismatch (expected {expected}, computed {computed})")]
    ChecksumMismatch { expected: u32, computed: u32 },

    #[error("Corrupt WAL file: {0}")]
    CorruptWal(String),

    #[error("Collection '{0}' already exists")]
    CollectionAlreadyExists(String),

    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("Document '{0}' not found")]
    DocumentNotFound(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Index not trained: {0}")]
    IndexNotTrained(String),

    #[error("Lock error: {0}")]
    Lock(String),
}

#[cfg(feature = "python")]
impl From<SutraError> for PyErr {
    fn from(err: SutraError) -> PyErr {
        match err {
            SutraError::Io(e) => PyIOError::new_err(e.to_string()),
            SutraError::DimensionMismatch { expected, got } => {
                PyValueError::new_err(format!("Dimension mismatch: expected {}, got {}", expected, got))
            }
            SutraError::InvalidHeader(msg) => PyValueError::new_err(format!("Corrupt file: {}", msg)),
            SutraError::ChecksumMismatch { expected, computed } => {
                PyValueError::new_err(format!("Integrity check failed: CRC32 mismatch ({} != {})", computed, expected))
            }
            SutraError::CorruptWal(msg) => PyIOError::new_err(msg),
            SutraError::CollectionAlreadyExists(name) => {
                PyValueError::new_err(format!("Collection '{}' already exists.", name))
            }
            SutraError::CollectionNotFound(name) => {
                PyKeyError::new_err(format!("Collection '{}' not found.", name))
            }
            SutraError::DocumentNotFound(id) => {
                PyKeyError::new_err(format!("Document '{}' not found.", id))
            }
            SutraError::InvalidQuery(msg) => PyValueError::new_err(msg),
            SutraError::Serialization(msg) => PyValueError::new_err(msg),
            SutraError::IndexNotTrained(msg) => PyRuntimeError::new_err(msg),
            SutraError::Lock(msg) => PyRuntimeError::new_err(msg),
        }
    }
}
