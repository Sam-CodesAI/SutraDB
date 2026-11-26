pub mod error;
pub mod distance;
pub mod storage;
pub mod bm25;
pub mod filter;
pub mod fusion;
pub mod index;
pub mod sparse;
pub mod colbert;
pub mod collection;
pub mod cluster;
pub mod grpc;
#[cfg(feature = "python")]
pub mod py_api;

pub use storage::validate_collection_name;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _sutradb(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_api::register_module(py, m)?;
    Ok(())
}
