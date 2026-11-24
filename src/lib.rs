pub mod error;
pub mod distance;
pub mod storage;
pub mod bm25;
pub mod filter;
pub mod fusion;
pub mod index;
pub mod collection;
pub mod py_api;

use pyo3::prelude::*;

#[pymodule]
fn _sutradb(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_api::register_module(py, m)?;
    Ok(())
}
