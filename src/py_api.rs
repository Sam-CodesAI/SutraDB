use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, RwLock};

use crate::collection::{NativeCollection, NativeDocument};
use crate::distance::bq::BQVector;
use crate::distance::simd::{compute_vector_score, dot_product};
use crate::distance::sq8::SQ8Vector;
use crate::distance::Metric;
use crate::error::SutraError;
use crate::filter::haversine_distance;
use crate::index::hnsw::HNSWConfig;
use crate::index::ivf::IVFConfig;

fn py_any_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    // Check bool before int because Python `bool` is a subclass of `int`.
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(obj.extract::<bool>()?));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::Number(i.into()));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(Value::Number(u.into()));
        }
    }
    if obj.is_instance_of::<PyFloat>() {
        let f = obj.extract::<f64>()?;
        return Ok(serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(Value::String(obj.extract::<String>()?));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key = if let Ok(s) = k.extract::<String>() {
                s
            } else {
                k.str()?.extract::<String>()?
            };
            map.insert(key, py_any_to_json(py, &v)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut vec = Vec::with_capacity(list.len());
        for item in list.iter() {
            vec.push(py_any_to_json(py, &item)?);
        }
        return Ok(Value::Array(vec));
    }
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        let mut vec = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            vec.push(py_any_to_json(py, &item)?);
        }
        return Ok(Value::Array(vec));
    }

    // Handle NumPy scalar types cleanly without round-tripping through Python json.dumps
    if let Ok(tp_name) = obj.get_type().name() {
        let name = tp_name.to_str().unwrap_or_default();
        if name == "bool_" {
            if let Ok(b) = obj.extract::<bool>() {
                return Ok(Value::Bool(b));
            }
        } else if name.starts_with("int") || name.starts_with("uint") {
            if let Ok(i) = obj.extract::<i64>() {
                return Ok(Value::Number(i.into()));
            }
            if let Ok(u) = obj.extract::<u64>() {
                return Ok(Value::Number(u.into()));
            }
        } else if name.starts_with("float") {
            if let Ok(f) = obj.extract::<f64>() {
                return Ok(serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null));
            }
        }
    }

    if let Ok(json_str) = py.import("json")?.call_method1("dumps", (obj,))?.extract::<String>() {
        if let Ok(v) = serde_json::from_str::<Value>(&json_str) {
            return Ok(v);
        }
    }
    Ok(Value::Null)
}

fn json_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    match val {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

#[inline]
fn sparse_key_to_u32(key: &str) -> u32 {
    if let Ok(idx) = key.parse::<u32>() {
        return idx;
    }
    // Deterministic 32-bit FNV-1a hash with high bit set so string keys never collide with common indices
    let mut h: u32 = 0x811c9dc5;
    for &b in key.as_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h | 0x8000_0000
}

fn parse_sparse_py(item: &Bound<'_, PyAny>) -> PyResult<Option<crate::sparse::SparseVector>> {
    if item.is_none() {
        return Ok(None);
    }
    if let Ok(dict) = item.downcast::<PyDict>() {
        let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let val = v.extract::<f32>()?;
            if !val.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "sparse_vector value must be a finite number",
                ));
            }
            let idx = if let Ok(i) = k.extract::<i64>() {
                if i < 0 || i > u32::MAX as i64 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "sparse_vector index out of range",
                    ));
                }
                i as u32
            } else if let Ok(s) = k.extract::<String>() {
                sparse_key_to_u32(&s)
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "sparse_vector key must be int or str",
                ));
            };
            pairs.push((idx, val));
        }
        pairs.sort_by_key(|p| p.0);
        let idxs = pairs.iter().map(|p| p.0).collect();
        let vals = pairs.iter().map(|p| p.1).collect();
        return Ok(Some(crate::sparse::SparseVector::try_new(idxs, vals)?));
    }
    Ok(None)
}

fn extract_vectors_2d(vectors: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(arr) = vectors.extract::<PyReadonlyArray2<'_, f32>>() {
        let view = arr.as_array();
        let shape = view.shape();
        let rows = shape[0];
        let cols = shape[1];
        if cols == 0 {
            return Ok(vec![Vec::new(); rows]);
        }
        if let Ok(slice) = arr.as_slice() {
            return Ok(slice.chunks_exact(cols).map(|c| c.to_vec()).collect());
        }
        return Ok(view
            .rows()
            .into_iter()
            .map(|r| r.iter().copied().collect())
            .collect());
    }
    vectors.extract::<Vec<Vec<f32>>>()
}

fn extract_vector_1d(vector: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(arr) = vector.extract::<PyReadonlyArray1<'_, f32>>() {
        if let Ok(slice) = arr.as_slice() {
            return Ok(slice.to_vec());
        }
        return Ok(arr.as_array().iter().copied().collect());
    }
    vector.extract::<Vec<f32>>()
}

fn native_doc_to_py(
    py: Python<'_>,
    doc: &NativeDocument,
    vector: Option<&[f32]>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("id", &doc.id)?;
    dict.set_item("text", &doc.text)?;
    dict.set_item("namespace", doc.namespace.as_deref())?;
    dict.set_item("metadata", json_to_py(py, &doc.metadata)?)?;
    if let Some(vec_slice) = vector {
        dict.set_item("vector", PyArray1::from_slice(py, vec_slice))?;
    } else {
        dict.set_item("vector", py.None())?;
    }
    if let Some(ref sv) = doc.sparse_vector {
        let sp_dict = PyDict::new(py);
        for (&idx, &val) in sv.indices.iter().zip(sv.values.iter()) {
            sp_dict.set_item(idx, val)?;
        }
        dict.set_item("sparse_vector", sp_dict)?;
    } else {
        dict.set_item("sparse_vector", py.None())?;
    }
    if let Some(ref tokens) = doc.colbert_tokens {
        dict.set_item("colbert_tokens", tokens)?;
    } else {
        dict.set_item("colbert_tokens", py.None())?;
    }
    Ok(dict.into_any().unbind())
}

fn search_results_to_py(
    py: Python<'_>,
    results: Vec<crate::collection::NativeSearchResult>,
) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        let dict = PyDict::new(py);
        dict.set_item("id", r.id)?;
        dict.set_item("score", r.score)?;
        dict.set_item("text", r.text)?;
        dict.set_item("dense_score", r.dense_score)?;
        dict.set_item("bm25_score", r.bm25_score)?;
        dict.set_item("sparse_score", r.sparse_score)?;
        dict.set_item("colbert_score", r.colbert_score)?;
        dict.set_item("namespace", r.namespace)?;
        dict.set_item("metadata", json_to_py(py, &r.metadata)?)?;
        out.push(dict.into_any().unbind());
    }
    Ok(out)
}

#[pyclass(name = "NativeCollection")]
pub struct PyNativeCollection {
    inner: Arc<RwLock<NativeCollection>>,
}

#[pymethods]
impl PyNativeCollection {
    #[new]
    #[pyo3(signature = (name, dimension, metric="cosine", storage_path=None, enable_wal=false, analyzer=None))]
    fn new(
        name: String,
        dimension: usize,
        metric: &str,
        storage_path: Option<String>,
        enable_wal: bool,
        analyzer: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let m = Metric::from_str(metric)?;
        let path = storage_path.map(PathBuf::from);

        let mut config = crate::bm25::AnalyzerConfig::default();
        if let Some(dict) = analyzer {
            if let Some(lc) = dict.get_item("lowercase")? {
                config.lowercase = lc.extract::<bool>()?;
            }
            if let Some(st) = dict.get_item("stemmer")? {
                config.stemmer = st.extract::<bool>()?;
            }
            if let Some(sw) = dict.get_item("stopwords")? {
                config.stopwords = sw.extract::<bool>()?;
            }
            if let Some(lang) = dict.get_item("language")? {
                config.language = lang.extract::<String>()?;
            }
            if let Some(si) = dict.get_item("split_identifiers")? {
                config.split_identifiers = si.extract::<bool>()?;
            }
            if let Some(cjk) = dict.get_item("cjk_bigrams")? {
                config.cjk_bigrams = cjk.extract::<bool>()?;
            }
        }
        let anz = crate::bm25::Analyzer::new(config);
        let col = NativeCollection::with_analyzer(name, dimension, m, path, enable_wal, anz)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(col)),
        })
    }

    #[pyo3(signature = (documents, vectors))]
    fn insert(
        &self,
        py: Python,
        documents: Vec<Bound<'_, PyAny>>,
        vectors: Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let vecs = extract_vectors_2d(&vectors)?;
        let mut native_docs = Vec::with_capacity(documents.len());

        for item in documents {
            if let Ok(d) = item.downcast::<PyDict>() {
                let id = d
                    .get_item("id")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                let text = d
                    .get_item("text")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_default();
                let namespace = d
                    .get_item("namespace")?
                    .map(|v| v.extract::<Option<String>>())
                    .transpose()?
                    .flatten();
                let metadata_val = if let Some(meta_item) = d.get_item("metadata")? {
                    py_any_to_json(py, &meta_item)?
                } else {
                    Value::Object(serde_json::Map::new())
                };
                let sparse_vec = if let Some(sp_item) = d.get_item("sparse_vector")? {
                    parse_sparse_py(&sp_item)?
                } else {
                    None
                };
                let colbert_tokens: Option<Vec<Vec<f32>>> =
                    if let Some(cb_item) = d.get_item("colbert_tokens")? {
                        cb_item.extract::<Option<Vec<Vec<f32>>>>()?
                    } else {
                        None
                    };
                native_docs.push(NativeDocument {
                    id,
                    namespace,
                    text,
                    metadata: metadata_val,
                    sparse_vector: sparse_vec,
                    colbert_tokens,
                });
            } else {
                // Support Python Document dataclass / objects directly without building intermediate PyDicts
                let id = item.getattr("id")?.extract::<String>()?;
                let text = item.getattr("text")?.extract::<String>()?;
                let namespace = item.getattr("namespace")?.extract::<Option<String>>()?;
                let meta_obj = item.getattr("metadata")?;
                let metadata_val = py_any_to_json(py, &meta_obj)?;
                let sp_obj = item.getattr("sparse_vector")?;
                let sparse_vec = parse_sparse_py(&sp_obj)?;
                let cb_obj = item.getattr("colbert_tokens")?;
                let colbert_tokens = cb_obj.extract::<Option<Vec<Vec<f32>>>>()?;
                native_docs.push(NativeDocument {
                    id,
                    namespace,
                    text,
                    metadata: metadata_val,
                    sparse_vector: sparse_vec,
                    colbert_tokens,
                });
            }
        }

        py.allow_threads(|| -> Result<usize, SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.insert(native_docs, vecs)
        })
        .map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (vectors, ids, texts, metadatas=None, namespaces=None, sparse_vectors=None, colbert_tokens=None))]
    fn insert_numpy(
        &self,
        py: Python,
        vectors: PyReadonlyArray2<'_, f32>,
        ids: Vec<String>,
        texts: Vec<String>,
        metadatas: Option<Vec<Bound<'_, PyAny>>>,
        namespaces: Option<Vec<Option<String>>>,
        sparse_vectors: Option<Vec<Option<Bound<'_, PyAny>>>>,
        colbert_tokens: Option<Vec<Option<Vec<Vec<f32>>>>>,
    ) -> PyResult<usize> {
        let view = vectors.as_array();
        let shape = view.shape();
        let rows = shape[0];
        let cols = shape[1];
        if rows != ids.len() || rows != texts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Batch length mismatch: vectors={}, ids={}, texts={}",
                rows,
                ids.len(),
                texts.len()
            )));
        }

        let flat_vecs: Vec<f32> = if cols == 0 {
            Vec::new()
        } else if let Ok(slice) = vectors.as_slice() {
            slice.to_vec()
        } else {
            let mut buf = Vec::with_capacity(rows * cols);
            for r in view.rows() {
                buf.extend(r.iter().copied());
            }
            buf
        };

        let mut native_docs = Vec::with_capacity(rows);
        for i in 0..rows {
            let metadata = match metadatas.as_ref().and_then(|m| m.get(i)) {
                Some(obj) => py_any_to_json(py, obj)?,
                None => Value::Object(serde_json::Map::new()),
            };
            let namespace = namespaces.as_ref().and_then(|n| n.get(i).cloned()).flatten();
            let sparse_vector = match sparse_vectors.as_ref().and_then(|s| s.get(i)) {
                Some(Some(sp_obj)) => parse_sparse_py(sp_obj)?,
                _ => None,
            };
            let cb_toks = colbert_tokens
                .as_ref()
                .and_then(|c| c.get(i).cloned())
                .flatten();

            native_docs.push(NativeDocument {
                id: ids[i].clone(),
                namespace,
                text: texts[i].clone(),
                metadata,
                sparse_vector,
                colbert_tokens: cb_toks,
            });
        }

        py.allow_threads(|| -> Result<usize, SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.insert_flat(native_docs, flat_vecs)
        })
        .map_err(Into::into)
    }

    #[pyo3(signature = (documents, vectors))]
    fn upsert(
        &self,
        py: Python,
        documents: Vec<Bound<'_, PyAny>>,
        vectors: Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        self.insert(py, documents, vectors)
    }

    #[pyo3(signature = (doc_id, metadata))]
    fn update_metadata(
        &self,
        py: Python,
        doc_id: &str,
        metadata: Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let meta_val = py_any_to_json(py, &metadata)?;
        py.allow_threads(|| -> Result<bool, SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.update_metadata(doc_id, meta_val)
        })
        .map_err(Into::into)
    }

    #[pyo3(signature = (doc_id))]
    fn get(&self, py: Python, doc_id: &str) -> PyResult<Option<PyObject>> {
        let found: Option<(NativeDocument, Vec<f32>)> = py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            if let Some(&idx) = guard.id_to_index.get(doc_id) {
                if idx < guard.tombstones.len() && guard.tombstones[idx] {
                    return Ok(None);
                }
                if let (Some(doc), Some(vec)) = (guard.documents.get(idx), guard.get_vector(idx)) {
                    return Ok(Some((doc.clone(), vec.into_owned())));
                }
            }
            Ok(None)
        })?;

        match found {
            Some((doc, vec)) => Ok(Some(native_doc_to_py(py, &doc, Some(&vec))?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (include_vectors=true))]
    fn get_all(&self, py: Python, include_vectors: bool) -> PyResult<Vec<PyObject>> {
        let rows: Vec<(NativeDocument, Option<Vec<f32>>)> =
            py.allow_threads(|| -> Result<_, SutraError> {
                let guard = self
                    .inner
                    .read()
                    .map_err(|e| SutraError::Lock(e.to_string()))?;
                let mut out = Vec::new();
                for (i, doc) in guard.documents.iter().enumerate() {
                    if i < guard.tombstones.len() && guard.tombstones[i] {
                        continue;
                    }
                    let vec = if include_vectors {
                        guard.get_vector(i).map(|v| v.into_owned())
                    } else {
                        None
                    };
                    out.push((doc.clone(), vec));
                }
                Ok(out)
            })?;

        let mut out = Vec::with_capacity(rows.len());
        for (doc, vec) in rows {
            out.push(native_doc_to_py(py, &doc, vec.as_deref())?);
        }
        Ok(out)
    }

    #[pyo3(signature = (ids))]
    fn delete(&self, py: Python, ids: Bound<'_, PyAny>) -> PyResult<usize> {
        let id_list: Vec<String> = if let Ok(single) = ids.extract::<String>() {
            vec![single]
        } else {
            ids.extract::<Vec<String>>()?
        };
        py.allow_threads(|| -> Result<usize, SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.delete(&id_list)
        })
        .map_err(Into::into)
    }

    fn count(&self, py: Python) -> PyResult<usize> {
        py.allow_threads(|| -> Result<usize, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            Ok(guard.count())
        })
        .map_err(Into::into)
    }

    fn stats(&self, py: Python) -> PyResult<PyObject> {
        let val = py.allow_threads(|| -> Result<Value, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            Ok(guard.stats())
        })?;
        json_to_py(py, &val)
    }

    #[pyo3(signature = (m=16, ef_construction=200, ef_search=64))]
    fn build_hnsw(&self, py: Python, m: usize, ef_construction: usize, ef_search: usize) -> PyResult<()> {
        let cfg = HNSWConfig {
            m,
            m0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            seed: 42,
        };
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.build_hnsw(Some(cfg))
        })
        .map_err(Into::into)
    }

    #[pyo3(signature = (n_lists=None, nprobe=16))]
    fn build_ivf(&self, py: Python, n_lists: Option<usize>, nprobe: usize) -> PyResult<()> {
        let cfg = n_lists.map(|nl| IVFConfig {
            n_lists: nl,
            nprobe,
            kmeans_iters: 25,
            seed: 42,
        });
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.build_ivf(cfg)
        })
        .map_err(Into::into)
    }

    fn enable_sq8(&self, py: Python) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.enable_sq8();
            Ok(())
        })
        .map_err(Into::into)
    }

    fn enable_bq(&self, py: Python) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.enable_bq();
            Ok(())
        })
        .map_err(Into::into)
    }

    #[pyo3(signature = (m=None, max_samples=None))]
    fn train_pq(&self, py: Python, m: Option<usize>, max_samples: Option<usize>) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.train_pq(m, max_samples)
        })
        .map_err(|e| match e {
            SutraError::IndexNotTrained(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            other => PyErr::from(other),
        })
    }

    fn enable_pq(&self, py: Python) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.enable_pq()
        })
        .map_err(|e| match e {
            SutraError::IndexNotTrained(msg) => pyo3::exceptions::PyValueError::new_err(msg),
            other => PyErr::from(other),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        vector=None,
        text=None,
        sparse_vector=None,
        colbert_tokens=None,
        filter=None,
        namespace=None,
        top_k=10,
        hybrid=true,
        rrf_constant=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        sparse_weight=1.0,
        colbert_weight=1.0,
        alpha=None,
        use_ann=false
    ))]
    fn query(
        &self,
        py: Python,
        vector: Option<Bound<'_, PyAny>>,
        text: Option<&str>,
        sparse_vector: Option<Bound<'_, PyDict>>,
        colbert_tokens: Option<Vec<Vec<f32>>>,
        filter: Option<Bound<'_, PyAny>>,
        namespace: Option<&str>,
        top_k: usize,
        hybrid: bool,
        rrf_constant: usize,
        dense_weight: f32,
        bm25_weight: f32,
        sparse_weight: f32,
        colbert_weight: f32,
        alpha: Option<f32>,
        use_ann: bool,
    ) -> PyResult<Vec<PyObject>> {
        let vector_vec = vector.map(|v| extract_vector_1d(&v)).transpose()?;
        let filter_val: Option<Value> = match filter {
            Some(f_obj) if !f_obj.is_none() => Some(py_any_to_json(py, &f_obj)?),
            _ => None,
        };
        let sparse_vec = match sparse_vector {
            Some(sp_dict) => parse_sparse_py(sp_dict.as_any())?,
            None => None,
        };

        let results = py.allow_threads(|| {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.query(
                vector_vec.as_deref(),
                text,
                sparse_vec.as_ref(),
                colbert_tokens.as_deref(),
                namespace,
                filter_val.as_ref(),
                top_k,
                hybrid,
                alpha,
                rrf_constant,
                dense_weight,
                bm25_weight,
                sparse_weight,
                colbert_weight,
                use_ann,
            )
        })?;

        search_results_to_py(py, results)
    }

    #[pyo3(signature = (query_tokens, namespace=None, filter=None, top_k=10))]
    fn query_colbert(
        &self,
        py: Python,
        query_tokens: Vec<Vec<f32>>,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyAny>>,
        top_k: usize,
    ) -> PyResult<Vec<PyObject>> {
        let filter_val: Option<Value> = match filter {
            Some(f_obj) if !f_obj.is_none() => Some(py_any_to_json(py, &f_obj)?),
            _ => None,
        };

        let results = py.allow_threads(|| {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.query_colbert(&query_tokens, namespace, filter_val.as_ref(), top_k)
        })?;

        search_results_to_py(py, results)
    }

    #[pyo3(signature = (num_centroids=64, max_iters=15))]
    fn train_colbert_centroids(
        &self,
        py: Python,
        num_centroids: usize,
        max_iters: usize,
    ) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.colbert_index.train_centroids(num_centroids, max_iters)
        })
        .map_err(Into::into)
    }

    #[pyo3(signature = (path=None))]
    fn save(&self, py: Python, path: Option<String>) -> PyResult<u64> {
        py.allow_threads(|| -> Result<u64, SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.save(path.as_deref())
        })
        .map_err(Into::into)
    }

    #[pyo3(signature = (path=None))]
    fn load(&self, py: Python, path: Option<String>) -> PyResult<usize> {
        py.allow_threads(|| -> Result<usize, SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.load(path.as_deref())
        })
        .map_err(Into::into)
    }

    #[pyo3(signature = (path=None))]
    fn verify_checksum(&self, py: Python, path: Option<String>) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            let p = match path {
                Some(ref raw) => PathBuf::from(raw),
                None => match guard.storage_path {
                    Some(ref sp) => sp.clone(),
                    None => {
                        return Err(SutraError::Io(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "No storage path",
                        )))
                    }
                },
            };
            let storage = crate::storage::MmapVectorStorage::open_lazy(&p)?;
            storage.verify_checksum()
        })
        .map_err(Into::into)
    }

    fn compact(&self, py: Python) -> PyResult<usize> {
        py.allow_threads(|| -> Result<usize, SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            let before = guard.documents.len();
            guard.compact()?;
            let after = guard.documents.len();
            Ok(before.saturating_sub(after))
        })
        .map_err(Into::into)
    }

    // --- Inspection & compatibility shim helpers for Python Collection ---

    fn vectors_numpy(&self, py: Python) -> PyResult<PyObject> {
        let (rows, cols, flat) = py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            let n = guard.documents.len();
            let dim = guard.dimension;
            let buf = if !guard.vectors_evicted {
                guard.vectors.data.clone()
            } else {
                let mut out = Vec::with_capacity(n * dim);
                for i in 0..n {
                    if let Some(v) = guard.get_vector(i) {
                        out.extend_from_slice(&v);
                    } else {
                        out.resize(out.len() + dim, 0.0);
                    }
                }
                out
            };
            Ok((n, dim, buf))
        })?;
        if rows == 0 {
            return Ok(PyArray2::<f32>::zeros(py, (0, cols), false)
                .into_any()
                .unbind());
        }
        let arr1 = PyArray1::from_vec(py, flat);
        let arr2 = arr1.reshape([rows, cols])?;
        Ok(arr2.into_any().unbind())
    }

    fn all_documents_raw(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let rows: Vec<(NativeDocument, Vec<f32>)> = py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            Ok(guard
                .documents
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    (
                        d.clone(),
                        guard.get_vector(i).map(|v| v.into_owned()).unwrap_or_default(),
                    )
                })
                .collect())
        })?;
        let mut out = Vec::with_capacity(rows.len());
        for (doc, vec) in rows {
            out.push(native_doc_to_py(py, &doc, Some(&vec))?);
        }
        Ok(out)
    }

    fn id_to_index(&self, py: Python) -> PyResult<HashMap<String, usize>> {
        py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            Ok(guard.id_to_index.clone())
        })
        .map_err(Into::into)
    }

    fn bm25_score_query(&self, py: Python, text: &str) -> PyResult<Vec<f32>> {
        py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            Ok(guard.bm25.score_query(text, None))
        })
        .map_err(Into::into)
    }

    fn hnsw_layers(&self, py: Python) -> PyResult<Option<Vec<HashMap<u32, Vec<u32>>>>> {
        py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            Ok(guard.hnsw.as_ref().map(|h| h.layers.clone()))
        })
        .map_err(Into::into)
    }

    fn sq8_vectors_info(&self, py: Python) -> PyResult<Vec<(Vec<u8>, f32, f32)>> {
        py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            let n = guard.sq8_vectors.len();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let (min_val, scale) = guard.sq8_vectors.params[i];
                out.push((guard.sq8_vectors.data_slice(i).to_vec(), min_val, scale));
            }
            Ok(out)
        })
        .map_err(Into::into)
    }

    fn bq_vectors_info(&self, py: Python) -> PyResult<Vec<(Vec<u64>, usize)>> {
        py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            let n = guard.bq_vectors.len();
            let dim = guard.bq_vectors.dim;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push((guard.bq_vectors.word_slice(i).to_vec(), dim));
            }
            Ok(out)
        })
        .map_err(Into::into)
    }

    fn pq_vectors_info(&self, py: Python) -> PyResult<Vec<Vec<u8>>> {
        py.allow_threads(|| -> Result<_, SutraError> {
            let guard = self
                .inner
                .read()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            let n = guard.pq_vectors.len();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(guard.pq_vectors.code_slice(i).to_vec());
            }
            Ok(out)
        })
        .map_err(Into::into)
    }

    #[pyo3(signature = (path=None))]
    fn replay_wal(&self, py: Python, path: Option<String>) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            if let Some(p) = path {
                guard.replay_wal_file(p)
            } else if let Some(ref sp) = guard.storage_path {
                let wal_p = sp.with_extension("wal");
                guard.replay_wal_file(wal_p)
            } else {
                Ok(())
            }
        })
        .map_err(Into::into)
    }

    fn write_wal_config(&self, py: Python) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            let cfg = serde_json::json!({
                "op": "config",
                "dimension": guard.dimension,
                "metric": guard.metric.as_str(),
                "enable_wal": true,
            });
            if let Some(ref mut wal) = guard.wal {
                wal.append_value(&cfg)?;
            }
            Ok(())
        })
        .map_err(Into::into)
    }

    fn attach_wal(&self, py: Python, path: &str) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            let wal = crate::storage::WriteAheadLog::open(path)?;
            guard.wal = Some(wal);
            Ok(())
        })
        .map_err(Into::into)
    }

    fn close_wal(&self, py: Python) -> PyResult<()> {
        py.allow_threads(|| -> Result<(), SutraError> {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| SutraError::Lock(e.to_string()))?;
            guard.wal = None;
            Ok(())
        })
        .map_err(Into::into)
    }
}

#[pyfunction]
fn simd_dot(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(SutraError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        }
        .into());
    }
    Ok(dot_product(&a, &b))
}

#[pyfunction]
fn simd_cosine(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    compute_vector_score(&a, &b, Metric::Cosine, false).map_err(Into::into)
}

#[pyfunction]
fn simd_euclidean(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    compute_vector_score(&a, &b, Metric::Euclidean, false).map_err(Into::into)
}

#[pyfunction]
fn sq8_quantize(v: Vec<f32>) -> (Vec<u8>, f32, f32) {
    let sq = SQ8Vector::quantize(&v);
    (sq.data, sq.min_val, sq.scale)
}

#[pyfunction]
fn bq_quantize(v: Vec<f32>) -> (Vec<u64>, usize) {
    let bq = BQVector::quantize(&v);
    (bq.bits, bq.dimension)
}

#[pyfunction]
fn bq_hamming(a: Vec<u64>, b: Vec<u64>) -> PyResult<u32> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Slice length mismatch in bq_hamming: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(BQVector::hamming_distance_words(&a, &b))
}

#[pyfunction]
fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    haversine_distance(lat1, lon1, lat2, lon2)
}

#[pyfunction]
fn tokenize_text(text: &str) -> Vec<String> {
    crate::bm25::tokenizer::tokenize(text)
}

#[pyfunction]
fn detect_simd() -> String {
    crate::distance::simd::detect_simd_capabilities().to_string()
}

#[pyfunction]
fn stem_word(word: &str, language: &str) -> String {
    crate::bm25::stemmer::stem(word, language)
}

#[pyfunction]
#[pyo3(signature = (text, language="english"))]
fn analyze_text(text: &str, language: &str) -> Vec<String> {
    let config = crate::bm25::AnalyzerConfig {
        language: language.to_string(),
        ..Default::default()
    };
    let anz = crate::bm25::Analyzer::new(config);
    anz.analyze(text)
}

#[pyfunction]
#[pyo3(signature = (query, vectors, metric="cosine", is_normalized=false))]
fn batch_vector_distances(
    query: Vec<f32>,
    vectors: Vec<Vec<f32>>,
    metric: &str,
    is_normalized: bool,
) -> PyResult<Vec<f32>> {
    let m = Metric::from_str(metric)?;
    crate::distance::simd::try_batch_distances(&query, &vectors, m, is_normalized).map_err(Into::into)
}

#[pyclass(name = "NativeConsistentHashRing")]
pub struct PyConsistentHashRing {
    inner: crate::cluster::ConsistentHashRing,
}

#[pymethods]
impl PyConsistentHashRing {
    #[new]
    #[pyo3(signature = (vnodes_per_node=64))]
    fn new(vnodes_per_node: usize) -> Self {
        Self {
            inner: crate::cluster::ConsistentHashRing::new(vnodes_per_node),
        }
    }

    fn add_node(&mut self, node_id: &str) {
        self.inner.add_node(node_id);
    }

    fn remove_node(&mut self, node_id: &str) -> bool {
        self.inner.remove_node(node_id)
    }

    fn get_node(&self, key: &str) -> Option<String> {
        self.inner.get_node(key).map(|s| s.to_string())
    }

    fn get_preference_list(&self, key: &str, count: usize) -> Vec<String> {
        self.inner
            .get_preference_list(key, count)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn get_nodes(&self) -> Vec<String> {
        self.inner.get_nodes()
    }
}

pub fn register_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNativeCollection>()?;
    m.add_class::<PyConsistentHashRing>()?;
    m.add_function(wrap_pyfunction!(simd_dot, m)?)?;
    m.add_function(wrap_pyfunction!(simd_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(simd_euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(sq8_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(bq_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(bq_hamming, m)?)?;
    m.add_function(wrap_pyfunction!(haversine, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_text, m)?)?;
    m.add_function(wrap_pyfunction!(detect_simd, m)?)?;
    m.add_function(wrap_pyfunction!(stem_word, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_text, m)?)?;
    m.add_function(wrap_pyfunction!(batch_vector_distances, m)?)?;
    Ok(())
}
