use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::str::FromStr;
use serde_json::Value;

use crate::collection::{NativeCollection, NativeDocument};
use crate::distance::bq::BQVector;
use crate::distance::simd::{dot_product, compute_vector_score};
use crate::distance::sq8::SQ8Vector;
use crate::distance::Metric;
use crate::error::SutraError;
use crate::filter::haversine_distance;
use crate::index::hnsw::HNSWConfig;
use crate::index::ivf::IVFConfig;

#[pyclass(name = "NativeCollection")]
pub struct PyNativeCollection {
    inner: NativeCollection,
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
        Ok(Self { inner: col })
    }

    #[pyo3(signature = (documents, vectors))]
    fn insert(&mut self, py: Python, documents: Vec<Bound<'_, PyDict>>, vectors: Vec<Vec<f32>>) -> PyResult<usize> {
        let mut native_docs = Vec::with_capacity(documents.len());
        for d in documents {
            let id = d.get_item("id")?.map(|v| v.extract::<String>()).transpose()?.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let text = d.get_item("text")?.map(|v| v.extract::<String>()).transpose()?.unwrap_or_default();
            let namespace = d.get_item("namespace")?.map(|v| v.extract::<String>()).transpose()?;
            
            let metadata_val: Value = if let Some(meta_item) = d.get_item("metadata")? {
                let json_str = py.import("json")?.call_method1("dumps", (meta_item,))?.extract::<String>()?;
                serde_json::from_str(&json_str).unwrap_or(Value::Object(serde_json::Map::new()))
            } else {
                Value::Object(serde_json::Map::new())
            };

            native_docs.push(NativeDocument {
                id,
                namespace,
                text,
                metadata: metadata_val,
            });
        }

        // Release the GIL during native vector insertion
        py.allow_threads(|| {
            self.inner.insert(native_docs, vectors).map_err(Into::into)
        })
    }

    #[pyo3(signature = (documents, vectors))]
    fn upsert(&mut self, py: Python, documents: Vec<Bound<'_, PyDict>>, vectors: Vec<Vec<f32>>) -> PyResult<usize> {
        self.insert(py, documents, vectors)
    }

    #[pyo3(signature = (doc_id, metadata))]
    fn update_metadata(&mut self, py: Python, doc_id: &str, metadata: Bound<'_, PyDict>) -> PyResult<bool> {
        let json_str = py.import("json")?.call_method1("dumps", (metadata,))?.extract::<String>()?;
        let meta_val: Value = serde_json::from_str(&json_str).unwrap_or(Value::Object(serde_json::Map::new()));
        self.inner.update_metadata(doc_id, meta_val).map_err(Into::into)
    }

    #[pyo3(signature = (ids))]
    fn delete(&mut self, py: Python, ids: Vec<String>) -> PyResult<usize> {
        py.allow_threads(|| {
            self.inner.delete(&ids).map_err(Into::into)
        })
    }

    fn count(&self) -> usize {
        self.inner.count()
    }

    fn stats(&self, py: Python) -> PyResult<PyObject> {
        let val = self.inner.stats();
        let json_str = serde_json::to_string(&val).map_err(|e| SutraError::Serialization(e.to_string()))?;
        let dict = py.import("json")?.call_method1("loads", (json_str,))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (m=16, ef_construction=64, ef_search=32))]
    fn build_hnsw(&mut self, py: Python, m: usize, ef_construction: usize, ef_search: usize) -> PyResult<()> {
        let cfg = HNSWConfig {
            m,
            m0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            seed: 42,
        };
        py.allow_threads(|| {
            self.inner.build_hnsw(Some(cfg)).map_err(Into::into)
        })
    }

    #[pyo3(signature = (n_lists=None, nprobe=16))]
    fn build_ivf(&mut self, py: Python, n_lists: Option<usize>, nprobe: usize) -> PyResult<()> {
        let cfg = n_lists.map(|nl| IVFConfig {
            n_lists: nl,
            nprobe,
            kmeans_iters: 25,
            seed: 42,
        });
        py.allow_threads(|| {
            self.inner.build_ivf(cfg).map_err(Into::into)
        })
    }

    fn enable_sq8(&mut self, py: Python) {
        py.allow_threads(|| {
            self.inner.enable_sq8();
        });
    }

    fn enable_bq(&mut self, py: Python) {
        py.allow_threads(|| {
            self.inner.enable_bq();
        });
    }

    #[pyo3(signature = (m=None, max_samples=None))]
    fn train_pq(&mut self, py: Python, m: Option<usize>, max_samples: Option<usize>) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.train_pq(m, max_samples).map_err(Into::into)
        })
    }

    fn enable_pq(&mut self, py: Python) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.enable_pq().map_err(Into::into)
        })
    }

    #[pyo3(signature = (
        vector=None,
        text=None,
        namespace=None,
        filter=None,
        top_k=10,
        hybrid=true,
        alpha=None,
        rrf_constant=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        use_ann=false
    ))]
    fn query(
        &mut self,
        py: Python,
        vector: Option<Vec<f32>>,
        text: Option<&str>,
        namespace: Option<&str>,
        filter: Option<Bound<'_, PyDict>>,
        top_k: usize,
        hybrid: bool,
        alpha: Option<f32>,
        rrf_constant: usize,
        dense_weight: f32,
        bm25_weight: f32,
        use_ann: bool,
    ) -> PyResult<Vec<PyObject>> {
        let filter_val: Option<Value> = if let Some(f_dict) = filter {
            let json_str = py.import("json")?.call_method1("dumps", (f_dict,))?.extract::<String>()?;
            Some(serde_json::from_str(&json_str).unwrap_or(Value::Null))
        } else {
            None
        };

        let results = py.allow_threads(|| {
            self.inner.query(
                vector.as_deref(),
                text,
                namespace,
                filter_val.as_ref(),
                top_k,
                hybrid,
                alpha,
                rrf_constant,
                dense_weight,
                bm25_weight,
                use_ann,
            )
        })?;

        let json_mod = py.import("json")?;
        let mut out = Vec::with_capacity(results.len());

        for r in results {
            let dict = PyDict::new(py);
            dict.set_item("id", r.id)?;
            dict.set_item("score", r.score)?;
            dict.set_item("text", r.text)?;
            dict.set_item("dense_score", r.dense_score)?;
            dict.set_item("bm25_score", r.bm25_score)?;
            dict.set_item("namespace", r.namespace)?;

            let meta_json = serde_json::to_string(&r.metadata).unwrap_or_default();
            let meta_py = json_mod.call_method1("loads", (meta_json,))?;
            dict.set_item("metadata", meta_py)?;

            out.push(dict.into());
        }

        Ok(out)
    }

    #[pyo3(signature = (path=None))]
    fn save(&mut self, py: Python, path: Option<String>) -> PyResult<u64> {
        py.allow_threads(|| {
            self.inner.save(path.as_deref()).map_err(Into::into)
        })
    }

    #[pyo3(signature = (path=None))]
    fn load(&mut self, py: Python, path: Option<String>) -> PyResult<usize> {
        py.allow_threads(|| {
            self.inner.load(path.as_deref()).map_err(Into::into)
        })
    }

    fn compact(&mut self, py: Python) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.compact().map_err(Into::into)
        })
    }
}

#[pyfunction]
fn simd_dot(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(SutraError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        }.into());
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
fn bq_hamming(a: Vec<u64>, b: Vec<u64>) -> u32 {
    BQVector::hamming_distance_words(&a, &b)
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
    let mut config = crate::bm25::AnalyzerConfig::default();
    config.language = language.to_string();
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
    Ok(crate::distance::simd::batch_distances(&query, &vectors, m, is_normalized))
}

pub fn register_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNativeCollection>()?;
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
