use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::task::JoinSet;
use tonic::{Request, Response, Status};

use crate::collection::{NativeCollection, NativeDocument};
use crate::distance::Metric;
use crate::error::SutraError;
use crate::grpc::proto::sutra_service_server::SutraService;
use crate::grpc::proto::*;
use crate::sparse::SparseVector;
use crate::storage::{MmapVectorStorage, WriteAheadLog};
use crate::validate_collection_name;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CollectionManifest {
    name: String,
    dimension: usize,
    metric: String,
    enable_wal: bool,
}

#[inline]
pub(crate) fn sutra_path_for(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{}.sutra", name))
}

#[inline]
fn wal_path_for(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{}.wal", name))
}

#[inline]
fn tmp_path_for(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{}.tmp", name))
}

#[inline]
fn manifest_path_for(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{}.meta.json", name))
}

#[inline]
fn manifest_tmp_path_for(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{}.meta.tmp", name))
}

fn write_manifest_atomic(dir: &Path, manifest: &CollectionManifest) -> std::io::Result<()> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }
    let payload = serde_json::to_vec_pretty(manifest)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp_path = manifest_tmp_path_for(dir, &manifest.name);
    let final_path = manifest_path_for(dir, &manifest.name);
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(&payload)?;
        f.flush()?;
        f.sync_data()?;
    }
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
}

pub(crate) fn write_manifest_for_collection(col: &NativeCollection) -> std::io::Result<()> {
    if let Some(ref sp) = col.storage_path {
        let dir = sp.parent().unwrap_or_else(|| Path::new("."));
        let manifest = CollectionManifest {
            name: col.name.clone(),
            dimension: col.dimension,
            metric: col.metric.as_str().to_string(),
            enable_wal: col.wal.is_some(),
        };
        write_manifest_atomic(dir, &manifest)?;
    }
    Ok(())
}

pub(crate) fn count_wal_records(wal_path: &Path) -> u64 {
    WriteAheadLog::count_records_at(wal_path)
}

pub(crate) fn wal_records_for_collection(col: &NativeCollection) -> u64 {
    if col.wal.is_none() {
        return 0;
    }
    match col.storage_path {
        Some(ref sp) => count_wal_records(&sp.with_extension("wal")),
        None => 0,
    }
}

pub(crate) fn sutra_err_to_status(err: SutraError) -> Status {
    match err {
        SutraError::DimensionMismatch { .. }
        | SutraError::InvalidQuery(_)
        | SutraError::Serialization(_) => Status::invalid_argument(err.to_string()),
        SutraError::CollectionNotFound(_) | SutraError::DocumentNotFound(_) => {
            Status::not_found(err.to_string())
        }
        SutraError::CollectionAlreadyExists(_) => Status::already_exists(err.to_string()),
        SutraError::IndexNotTrained(_) => Status::failed_precondition(err.to_string()),
        SutraError::Io(_)
        | SutraError::InvalidHeader(_)
        | SutraError::ChecksumMismatch { .. }
        | SutraError::CorruptWal(_)
        | SutraError::Lock(_) => Status::internal(err.to_string()),
    }
}

fn canonicalize_allowing_nonexistent(path: &Path) -> Result<PathBuf, Box<Status>> {
    if path.exists() {
        return path
            .canonicalize()
            .map_err(|e| Box::new(Status::invalid_argument(format!("Invalid data_dir path: {}", e))));
    }
    let mut missing_parts: Vec<std::ffi::OsString> = Vec::new();
    let mut current = path;
    while !current.exists() {
        if let Some(name) = current.file_name() {
            missing_parts.push(name.to_os_string());
        } else {
            return Err(Box::new(Status::invalid_argument("Invalid data_dir path")));
        }
        current = match current.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
    }
    let mut resolved = current
        .canonicalize()
        .map_err(|e| Box::new(Status::invalid_argument(format!("Invalid data_dir path: {}", e))))?;
    for part in missing_parts.into_iter().rev() {
        resolved.push(part);
    }
    Ok(resolved)
}

fn infer_dimension_from_wal(wal_path: &Path) -> Result<Option<usize>, SutraError> {
    let mut wal = WriteAheadLog::open(wal_path)?;
    let records = wal.replay()?;
    for rec in records {
        if rec.op == "insert" {
            for doc in rec.docs {
                if let Some(arr) = doc.get("vector").and_then(|v| v.as_array()) {
                    if !arr.is_empty() {
                        return Ok(Some(arr.len()));
                    }
                }
            }
        }
    }
    Ok(None)
}

fn recover_collection_from_dir(dir: &Path, name: &str) -> Result<Option<NativeCollection>, SutraError> {
    let sutra_path = sutra_path_for(dir, name);
    let wal_path = wal_path_for(dir, name);
    let meta_path = manifest_path_for(dir, name);

    let manifest: Option<CollectionManifest> = if meta_path.is_file() {
        match std::fs::read(&meta_path) {
            Ok(bytes) => match serde_json::from_slice::<CollectionManifest>(&bytes) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!(
                        "[SutraDB] Warning: corrupt manifest '{}': {}. Attempting fallback recovery.",
                        meta_path.display(),
                        e
                    );
                    None
                }
            },
            Err(e) => {
                eprintln!(
                    "[SutraDB] Warning: could not read manifest '{}': {}. Attempting fallback recovery.",
                    meta_path.display(),
                    e
                );
                None
            }
        }
    } else {
        None
    };

    if sutra_path.is_file() {
        let storage = MmapVectorStorage::open(&sutra_path)?;
        let header = storage.header().clone();
        let dim = header.dimension as usize;
        if dim == 0 {
            return Err(SutraError::InvalidHeader(format!(
                "Snapshot '{}' has dimension 0",
                sutra_path.display()
            )));
        }
        let metric = Metric::from_str(&header.metric).unwrap_or(Metric::Cosine);
        drop(storage);

        let enable_wal = manifest.as_ref().map(|m| m.enable_wal).unwrap_or(true);
        let mut col = NativeCollection::new(
            name.to_string(),
            dim,
            metric,
            Some(sutra_path.clone()),
            enable_wal,
        )?;
        col.load(Some(&sutra_path))?;
        return Ok(Some(col));
    }

    if let Some(m) = manifest {
        if m.dimension == 0 {
            return Err(SutraError::InvalidHeader(format!(
                "Manifest '{}' has dimension 0",
                meta_path.display()
            )));
        }
        let metric = Metric::from_str(&m.metric).unwrap_or(Metric::Cosine);
        let col = NativeCollection::new(
            name.to_string(),
            m.dimension,
            metric,
            Some(sutra_path),
            m.enable_wal,
        )?;
        return Ok(Some(col));
    }

    if wal_path.is_file() {
        if let Some(dim) = infer_dimension_from_wal(&wal_path)? {
            let col = NativeCollection::new(
                name.to_string(),
                dim,
                Metric::Cosine,
                Some(sutra_path),
                true,
            )?;
            return Ok(Some(col));
        } else {
            eprintln!(
                "[SutraDB] Warning: orphan WAL '{}' has no vector inserts or manifest; skipping.",
                wal_path.display()
            );
            return Ok(None);
        }
    }

    Ok(None)
}

pub type CollectionHandle = Arc<RwLock<NativeCollection>>;
pub type CollectionMap = Arc<RwLock<HashMap<String, CollectionHandle>>>;

#[derive(Clone)]
pub struct SutraGrpcService {
    collections: CollectionMap,
    start_time: Instant,
    default_data_dir: Option<PathBuf>,
}

impl SutraGrpcService {
    pub fn new(default_data_dir: Option<PathBuf>) -> Self {
        let mut recovered = HashMap::new();

        if let Some(ref dir) = default_data_dir {
            if dir.is_dir() {
                match std::fs::read_dir(dir) {
                    Ok(entries) => {
                        let mut candidates = BTreeSet::new();
                        for entry_res in entries {
                            let entry = match entry_res {
                                Ok(e) => e,
                                Err(e) => {
                                    eprintln!(
                                        "[SutraDB] Warning: error reading entry in '{}': {}",
                                        dir.display(),
                                        e
                                    );
                                    continue;
                                }
                            };
                            let path = entry.path();
                            if !path.is_file() {
                                continue;
                            }
                            let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
                                continue;
                            };
                            let candidate = if let Some(stem) = fname.strip_suffix(".meta.json") {
                                Some(stem)
                            } else if let Some(stem) = fname.strip_suffix(".sutra") {
                                Some(stem)
                            } else {
                                fname.strip_suffix(".wal")
                            };
                            if let Some(name) = candidate {
                                if validate_collection_name(name).is_ok() {
                                    candidates.insert(name.to_string());
                                }
                            }
                        }

                        for name in candidates {
                            match recover_collection_from_dir(dir, &name) {
                                Ok(Some(col)) => {
                                    recovered.insert(name, Arc::new(RwLock::new(col)));
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    eprintln!(
                                        "[SutraDB] Warning: failed to recover collection '{}' from '{}': {}",
                                        name,
                                        dir.display(),
                                        e
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[SutraDB] Warning: failed to scan data directory '{}': {}",
                            dir.display(),
                            e
                        );
                    }
                }
            }
        }

        Self {
            collections: Arc::new(RwLock::new(recovered)),
            start_time: Instant::now(),
            default_data_dir,
        }
    }

    pub fn with_collection(name: String, collection: NativeCollection) -> Self {
        let mut map = HashMap::new();
        map.insert(name, Arc::new(RwLock::new(collection)));
        Self {
            collections: Arc::new(RwLock::new(map)),
            start_time: Instant::now(),
            default_data_dir: None,
        }
    }

    pub async fn add_collection(&self, name: String, collection: NativeCollection) {
        let mut map = self.collections.write().await;
        map.insert(name, Arc::new(RwLock::new(collection)));
    }

    pub fn collections(&self) -> &CollectionMap {
        &self.collections
    }

    pub fn default_data_dir(&self) -> Option<&PathBuf> {
        self.default_data_dir.as_ref()
    }

    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    fn resolve_confined_data_dir(&self, req_data_dir: Option<&str>) -> Result<Option<PathBuf>, Box<Status>> {
        let raw = match req_data_dir {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => return Ok(self.default_data_dir.clone()),
        };

        if raw.contains('\0') {
            return Err(Box::new(Status::invalid_argument(
                "Invalid data_dir: contains NUL byte",
            )));
        }

        let base_dir = self.default_data_dir.as_ref().ok_or_else(|| {
            Box::new(Status::invalid_argument(
                "Custom data_dir is disallowed when server has no default_data_dir",
            ))
        })?;

        let req_path = Path::new(raw);
        if req_path
            .components()
            .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(Box::new(Status::invalid_argument(
                "Invalid data_dir: path traversal ('..') is not allowed",
            )));
        }

        if !base_dir.exists() {
            std::fs::create_dir_all(base_dir).map_err(|e| {
                Box::new(Status::internal(format!(
                    "Failed to create default_data_dir: {}",
                    e
                )))
            })?;
        }
        let canonical_base = base_dir.canonicalize().map_err(|e| {
            Box::new(Status::internal(format!(
                "Failed to canonicalize default_data_dir: {}",
                e
            )))
        })?;

        let candidate = if req_path.is_absolute() {
            req_path.to_path_buf()
        } else {
            canonical_base.join(req_path)
        };

        let resolved = canonicalize_allowing_nonexistent(&candidate)?;
        if !resolved.starts_with(&canonical_base) {
            return Err(Box::new(Status::invalid_argument(
                "Invalid data_dir: path escapes server default_data_dir",
            )));
        }

        Ok(Some(resolved))
    }

    pub(crate) async fn create_collection_internal(
        &self,
        name: &str,
        dimension: usize,
        metric_str: &str,
        data_dir: Option<&str>,
        enable_wal: bool,
    ) -> Result<(bool, Option<serde_json::Value>), Box<Status>> {
        validate_collection_name(name).map_err(|e| Box::new(sutra_err_to_status(e)))?;
        if dimension == 0 {
            return Err(Box::new(Status::invalid_argument(
                "dimension must be greater than 0",
            )));
        }
        let metric = Metric::from_str(metric_str).map_err(|e| Box::new(sutra_err_to_status(e)))?;
        let resolved_dir = self.resolve_confined_data_dir(data_dir)?;
        let storage_path = resolved_dir.as_ref().map(|dir| sutra_path_for(dir, name));

        let name_owned = name.to_string();
        {
            let map = self.collections.read().await;
            if map.contains_key(&name_owned) {
                return Ok((false, None));
            }
        }

        let sp_clone = storage_path.clone();
        let name_for_blocking = name_owned.clone();
        let (col, stats) = tokio::task::spawn_blocking(move || -> Result<(NativeCollection, serde_json::Value), Box<Status>> {
            let col = NativeCollection::new(
                name_for_blocking.clone(),
                dimension,
                metric,
                sp_clone.clone(),
                enable_wal,
            )
            .map_err(|e| Box::new(sutra_err_to_status(e)))?;

            if let Some(ref sp) = sp_clone {
                let dir = sp.parent().unwrap_or_else(|| Path::new("."));
                let manifest = CollectionManifest {
                    name: name_for_blocking,
                    dimension,
                    metric: metric.as_str().to_string(),
                    enable_wal,
                };
                write_manifest_atomic(dir, &manifest).map_err(|e| {
                    Box::new(Status::internal(format!(
                        "Failed to write collection manifest: {}",
                        e
                    )))
                })?;
            }

            let mut stats = col.stats();
            if let Some(obj) = stats.as_object_mut() {
                obj.insert(
                    "wal_records".to_string(),
                    serde_json::Value::from(wal_records_for_collection(&col)),
                );
                obj.insert(
                    "is_dirty".to_string(),
                    serde_json::Value::from(col.dirty_index),
                );
            }
            Ok((col, stats))
        })
        .await
        .map_err(|e| Box::new(Status::internal(format!("create_collection task panicked: {}", e))))??;

        let mut map = self.collections.write().await;
        if map.contains_key(&name_owned) {
            return Ok((false, None));
        }

        map.insert(name_owned, Arc::new(RwLock::new(col)));
        Ok((true, Some(stats)))
    }

    pub(crate) async fn drop_collection_internal(&self, name: &str) -> Result<bool, Box<Status>> {
        validate_collection_name(name).map_err(|e| Box::new(sutra_err_to_status(e)))?;

        let mut map = self.collections.write().await;
        let removed_arc = map.remove(name);
        let removed_from_mem = removed_arc.is_some();

        let storage_path = if let Some(col_arc) = removed_arc {
            let sp = {
                let guard = col_arc.read().await;
                guard.storage_path.clone()
            };
            drop(col_arc);
            sp
        } else {
            None
        };

        let mut dirs_to_clean: Vec<PathBuf> = Vec::new();
        if let Some(ref sp) = storage_path {
            if let Some(parent) = sp.parent() {
                dirs_to_clean.push(parent.to_path_buf());
            }
        }
        if let Some(ref default_dir) = self.default_data_dir {
            if !dirs_to_clean.iter().any(|d| d == default_dir) {
                dirs_to_clean.push(default_dir.clone());
            }
        }

        let mut removed_any_file = false;
        let mut unlink = |p: PathBuf| -> Result<(), Box<Status>> {
            match std::fs::remove_file(&p) {
                Ok(()) => {
                    removed_any_file = true;
                    Ok(())
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(Box::new(Status::internal(format!(
                    "Failed to delete collection artifact '{}': {}",
                    p.display(),
                    e
                )))),
            }
        };

        for dir in dirs_to_clean {
            unlink(sutra_path_for(&dir, name))?;
            unlink(wal_path_for(&dir, name))?;
            unlink(tmp_path_for(&dir, name))?;
            unlink(manifest_path_for(&dir, name))?;
            unlink(manifest_tmp_path_for(&dir, name))?;
        }

        if let Some(sp) = storage_path {
            unlink(sp.with_extension("wal"))?;
            unlink(sp.with_extension("tmp"))?;
            unlink(sp)?;
        }

        Ok(removed_from_mem || removed_any_file)
    }

    pub(crate) async fn get_collection_arc(&self, name: &str) -> Result<CollectionHandle, Box<Status>> {
        validate_collection_name(name).map_err(|e| Box::new(sutra_err_to_status(e)))?;
        let map = self.collections.read().await;
        map.get(name).cloned().ok_or_else(|| {
            Box::new(Status::not_found(format!("Collection '{}' not found", name)))
        })
    }

    pub(crate) async fn save_collection_internal(
        &self,
        name: &str,
    ) -> Result<(PathBuf, u64), Box<Status>> {
        let col_arc = self.get_collection_arc(name).await?;
        let default_data_dir = self.default_data_dir.clone();
        let col_name = name.to_string();
        let mut col = col_arc.write_owned().await;
        tokio::task::spawn_blocking(move || -> Result<(PathBuf, u64), Box<Status>> {
            if col.storage_path.is_none() {
                if let Some(ref dir) = default_data_dir {
                    col.storage_path = Some(sutra_path_for(dir, &col_name));
                } else {
                    return Err(Box::new(Status::failed_precondition(format!(
                        "Collection '{}' has no storage_path configured",
                        col_name
                    ))));
                }
            }

            let bytes_written = col
                .save::<PathBuf>(None)
                .map_err(|e| Box::new(sutra_err_to_status(e)))?;
            write_manifest_for_collection(&col).map_err(|e| {
                Box::new(Status::internal(format!("Failed to refresh manifest: {}", e)))
            })?;
            let saved_path = col
                .storage_path
                .clone()
                .unwrap_or_else(|| PathBuf::from(format!("{}.sutra", col_name)));
            Ok((saved_path, bytes_written))
        })
        .await
        .map_err(|e| Box::new(Status::internal(format!("save task panicked: {}", e))))?
    }

    pub(crate) async fn compact_collection_internal(&self, name: &str) -> Result<usize, Box<Status>> {
        let col_arc = self.get_collection_arc(name).await?;
        let mut col = col_arc.write_owned().await;
        tokio::task::spawn_blocking(move || -> Result<usize, Box<Status>> {
            col.compact()
                .map_err(|e| Box::new(sutra_err_to_status(e)))?;
            Ok(col.count())
        })
        .await
        .map_err(|e| Box::new(Status::internal(format!("compact task panicked: {}", e))))?
    }
}

#[tonic::async_trait]
impl SutraService for SutraGrpcService {
    async fn health(&self, _request: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        let count = self.collections.read().await.len() as u32;
        let uptime = self.start_time.elapsed().as_secs();
        Ok(Response::new(HealthResponse {
            status: "SERVING".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            collection_count: count,
            uptime_seconds: uptime,
        }))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let (created, _) = self
            .create_collection_internal(
                &req.collection_name,
                req.dimension as usize,
                &req.metric,
                req.data_dir.as_deref(),
                true,
            )
            .await
            .map_err(|e| *e)?;

        if !created {
            return Ok(Response::new(CreateCollectionResponse {
                success: false,
                message: format!("Collection '{}' already exists", req.collection_name),
            }));
        }

        Ok(Response::new(CreateCollectionResponse {
            success: true,
            message: format!("Collection '{}' created successfully", req.collection_name),
        }))
    }

    async fn drop_collection(
        &self,
        request: Request<DropCollectionRequest>,
    ) -> Result<Response<DropCollectionResponse>, Status> {
        let req = request.into_inner();
        let removed = self
            .drop_collection_internal(&req.collection_name)
            .await
            .map_err(|e| *e)?;
        Ok(Response::new(DropCollectionResponse { success: removed }))
    }

    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let map = self.collections.read().await;
        let mut collection_names: Vec<String> = map.keys().cloned().collect();
        collection_names.sort();
        Ok(Response::new(ListCollectionsResponse { collection_names }))
    }

    async fn get_stats(
        &self,
        request: Request<CollectionStatsRequest>,
    ) -> Result<Response<CollectionStatsResponse>, Status> {
        let req = request.into_inner();
        let col_arc = self
            .get_collection_arc(&req.collection_name)
            .await
            .map_err(|e| *e)?;

        let col = col_arc.read_owned().await;
        let resp = tokio::task::spawn_blocking(move || {
            let wal_records = wal_records_for_collection(&col);
            CollectionStatsResponse {
                name: col.name.clone(),
                count: col.count() as u64,
                dimension: col.dimension as u32,
                metric: col.metric.as_str().to_string(),
                is_dirty: col.dirty_index,
                wal_records,
            }
        })
        .await
        .map_err(|e| Status::internal(format!("get_stats task panicked: {}", e)))?;
        Ok(Response::new(resp))
    }

    async fn insert(&self, request: Request<InsertRequest>) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let col_arc = self
            .get_collection_arc(&req.collection_name)
            .await
            .map_err(|e| *e)?;

        let mut docs = Vec::with_capacity(req.documents.len());
        let mut vecs = Vec::with_capacity(req.documents.len());

        for d in req.documents {
            if d.id.trim().is_empty() {
                return Err(Status::invalid_argument("Document id cannot be empty"));
            }

            let metadata = match d.metadata_json {
                Some(ref json_str) if !json_str.trim().is_empty() => serde_json::from_str(json_str)
                    .map_err(|e| {
                        Status::invalid_argument(format!(
                            "Invalid metadata_json for document '{}': {}",
                            d.id, e
                        ))
                    })?,
                _ => serde_json::Value::Object(serde_json::Map::new()),
            };

            let sparse_vector = match d.sparse_vector {
                Some(sv) => Some(SparseVector::try_new(sv.indices, sv.values).map_err(|e| {
                    Status::invalid_argument(format!(
                        "Invalid sparse_vector for document '{}': {}",
                        d.id, e
                    ))
                })?),
                None => None,
            };

            let colbert_tokens = d.colbert_tokens.map(|mv| {
                mv.vectors.into_iter().map(|v| v.values).collect()
            });

            docs.push(NativeDocument {
                id: d.id,
                namespace: d.namespace,
                text: d.text.unwrap_or_default(),
                metadata,
                sparse_vector,
                colbert_tokens,
            });
            vecs.push(d.vector);
        }

        let mut col = col_arc.write_owned().await;
        let inserted = tokio::task::spawn_blocking(move || col.insert(docs, vecs))
            .await
            .map_err(|e| Status::internal(format!("insert task panicked: {}", e)))?
            .map_err(sutra_err_to_status)?;

        Ok(Response::new(InsertResponse {
            inserted_count: inserted as u64,
            success: true,
            message: format!("Successfully inserted {} documents", inserted),
        }))
    }

    async fn delete(&self, request: Request<DeleteRequest>) -> Result<Response<DeleteResponse>, Status> {
        let req = request.into_inner();
        let col_arc = self
            .get_collection_arc(&req.collection_name)
            .await
            .map_err(|e| *e)?;

        let ids = req.ids;
        let mut col = col_arc.write_owned().await;
        let deleted = tokio::task::spawn_blocking(move || col.delete(&ids))
            .await
            .map_err(|e| Status::internal(format!("delete task panicked: {}", e)))?
            .map_err(sutra_err_to_status)?;

        Ok(Response::new(DeleteResponse {
            deleted_count: deleted as u64,
            success: true,
        }))
    }

    async fn query(&self, request: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let col_arc = self
            .get_collection_arc(&req.collection_name)
            .await
            .map_err(|e| *e)?;

        let filter_val = match req.filter_json {
            Some(ref s) if !s.trim().is_empty() => {
                let parsed: serde_json::Value = serde_json::from_str(s)
                    .map_err(|e| Status::invalid_argument(format!("Invalid filter JSON: {}", e)))?;
                Some(parsed)
            }
            _ => None,
        };

        let sparse_vec = match req.sparse_vector {
            Some(sv) => Some(
                SparseVector::try_new(sv.indices, sv.values)
                    .map_err(|e| Status::invalid_argument(format!("Invalid sparse_vector: {}", e)))?,
            ),
            None => None,
        };

        let colbert_tokens: Option<Vec<Vec<f32>>> = req.colbert_tokens.map(|mv| {
            mv.vectors.into_iter().map(|v| v.values).collect()
        });
        let vector_owned: Option<Vec<f32>> = if !req.vector.is_empty() {
            Some(req.vector)
        } else {
            None
        };
        let text_owned: Option<String> = req
            .text
            .filter(|s| !s.trim().is_empty());
        let namespace_owned: Option<String> = req.namespace.filter(|s| !s.is_empty());

        let has_any_signal = vector_owned.is_some()
            || text_owned.is_some()
            || sparse_vec.as_ref().is_some_and(|sv| !sv.is_empty())
            || colbert_tokens.as_ref().is_some_and(|ct| !ct.is_empty());
        if !has_any_signal {
            return Err(Status::invalid_argument(
                "Query must specify at least one of vector, text, sparse_vector, or colbert_tokens",
            ));
        }

        let top_k = if req.top_k > 0 { req.top_k as usize } else { 10 };
        let d_w = if req.dense_weight > 0.0 {
            req.dense_weight
        } else {
            1.0
        };
        let b_w = if req.bm25_weight > 0.0 {
            req.bm25_weight
        } else {
            1.0
        };
        let s_w = if req.sparse_weight > 0.0 {
            req.sparse_weight
        } else {
            1.0
        };
        let c_w = if req.colbert_weight > 0.0 {
            req.colbert_weight
        } else {
            1.0
        };
        let hybrid = req.hybrid;
        let use_ann = req.use_ann;

        let col = col_arc.read_owned().await;
        let search_results = tokio::task::spawn_blocking(move || {
            col.query(
                vector_owned.as_deref(),
                text_owned.as_deref(),
                sparse_vec.as_ref(),
                colbert_tokens.as_deref(),
                namespace_owned.as_deref(),
                filter_val.as_ref(),
                top_k,
                hybrid,
                None,
                60,
                d_w,
                b_w,
                s_w,
                c_w,
                use_ann,
            )
        })
        .await
        .map_err(|e| Status::internal(format!("query task panicked: {}", e)))?
        .map_err(sutra_err_to_status)?;

        let results = search_results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                score: r.score,
                text: if r.text.is_empty() { None } else { Some(r.text) },
                metadata_json: serde_json::to_string(&r.metadata).ok(),
                dense_score: r.dense_score,
                bm25_score: r.bm25_score,
                sparse_score: r.sparse_score,
                colbert_score: r.colbert_score,
                namespace: r.namespace,
            })
            .collect();

        let latency_micros = start.elapsed().as_micros() as u64;
        Ok(Response::new(QueryResponse {
            results,
            latency_micros,
        }))
    }

    async fn batch_query(
        &self,
        request: Request<BatchQueryRequest>,
    ) -> Result<Response<BatchQueryResponse>, Status> {
        let req = request.into_inner();
        let default_col = req.collection_name.trim().to_string();
        if !default_col.is_empty() {
            validate_collection_name(&default_col).map_err(sutra_err_to_status)?;
        }

        if req.queries.is_empty() {
            return Ok(Response::new(BatchQueryResponse {
                responses: Vec::new(),
            }));
        }

        let total = req.queries.len();
        let mut join_set = JoinSet::new();

        for (idx, mut query_req) in req.queries.into_iter().enumerate() {
            if query_req.collection_name.trim().is_empty() {
                query_req.collection_name = default_col.clone();
            }
            validate_collection_name(&query_req.collection_name).map_err(sutra_err_to_status)?;

            let svc = self.clone();
            join_set.spawn(async move {
                let res = svc.query(Request::new(query_req)).await?.into_inner();
                Ok::<(usize, QueryResponse), Status>((idx, res))
            });
        }

        let mut ordered: Vec<Option<QueryResponse>> = vec![None; total];
        while let Some(joined) = join_set.join_next().await {
            let (idx, res) = joined
                .map_err(|e| Status::internal(format!("batch_query worker failed: {}", e)))??;
            ordered[idx] = Some(res);
        }

        let responses = ordered.into_iter().flatten().collect();
        Ok(Response::new(BatchQueryResponse { responses }))
    }

    async fn batch_distances(
        &self,
        request: Request<BatchDistanceRequest>,
    ) -> Result<Response<BatchDistanceResponse>, Status> {
        let req = request.into_inner();
        let dim = req.dimension as usize;
        if dim == 0 {
            return Err(Status::invalid_argument("Dimension must be greater than 0"));
        }
        if req.query_vector.len() != dim {
            return Err(Status::invalid_argument(format!(
                "Query vector length ({}) does not match dimension ({})",
                req.query_vector.len(),
                dim
            )));
        }
        if !req.target_vectors.len().is_multiple_of(dim) {
            return Err(Status::invalid_argument(format!(
                "Target vectors length ({}) is not a multiple of dimension ({})",
                req.target_vectors.len(),
                dim
            )));
        }

        let metric = Metric::from_str(&req.metric).map_err(sutra_err_to_status)?;
        let query_vec = req.query_vector;
        let target_vecs = req.target_vectors;
        let scores = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, SutraError> {
            let num_targets = target_vecs.len() / dim;
            let mut scores = Vec::with_capacity(num_targets);

            let q_vec = if metric == Metric::Cosine {
                crate::distance::simd::normalize_copy(&query_vec)
            } else {
                query_vec
            };

            // Target chunks may be unnormalized, so always pass is_normalized = false.
            for chunk in target_vecs.chunks_exact(dim) {
                let score = crate::distance::simd::compute_vector_score(&q_vec, chunk, metric, false)?;
                scores.push(score);
            }
            Ok(scores)
        })
        .await
        .map_err(|e| Status::internal(format!("batch_distances task panicked: {}", e)))?
        .map_err(sutra_err_to_status)?;

        Ok(Response::new(BatchDistanceResponse { scores }))
    }

    async fn save(&self, request: Request<SaveRequest>) -> Result<Response<SaveResponse>, Status> {
        let req = request.into_inner();
        let (path, bytes_written) = self
            .save_collection_internal(&req.collection_name)
            .await
            .map_err(|e| *e)?;
        Ok(Response::new(SaveResponse {
            success: true,
            message: format!(
                "Collection '{}' saved ({} bytes)",
                req.collection_name, bytes_written
            ),
            path: path.to_string_lossy().into_owned(),
            bytes_written,
        }))
    }

    async fn compact(
        &self,
        request: Request<CompactRequest>,
    ) -> Result<Response<CompactResponse>, Status> {
        let req = request.into_inner();
        let count = self
            .compact_collection_internal(&req.collection_name)
            .await
            .map_err(|e| *e)?;
        Ok(Response::new(CompactResponse {
            success: true,
            count: count as u64,
            message: format!(
                "Collection '{}' compacted ({} active documents)",
                req.collection_name, count
            ),
        }))
    }
}
