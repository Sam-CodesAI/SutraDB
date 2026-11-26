use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, RwLock};
use tokio::task::JoinSet;
use tonic::Status;

use crate::collection::{NativeCollection, NativeDocument};
use crate::grpc::service::{
    sutra_err_to_status, wal_records_for_collection, write_manifest_for_collection,
    SutraGrpcService,
};
use crate::index::hnsw::HNSWConfig;
use crate::sparse::SparseVector;

pub const MAX_PAYLOAD_SIZE: usize = 10 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const IDLE_READ_TIMEOUT: Duration = Duration::from_secs(30);

struct HttpRequest {
    method: String,
    target: String,
    version: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

struct HttpResponse {
    status_code: u16,
    reason: &'static str,
    content_type: &'static str,
    body: Vec<u8>,
    force_close: bool,
}

impl HttpResponse {
    fn json(status_code: u16, value: &Value) -> Self {
        let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
        Self {
            status_code,
            reason: reason_phrase(status_code),
            content_type: "application/json",
            body,
            force_close: false,
        }
    }

    fn text(status_code: u16, content_type: &'static str, text: String) -> Self {
        Self {
            status_code,
            reason: reason_phrase(status_code),
            content_type,
            body: text.into_bytes(),
            force_close: false,
        }
    }

    fn empty(status_code: u16) -> Self {
        Self {
            status_code,
            reason: reason_phrase(status_code),
            content_type: "text/plain; charset=utf-8",
            body: Vec::new(),
            force_close: false,
        }
    }

    fn error(status_code: u16, msg: impl Into<String>) -> Self {
        let s = msg.into();
        let val = serde_json::json!({
            "error": s,
            "detail": s,
        });
        Self::json(status_code, &val)
    }

    fn with_force_close(mut self) -> Self {
        self.force_close = true;
        self
    }
}

fn reason_phrase(code: u16) -> &'static str {
    match code {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

fn status_to_http(status: Status) -> HttpResponse {
    let code = match status.code() {
        tonic::Code::InvalidArgument | tonic::Code::FailedPrecondition => 400,
        tonic::Code::NotFound => 404,
        tonic::Code::AlreadyExists => 409,
        _ => 500,
    };
    HttpResponse::error(code, status.message())
}

pub async fn serve_http_bridge(
    listener: TcpListener,
    service: SutraGrpcService,
    mut shutdown: watch::Receiver<bool>,
) -> std::io::Result<()> {
    loop {
        if *shutdown.borrow() {
            break;
        }
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
            accept_res = listener.accept() => {
                match accept_res {
                    Ok((stream, _peer)) => {
                        let svc = service.clone();
                        let mut conn_shutdown = shutdown.clone();
                        tokio::spawn(async move {
                            tokio::select! {
                                _ = handle_connection(stream, svc) => {}
                                _ = conn_shutdown.changed() => {}
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("[SutraDB] HTTP bridge accept error: {}", e);
                    }
                }
            }
        }
    }
    Ok(())
}

async fn handle_connection(mut stream: TcpStream, service: SutraGrpcService) {
    let mut buf: Vec<u8> = Vec::with_capacity(8192);

    loop {
        match read_http_request(&mut stream, &mut buf).await {
            Ok(Some(req)) => {
                let conn_hdr = req
                    .headers
                    .get("connection")
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_default();
                let wants_close = if req.version == "HTTP/1.0" {
                    !conn_hdr.contains("keep-alive")
                } else {
                    conn_hdr.contains("close")
                };

                let resp = route_request(&service, req).await;
                let close_after = wants_close || resp.force_close;

                if write_http_response(&mut stream, &resp, !close_after)
                    .await
                    .is_err()
                {
                    break;
                }
                if close_after {
                    break;
                }
            }
            Ok(None) => break,
            Err(early_resp) => {
                let _ = write_http_response(&mut stream, &early_resp, false).await;
                break;
            }
        }
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

async fn read_http_request(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
) -> Result<Option<HttpRequest>, HttpResponse> {
    let header_end = loop {
        if let Some(pos) = find_header_end(buf) {
            break pos;
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err(
                HttpResponse::error(431, "Request headers exceed maximum allowed size")
                    .with_force_close(),
            );
        }

        let mut tmp = [0u8; 4096];
        let n = match tokio::time::timeout(IDLE_READ_TIMEOUT, stream.read(&mut tmp)).await {
            Ok(Ok(n)) => n,
            Ok(Err(_)) | Err(_) => return Ok(None),
        };
        if n == 0 {
            if buf.is_empty() {
                return Ok(None);
            }
            return Err(HttpResponse::error(400, "Incomplete HTTP request").with_force_close());
        }
        buf.extend_from_slice(&tmp[..n]);
    };

    let head_bytes = buf[..header_end].to_vec();
    buf.drain(..header_end + 4);

    let head_str = std::str::from_utf8(&head_bytes)
        .map_err(|_| HttpResponse::error(400, "Invalid UTF-8 in HTTP headers").with_force_close())?;

    let mut lines = head_str.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| HttpResponse::error(400, "Empty HTTP request").with_force_close())?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| HttpResponse::error(400, "Missing HTTP method").with_force_close())?
        .to_ascii_uppercase();
    let target = parts
        .next()
        .ok_or_else(|| HttpResponse::error(400, "Missing request target").with_force_close())?
        .to_string();
    let version = parts
        .next()
        .ok_or_else(|| HttpResponse::error(400, "Missing HTTP version").with_force_close())?
        .to_ascii_uppercase();
    if parts.next().is_some() || (version != "HTTP/1.1" && version != "HTTP/1.0") {
        return Err(HttpResponse::error(400, "Unsupported HTTP request line").with_force_close());
    }

    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            return Err(HttpResponse::error(400, "Malformed HTTP header").with_force_close());
        };
        headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
    }

    let is_chunked = headers
        .get("transfer-encoding")
        .map(|te| te.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false);

    let body = if is_chunked {
        read_chunked_body(stream, buf).await?
    } else if let Some(cl_str) = headers.get("content-length") {
        let content_length = cl_str.parse::<usize>().map_err(|_| {
            HttpResponse::error(400, "Invalid Content-Length header").with_force_close()
        })?;
        if content_length > MAX_PAYLOAD_SIZE {
            return Err(HttpResponse::error(
                413,
                "Payload exceeds maximum allowed size of 10MB",
            )
            .with_force_close());
        }
        read_exact_body(stream, buf, content_length).await?
    } else {
        Vec::new()
    };

    Ok(Some(HttpRequest {
        method,
        target,
        version,
        headers,
        body,
    }))
}

async fn read_exact_body(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    content_length: usize,
) -> Result<Vec<u8>, HttpResponse> {
    while buf.len() < content_length {
        let mut tmp = [0u8; 8192];
        let n = match tokio::time::timeout(IDLE_READ_TIMEOUT, stream.read(&mut tmp)).await {
            Ok(Ok(n)) => n,
            Ok(Err(_)) | Err(_) => {
                return Err(
                    HttpResponse::error(400, "Connection closed while reading body")
                        .with_force_close(),
                );
            }
        };
        if n == 0 {
            return Err(HttpResponse::error(400, "Unexpected EOF in request body").with_force_close());
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = buf.drain(..content_length).collect();
    Ok(body)
}

async fn read_line_from_buf(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
) -> Result<String, HttpResponse> {
    loop {
        if let Some(pos) = buf.windows(2).position(|w| w == b"\r\n") {
            let line_bytes: Vec<u8> = buf.drain(..pos).collect();
            buf.drain(..2);
            return String::from_utf8(line_bytes).map_err(|_| {
                HttpResponse::error(400, "Invalid UTF-8 in chunked framing").with_force_close()
            });
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err(HttpResponse::error(400, "Chunk header too large").with_force_close());
        }
        let mut tmp = [0u8; 4096];
        let n = match tokio::time::timeout(IDLE_READ_TIMEOUT, stream.read(&mut tmp)).await {
            Ok(Ok(n)) => n,
            Ok(Err(_)) | Err(_) => {
                return Err(HttpResponse::error(400, "Timeout reading chunked body").with_force_close());
            }
        };
        if n == 0 {
            return Err(HttpResponse::error(400, "Unexpected EOF in chunked body").with_force_close());
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

async fn read_chunked_body(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
) -> Result<Vec<u8>, HttpResponse> {
    let mut decoded = Vec::new();
    loop {
        let size_line = read_line_from_buf(stream, buf).await?;
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let chunk_size = usize::from_str_radix(size_hex, 16).map_err(|_| {
            HttpResponse::error(400, "Invalid chunk size in Transfer-Encoding: chunked")
                .with_force_close()
        })?;
        if chunk_size == 0 {
            // Drain trailing headers until empty line
            loop {
                let trailer = read_line_from_buf(stream, buf).await?;
                if trailer.is_empty() {
                    break;
                }
            }
            break;
        }
        if decoded.len().saturating_add(chunk_size) > MAX_PAYLOAD_SIZE {
            return Err(HttpResponse::error(
                413,
                "Payload exceeds maximum allowed size of 10MB",
            )
            .with_force_close());
        }
        let chunk_bytes = read_exact_body(stream, buf, chunk_size).await?;
        decoded.extend_from_slice(&chunk_bytes);
        // Consume CRLF after chunk payload
        let crlf = read_exact_body(stream, buf, 2).await?;
        if crlf != b"\r\n" {
            return Err(HttpResponse::error(400, "Malformed chunk terminator").with_force_close());
        }
    }
    Ok(decoded)
}

async fn write_http_response(
    stream: &mut TcpStream,
    resp: &HttpResponse,
    keep_alive: bool,
) -> std::io::Result<()> {
    let conn_val = if keep_alive { "keep-alive" } else { "close" };
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, PUT, PATCH, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n",
        resp.status_code,
        resp.reason,
        resp.content_type,
        resp.body.len(),
        conn_val
    );
    stream.write_all(head.as_bytes()).await?;
    if !resp.body.is_empty() {
        stream.write_all(&resp.body).await?;
    }
    stream.flush().await
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn percent_decode(input: &str, plus_as_space: bool) -> Result<String, ()> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                if i + 2 >= bytes.len() {
                    return Err(());
                }
                let hi = hex_val(bytes[i + 1]).ok_or(())?;
                let lo = hex_val(bytes[i + 2]).ok_or(())?;
                out.push((hi << 4) | lo);
                i += 3;
            }
            b'+' if plus_as_space => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| ())
}

fn parse_query_params(raw_query: &str) -> Result<HashMap<String, String>, ()> {
    let mut params = HashMap::new();
    if raw_query.is_empty() {
        return Ok(params);
    }
    for pair in raw_query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let key = percent_decode(k, true)?;
        let val = percent_decode(v, true)?;
        params.insert(key, val);
    }
    Ok(params)
}

fn col_stats_json(col: &NativeCollection) -> Value {
    let mut val = col.stats();
    if let Some(obj) = val.as_object_mut() {
        obj.insert(
            "wal_records".to_string(),
            Value::from(wal_records_for_collection(col)),
        );
        obj.insert("is_dirty".to_string(), Value::from(col.dirty_index));
    }
    val
}

fn sparse_to_json(sv: &Option<SparseVector>) -> Value {
    match sv {
        None => Value::Null,
        Some(s) => {
            let mut m = Map::new();
            for (&idx, &val) in s.indices.iter().zip(s.values.iter()) {
                m.insert(idx.to_string(), Value::from(val));
            }
            Value::Object(m)
        }
    }
}

fn doc_to_json(col: &NativeCollection, idx: usize, include_vector: bool) -> Value {
    let doc = &col.documents[idx];
    let mut obj = Map::new();
    obj.insert("id".to_string(), Value::String(doc.id.clone()));
    obj.insert("text".to_string(), Value::String(doc.text.clone()));
    obj.insert("metadata".to_string(), doc.metadata.clone());
    obj.insert(
        "namespace".to_string(),
        match &doc.namespace {
            Some(ns) => Value::String(ns.clone()),
            None => Value::Null,
        },
    );
    obj.insert(
        "sparse_vector".to_string(),
        sparse_to_json(&doc.sparse_vector),
    );
    obj.insert(
        "colbert_tokens".to_string(),
        match &doc.colbert_tokens {
            Some(toks) => serde_json::to_value(toks).unwrap_or(Value::Null),
            None => Value::Null,
        },
    );
    if include_vector {
        obj.insert(
            "vector".to_string(),
            col.get_vector(idx)
                .map(|v| serde_json::to_value(v.as_ref()).unwrap_or(Value::Null))
                .unwrap_or(Value::Null),
        );
    }
    Value::Object(obj)
}

fn parse_f32_vec(val: &Value, field_name: &str) -> Result<Vec<f32>, String> {
    let arr = val
        .as_array()
        .ok_or_else(|| format!("'{}' must be an array of numbers", field_name))?;
    let mut out = Vec::with_capacity(arr.len());
    for elem in arr {
        let num = elem
            .as_f64()
            .ok_or_else(|| format!("'{}' must contain only finite numbers", field_name))?;
        let f = num as f32;
        if !f.is_finite() {
            return Err(format!("'{}' contains non-finite float", field_name));
        }
        out.push(f);
    }
    Ok(out)
}

fn parse_sparse_vector_json(val: &Value) -> Result<Option<SparseVector>, String> {
    if val.is_null() {
        return Ok(None);
    }
    let obj = val
        .as_object()
        .ok_or_else(|| "sparse_vector must be a JSON object".to_string())?;

    if obj.contains_key("indices") || obj.contains_key("values") {
        let indices_val = obj
            .get("indices")
            .ok_or_else(|| "sparse_vector must contain 'indices' and 'values'".to_string())?;
        let values_val = obj
            .get("values")
            .ok_or_else(|| "sparse_vector must contain 'indices' and 'values'".to_string())?;

        let idx_arr = indices_val
            .as_array()
            .ok_or_else(|| "sparse_vector.indices must be an array of integers".to_string())?;
        let val_arr = values_val
            .as_array()
            .ok_or_else(|| "sparse_vector.values must be an array of numbers".to_string())?;

        let mut indices = Vec::with_capacity(idx_arr.len());
        for item in idx_arr {
            let u = item
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| "sparse_vector.indices must contain u32 integers".to_string())?;
            indices.push(u);
        }

        let mut values = Vec::with_capacity(val_arr.len());
        for item in val_arr {
            let f = item
                .as_f64()
                .map(|v| v as f32)
                .filter(|v| v.is_finite())
                .ok_or_else(|| "sparse_vector.values must contain finite numbers".to_string())?;
            values.push(f);
        }

        let sv = SparseVector::try_new(indices, values).map_err(|e| e.to_string())?;
        return Ok(Some(sv));
    }

    let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let idx = k
            .parse::<u32>()
            .map_err(|_| format!("Invalid sparse_vector index key '{}'", k))?;
        let weight = v
            .as_f64()
            .map(|x| x as f32)
            .filter(|x| x.is_finite())
            .ok_or_else(|| format!("Invalid sparse_vector weight for key '{}'", k))?;
        pairs.push((idx, weight));
    }
    pairs.sort_by_key(|p| p.0);
    let indices = pairs.iter().map(|p| p.0).collect();
    let values = pairs.iter().map(|p| p.1).collect();
    let sv = SparseVector::try_new(indices, values).map_err(|e| e.to_string())?;
    Ok(Some(sv))
}

fn parse_colbert_tokens_json(val: &Value) -> Result<Option<Vec<Vec<f32>>>, String> {
    if val.is_null() {
        return Ok(None);
    }
    let arr = val
        .as_array()
        .ok_or_else(|| "colbert_tokens must be an array of token vectors".to_string())?;
    let mut tokens = Vec::with_capacity(arr.len());
    for tok_val in arr {
        tokens.push(parse_f32_vec(tok_val, "colbert_tokens[]")?);
    }
    Ok(Some(tokens))
}

fn parse_json_body(body: &[u8]) -> Result<Value, HttpResponse> {
    if body.is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    serde_json::from_slice(body)
        .map_err(|e| HttpResponse::error(400, format!("Invalid JSON body: {}", e)))
}

fn parse_optional_json_body(body: &[u8]) -> Result<Value, HttpResponse> {
    if body.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(Value::Object(Map::new()));
    }
    serde_json::from_slice(body)
        .map_err(|e| HttpResponse::error(400, format!("Invalid JSON body: {}", e)))
}

async fn execute_json_query(
    col_arc: Arc<RwLock<NativeCollection>>,
    payload: &Value,
) -> Result<Value, HttpResponse> {
    let start = Instant::now();
    let obj = payload
        .as_object()
        .ok_or_else(|| HttpResponse::error(400, "Query payload must be a JSON object"))?;

    let vector = match obj.get("vector") {
        Some(v) if !v.is_null() => Some(parse_f32_vec(v, "vector").map_err(|e| HttpResponse::error(400, e))?),
        _ => None,
    };

    let text = match obj.get("text") {
        Some(v) if !v.is_null() => {
            let s = v
                .as_str()
                .ok_or_else(|| HttpResponse::error(400, "'text' must be a string"))?;
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        }
        _ => None,
    };

    let sparse_vector = match obj.get("sparse_vector") {
        Some(v) => parse_sparse_vector_json(v).map_err(|e| HttpResponse::error(400, e))?,
        None => None,
    };

    let colbert_tokens = match obj.get("colbert_tokens") {
        Some(v) => parse_colbert_tokens_json(v).map_err(|e| HttpResponse::error(400, e))?,
        None => None,
    };

    let has_any_signal = vector.is_some()
        || text.is_some()
        || sparse_vector.as_ref().is_some_and(|sv| !sv.is_empty())
        || colbert_tokens.as_ref().is_some_and(|ct| !ct.is_empty());
    if !has_any_signal {
        return Err(HttpResponse::error(
            400,
            "Query must specify at least one of vector, text, sparse_vector, or colbert_tokens",
        ));
    }

    let filter_owned = match obj.get("filter") {
        Some(v) if !v.is_null() => {
            if !v.is_object() {
                return Err(HttpResponse::error(400, "'filter' must be a JSON object"));
            }
            Some(v.clone())
        }
        _ => None,
    };

    let namespace_owned = match obj.get("namespace") {
        Some(v) if !v.is_null() => {
            let s = v
                .as_str()
                .ok_or_else(|| HttpResponse::error(400, "'namespace' must be a string"))?;
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        }
        _ => None,
    };

    let top_k: usize = match obj.get("top_k") {
        Some(v) if !v.is_null() => {
            let k = v
                .as_u64()
                .ok_or_else(|| HttpResponse::error(400, "'top_k' must be a positive integer"))?;
            if k == 0 || k > 10_000 {
                return Err(HttpResponse::error(400, "'top_k' must be between 1 and 10000"));
            }
            k as usize
        }
        _ => 10,
    };

    let hybrid = match obj.get("hybrid") {
        Some(v) if !v.is_null() => v
            .as_bool()
            .ok_or_else(|| HttpResponse::error(400, "'hybrid' must be a boolean"))?,
        _ => true,
    };

    let alpha = match obj.get("alpha") {
        Some(v) if !v.is_null() => {
            let a = v
                .as_f64()
                .map(|x| x as f32)
                .filter(|x| x.is_finite() && (0.0..=1.0).contains(x))
                .ok_or_else(|| HttpResponse::error(400, "'alpha' must be between 0.0 and 1.0"))?;
            Some(a)
        }
        _ => None,
    };

    let rrf_constant: usize = match obj.get("rrf_constant") {
        Some(v) if !v.is_null() => {
            let c = v
                .as_u64()
                .ok_or_else(|| HttpResponse::error(400, "'rrf_constant' must be >= 1"))?;
            if c == 0 {
                return Err(HttpResponse::error(400, "'rrf_constant' must be >= 1"));
            }
            c as usize
        }
        _ => 60,
    };

    let parse_weight = |key: &str| -> Result<f32, HttpResponse> {
        match obj.get(key) {
            Some(v) if !v.is_null() => v
                .as_f64()
                .map(|x| x as f32)
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| HttpResponse::error(400, format!("'{}' must be >= 0.0", key))),
            _ => Ok(1.0),
        }
    };

    let dense_weight = parse_weight("dense_weight")?;
    let bm25_weight = parse_weight("bm25_weight")?;
    let sparse_weight = parse_weight("sparse_weight")?;
    let colbert_weight = parse_weight("colbert_weight")?;

    let use_ann = match obj.get("use_ann") {
        Some(v) if !v.is_null() => v
            .as_bool()
            .ok_or_else(|| HttpResponse::error(400, "'use_ann' must be a boolean"))?,
        _ => false,
    };

    let col = col_arc.read_owned().await;
    let search_results = tokio::task::spawn_blocking(move || {
        col.query(
            vector.as_deref(),
            text.as_deref(),
            sparse_vector.as_ref(),
            colbert_tokens.as_deref(),
            namespace_owned.as_deref(),
            filter_owned.as_ref(),
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
    })
    .await
    .map_err(|e| HttpResponse::error(500, format!("query task panicked: {}", e)))?
    .map_err(|e| status_to_http(sutra_err_to_status(e)))?;

    let results_json: Vec<Value> = search_results
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "score": r.score,
                "text": r.text,
                "metadata": r.metadata,
                "dense_score": r.dense_score,
                "bm25_score": r.bm25_score,
                "sparse_score": r.sparse_score,
                "colbert_score": r.colbert_score,
                "namespace": r.namespace,
            })
        })
        .collect();

    let latency_micros = start.elapsed().as_micros() as u64;
    Ok(serde_json::json!({
        "count": results_json.len(),
        "results": results_json,
        "latency_micros": latency_micros,
    }))
}

async fn route_request(service: &SutraGrpcService, req: HttpRequest) -> HttpResponse {
    if req.method == "OPTIONS" {
        return HttpResponse::empty(204);
    }

    let (raw_path, raw_query) = match req.target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (req.target.as_str(), ""),
    };

    let query_params = match parse_query_params(raw_query) {
        Ok(qp) => qp,
        Err(()) => return HttpResponse::error(400, "Malformed query string encoding"),
    };

    let trimmed_path = raw_path.trim_end_matches('/');
    let segments: Vec<String> = if trimmed_path.is_empty() || trimmed_path == "/" {
        Vec::new()
    } else {
        let mut decoded = Vec::new();
        for seg in trimmed_path.trim_start_matches('/').split('/') {
            match percent_decode(seg, false) {
                Ok(s) => decoded.push(s),
                Err(()) => return HttpResponse::error(400, "Malformed percent-encoding in URL path"),
            }
        }
        decoded
    };

    let seg_refs: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();

    match (req.method.as_str(), seg_refs.as_slice()) {
        // GET / or GET /health
        ("GET", []) | ("GET", ["health"]) => {
            let map = service.collections().read().await;
            let mut names: Vec<String> = map.keys().cloned().collect();
            names.sort();
            HttpResponse::json(
                200,
                &serde_json::json!({
                    "status": "healthy",
                    "version": env!("CARGO_PKG_VERSION"),
                    "collections": names,
                    "uptime_seconds": service.start_time().elapsed().as_secs(),
                }),
            )
        }

        // GET /metrics
        ("GET", ["metrics"]) => {
            let pairs: Vec<(String, Arc<RwLock<NativeCollection>>)> = {
                let map = service.collections().read().await;
                let mut list: Vec<_> = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                list.sort_by(|a, b| a.0.cmp(&b.0));
                list
            };

            let mut lines = vec![
                "# HELP sutradb_collections_total Total number of collections".to_string(),
                "# TYPE sutradb_collections_total gauge".to_string(),
                format!("sutradb_collections_total {}", pairs.len()),
            ];

            for (name, col_arc) in pairs {
                let col = col_arc.read().await;
                let st = col.stats();
                let count = st.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
                let mem_bytes = st
                    .get("vector_memory_bytes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                lines.push(format!(
                    "sutradb_documents_count{{collection=\"{}\"}} {}",
                    name, count
                ));
                lines.push(format!(
                    "sutradb_vector_memory_bytes{{collection=\"{}\"}} {}",
                    name, mem_bytes
                ));
            }

            HttpResponse::text(
                200,
                "text/plain; version=0.0.4",
                format!("{}\n", lines.join("\n")),
            )
        }

        // GET /collections
        ("GET", ["collections"]) => {
            let pairs: Vec<(String, Arc<RwLock<NativeCollection>>)> = {
                let map = service.collections().read().await;
                let mut list: Vec<_> = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                list.sort_by(|a, b| a.0.cmp(&b.0));
                list
            };

            let mut cols = Vec::with_capacity(pairs.len());
            for (_, col_arc) in pairs {
                let col = col_arc.read().await;
                cols.push(col_stats_json(&col));
            }
            HttpResponse::json(200, &serde_json::json!({ "collections": cols }))
        }

        // POST /collections
        ("POST", ["collections"]) => {
            let body = match parse_json_body(&req.body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let Some(obj) = body.as_object() else {
                return HttpResponse::error(400, "Request body must be a JSON object");
            };
            let Some(name) = obj.get("name").and_then(|v| v.as_str()) else {
                return HttpResponse::error(400, "Field 'name' (string) is required");
            };
            let Some(dim_u64) = obj.get("dimension").and_then(|v| v.as_u64()) else {
                return HttpResponse::error(400, "Field 'dimension' (positive integer) is required");
            };
            let dimension = match usize::try_from(dim_u64) {
                Ok(d) if d > 0 => d,
                _ => return HttpResponse::error(400, "Field 'dimension' must be > 0"),
            };
            let metric = match obj.get("metric") {
                Some(v) if !v.is_null() => match v.as_str() {
                    Some(s) => s,
                    None => return HttpResponse::error(400, "Field 'metric' must be a string"),
                },
                _ => "cosine",
            };
            let enable_wal = match obj.get("enable_wal") {
                Some(v) if !v.is_null() => match v.as_bool() {
                    Some(b) => b,
                    None => return HttpResponse::error(400, "Field 'enable_wal' must be a boolean"),
                },
                _ => false,
            };

            match service
                .create_collection_internal(name, dimension, metric, None, enable_wal)
                .await
            {
                Ok((true, Some(stats))) => HttpResponse::json(
                    201,
                    &serde_json::json!({
                        "message": "Collection created",
                        "stats": stats,
                    }),
                ),
                Ok((false, _)) => HttpResponse::error(
                    409,
                    format!("Collection '{}' already exists", name),
                ),
                Ok((true, None)) => HttpResponse::error(500, "Internal creation error"),
                Err(status) => status_to_http(*status),
            }
        }

        // DELETE /collections/{name}
        ("DELETE", ["collections", name]) => {
            match service.drop_collection_internal(name).await {
                Ok(true) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "message": format!("Collection '{}' dropped successfully", name)
                    }),
                ),
                Ok(false) => HttpResponse::error(404, format!("Collection '{}' not found", name)),
                Err(status) => status_to_http(*status),
            }
        }

        // GET /collections/{name}/stats
        ("GET", ["collections", name, "stats"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let col = col_arc.read().await;
            HttpResponse::json(200, &col_stats_json(&col))
        }

        // GET /collections/{name}/documents
        ("GET", ["collections", name, "documents"]) => {
            let limit: usize = match query_params.get("limit") {
                Some(s) => match s.parse::<usize>() {
                    Ok(v) if (1..=10_000).contains(&v) => v,
                    _ => return HttpResponse::error(400, "'limit' must be between 1 and 10000"),
                },
                None => 50,
            };
            let offset: usize = match query_params.get("offset") {
                Some(s) => match s.parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => return HttpResponse::error(400, "'offset' must be >= 0"),
                },
                None => 0,
            };

            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let col = col_arc.read().await;
            let active_indices: Vec<usize> = (0..col.documents.len())
                .filter(|&i| i >= col.tombstones.len() || !col.tombstones[i])
                .collect();
            let total = active_indices.len();
            let start = offset.min(total);
            let end = start.saturating_add(limit).min(total);
            let docs: Vec<Value> = active_indices[start..end]
                .iter()
                .map(|&idx| doc_to_json(&col, idx, false))
                .collect();

            HttpResponse::json(
                200,
                &serde_json::json!({
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "documents": docs,
                }),
            )
        }

        // GET /collections/{name}/documents/{id}
        ("GET", ["collections", name, "documents", doc_id]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let col = col_arc.read().await;
            match col.id_to_index.get(*doc_id).copied() {
                Some(idx) if idx >= col.tombstones.len() || !col.tombstones[idx] => {
                    HttpResponse::json(200, &doc_to_json(&col, idx, true))
                }
                _ => HttpResponse::error(
                    404,
                    format!("Document '{}' not found in collection '{}'", doc_id, name),
                ),
            }
        }

        // GET /collections/{name}/sample or /collections/{name}/sample_vectors
        ("GET", ["collections", name, "sample"])
        | ("GET", ["collections", name, "sample_vectors"]) => {
            let limit: usize = match query_params.get("limit") {
                Some(s) => match s.parse::<usize>() {
                    Ok(v) if (1..=10_000).contains(&v) => v,
                    _ => return HttpResponse::error(400, "'limit' must be between 1 and 10000"),
                },
                None => 100,
            };

            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let col = col_arc.read().await;
            let mut vectors = Vec::new();
            for (idx, doc) in col.documents.iter().enumerate() {
                if idx < col.tombstones.len() && col.tombstones[idx] {
                    continue;
                }
                let preview_vec: Vec<f32> = col
                    .get_vector(idx)
                    .map(|v| v.iter().take(128).copied().collect())
                    .unwrap_or_default();
                let preview_text: String = doc.text.chars().take(100).collect();
                vectors.push(serde_json::json!({
                    "id": doc.id,
                    "vector": preview_vec,
                    "text": preview_text,
                    "metadata": doc.metadata,
                }));
                if vectors.len() >= limit {
                    break;
                }
            }

            HttpResponse::json(
                200,
                &serde_json::json!({
                    "collection": name,
                    "count": vectors.len(),
                    "vectors": vectors,
                }),
            )
        }

        // POST /collections/{name}/insert | /documents | /upsert
        ("POST", ["collections", name, action @ ("insert" | "documents" | "upsert")]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let body = match parse_json_body(&req.body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };

            let raw_docs: Vec<Value> = match body {
                Value::Array(arr) => arr,
                Value::Object(mut obj) => {
                    if let Some(docs_val) = obj.remove("documents") {
                        match docs_val {
                            Value::Array(arr) => arr,
                            _ => return HttpResponse::error(400, "'documents' must be an array"),
                        }
                    } else if obj.contains_key("vector")
                        || obj.contains_key("text")
                        || obj.contains_key("sparse_vector")
                        || obj.contains_key("colbert_tokens")
                    {
                        vec![Value::Object(obj)]
                    } else {
                        Vec::new()
                    }
                }
                _ => return HttpResponse::error(400, "Invalid JSON payload for insert"),
            };

            if raw_docs.is_empty() {
                return HttpResponse::error(400, "Missing 'documents' list");
            }

            let mut native_docs = Vec::with_capacity(raw_docs.len());
            let mut vectors = Vec::with_capacity(raw_docs.len());
            let batch_len = raw_docs.len();

            for item in raw_docs {
                let Some(d_obj) = item.as_object() else {
                    return HttpResponse::error(400, "Each document must be a JSON object");
                };
                let id = match d_obj.get("id") {
                    Some(v) if !v.is_null() => {
                        let Some(s) = v.as_str() else {
                            return HttpResponse::error(400, "Document 'id' must be a string");
                        };
                        if s.trim().is_empty() {
                            return HttpResponse::error(400, "Document 'id' cannot be empty");
                        }
                        s.to_string()
                    }
                    _ => uuid::Uuid::new_v4().to_string(),
                };

                let Some(vec_val) = d_obj.get("vector") else {
                    return HttpResponse::error(
                        400,
                        format!("Document '{}' is missing a 'vector' array", id),
                    );
                };
                let vec = match parse_f32_vec(vec_val, "vector") {
                    Ok(v) => v,
                    Err(e) => return HttpResponse::error(400, e),
                };

                let text = match d_obj.get("text") {
                    Some(v) if !v.is_null() => match v.as_str() {
                        Some(s) => s.to_string(),
                        None => return HttpResponse::error(400, "Document 'text' must be a string"),
                    },
                    _ => String::new(),
                };

                let namespace = match d_obj.get("namespace") {
                    Some(v) if !v.is_null() => match v.as_str() {
                        Some(s) => Some(s.to_string()),
                        None => {
                            return HttpResponse::error(400, "Document 'namespace' must be a string")
                        }
                    },
                    _ => None,
                };

                let metadata = match d_obj.get("metadata") {
                    Some(v) if !v.is_null() => {
                        if !v.is_object() {
                            return HttpResponse::error(
                                400,
                                "Document 'metadata' must be a JSON object",
                            );
                        }
                        v.clone()
                    }
                    _ => Value::Object(Map::new()),
                };

                let sparse_vector = match d_obj.get("sparse_vector") {
                    Some(v) => match parse_sparse_vector_json(v) {
                        Ok(sv) => sv,
                        Err(e) => return HttpResponse::error(400, e),
                    },
                    None => None,
                };

                let colbert_tokens = match d_obj.get("colbert_tokens") {
                    Some(v) => match parse_colbert_tokens_json(v) {
                        Ok(ct) => ct,
                        Err(e) => return HttpResponse::error(400, e),
                    },
                    None => None,
                };

                native_docs.push(NativeDocument {
                    id,
                    namespace,
                    text,
                    metadata,
                    sparse_vector,
                    colbert_tokens,
                });
                vectors.push(vec);
            }

            let is_upsert = *action == "upsert";
            let mut col = col_arc.write_owned().await;
            let res = tokio::task::spawn_blocking(move || {
                if is_upsert {
                    col.upsert(native_docs, vectors)
                } else {
                    col.insert(native_docs, vectors)
                }
            })
            .await;
            let res = match res {
                Ok(r) => r,
                Err(e) => return HttpResponse::error(500, format!("insert task panicked: {}", e)),
            };
            match res {
                Ok(total_count) => {
                    let key = if is_upsert {
                        "upserted"
                    } else {
                        "inserted"
                    };
                    HttpResponse::json(
                        200,
                        &serde_json::json!({
                            key: batch_len,
                            "total_count": total_count,
                        }),
                    )
                }
                Err(e) => status_to_http(sutra_err_to_status(e)),
            }
        }

        // DELETE /collections/{name}/documents/{id}
        ("DELETE", ["collections", name, "documents", doc_id]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let id_owned = (*doc_id).to_string();
            let mut col = col_arc.write_owned().await;
            let res = tokio::task::spawn_blocking(move || col.delete(&[id_owned])).await;
            match res {
                Ok(Ok(0)) => HttpResponse::error(
                    404,
                    format!("Document '{}' not found in collection '{}'", doc_id, name),
                ),
                Ok(Ok(deleted)) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "deleted": deleted,
                        "message": format!("Document '{}' deleted successfully", doc_id),
                    }),
                ),
                Ok(Err(e)) => status_to_http(sutra_err_to_status(e)),
                Err(e) => HttpResponse::error(500, format!("delete task panicked: {}", e)),
            }
        }

        // DELETE /collections/{name}/documents
        ("DELETE", ["collections", name, "documents"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let body = match parse_json_body(&req.body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let ids_val = match body {
                Value::Array(arr) => arr,
                Value::Object(mut obj) => match obj.remove("ids") {
                    Some(Value::Array(arr)) => arr,
                    _ => {
                        return HttpResponse::error(400, "Missing 'ids' list for batch deletion")
                    }
                },
                _ => return HttpResponse::error(400, "Missing 'ids' list for batch deletion"),
            };
            let mut ids = Vec::with_capacity(ids_val.len());
            for v in ids_val {
                let Some(s) = v.as_str() else {
                    return HttpResponse::error(400, "All 'ids' entries must be strings");
                };
                ids.push(s.to_string());
            }
            if ids.is_empty() {
                return HttpResponse::error(400, "Missing 'ids' list for batch deletion");
            }

            let mut col = col_arc.write_owned().await;
            match tokio::task::spawn_blocking(move || col.delete(&ids)).await {
                Ok(Ok(deleted)) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "deleted": deleted,
                        "message": format!("Deleted {} documents", deleted),
                    }),
                ),
                Ok(Err(e)) => status_to_http(sutra_err_to_status(e)),
                Err(e) => HttpResponse::error(500, format!("batch delete task panicked: {}", e)),
            }
        }

        // PATCH /collections/{name}/documents/{id}/metadata
        ("PATCH", ["collections", name, "documents", doc_id, "metadata"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let body = match parse_json_body(&req.body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let Some(meta) = body.as_object().and_then(|o| o.get("metadata")).cloned() else {
                return HttpResponse::error(400, "Field 'metadata' (JSON object) is required");
            };
            if !meta.is_object() {
                return HttpResponse::error(400, "Field 'metadata' must be a JSON object");
            }

            let id_owned = (*doc_id).to_string();
            let mut col = col_arc.write_owned().await;
            match tokio::task::spawn_blocking(move || col.update_metadata(&id_owned, meta)).await {
                Ok(Ok(true)) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "updated": true,
                        "message": format!("Document '{}' metadata updated", doc_id),
                    }),
                ),
                Ok(Ok(false)) => HttpResponse::error(
                    404,
                    format!("Document '{}' not found in collection '{}'", doc_id, name),
                ),
                Ok(Err(e)) => status_to_http(sutra_err_to_status(e)),
                Err(e) => HttpResponse::error(500, format!("update_metadata task panicked: {}", e)),
            }
        }

        // POST /collections/{name}/query
        ("POST", ["collections", name, "query"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let body = match parse_json_body(&req.body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            match execute_json_query(col_arc, &body).await {
                Ok(resp_val) => HttpResponse::json(200, &resp_val),
                Err(resp) => resp,
            }
        }

        // POST /collections/{name}/batch_query
        ("POST", ["collections", name, "batch_query"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let body = match parse_json_body(&req.body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let Some(queries) = body.as_object().and_then(|o| o.get("queries")).and_then(|q| q.as_array()).cloned() else {
                return HttpResponse::error(400, "Field 'queries' (array) is required");
            };
            if queries.is_empty() {
                return HttpResponse::json(200, &serde_json::json!({ "responses": [] }));
            }

            let total = queries.len();
            let mut join_set = JoinSet::new();
            for (idx, q) in queries.into_iter().enumerate() {
                let c = col_arc.clone();
                join_set.spawn(async move {
                    let res = execute_json_query(c, &q).await;
                    (idx, res)
                });
            }

            let mut ordered: Vec<Option<Value>> = vec![None; total];
            while let Some(joined) = join_set.join_next().await {
                match joined {
                    Ok((idx, Ok(val))) => ordered[idx] = Some(val),
                    Ok((_, Err(err_resp))) => return err_resp,
                    Err(e) => {
                        return HttpResponse::error(
                            500,
                            format!("batch_query worker failed: {}", e),
                        )
                    }
                }
            }
            let responses: Vec<Value> = ordered.into_iter().flatten().collect();
            HttpResponse::json(200, &serde_json::json!({ "responses": responses }))
        }

        // POST /collections/{name}/hnsw
        ("POST", ["collections", name, "hnsw"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let body = match parse_optional_json_body(&req.body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let Some(obj) = body.as_object() else {
                return HttpResponse::error(400, "HNSW payload must be a JSON object");
            };

            let m: usize = match obj.get("m") {
                Some(v) if !v.is_null() => match v.as_u64() {
                    Some(n) if (4..=128).contains(&n) => n as usize,
                    _ => return HttpResponse::error(400, "'m' must be between 4 and 128"),
                },
                _ => 16,
            };
            let ef_construction: usize = match obj.get("ef_construction") {
                Some(v) if !v.is_null() => match v.as_u64() {
                    Some(n) if n >= 8 => n as usize,
                    _ => return HttpResponse::error(400, "'ef_construction' must be >= 8"),
                },
                _ => 64,
            };
            let ef_search: usize = match obj.get("ef_search") {
                Some(v) if !v.is_null() => match v.as_u64() {
                    Some(n) if n >= 4 => n as usize,
                    _ => return HttpResponse::error(400, "'ef_search' must be >= 4"),
                },
                _ => 32,
            };

            let cfg = HNSWConfig {
                m,
                m0: m * 2,
                ef_construction,
                ef_search,
                ml: 1.0 / (m as f64).ln(),
                seed: 42,
            };
            let mut col = col_arc.write_owned().await;
            match tokio::task::spawn_blocking(move || col.build_hnsw(Some(cfg)).map(|()| col.count())).await {
                Ok(Ok(count)) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "message": "HNSW index built",
                        "config": {
                            "m": m,
                            "ef_construction": ef_construction,
                            "ef_search": ef_search,
                            "count": count,
                        }
                    }),
                ),
                Ok(Err(e)) => status_to_http(sutra_err_to_status(e)),
                Err(e) => HttpResponse::error(500, format!("build_hnsw task panicked: {}", e)),
            }
        }

        // POST /collections/{name}/sq8
        ("POST", ["collections", name, "sq8"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let mut col = col_arc.write_owned().await;
            match tokio::task::spawn_blocking(move || {
                col.enable_sq8();
                col_stats_json(&col)
            })
            .await
            {
                Ok(stats) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "message": "SQ8 quantization enabled",
                        "stats": stats,
                    }),
                ),
                Err(e) => HttpResponse::error(500, format!("enable_sq8 task panicked: {}", e)),
            }
        }

        // POST /collections/{name}/bq
        ("POST", ["collections", name, "bq"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let mut col = col_arc.write_owned().await;
            match tokio::task::spawn_blocking(move || {
                col.enable_bq();
                col_stats_json(&col)
            })
            .await
            {
                Ok(stats) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "message": "1-Bit Binary Quantization (BQ) enabled (32x RAM reduction)",
                        "stats": stats,
                    }),
                ),
                Err(e) => HttpResponse::error(500, format!("enable_bq task panicked: {}", e)),
            }
        }

        // POST /collections/{name}/pq/train
        ("POST", ["collections", name, "pq", "train"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let body = match parse_optional_json_body(&req.body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let Some(obj) = body.as_object() else {
                return HttpResponse::error(400, "PQ train payload must be a JSON object");
            };
            let m = match obj.get("m") {
                Some(v) if !v.is_null() => match v.as_u64() {
                    Some(n) => Some(n as usize),
                    None => return HttpResponse::error(400, "'m' must be a non-negative integer"),
                },
                _ => None,
            };
            let max_samples = match obj.get("max_samples") {
                Some(v) if !v.is_null() => match v.as_u64() {
                    Some(n) if n > 0 => Some(n as usize),
                    _ => return HttpResponse::error(400, "'max_samples' must be > 0"),
                },
                _ => None,
            };

            let mut col = col_arc.write_owned().await;
            match tokio::task::spawn_blocking(move || {
                col.train_pq(m, max_samples)?;
                Ok::<Value, crate::error::SutraError>(col_stats_json(&col))
            })
            .await
            {
                Ok(Ok(stats)) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "message": "PQ codebook trained",
                        "stats": stats,
                    }),
                ),
                Ok(Err(e)) => status_to_http(sutra_err_to_status(e)),
                Err(e) => HttpResponse::error(500, format!("train_pq task panicked: {}", e)),
            }
        }

        // POST /collections/{name}/pq
        ("POST", ["collections", name, "pq"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let mut col = col_arc.write_owned().await;
            match tokio::task::spawn_blocking(move || {
                col.enable_pq()?;
                Ok::<Value, crate::error::SutraError>(col_stats_json(&col))
            })
            .await
            {
                Ok(Ok(stats)) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "message": "Product Quantization (PQ) enabled (up to 64x RAM reduction)",
                        "stats": stats,
                    }),
                ),
                Ok(Err(e)) => status_to_http(sutra_err_to_status(e)),
                Err(e) => HttpResponse::error(500, format!("enable_pq task panicked: {}", e)),
            }
        }

        // POST /collections/{name}/save
        ("POST", ["collections", name, "save"]) => {
            match service.save_collection_internal(name).await {
                Ok((_path, bytes)) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "message": "Snapshot saved",
                        "bytes": bytes,
                    }),
                ),
                Err(status) => status_to_http(*status),
            }
        }

        // POST /collections/{name}/compact
        ("POST", ["collections", name, "compact"]) => {
            let col_arc = match service.get_collection_arc(name).await {
                Ok(c) => c,
                Err(status) => return status_to_http(*status),
            };
            let mut col = col_arc.write_owned().await;
            match tokio::task::spawn_blocking(move || {
                col.compact()?;
                let _ = write_manifest_for_collection(&col);
                Ok::<Value, crate::error::SutraError>(col_stats_json(&col))
            })
            .await
            {
                Ok(Ok(stats)) => HttpResponse::json(
                    200,
                    &serde_json::json!({
                        "message": "Collection compacted",
                        "stats": stats,
                    }),
                ),
                Ok(Err(e)) => status_to_http(sutra_err_to_status(e)),
                Err(e) => HttpResponse::error(500, format!("compact task panicked: {}", e)),
            }
        }

        _ => HttpResponse::error(404, "Endpoint not found"),
    }
}
