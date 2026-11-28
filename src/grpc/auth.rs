use std::collections::HashMap;
use serde::Deserialize;
use tonic::metadata::MetadataMap;
use tonic::Status;
use crate::error::SutraError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Read = 1,
    Write = 2,
    Admin = 3,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawPolicyInput {
    RoleOnly(Role),
    Detailed {
        role: Role,
        #[serde(default)]
        collections: Option<Vec<String>>,
        #[serde(default)]
        prefixes: Option<Vec<String>>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawRbacDocument {
    Wrapped {
        keys: HashMap<String, RawPolicyInput>,
    },
    Flat(HashMap<String, RawPolicyInput>),
}

#[derive(Debug, Clone)]
pub struct ApiKeyPolicy {
    pub key: String,
    pub role: Role,
    pub collections: Option<Vec<String>>,
    pub prefixes: Option<Vec<String>>,
}

impl ApiKeyPolicy {
    pub fn allows_collection(&self, collection_name: &str) -> bool {
        if self.collections.is_none() && self.prefixes.is_none() {
            return true;
        }
        if let Some(ref cols) = self.collections {
            for pat in cols {
                if pat == "*" || pat == collection_name {
                    return true;
                }
                if let Some(prefix) = pat.strip_suffix('*') {
                    if collection_name.starts_with(prefix) {
                        return true;
                    }
                }
            }
        }
        if let Some(ref prefs) = self.prefixes {
            for p in prefs {
                if collection_name.starts_with(p) {
                    return true;
                }
            }
        }
        false
    }
}

/// HTTP-oriented auth failure mapped to 401/403 status codes.
#[derive(Debug, Clone)]
pub struct HttpAuthError {
    pub status: u16,
    pub message: String,
}

impl HttpAuthError {
    fn from_status(status: Status) -> Self {
        let code = match status.code() {
            tonic::Code::Unauthenticated => 401,
            tonic::Code::PermissionDenied => 403,
            _ => 500,
        };
        Self {
            status: code,
            message: status.message().to_string(),
        }
    }
}

/// Constant-time byte comparison that inspects all bytes up to `max(a.len(), b.len())`
/// without early-returning on mismatch or length difference.
#[inline]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff: usize = a.len() ^ b.len();
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let x = a.get(i).copied().unwrap_or(0) as usize;
        let y = b.get(i).copied().unwrap_or(0) as usize;
        diff |= x ^ y;
    }
    std::hint::black_box(diff) == 0
}

#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    policies: Vec<ApiKeyPolicy>,
}

impl AuthConfig {
    /// Loads auth policy from `SUTRA_API_KEY` (admin key) and optional `SUTRA_RBAC_CONFIG` JSON.
    pub fn from_env() -> Result<Self, SutraError> {
        let admin_key = std::env::var("SUTRA_API_KEY").ok();
        let rbac_json = std::env::var("SUTRA_RBAC_CONFIG").ok();
        Self::from_parts(admin_key.as_deref(), rbac_json.as_deref())
    }

    /// Constructs auth policy from explicit admin key and optional JSON RBAC string.
    pub fn from_parts(
        admin_key: Option<&str>,
        rbac_json: Option<&str>,
    ) -> Result<Self, SutraError> {
        let mut policies_map: HashMap<String, ApiKeyPolicy> = HashMap::new();

        if let Some(raw_admin) = admin_key {
            let trimmed = raw_admin.trim();
            if !trimmed.is_empty() {
                policies_map.insert(
                    trimmed.to_string(),
                    ApiKeyPolicy {
                        key: trimmed.to_string(),
                        role: Role::Admin,
                        collections: None,
                        prefixes: None,
                    },
                );
            }
        }

        if let Some(raw_rbac) = rbac_json {
            let trimmed = raw_rbac.trim();
            if !trimmed.is_empty() {
                let parsed: RawRbacDocument = serde_json::from_str(trimmed).map_err(|e| {
                    SutraError::Serialization(format!("Invalid SUTRA_RBAC_CONFIG JSON: {}", e))
                })?;
                let entries = match parsed {
                    RawRbacDocument::Wrapped { keys } => keys,
                    RawRbacDocument::Flat(map) => map,
                };
                for (k, raw_policy) in entries {
                    let key_trimmed = k.trim();
                    if key_trimmed.is_empty() {
                        return Err(SutraError::InvalidQuery(
                            "SUTRA_RBAC_CONFIG contains an empty API key".into(),
                        ));
                    }
                    let policy = match raw_policy {
                        RawPolicyInput::RoleOnly(role) => ApiKeyPolicy {
                            key: key_trimmed.to_string(),
                            role,
                            collections: None,
                            prefixes: None,
                        },
                        RawPolicyInput::Detailed {
                            role,
                            collections,
                            prefixes,
                        } => ApiKeyPolicy {
                            key: key_trimmed.to_string(),
                            role,
                            collections,
                            prefixes,
                        },
                    };
                    policies_map.insert(key_trimmed.to_string(), policy);
                }
            }
        }

        let mut policies: Vec<ApiKeyPolicy> = policies_map.into_values().collect();
        policies.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(Self { policies })
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        !self.policies.is_empty()
    }

    #[allow(clippy::result_large_err)]
    fn authenticate_token(&self, token: Option<&str>) -> Result<Option<ApiKeyPolicy>, Status> {
        if !self.is_enabled() {
            return Ok(None);
        }
        let provided = match token {
            Some(t) if !t.trim().is_empty() => t.trim(),
            _ => return Err(Status::unauthenticated("Missing API key")),
        };

        // Always evaluate all configured keys in constant time to avoid timing side channels.
        let mut matched: Option<usize> = None;
        for (idx, policy) in self.policies.iter().enumerate() {
            if constant_time_eq(policy.key.as_bytes(), provided.as_bytes()) {
                matched = Some(idx);
            }
        }

        match matched {
            Some(idx) => Ok(Some(self.policies[idx].clone())),
            None => Err(Status::unauthenticated("Invalid API key")),
        }
    }

    pub fn extract_token_from_http_headers(headers: &HashMap<String, String>) -> Option<String> {
        if let Some(auth_val) = headers.get("authorization") {
            let trimmed = auth_val.trim();
            if trimmed.len() >= 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
                let tok = trimmed[7..].trim();
                if !tok.is_empty() {
                    return Some(tok.to_string());
                }
            }
        }
        if let Some(api_key) = headers.get("x-api-key") {
            let tok = api_key.trim();
            if !tok.is_empty() {
                return Some(tok.to_string());
            }
        }
        None
    }

    pub fn extract_token_from_grpc_metadata(metadata: &MetadataMap) -> Option<String> {
        if let Some(auth_val) = metadata.get("authorization").and_then(|v| v.to_str().ok()) {
            let trimmed = auth_val.trim();
            if trimmed.len() >= 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
                let tok = trimmed[7..].trim();
                if !tok.is_empty() {
                    return Some(tok.to_string());
                }
            }
        }
        if let Some(api_key) = metadata.get("x-api-key").and_then(|v| v.to_str().ok()) {
            let tok = api_key.trim();
            if !tok.is_empty() {
                return Some(tok.to_string());
            }
        }
        None
    }

    #[allow(clippy::result_large_err)]
    pub fn authorize_grpc_metadata(
        &self,
        metadata: &MetadataMap,
        required_role: Role,
        collection: Option<&str>,
    ) -> Result<Option<ApiKeyPolicy>, Status> {
        if !self.is_enabled() {
            return Ok(None);
        }
        let token = Self::extract_token_from_grpc_metadata(metadata);
        let policy = self
            .authenticate_token(token.as_deref())?
            .expect("Policy present when auth enabled");
        if policy.role < required_role {
            return Err(Status::permission_denied(format!(
                "Insufficient role: requires {:?}, got {:?}",
                required_role, policy.role
            )));
        }
        if let Some(col_name) = collection {
            if !policy.allows_collection(col_name) {
                return Err(Status::permission_denied(format!(
                    "API key is not authorized for collection '{}'",
                    col_name
                )));
            }
        }
        Ok(Some(policy))
    }

    pub fn authorize_http(
        &self,
        headers: &HashMap<String, String>,
        required_role: Role,
        collection: Option<&str>,
    ) -> Result<Option<ApiKeyPolicy>, HttpAuthError> {
        if !self.is_enabled() {
            return Ok(None);
        }
        let token = Self::extract_token_from_http_headers(headers);
        let policy = self
            .authenticate_token(token.as_deref())
            .map_err(HttpAuthError::from_status)?
            .expect("Policy present when auth enabled");
        if policy.role < required_role {
            return Err(HttpAuthError {
                status: 403,
                message: format!(
                    "Insufficient role: requires {:?}, got {:?}",
                    required_role, policy.role
                ),
            });
        }
        if let Some(col_name) = collection {
            if !policy.allows_collection(col_name) {
                return Err(HttpAuthError {
                    status: 403,
                    message: format!("API key is not authorized for collection '{}'", col_name),
                });
            }
        }
        Ok(Some(policy))
    }
}
