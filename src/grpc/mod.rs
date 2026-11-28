#[allow(clippy::all, dead_code, unused_imports, non_snake_case)]
pub mod proto {
    include!("sutra_gen.rs");
}

pub mod auth;
pub mod http_bridge;
pub mod metrics;
pub mod service;

pub use auth::{ApiKeyPolicy, AuthConfig, Role};
pub use metrics::MetricsRegistry;
pub use proto::sutra_service_server::{SutraService, SutraServiceServer};
pub use proto::sutra_service_client::SutraServiceClient;
pub use proto::*;
pub use service::SutraGrpcService;
