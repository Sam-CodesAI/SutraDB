#[allow(clippy::all, dead_code, unused_imports, non_snake_case)]
pub mod proto {
    include!("sutra_gen.rs");
}

pub mod service;

pub use proto::sutra_service_server::{SutraService, SutraServiceServer};
pub use proto::sutra_service_client::SutraServiceClient;
pub use proto::*;
pub use service::SutraGrpcService;
