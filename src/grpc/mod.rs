pub mod proto {
    tonic::include_proto!("sutra");
}

pub mod service;

pub use proto::sutra_service_server::{SutraService, SutraServiceServer};
pub use proto::sutra_service_client::SutraServiceClient;
pub use proto::*;
pub use service::SutraGrpcService;
