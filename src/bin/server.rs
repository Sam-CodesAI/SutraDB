use std::net::SocketAddr;
use std::path::PathBuf;
use clap::Parser;
use tonic::transport::Server;
use _sutradb::grpc::{SutraGrpcService, SutraServiceServer};

#[derive(Parser, Debug)]
#[command(name = "sutradb-server")]
#[command(author = "Samarth Nimangre <samarthknimangre@gmail.com>")]
#[command(version = "5.1.0")]
#[command(about = "SutraDB Industrial High-Throughput Binary gRPC Vector Engine", long_about = None)]
struct Args {
    /// Host interface to bind to
    #[arg(short = 'H', long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 50051)]
    port: u16,

    /// Persistent data directory for collections and WAL logs
    #[arg(short, long, default_value = "./sutra_data")]
    data_dir: PathBuf,
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(e) => {
                eprintln!("[SutraDB] Warning: Failed to install SIGTERM handler: {}", e);
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            println!("\n[SutraDB] Received SIGINT (Ctrl+C). Initiating graceful shutdown...");
        },
        _ = terminate => {
            println!("\n[SutraDB] Received SIGTERM. Initiating graceful shutdown...");
        },
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    println!("========================================================");
    println!("  SutraDB v4.0 Industrial Binary gRPC Engine");
    println!("  Listening on   : http://{}", addr);
    println!("  Data directory : {:?}", args.data_dir);
    println!("  Protocol       : HTTP/2 gRPC Streaming over Protobuf");
    println!("========================================================");

    if !args.data_dir.exists() {
        std::fs::create_dir_all(&args.data_dir)?;
    }

    let service = SutraGrpcService::new(Some(args.data_dir));
    let server = SutraServiceServer::new(service);

    Server::builder()
        .add_service(server)
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    println!("[SutraDB] Server shutdown complete.");
    Ok(())
}
