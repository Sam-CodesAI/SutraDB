use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use clap::Parser;
use tonic::transport::Server;
use _sutradb::grpc::{http_bridge, SutraGrpcService, SutraServiceServer};

#[derive(Parser, Debug)]
#[command(name = "sutradb-server")]
#[command(author = "Samarth Nimangre <samarthknimangre@gmail.com>")]
#[command(version = env!("CARGO_PKG_VERSION"))]
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

    /// Optional loopback HTTP/JSON control-plane bridge address (env: SUTRA_HTTP_BRIDGE_ADDR)
    #[arg(long, value_name = "ADDR")]
    http_bridge_addr: Option<String>,
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
    let bridge_addr_str = args
        .http_bridge_addr
        .or_else(|| {
            std::env::var("SUTRA_HTTP_BRIDGE_ADDR")
                .ok()
                .filter(|s| !s.trim().is_empty())
        });

    let bridge_listener = if let Some(ref b_str) = bridge_addr_str {
        let b_addr: SocketAddr = b_str.trim().parse()?;
        Some((b_addr, tokio::net::TcpListener::bind(b_addr).await?))
    } else {
        None
    };

    println!("========================================================");
    println!(
        "  SutraDB v{} Industrial Binary gRPC Engine",
        env!("CARGO_PKG_VERSION")
    );
    println!("  Listening on   : http://{}", addr);
    if let Some((b_addr, _)) = &bridge_listener {
        println!("  HTTP Bridge    : http://{}", b_addr);
    }
    println!("  Data directory : {:?}", args.data_dir);
    println!("  Protocol       : HTTP/2 gRPC Streaming over Protobuf");
    println!("========================================================");
    let _ = std::io::stdout().flush();

    if !args.data_dir.exists() {
        std::fs::create_dir_all(&args.data_dir)?;
    }

    let service = SutraGrpcService::new(Some(args.data_dir));
    let server = SutraServiceServer::new(service.clone());

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut grpc_shutdown_rx = shutdown_rx.clone();

    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    let bridge_handle = bridge_listener.map(|(_, listener)| {
        let bridge_service = service.clone();
        let bridge_rx = shutdown_rx.clone();
        tokio::spawn(async move {
            if let Err(e) = http_bridge::serve_http_bridge(listener, bridge_service, bridge_rx).await
            {
                eprintln!("[SutraDB] HTTP bridge error: {}", e);
            }
        })
    });

    Server::builder()
        .add_service(server)
        .serve_with_shutdown(addr, async move {
            while !*grpc_shutdown_rx.borrow() {
                if grpc_shutdown_rx.changed().await.is_err() {
                    break;
                }
            }
        })
        .await?;

    if let Some(handle) = bridge_handle {
        let _ = handle.await;
    }

    println!("[SutraDB] Server shutdown complete.");
    Ok(())
}
