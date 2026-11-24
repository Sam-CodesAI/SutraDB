fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/sutra.proto");

    // Only compile protobufs if protoc is detected or explicit override is set
    let has_protoc = std::env::var("PROTOC").is_ok()
        || std::process::Command::new("protoc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

    if has_protoc {
        if let Err(e) = tonic_build::compile_protos("proto/sutra.proto") {
            println!("cargo:warning=tonic_build failed: {e}. Falling back to pre-generated src/grpc/sutra_gen.rs");
        }
    } else {
        println!("cargo:warning=protoc not found. Using pre-generated src/grpc/sutra_gen.rs");
    }

    Ok(())
}
