//! End-to-end Qwen3.8 Flash-Next resident-program sweep.
//!
//! A step spans 49 graph replays and 48 host-resolved streaming rounds, so it uses a dedicated
//! host-observed timing harness.

use std::error::Error;
use std::path::PathBuf;

#[cfg(feature = "device")]
use tuisko_qual::{
    benchmark_qwen38_flash_next_resident_model, print_qwen38_flash_next_resident_benchmark,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(feature = "device")]
fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let snapshot = arguments
        .next()
        .ok_or("usage: bench-qwen38-flash-next-resident SNAPSHOT")?;
    let report = benchmark_qwen38_flash_next_resident_model(&PathBuf::from(snapshot))?;
    print_qwen38_flash_next_resident_benchmark(&report);

    Ok(())
}

#[cfg(not(feature = "device"))]
fn run() -> Result<(), Box<dyn Error>> {
    let _ = PathBuf::new();
    Err("bench-qwen38-flash-next-resident requires the `device` feature".into())
}
