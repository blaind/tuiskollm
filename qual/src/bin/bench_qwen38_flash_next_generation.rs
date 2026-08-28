//! Qwen3.8 Flash-Next generation benchmark entry point.

use std::error::Error;
use std::path::PathBuf;

#[cfg(feature = "device")]
use tuisko_qual::{
    benchmark_qwen38_flash_next_generation, print_qwen38_flash_next_generation_benchmark,
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
        .ok_or("usage: bench-qwen38-flash-next-generation SNAPSHOT")?;
    if arguments.next().is_some() {
        return Err("usage: bench-qwen38-flash-next-generation SNAPSHOT".into());
    }
    let report = benchmark_qwen38_flash_next_generation(&PathBuf::from(snapshot))?;
    print_qwen38_flash_next_generation_benchmark(&report);

    Ok(())
}

#[cfg(not(feature = "device"))]
fn run() -> Result<(), Box<dyn Error>> {
    let _ = PathBuf::new();

    Err("bench-qwen38-flash-next-generation requires the `device` feature".into())
}
