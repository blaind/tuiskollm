//! Times sequential and grouped prompt admission on the Qwen3.8 Flash-Next owner.

use std::error::Error;
use std::path::PathBuf;

#[cfg(feature = "device")]
use tuisko_qual::{
    benchmark_qwen38_flash_next_prompt_prime, print_qwen38_flash_next_prompt_prime_benchmark,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(feature = "device")]
fn run() -> Result<(), Box<dyn Error>> {
    let snapshot = std::env::args_os()
        .nth(1)
        .ok_or("usage: bench-qwen38-flash-next-prompt-prime SNAPSHOT")?;
    let report = benchmark_qwen38_flash_next_prompt_prime(&PathBuf::from(snapshot))?;
    print_qwen38_flash_next_prompt_prime_benchmark(&report);

    Ok(())
}

#[cfg(not(feature = "device"))]
fn run() -> Result<(), Box<dyn Error>> {
    let _ = PathBuf::new();
    Err("bench-qwen38-flash-next-prompt-prime requires the `device` feature".into())
}
