//! Fresh-process startup benchmark entry point.

use std::process::ExitCode;

fn main() -> ExitCode {
    match tuisko_qual::run_startup_benchmark_cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!();
            eprintln!("============================================================");
            eprintln!("TUISKO STARTUP BENCHMARK FAILED");
            eprintln!("============================================================");
            eprintln!("{error}");
            eprintln!("============================================================");
            ExitCode::FAILURE
        }
    }
}
