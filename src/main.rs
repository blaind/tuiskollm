//! Rust-owned inference server.

use std::ffi::OsString;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use tuisko_serve::ServerConfig;

const DEFAULT_ADDRESS: &str = "127.0.0.1:8000";
const USAGE: &str = "TuiskoLLM exact-target inference server\n\nUsage:\n  tuiskollm serve SNAPSHOT [ADDRESS]\n  tuiskollm --help\n  tuiskollm --version\n\nADDRESS defaults to 127.0.0.1:8000.";

#[derive(Debug, Eq, PartialEq)]
enum Command {
    Serve(ServerConfig),
    Help,
    Version,
}

fn main() -> ExitCode {
    match parse_args(std::env::args_os().skip(1)) {
        Ok(Command::Serve(config)) => match tuisko_serve::run(config) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("tuiskollm: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(Command::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("tuiskollm {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("tuiskollm: {error}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        return Err("missing command".into());
    };
    if command == "--help" || command == "-h" {
        require_end(args)?;
        return Ok(Command::Help);
    }
    if command == "--version" || command == "-V" {
        require_end(args)?;
        return Ok(Command::Version);
    }
    if command != "serve" {
        return Err(format!("unknown command `{}`", command.to_string_lossy()));
    }

    let snapshot = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "serve requires SNAPSHOT".to_owned())?;
    let address = match args.next() {
        Some(address) => address
            .to_str()
            .ok_or_else(|| "ADDRESS must be valid UTF-8".to_owned())?
            .parse::<SocketAddr>()
            .map_err(|error| format!("invalid ADDRESS: {error}"))?,
        None => DEFAULT_ADDRESS
            .parse()
            .expect("the checked default address is valid"),
    };
    require_end(args)?;
    Ok(Command::Serve(ServerConfig { snapshot, address }))
}

fn require_end(mut args: impl Iterator<Item = OsString>) -> Result<(), String> {
    match args.next() {
        None => Ok(()),
        Some(argument) => Err(format!(
            "unexpected argument `{}`",
            argument.to_string_lossy()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, DEFAULT_ADDRESS, parse_args};
    use std::ffi::OsString;
    use std::net::SocketAddr;
    use std::path::PathBuf;

    fn parse(args: &[&str]) -> Result<Command, String> {
        parse_args(args.iter().map(OsString::from))
    }

    #[test]
    fn serve_defaults_to_loopback_and_preserves_the_snapshot_path() {
        let command = parse(&["serve", "/models/pinned"]).unwrap();
        assert_eq!(
            command,
            Command::Serve(tuisko_serve::ServerConfig {
                snapshot: PathBuf::from("/models/pinned"),
                address: DEFAULT_ADDRESS.parse::<SocketAddr>().unwrap(),
            })
        );
    }

    #[test]
    fn serve_accepts_one_explicit_numeric_socket_address() {
        let command = parse(&["serve", "snapshot", "0.0.0.0:9123"]).unwrap();
        let Command::Serve(config) = command else {
            panic!("expected serve command");
        };
        assert_eq!(config.address, "0.0.0.0:9123".parse().unwrap());
    }

    #[test]
    fn malformed_or_ambiguous_commands_are_refused() {
        assert!(parse(&[]).unwrap_err().contains("missing command"));
        assert!(parse(&["server"]).unwrap_err().contains("unknown command"));
        assert!(parse(&["serve"]).unwrap_err().contains("requires SNAPSHOT"));
        assert!(
            parse(&["serve", "snapshot", "localhost:8000"])
                .unwrap_err()
                .contains("invalid ADDRESS")
        );
        assert!(
            parse(&["serve", "snapshot", "127.0.0.1:8000", "extra"])
                .unwrap_err()
                .contains("unexpected argument")
        );
    }

    #[test]
    fn informational_commands_accept_no_trailing_arguments() {
        assert_eq!(parse(&["--help"]).unwrap(), Command::Help);
        assert_eq!(parse(&["-V"]).unwrap(), Command::Version);
        assert!(parse(&["--help", "extra"]).is_err());
    }
}
