//! Rust-owned inference server.

mod hf;

use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use tuisko_serve::ServerConfig;

const DEFAULT_ADDRESS: &str = "127.0.0.1:8000";
const USAGE: &str = "TuiskoLLM exact-target inference server\n\nUsage:\n  tuiskollm serve [SNAPSHOT] [ADDRESS]\n  tuiskollm --help\n  tuiskollm --version\n\nWithout SNAPSHOT, the pinned Hugging Face cache entry is used. ADDRESS defaults to 127.0.0.1:8000.";

#[derive(Debug, Eq, PartialEq)]
struct ServeArgs {
    snapshot: Option<PathBuf>,
    address: SocketAddr,
}

#[derive(Debug, Eq, PartialEq)]
enum Command {
    Serve(ServeArgs),
    Help,
    Version,
}

fn main() -> ExitCode {
    match parse_args(std::env::args_os().skip(1)) {
        Ok(Command::Serve(args)) => match run_serve(args) {
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

fn run_serve(args: ServeArgs) -> Result<(), String> {
    let resolution = hf::resolve_snapshot(args.snapshot)?;
    if let Some(provisioning) = resolution.provisioning {
        let stdout = std::io::stdout();
        let color = stdout.is_terminal() && std::env::var_os("NO_COLOR").is_none();
        let mut stdout = stdout.lock();
        stdout
            .write_all(render_provisioning(&provisioning, color).as_bytes())
            .map_err(|error| format!("writing startup output: {error}"))?;
        stdout
            .flush()
            .map_err(|error| format!("flushing startup output: {error}"))?;
    }
    tuisko_serve::run(ServerConfig {
        snapshot: resolution.path,
        address: args.address,
    })
    .map_err(|error| error.to_string())
}

fn render_provisioning(provisioning: &hf::Provisioning, color: bool) -> String {
    let (ok, reset) = if color {
        ("\x1b[32m", "\x1b[0m")
    } else {
        ("", "")
    };
    format!(
        "{ok}OK{reset} snapshot     {:>7.1} ms · {} files · {:.2} GiB\n",
        provisioning.elapsed.as_secs_f64() * 1_000.0,
        provisioning.files,
        provisioning.bytes as f64 / (1_u64 << 30) as f64,
    )
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

    let snapshot = args.next().map(PathBuf::from);
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
    Ok(Command::Serve(ServeArgs { snapshot, address }))
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
    use super::{Command, DEFAULT_ADDRESS, ServeArgs, parse_args, render_provisioning};
    use std::ffi::OsString;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::Duration;

    fn parse(args: &[&str]) -> Result<Command, String> {
        parse_args(args.iter().map(OsString::from))
    }

    #[test]
    fn serve_defaults_to_loopback_and_preserves_the_snapshot_path() {
        let command = parse(&["serve", "/models/pinned"]).unwrap();
        assert_eq!(
            command,
            Command::Serve(ServeArgs {
                snapshot: Some(PathBuf::from("/models/pinned")),
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
        assert_eq!(
            parse(&["serve"]).unwrap(),
            Command::Serve(ServeArgs {
                snapshot: None,
                address: DEFAULT_ADDRESS.parse().unwrap(),
            })
        );
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

    #[test]
    fn provisioning_output_derives_every_reported_value() {
        let provisioning = super::hf::Provisioning {
            elapsed: Duration::from_micros(338_500),
            files: 7,
            bytes: 23_437_778_955,
        };
        assert_eq!(
            render_provisioning(&provisioning, false),
            "OK snapshot       338.5 ms · 7 files · 21.83 GiB\n"
        );
        assert!(render_provisioning(&provisioning, true).starts_with("\x1b[32mOK\x1b[0m"));
    }
}
