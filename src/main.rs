//! Rust-owned inference server.

use clap::{Args, Parser, Subcommand, error::ErrorKind};
use std::io::{IsTerminal, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};
use tuisko_provision::{Provisioning, ProvisioningProgress, ProvisioningStage, SnapshotResolution};
use tuisko_serve::{ServerConfig, ServerModel};

const DEFAULT_ADDRESS: &str = "127.0.0.1:8000";

#[derive(Debug, Parser)]
#[command(
    name = "tuiskollm",
    version,
    about = "TuiskoLLM exact-target inference server"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand, Eq, PartialEq)]
enum Command {
    /// Serves one exact model through the OpenAI-compatible API.
    Serve(ServeArgs),
}

#[derive(Debug, Args, Eq, PartialEq)]
struct ServeArgs {
    /// Exact Hugging Face model ID.
    #[arg(
        value_name = "MODEL",
        long_help = "Exact Hugging Face model ID.\n\nSupported models:\n  unsloth/Qwen3.8-27B-NVFP4             automatic download\n  AxionML/Qwen3.5-9B-NVFP4              --snapshot required\n  nvidia/Qwen3.6-35B-A3B-NVFP4          --snapshot required"
    )]
    model: ServerModel,
    /// Existing admitted snapshot; Qwen3.8 downloads automatically when omitted.
    #[arg(long, value_name = "SNAPSHOT")]
    snapshot: Option<PathBuf>,
    /// TCP listen address.
    #[arg(long, value_name = "ADDRESS", default_value = DEFAULT_ADDRESS)]
    address: SocketAddr,
}

fn main() -> ExitCode {
    let command = match parse_args(std::env::args_os().skip(1)) {
        Ok(command) => command,
        Err(error) => error.exit(),
    };
    match command {
        Command::Serve(args) => match run_serve(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("tuiskollm: {error}");
                ExitCode::FAILURE
            }
        },
    }
}

fn run_serve(args: ServeArgs) -> Result<(), String> {
    let stdout = std::io::stdout();
    let interactive = stdout.is_terminal();
    let color = interactive && std::env::var_os("NO_COLOR").is_none();
    let resolution = {
        let mut stdout = stdout.lock();
        let mut display = ProvisioningDisplay::new(interactive, color);
        let resolution = match (args.model, args.snapshot) {
            (_, Some(path)) => Ok(SnapshotResolution {
                path,
                provisioning: None,
            }),
            (ServerModel::Qwen38, None) => {
                tuisko_provision::resolve_snapshot_with_progress(None, |progress| {
                    display
                        .update(&mut stdout, progress)
                        .map_err(|error| format!("writing provisioning progress: {error}"))
                })
            }
            (model, None) => Err(format!(
                "automatic download is not available for {}; pass --snapshot SNAPSHOT",
                model.model_id(),
            )),
        };
        display
            .finish(&mut stdout)
            .map_err(|error| format!("finishing provisioning progress: {error}"))?;
        let resolution = resolution?;
        if let Some(provisioning) = &resolution.provisioning {
            stdout
                .write_all(render_provisioning(provisioning, color).as_bytes())
                .map_err(|error| format!("writing startup output: {error}"))?;
            stdout
                .flush()
                .map_err(|error| format!("flushing startup output: {error}"))?;
        }
        resolution
    };
    tuisko_serve::run(ServerConfig {
        model: args.model,
        snapshot: resolution.path,
        address: args.address,
    })
    .map_err(|error| error.to_string())
}

struct ProvisioningDisplay {
    interactive: bool,
    color: bool,
    last_draw: Option<Instant>,
    last_stage: Option<(ProvisioningStage, &'static str)>,
    active: bool,
}

impl ProvisioningDisplay {
    fn new(interactive: bool, color: bool) -> Self {
        Self {
            interactive,
            color,
            last_draw: None,
            last_stage: None,
            active: false,
        }
    }

    fn update(
        &mut self,
        output: &mut impl Write,
        progress: ProvisioningProgress,
    ) -> std::io::Result<()> {
        let stage = (progress.stage(), progress.file());
        let stage_changed = self.last_stage != Some(stage);
        let complete = progress.file_bytes() == progress.file_total();
        if self.interactive
            && !stage_changed
            && !complete
            && self
                .last_draw
                .is_some_and(|draw| draw.elapsed() < Duration::from_millis(50))
        {
            return Ok(());
        }
        if !self.interactive && !stage_changed {
            return Ok(());
        }

        let line = render_provisioning_progress(
            progress.stage(),
            progress.file(),
            progress.file_bytes(),
            progress.file_total(),
            progress.completed_bytes(),
            progress.total_bytes(),
            self.color,
        );
        if self.interactive {
            output.write_all(b"\r\x1b[2K")?;
            output.write_all(line.as_bytes())?;
        } else {
            writeln!(output, "{line}")?;
        }
        output.flush()?;
        self.last_draw = Some(Instant::now());
        self.last_stage = Some(stage);
        self.active = true;
        Ok(())
    }

    fn finish(&mut self, output: &mut impl Write) -> std::io::Result<()> {
        if self.interactive && self.active {
            output.write_all(b"\r\x1b[2K")?;
            output.flush()?;
        }
        Ok(())
    }
}

fn render_provisioning_progress(
    stage: ProvisioningStage,
    file: &str,
    file_bytes: u64,
    file_total: u64,
    completed_bytes: u64,
    total_bytes: u64,
    color: bool,
) -> String {
    let (label, escape) = match stage {
        ProvisioningStage::Verifying => ("VERIFYING", "\x1b[1;36m"),
        ProvisioningStage::Downloading => ("FETCHING", "\x1b[1;33m"),
        ProvisioningStage::Finalizing => ("FINALIZING", "\x1b[1;33m"),
    };
    let (escape, reset) = if color { (escape, "\x1b[0m") } else { ("", "") };
    if stage == ProvisioningStage::Finalizing {
        return format!("{escape}{label}{reset} {file}…");
    }

    const BAR_WIDTH: u64 = 20;
    let filled = file_bytes
        .min(file_total)
        .saturating_mul(BAR_WIDTH)
        .checked_div(file_total)
        .unwrap_or(0) as usize;
    let bar = format!(
        "{}{}",
        "█".repeat(filled),
        "░".repeat(BAR_WIDTH as usize - filled)
    );
    format!(
        "{escape}{label}{reset} {file}  {bar}  {:.2} / {:.2} GiB · total {:.2} / {:.2} GiB",
        gibibytes(file_bytes),
        gibibytes(file_total),
        gibibytes(completed_bytes),
        gibibytes(total_bytes),
    )
}

fn render_provisioning(provisioning: &Provisioning, color: bool) -> String {
    let (ok, reset) = if color {
        ("\x1b[32m", "\x1b[0m")
    } else {
        ("", "")
    };
    format!(
        "{ok}OK{reset} snapshot     {:>7.1} ms · {} files · {:.2} GiB\n",
        provisioning.elapsed.as_secs_f64() * 1_000.0,
        provisioning.files,
        gibibytes(provisioning.bytes),
    )
}

fn gibibytes(bytes: u64) -> f64 {
    bytes as f64 / (1_u64 << 30) as f64
}

fn parse_args<I, T>(args: I) -> Result<Command, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let arguments = std::iter::once(std::ffi::OsString::from("tuiskollm"))
        .chain(args.into_iter().map(Into::into));
    let cli = Cli::try_parse_from(arguments)?;
    match cli.command {
        Command::Serve(args) if args.model != ServerModel::Qwen38 && args.snapshot.is_none() => {
            Err(clap::Error::raw(
                ErrorKind::MissingRequiredArgument,
                format!("{} requires --snapshot SNAPSHOT", args.model.model_id()),
            ))
        }
        command => Ok(command),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Command, DEFAULT_ADDRESS, ServeArgs, parse_args, render_provisioning,
        render_provisioning_progress,
    };
    use clap::error::ErrorKind;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::Duration;
    use tuisko_provision::{Provisioning, ProvisioningStage};
    use tuisko_serve::ServerModel;

    const QWEN38: &str = "unsloth/Qwen3.8-27B-NVFP4";
    const QWEN35: &str = "AxionML/Qwen3.5-9B-NVFP4";

    fn parse(args: &[&str]) -> Result<Command, clap::Error> {
        parse_args(args.iter().copied())
    }

    #[test]
    fn serve_requires_an_exact_model_and_defaults_to_loopback() {
        let command = parse(&["serve", QWEN38]).unwrap();
        assert_eq!(
            command,
            Command::Serve(ServeArgs {
                model: ServerModel::Qwen38,
                snapshot: None,
                address: DEFAULT_ADDRESS.parse::<SocketAddr>().unwrap(),
            })
        );
        assert_eq!(
            parse(&["serve"]).unwrap_err().kind(),
            ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn serve_accepts_named_options_in_any_order() {
        for arguments in [
            vec![
                "serve",
                QWEN35,
                "--snapshot",
                "/models/pinned",
                "--address",
                "0.0.0.0:9123",
            ],
            vec![
                "serve",
                QWEN35,
                "--address",
                "0.0.0.0:9123",
                "--snapshot",
                "/models/pinned",
            ],
        ] {
            assert_eq!(
                parse(&arguments).unwrap(),
                Command::Serve(ServeArgs {
                    model: ServerModel::Qwen35,
                    snapshot: Some(PathBuf::from("/models/pinned")),
                    address: "0.0.0.0:9123".parse().unwrap(),
                })
            );
        }
    }

    #[test]
    fn malformed_or_ambiguous_commands_are_refused() {
        assert_eq!(
            parse(&[]).unwrap_err().kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        assert_eq!(
            parse(&["server"]).unwrap_err().kind(),
            ErrorKind::InvalidSubcommand
        );
        assert!(
            parse(&["serve", "unsloth/Qwen3.8-Flash-Next-GGUF"])
                .unwrap_err()
                .to_string()
                .contains("unsupported model")
        );
        assert!(
            parse(&["serve", QWEN35])
                .unwrap_err()
                .to_string()
                .contains("AxionML/Qwen3.5-9B-NVFP4 requires --snapshot SNAPSHOT")
        );
        assert!(
            parse(&["serve", QWEN38, "--address", "localhost:8000"])
                .unwrap_err()
                .to_string()
                .contains("invalid socket address")
        );
        assert_eq!(
            parse(&["serve", QWEN38, "snapshot"]).unwrap_err().kind(),
            ErrorKind::UnknownArgument
        );
        for option in ["--snapshot", "--address"] {
            assert_eq!(
                parse(&["serve", QWEN38, option]).unwrap_err().kind(),
                ErrorKind::InvalidValue
            );
        }
        assert_eq!(
            parse(&["serve", QWEN38, "--snapshot", "one", "--snapshot", "two",])
                .unwrap_err()
                .kind(),
            ErrorKind::ArgumentConflict
        );
        assert_eq!(
            parse(&["serve", "--model", QWEN38]).unwrap_err().kind(),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn clap_owns_help_and_version_output() {
        assert_eq!(
            parse(&["--help"]).unwrap_err().kind(),
            ErrorKind::DisplayHelp
        );
        let serve_help = parse(&["serve", "--help"]).unwrap_err();
        assert_eq!(serve_help.kind(), ErrorKind::DisplayHelp);
        assert!(
            serve_help
                .to_string()
                .contains("Usage: tuiskollm serve [OPTIONS] <MODEL>")
        );
        for model in [QWEN38, QWEN35, "nvidia/Qwen3.6-35B-A3B-NVFP4"] {
            assert!(serve_help.to_string().contains(model));
        }
        assert_eq!(
            parse(&["-V"]).unwrap_err().kind(),
            ErrorKind::DisplayVersion
        );
    }

    #[test]
    fn provisioning_output_derives_every_reported_value() {
        let provisioning = Provisioning {
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

    #[test]
    fn provisioning_progress_reports_exact_file_and_total_bytes() {
        let fetching = render_provisioning_progress(
            ProvisioningStage::Downloading,
            "model.safetensors",
            3 << 30,
            4 << 30,
            5 << 30,
            8 << 30,
            false,
        );
        assert!(fetching.starts_with("FETCHING model.safetensors"));
        assert!(fetching.contains("███████████████░░░░░"));
        assert!(fetching.contains("3.00 / 4.00 GiB · total 5.00 / 8.00 GiB"));

        let verifying = render_provisioning_progress(
            ProvisioningStage::Verifying,
            "model.safetensors",
            1,
            2,
            1,
            2,
            true,
        );
        assert!(verifying.starts_with("\x1b[1;36mVERIFYING\x1b[0m"));
        let finalizing = render_provisioning_progress(
            ProvisioningStage::Finalizing,
            "tokenizer.json",
            2,
            2,
            2,
            2,
            false,
        );
        assert_eq!(finalizing, "FINALIZING tokenizer.json…");
    }
}
