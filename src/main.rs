//! Rust-owned inference server.

use clap::{Args, Parser, Subcommand};
use std::io::{IsTerminal, Write};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};
use tuisko_model::{Arch, Qwen38FlashNext};
use tuisko_provision::{
    ProvisionedModel, Provisioning, ProvisioningProgress, ProvisioningStage, SnapshotResolution,
};
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
        long_help = "Exact Hugging Face model ID.\n\nSupported models:\n  unsloth/Qwen3.8-27B-NVFP4\n  AxionML/Qwen3.5-9B-NVFP4\n  nvidia/Qwen3.6-35B-A3B-NVFP4\n  RadixArk/Qwen3.8-Flash-Next-NVFP4"
    )]
    model: ServerModel,
    /// Existing admitted snapshot; overrides automatic Hugging Face resolution.
    #[arg(long, value_name = "SNAPSHOT")]
    snapshot: Option<PathBuf>,
    /// TCP listen address.
    #[arg(
        long,
        value_name = "ADDRESS",
        default_value = DEFAULT_ADDRESS,
        value_parser = parse_address
    )]
    address: SocketAddr,
    /// Environment variable containing the lifecycle-route bearer token.
    #[arg(long, value_name = "ENV")]
    admin_token_env: Option<String>,
    /// Prints live inference progress as newline-delimited snapshots instead of terminal redraws.
    #[arg(long)]
    progress_lines: bool,
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
    if !args.address.ip().is_loopback() && args.admin_token_env.is_none() {
        return Err("a non-loopback listener requires --admin-token-env".into());
    }
    if let Some(variable) = args.admin_token_env.as_deref() {
        std::env::var(variable).map_err(|_| {
            format!("admin token environment variable `{variable}` is unset or not Unicode")
        })?;
    }
    let stdout = std::io::stdout();
    let interactive = stdout.is_terminal();
    let color = interactive && std::env::var_os("NO_COLOR").is_none();
    let resolution = {
        let mut stdout = stdout.lock();
        let mut display = ProvisioningDisplay::new(interactive, color);
        let resolution = match args.model {
            ServerModel::Qwen38FlashNext => resolve_qwen38_flash_next_snapshot(args.snapshot),
            ServerModel::Qwen38 | ServerModel::Qwen35 | ServerModel::Qwen36 => {
                let model = match args.model {
                    ServerModel::Qwen38 => ProvisionedModel::Qwen38,
                    ServerModel::Qwen35 => ProvisionedModel::Qwen35,
                    ServerModel::Qwen36 => ProvisionedModel::Qwen36,
                    ServerModel::Qwen38FlashNext => unreachable!(),
                };
                tuisko_provision::resolve_snapshot_with_progress(model, args.snapshot, |progress| {
                    display
                        .update(&mut stdout, progress)
                        .map_err(|error| format!("writing provisioning progress: {error}"))
                })
            }
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
        admin_token_env: args.admin_token_env,
        progress_lines: args.progress_lines,
    })
    .map_err(|error| error.to_string())
}

fn resolve_qwen38_flash_next_snapshot(
    snapshot: Option<PathBuf>,
) -> Result<SnapshotResolution, String> {
    let path = snapshot.ok_or_else(|| {
        format!(
            "{} requires --snapshot at pinned revision {}",
            Qwen38FlashNext::MODEL_ID,
            Qwen38FlashNext::REVISION
        )
    })?;
    if !path.is_dir() {
        return Err(format!("snapshot `{}` is not a directory", path.display()));
    }
    let revision = path.file_name().and_then(|name| name.to_str());
    if revision != Some(Qwen38FlashNext::REVISION) {
        return Err(format!(
            "snapshot revision {revision:?} is not {}",
            Qwen38FlashNext::REVISION
        ));
    }

    Ok(SnapshotResolution {
        path,
        provisioning: None,
    })
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
    Ok(Cli::try_parse_from(arguments)?.command)
}

fn parse_address(address: &str) -> Result<SocketAddr, String> {
    if let Ok(address) = address.parse() {
        return Ok(address);
    }

    address
        .to_socket_addrs()
        .map_err(|error| format!("invalid ADDRESS `{address}`: {error}"))?
        .next()
        .ok_or_else(|| format!("ADDRESS `{address}` resolved to no socket addresses"))
}

#[cfg(test)]
mod tests {
    use super::{
        Command, DEFAULT_ADDRESS, ServeArgs, parse_args, render_provisioning,
        render_provisioning_progress, resolve_qwen38_flash_next_snapshot,
    };
    use clap::error::ErrorKind;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::Duration;
    use tuisko_provision::{Provisioning, ProvisioningStage};
    use tuisko_serve::ServerModel;

    const QWEN38: &str = "unsloth/Qwen3.8-27B-NVFP4";
    const QWEN35: &str = "AxionML/Qwen3.5-9B-NVFP4";
    const QWEN36: &str = "nvidia/Qwen3.6-35B-A3B-NVFP4";
    const QWEN38_FLASH_NEXT: &str = "RadixArk/Qwen3.8-Flash-Next-NVFP4";

    fn parse(args: &[&str]) -> Result<Command, clap::Error> {
        parse_args(args.iter().copied())
    }

    #[test]
    fn serve_requires_an_exact_model_and_defaults_to_loopback() {
        for (model_id, model) in [
            (QWEN38, ServerModel::Qwen38),
            (QWEN35, ServerModel::Qwen35),
            (QWEN36, ServerModel::Qwen36),
            (QWEN38_FLASH_NEXT, ServerModel::Qwen38FlashNext),
        ] {
            let command = parse(&["serve", model_id]).unwrap();
            assert_eq!(
                command,
                Command::Serve(ServeArgs {
                    model,
                    snapshot: None,
                    address: DEFAULT_ADDRESS.parse::<SocketAddr>().unwrap(),
                    admin_token_env: None,
                    progress_lines: false,
                })
            );
        }
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
                "--admin-token-env",
                "TUISKO_ADMIN_TOKEN",
            ],
            vec![
                "serve",
                QWEN35,
                "--address",
                "0.0.0.0:9123",
                "--snapshot",
                "/models/pinned",
                "--admin-token-env",
                "TUISKO_ADMIN_TOKEN",
            ],
        ] {
            assert_eq!(
                parse(&arguments).unwrap(),
                Command::Serve(ServeArgs {
                    model: ServerModel::Qwen35,
                    snapshot: Some(PathBuf::from("/models/pinned")),
                    address: "0.0.0.0:9123".parse().unwrap(),
                    admin_token_env: Some("TUISKO_ADMIN_TOKEN".into()),
                    progress_lines: false,
                })
            );
        }
    }

    #[test]
    fn serve_accepts_newline_delimited_progress() {
        let Command::Serve(config) = parse(&["serve", QWEN38, "--progress-lines"]).unwrap();
        assert!(config.progress_lines);
    }

    #[test]
    fn qwen38_flash_next_requires_an_explicit_pinned_snapshot() {
        let error = resolve_qwen38_flash_next_snapshot(None).err().unwrap();
        assert!(error.contains(QWEN38_FLASH_NEXT));
        assert!(error.contains("--snapshot"));
    }

    #[test]
    fn serve_resolves_a_hostname_socket_address() {
        let Command::Serve(config) =
            parse(&["serve", QWEN38, "--address", "localhost:8000"]).unwrap();
        assert!(config.address.ip().is_loopback());
        assert_eq!(config.address.port(), 8000);
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
            parse(&["serve", QWEN38, "--address", "localhost"])
                .unwrap_err()
                .to_string()
                .contains("invalid ADDRESS")
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
        for model in [QWEN38, QWEN35, QWEN36, QWEN38_FLASH_NEXT] {
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
