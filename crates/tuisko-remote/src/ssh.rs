//! Direct Rust SSH and SFTP transport for an ephemeral pod.

use std::future::Future;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use flate2::Compression;
use flate2::write::GzEncoder;
use russh::client;
use russh::keys::{HashAlg, PrivateKeyWithHashAlg, PublicKeyOrCertificate};
use russh::{ChannelMsg, Disconnect};
use russh_sftp::client::SftpSession;
use tokio::io::AsyncWriteExt;
use tokio::runtime::{Builder, Runtime};

use crate::{RemoteError, RemoteResult};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Default)]
struct Client {
    host_key: Option<String>,
}

impl Client {
    fn accept_host_key(&mut self, fingerprint: String) -> bool {
        match &self.host_key {
            Some(expected) => expected == &fingerprint,
            None => {
                self.host_key = Some(fingerprint);
                true
            }
        }
    }
}

impl client::Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &PublicKeyOrCertificate,
    ) -> Result<bool, Self::Error> {
        let fingerprint = server_public_key
            .public_key()
            .fingerprint(HashAlg::Sha256)
            .to_string();
        Ok(self.accept_host_key(fingerprint))
    }
}

/// One authenticated SSH session to an API-provided direct pod endpoint.
pub struct Ssh {
    runtime: Runtime,
    session: client::Handle<Client>,
}

impl Ssh {
    /// Validates that the private key can be decoded before renting a pod.
    pub fn validate_key(key_file: &Path) -> RemoteResult<()> {
        load_key(key_file).map(|_| ())
    }

    /// Connects to one direct pod endpoint using an OpenSSH private key.
    pub fn connect(key_file: &Path, host: &str, port: u16, user: &str) -> RemoteResult<Self> {
        let key = load_key(key_file)?;
        let runtime = Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .map_err(|source| RemoteError::SshIo {
                operation: "creating the async runtime",
                source,
            })?;
        let config = Arc::new(client::Config {
            keepalive_interval: Some(Duration::from_secs(10)),
            keepalive_max: 3,
            nodelay: true,
            ..client::Config::default()
        });
        let connection = runtime.block_on(async {
            tokio::time::timeout(
                CONNECT_TIMEOUT,
                client::connect(config, (host, port), Client::default()),
            )
            .await
        });
        let mut session = match connection {
            Ok(Ok(session)) => session,
            Ok(Err(source)) => {
                return Err(RemoteError::Ssh {
                    operation: "connect",
                    source,
                });
            }
            Err(_) => return Err(timeout_error("connect", CONNECT_TIMEOUT.as_secs())),
        };
        let authentication = runtime.block_on(async {
            tokio::time::timeout(
                CONNECT_TIMEOUT,
                session
                    .authenticate_publickey(user, PrivateKeyWithHashAlg::new(Arc::new(key), None)),
            )
            .await
        });
        match authentication {
            Ok(Ok(result)) if result.success() => {}
            Ok(Ok(_)) => {
                return Err(RemoteError::Precheck {
                    detail: "SSH public-key authentication was rejected".to_owned(),
                });
            }
            Ok(Err(source)) => {
                return Err(RemoteError::Ssh {
                    operation: "authenticate",
                    source,
                });
            }
            Err(_) => return Err(timeout_error("authenticate", CONNECT_TIMEOUT.as_secs())),
        }

        Ok(Self { runtime, session })
    }

    /// Executes one command through the pod's SSH server.
    pub fn run(&self, remote_command: &str, timeout_secs: u64) -> RemoteResult<(u32, String)> {
        self.block_with_timeout(
            timeout_secs,
            "command",
            run_command(&self.session, remote_command),
        )
    }

    /// Uploads a gzip-compressed executable through SFTP and sets mode 0755.
    pub fn put_file(&self, local: &Path, remote: &str) -> RemoteResult<()> {
        let bytes = std::fs::read(local).map_err(|source| RemoteError::Read {
            what: format!("local file {}", local.display()),
            source,
        })?;
        let compressed = gzip(&bytes)?;
        let remote_compressed = format!("{remote}.gz");
        let quoted_remote = shell_quote(remote)?;
        let quoted_remote_compressed = shell_quote(&remote_compressed)?;
        self.block_with_timeout(240, "SFTP upload", async {
            let sftp = open_sftp(&self.session, "open upload channel").await?;
            let mut file = sftp
                .create(remote_compressed.clone())
                .await
                .map_err(|source| RemoteError::Sftp {
                    operation: "create upload",
                    source,
                })?;
            file.write_all(&compressed)
                .await
                .map_err(|source| RemoteError::SshIo {
                    operation: "write upload",
                    source,
                })?;
            file.close().await.map_err(|source| RemoteError::SshIo {
                operation: "close upload file",
                source,
            })?;
            sftp.close().await.map_err(|source| RemoteError::Sftp {
                operation: "close upload",
                source,
            })
        })?;

        let command = format!(
            "gzip -dc {quoted_remote_compressed} > {quoted_remote} && \
             chmod 0755 {quoted_remote} && test $(wc -c < {quoted_remote}) -eq {} && \
             rm -f {quoted_remote_compressed}",
            bytes.len()
        );
        let (status, output) = self.run(&command, 60)?;
        if status != 0 {
            return Err(RemoteError::SshIo {
                operation: "expand SFTP artifact",
                source: std::io::Error::other(output),
            });
        }
        println!(
            "uploaded {} bytes as {} compressed bytes via SFTP",
            bytes.len(),
            compressed.len()
        );

        Ok(())
    }

    /// Downloads one report through SFTP.
    pub fn get_file(&self, remote: &str, local: &Path) -> RemoteResult<()> {
        let bytes = self.block_with_timeout(120, "SFTP download", async {
            let sftp = open_sftp(&self.session, "open download channel").await?;
            let bytes = sftp
                .read(remote)
                .await
                .map_err(|source| RemoteError::Sftp {
                    operation: "read download",
                    source,
                })?;
            sftp.close().await.map_err(|source| RemoteError::Sftp {
                operation: "close download",
                source,
            })?;
            Ok(bytes)
        })?;
        std::fs::write(local, bytes).map_err(|source| RemoteError::Write {
            path: local.to_path_buf(),
            source,
        })
    }

    fn block_with_timeout<T>(
        &self,
        timeout_secs: u64,
        operation: &'static str,
        future: impl Future<Output = RemoteResult<T>>,
    ) -> RemoteResult<T> {
        self.runtime.block_on(async {
            tokio::time::timeout(Duration::from_secs(timeout_secs), future)
                .await
                .map_err(|_| timeout_error(operation, timeout_secs))?
        })
    }
}

fn load_key(key_file: &Path) -> RemoteResult<russh::keys::PrivateKey> {
    russh::keys::load_secret_key(key_file, None).map_err(|source| RemoteError::SshKey { source })
}

impl Drop for Ssh {
    fn drop(&mut self) {
        self.runtime.block_on(async {
            tokio::time::timeout(
                Duration::from_secs(1),
                self.session.disconnect(Disconnect::ByApplication, "", ""),
            )
            .await
            .ok();
        });
    }
}

async fn run_command(
    session: &client::Handle<Client>,
    remote_command: &str,
) -> RemoteResult<(u32, String)> {
    let mut channel = session
        .channel_open_session()
        .await
        .map_err(|source| RemoteError::Ssh {
            operation: "open command channel",
            source,
        })?;
    channel
        .exec(true, remote_command)
        .await
        .map_err(|source| RemoteError::Ssh {
            operation: "execute command",
            source,
        })?;

    let mut status = None;
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    while let Some(message) = channel.wait().await {
        match message {
            ChannelMsg::Data { data } => stdout.extend_from_slice(&data),
            ChannelMsg::ExtendedData { data, ext: 1 } => stderr.extend_from_slice(&data),
            ChannelMsg::ExitStatus { exit_status } => status = Some(exit_status),
            _ => {}
        }
    }
    let mut output = String::from_utf8_lossy(&stdout).into_owned();
    if !stderr.is_empty() {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&String::from_utf8_lossy(&stderr));
    }

    Ok((status.unwrap_or(1), output))
}

async fn open_sftp(
    session: &client::Handle<Client>,
    operation: &'static str,
) -> RemoteResult<SftpSession> {
    let channel = session
        .channel_open_session()
        .await
        .map_err(|source| RemoteError::Ssh { operation, source })?;
    channel
        .request_subsystem(true, "sftp")
        .await
        .map_err(|source| RemoteError::Ssh { operation, source })?;
    SftpSession::new(channel.into_stream())
        .await
        .map_err(|source| RemoteError::Sftp { operation, source })
}

fn gzip(bytes: &[u8]) -> RemoteResult<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(bytes)
        .and_then(|()| encoder.finish())
        .map_err(|source| RemoteError::SshIo {
            operation: "compress upload artifact",
            source,
        })
}

fn shell_quote(value: &str) -> RemoteResult<String> {
    if value.contains(['\n', '\r', '\0']) {
        return Err(RemoteError::Precheck {
            detail: "remote path contains a control character".to_owned(),
        });
    }
    Ok(format!("'{}'", value.replace('\'', "'\"'\"'")))
}

fn timeout_error(operation: &'static str, timeout_secs: u64) -> RemoteError {
    RemoteError::SshIo {
        operation,
        source: std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!("operation exceeded {timeout_secs}s"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::GzDecoder;

    use super::{Client, gzip, shell_quote};

    #[test]
    fn host_key_is_pinned_for_the_session() {
        let mut client = Client::default();
        assert!(client.accept_host_key("SHA256:first".to_owned()));
        assert!(client.accept_host_key("SHA256:first".to_owned()));
        assert!(!client.accept_host_key("SHA256:changed".to_owned()));
    }

    #[test]
    fn remote_paths_are_quoted_or_rejected() {
        assert_eq!(shell_quote("it's").expect("safe path"), "'it'\"'\"'s'");
        assert!(shell_quote("bad\npath").is_err());
    }

    #[test]
    fn compressed_upload_is_lossless() {
        let input = b"tuiskollm remote artifact".repeat(128);
        let compressed = gzip(&input).expect("compression succeeds");
        let mut decoded = Vec::new();
        GzDecoder::new(compressed.as_slice())
            .read_to_end(&mut decoded)
            .expect("valid gzip stream");

        assert_eq!(decoded, input);
    }
}
