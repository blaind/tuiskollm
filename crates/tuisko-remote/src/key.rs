//! API key resolution. The key never appears in error messages.

use std::path::PathBuf;

use crate::{RemoteError, RemoteResult};

/// Environment variable carrying the RunPod API key.
const ENV_VAR: &str = "RUNPOD_API_KEY";

/// Directory levels to walk upward when hunting for a `.env` file.
const DOTENV_MAX_DEPTH: usize = 10;

/// Credentials file path under the user's home directory.
fn credentials_path() -> RemoteResult<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| RemoteError::Read {
        what: "HOME environment variable".to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "HOME is not set"),
    })?;
    Ok(PathBuf::from(home).join(".runpod").join("credentials.json"))
}

/// Resolves the API key in precedence order:
/// `RUNPOD_API_KEY` env var, `~/.runpod/credentials.json`, then a `.env`
/// file found by walking up from the current directory. The credentials file
/// is the only step `resolve_env` does not share.
pub fn resolve_api_key() -> RemoteResult<String> {
    process_env(ENV_VAR)
        .or_else(credentials_file_key)
        .or_else(|| dotenv_value(ENV_VAR))
        .ok_or(RemoteError::MissingKey)
}

/// Resolves a configuration value: the process environment first, then
/// the nearest `.env` file (so `.env` works for every `RUNPOD_*` knob,
/// not only the API key).
pub fn resolve_env(name: &str) -> Option<String> {
    process_env(name).or_else(|| dotenv_value(name))
}

/// One value from the process environment, trimmed, if it is set non-empty.
fn process_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// One value from the nearest `.env` above the current directory.
fn dotenv_value(name: &str) -> Option<String> {
    let raw = std::fs::read_to_string(find_dotenv()?).ok()?;
    dotenv_lookup(&raw, name)
}

/// The first non-empty binding of `name` in one `.env` file's contents.
fn dotenv_lookup(raw: &str, name: &str) -> Option<String> {
    raw.lines()
        .filter_map(parse_dotenv_line)
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
        .filter(|value| !value.is_empty())
}

/// The key from `~/.runpod/credentials.json`, if the file has one.
fn credentials_file_key() -> Option<String> {
    let path = credentials_path().ok()?;
    let raw = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("api_key")
        .and_then(serde_json::Value::as_str)
        .map(|key| key.trim().to_owned())
        .filter(|key| !key.is_empty())
}

/// Finds the nearest `.env` walking upward from the current directory.
fn find_dotenv() -> Option<std::path::PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..DOTENV_MAX_DEPTH {
        let candidate = dir.join(".env");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
    None
}

/// Parses one `KEY=VALUE` line; quotes around the value are stripped.
fn parse_dotenv_line(raw: &str) -> Option<(String, String)> {
    let line = raw.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (key, value) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let mut value = value.trim();
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        value = &value[1..value.len() - 1];
    }
    Some((key.to_owned(), value.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::{dotenv_lookup, parse_dotenv_line};

    /// The API key and every other `RUNPOD_*` knob now share one `.env`
    /// lookup; it must still select the first non-empty binding of the
    /// requested name and nothing else.
    #[test]
    fn dotenv_lookup_selects_one_named_binding() {
        const FILE: &str = "# comment\nRUNPOD_API_KEY=\"abc123\"\nRUNPOD_SSH_KEY_FILE=/tmp/id\nRUNPOD_API_KEY=later\nRUNPOD_EMPTY=\n";

        assert_eq!(
            dotenv_lookup(FILE, "RUNPOD_API_KEY").as_deref(),
            Some("abc123")
        );
        assert_eq!(
            dotenv_lookup(FILE, "RUNPOD_SSH_KEY_FILE").as_deref(),
            Some("/tmp/id")
        );
        assert_eq!(dotenv_lookup(FILE, "RUNPOD_EMPTY"), None);
        assert_eq!(dotenv_lookup(FILE, "RUNPOD_ABSENT"), None);
    }

    #[test]
    fn plain_value_is_parsed() {
        let (key, value) = parse_dotenv_line("RUNPOD_API_KEY=abc123").expect("valid line");
        assert_eq!(key, "RUNPOD_API_KEY");
        assert_eq!(value, "abc123");
    }

    #[test]
    fn surrounding_quotes_are_stripped() {
        let (_, value) = parse_dotenv_line("RUNPOD_API_KEY=\"abc123\"").expect("valid line");
        assert_eq!(value, "abc123");
        let (_, value) = parse_dotenv_line("RUNPOD_API_KEY='abc123'").expect("valid line");
        assert_eq!(value, "abc123");
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        assert_eq!(parse_dotenv_line("# a comment"), None);
        assert_eq!(parse_dotenv_line("   "), None);
        assert_eq!(parse_dotenv_line(""), None);
    }

    #[test]
    fn malformed_lines_are_rejected() {
        assert_eq!(parse_dotenv_line("NO_EQUALS_SIGN"), None);
        assert_eq!(parse_dotenv_line("BAD-KEY=value"), None);
        assert_eq!(parse_dotenv_line("=value"), None);
    }
}
