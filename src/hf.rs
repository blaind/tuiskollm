use std::ffi::OsString;
use std::path::{Path, PathBuf};
use tuisko_model::{Arch, Qwen38_27B};

pub fn resolve_snapshot(explicit: Option<PathBuf>) -> Result<PathBuf, String> {
    resolve_snapshot_with(
        explicit,
        |name| std::env::var_os(name),
        |path| path.is_dir(),
    )
}

fn resolve_snapshot_with(
    explicit: Option<PathBuf>,
    environment: impl Fn(&str) -> Option<OsString>,
    is_directory: impl Fn(&Path) -> bool,
) -> Result<PathBuf, String> {
    if let Some(snapshot) = explicit {
        return Ok(snapshot);
    }
    if let Some(snapshot) = nonempty_environment(&environment, "TUISKO_SNAPSHOT") {
        return Ok(snapshot.into());
    }

    let cache = hub_cache(&environment)?;
    let repository = format!("models--{}", Qwen38_27B::MODEL_ID.replace('/', "--"));
    let snapshot = cache
        .join(repository)
        .join("snapshots")
        .join(Qwen38_27B::REVISION);
    if !is_directory(&snapshot) {
        return Err(format!(
            "the pinned snapshot is not cached at {}; pass SNAPSHOT or run `hf download {} --revision {}`",
            snapshot.display(),
            Qwen38_27B::MODEL_ID,
            Qwen38_27B::REVISION,
        ));
    }
    Ok(snapshot)
}

fn hub_cache(environment: &impl Fn(&str) -> Option<OsString>) -> Result<PathBuf, String> {
    if let Some(cache) = nonempty_environment(environment, "HF_HUB_CACHE") {
        return Ok(cache.into());
    }
    if let Some(home) = nonempty_environment(environment, "HF_HOME") {
        return Ok(PathBuf::from(home).join("hub"));
    }
    if let Some(cache) = nonempty_environment(environment, "XDG_CACHE_HOME") {
        return Ok(PathBuf::from(cache).join("huggingface/hub"));
    }
    if let Some(home) = nonempty_environment(environment, "HOME") {
        return Ok(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    Err("cannot locate the Hugging Face cache; set HF_HUB_CACHE or pass SNAPSHOT".into())
}

fn nonempty_environment(
    environment: &impl Fn(&str) -> Option<OsString>,
    name: &str,
) -> Option<OsString> {
    environment(name).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::resolve_snapshot_with;
    use std::collections::BTreeMap;
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    fn environment(values: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let values = values
            .iter()
            .map(|&(name, value)| (name.to_owned(), OsString::from(value)))
            .collect::<BTreeMap<_, _>>();
        move |name| values.get(name).cloned()
    }

    #[test]
    fn explicit_and_tuisko_paths_precede_the_hub_cache() {
        let explicit = resolve_snapshot_with(
            Some("/models/explicit".into()),
            environment(&[("TUISKO_SNAPSHOT", "/models/environment")]),
            |_| false,
        )
        .unwrap();
        assert_eq!(explicit, Path::new("/models/explicit"));

        let configured = resolve_snapshot_with(
            None,
            environment(&[("TUISKO_SNAPSHOT", "/models/environment")]),
            |_| false,
        )
        .unwrap();
        assert_eq!(configured, Path::new("/models/environment"));
    }

    #[test]
    fn hub_environment_precedence_and_pinned_identity_are_exact() {
        for (values, root) in [
            (
                vec![
                    ("HF_HUB_CACHE", "/cache/direct"),
                    ("HF_HOME", "/cache/home"),
                    ("XDG_CACHE_HOME", "/cache/xdg"),
                    ("HOME", "/home/user"),
                ],
                "/cache/direct",
            ),
            (vec![("HF_HOME", "/cache/home")], "/cache/home/hub"),
            (
                vec![("XDG_CACHE_HOME", "/cache/xdg")],
                "/cache/xdg/huggingface/hub",
            ),
            (
                vec![("HOME", "/home/user")],
                "/home/user/.cache/huggingface/hub",
            ),
        ] {
            let expected = PathBuf::from(root)
                .join("models--unsloth--Qwen3.8-27B-NVFP4/snapshots")
                .join("16b6615af3548b88e2d8e382457bc705b00479cf");
            let actual =
                resolve_snapshot_with(None, environment(&values), |path| path == expected).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn missing_snapshot_names_the_exact_recovery_command() {
        let error =
            resolve_snapshot_with(None, environment(&[("HF_HUB_CACHE", "/cache")]), |_| false)
                .unwrap_err();
        assert!(error.contains("models--unsloth--Qwen3.8-27B-NVFP4/snapshots"));
        assert!(error.contains("hf download unsloth/Qwen3.8-27B-NVFP4"));
        assert!(error.contains("16b6615af3548b88e2d8e382457bc705b00479cf"));
    }
}
