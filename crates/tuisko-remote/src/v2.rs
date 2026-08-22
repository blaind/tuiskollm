//! Minimal RunPod REST API v2 client for remote-runner pods.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::{GpuTarget, RemoteError, RemoteResult};

pub(crate) const DEFAULT_IMAGE: &str = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404";
pub(crate) const POD_NAME_PREFIX: &str = "tuiskollm-gate";
pub(crate) const REMOTE_WORKDIR: &str = "/tmp/tuiskollm";

const API_BASE: &str = "https://api.runpod.io/v2";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Authenticated client for the RunPod endpoints used by the runner.
pub(crate) struct V2 {
    key: String,
    agent: ureq::Agent,
}

impl V2 {
    pub(crate) fn new(key: String) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .http_status_as_error(false)
            .build();

        Self {
            key,
            agent: ureq::Agent::new_with_config(config),
        }
    }

    pub(crate) fn create_gate_pod(
        &self,
        name: &str,
        image: &str,
        gpu: GpuTarget,
    ) -> RemoteResult<Pod> {
        let response = self
            .agent
            .post(format!("{API_BASE}/pods"))
            .header("Authorization", self.authorization().as_str())
            .send_json(serde_json::json!({
                "name": name,
                "image": image,
                "gpu": { "id": gpu.device_name(), "count": 1 },
                "cloud": "SECURE",
                "disk": 50,
                "ports": ["22/tcp"],
                "startSsh": true,
            }));

        parse_json("create pod", response)
    }

    pub(crate) fn list_pods(&self) -> RemoteResult<Vec<Pod>> {
        let response = self
            .agent
            .get(format!("{API_BASE}/pods"))
            .header("Authorization", self.authorization().as_str())
            .call();
        let body = parse_json::<serde_json::Value>("list pods", response)?;

        parse_pod_list(&body)
    }

    pub(crate) fn get_pod(&self, pod_id: &str) -> RemoteResult<Pod> {
        let response = self
            .agent
            .get(format!("{API_BASE}/pods/{pod_id}"))
            .header("Authorization", self.authorization().as_str())
            .call();

        parse_json("get pod", response)
    }

    pub(crate) fn delete_pod(&self, pod_id: &str) -> RemoteResult<()> {
        let response = self
            .agent
            .delete(format!("{API_BASE}/pods/{pod_id}"))
            .header("Authorization", self.authorization().as_str())
            .call()
            .map_err(|source| RemoteError::Network {
                operation: "delete pod",
                source,
            })?;
        let status = response.status().as_u16();
        if (200..300).contains(&status) {
            return Ok(());
        }

        Err(RemoteError::Api {
            operation: "delete pod",
            status,
            body: response.into_body().read_to_string().unwrap_or_default(),
        })
    }

    fn authorization(&self) -> String {
        format!("Bearer {}", self.key)
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct Pod {
    pub(crate) id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) status: Option<String>,
    pub(crate) ssh: Option<SshRoutes>,
    #[serde(flatten)]
    pub(crate) extras: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct SshRoutes {
    pub(crate) direct: Option<SshEndpoint>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct SshEndpoint {
    pub(crate) host: Option<String>,
    pub(crate) port: Option<u16>,
    pub(crate) username: Option<String>,
}

pub(crate) fn wait_until_ssh(v2: &V2, pod_id: &str, deadline: Instant) -> RemoteResult<SshRoutes> {
    let started = Instant::now();
    loop {
        let pod = v2.get_pod(pod_id)?;
        if pod.status.as_deref() == Some("RUNNING")
            && let Some(ssh) = pod.ssh.as_ref().filter(|ssh| ssh.direct.is_some())
        {
            return Ok(ssh.clone());
        }
        if matches!(
            pod.status.as_deref(),
            Some("ERROR" | "TERMINATED" | "EXITED")
        ) {
            return Err(RemoteError::Api {
                operation: "get pod",
                status: 0,
                body: format!("pod left provisioning at status {:?}", pod.status),
            });
        }
        if Instant::now() >= deadline {
            return Err(RemoteError::SshRouteUnavailable {
                seconds: started.elapsed().as_secs(),
                status: pod.status.unwrap_or_else(|| "missing".to_owned()),
            });
        }

        println!(
            "pod status: {:?}, direct ssh: {}",
            pod.status,
            pod.ssh.as_ref().is_some_and(|ssh| ssh.direct.is_some())
        );
        std::thread::sleep(POLL_INTERVAL);
    }
}

pub(crate) fn is_missing(error: &RemoteError) -> bool {
    matches!(error, RemoteError::Api { status: 404, .. })
}

fn parse_json<T: serde::de::DeserializeOwned>(
    operation: &'static str,
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> RemoteResult<T> {
    let response = response.map_err(|source| RemoteError::Network { operation, source })?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(RemoteError::Api {
            operation,
            status,
            body: response.into_body().read_to_string().unwrap_or_default(),
        });
    }
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|source| RemoteError::Network { operation, source })?;

    serde_json::from_str(&body).map_err(|source| RemoteError::Json { source })
}

fn parse_pod_list(body: &serde_json::Value) -> RemoteResult<Vec<Pod>> {
    let items = match body {
        serde_json::Value::Array(items) => items.clone(),
        serde_json::Value::Object(map) => map
            .get("pods")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default(),
        _ => Vec::new(),
    };

    items
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()
        .map_err(|source| RemoteError::Json { source })
}

#[cfg(test)]
mod tests {
    use super::{SshRoutes, parse_pod_list};

    #[test]
    fn direct_route_parses() {
        let routes: SshRoutes = serde_json::from_str(
            r#"{"proxy":{"host":"ssh.runpod.io","port":22,"username":"pod-route"},"direct":{"host":"1.2.3.4","port":32122,"username":"root"}}"#,
        )
        .expect("valid routes");
        let direct = routes.direct.expect("direct route");
        assert_eq!(direct.host.as_deref(), Some("1.2.3.4"));
        assert_eq!(direct.port, Some(32122));
        assert_eq!(direct.username.as_deref(), Some("root"));
    }

    #[test]
    fn pod_list_accepts_bare_and_wrapped_arrays() {
        let bare = serde_json::json!([{"id": "a"}]);
        let wrapped = serde_json::json!({"pods": [{"id": "b"}]});

        assert_eq!(parse_pod_list(&bare).expect("bare list").len(), 1);
        assert_eq!(parse_pod_list(&wrapped).expect("wrapped list").len(), 1);
    }
}
