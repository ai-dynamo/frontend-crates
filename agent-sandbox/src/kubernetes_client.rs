// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete Kubernetes Agent Sandbox control-plane and router clients.

use std::collections::BTreeMap;
use std::time::Duration;

use base64::Engine;
use chrono::Utc;
use dynamo_agent_rt::BoxFuture;
use kube::api::{DeleteParams, PostParams, PropagationPolicy};
use kube::core::{ApiResource, DynamicObject, GroupVersionKind};
use kube::{Api, Client};
use reqwest::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use crate::{
    AgentSandboxControlPlane, Artifact, ArtifactRef, ExecutionRecord, SandboxClaimHandle,
    SandboxClaimRequest, SandboxSupervisor, ScopedExecutionId, StartExecution,
};

const WORKSPACE_FINGERPRINT_ANNOTATION: &str = "agent-rt.nvidia.com/workspace-fingerprint";
const MANAGED_BY_LABEL: &str = "app.kubernetes.io/managed-by";

#[derive(Debug, Clone)]
pub struct KubeAgentSandboxControlPlaneConfig {
    pub ready_timeout: Duration,
    pub poll_interval: Duration,
}

impl Default for KubeAgentSandboxControlPlaneConfig {
    fn default() -> Self {
        Self {
            ready_timeout: Duration::from_secs(60),
            poll_interval: Duration::from_millis(200),
        }
    }
}

#[derive(Debug, Error)]
pub enum KubeControlPlaneError {
    #[error("Kubernetes API failed: {0}")]
    Kube(#[from] kube::Error),
    #[error("sandbox claim exists with a different workspace or warm pool")]
    ClaimConflict,
    #[error("sandbox claim did not become ready within {0:?}")]
    ReadyTimeout(Duration),
    #[error("ready sandbox claim does not contain status.sandbox.name")]
    MissingSandboxIdentity,
    #[error("sandbox workspace TTL is outside the supported timestamp range")]
    InvalidTtl,
}

#[derive(Clone)]
pub struct KubeAgentSandboxControlPlane {
    client: Client,
    config: KubeAgentSandboxControlPlaneConfig,
    resource: ApiResource,
}

impl KubeAgentSandboxControlPlane {
    pub fn new(client: Client, config: KubeAgentSandboxControlPlaneConfig) -> Self {
        let gvk = GroupVersionKind::gvk("extensions.agents.x-k8s.io", "v1beta1", "SandboxClaim");
        Self {
            client,
            config,
            resource: ApiResource::from_gvk_with_plural(&gvk, "sandboxclaims"),
        }
    }

    fn api(&self, namespace: &str) -> Api<DynamicObject> {
        Api::namespaced_with(self.client.clone(), namespace, &self.resource)
    }

    fn claim(&self, request: &SandboxClaimRequest) -> Result<DynamicObject, KubeControlPlaneError> {
        let ttl = chrono::Duration::from_std(request.expires_after)
            .map_err(|_| KubeControlPlaneError::InvalidTtl)?;
        let shutdown_time = (Utc::now() + ttl).to_rfc3339();
        let mut claim = DynamicObject::new(&request.claim_name, &self.resource).data(json!({
            "spec": {
                "warmPoolRef": {"name": request.warm_pool},
                "lifecycle": {
                    "shutdownTime": shutdown_time,
                    "shutdownPolicy": "DeleteForeground",
                    "ttlSecondsAfterFinished": 0
                },
                "additionalPodMetadata": {
                    "labels": {
                        MANAGED_BY_LABEL: "agent-rt-sandbox",
                        "agent-rt.nvidia.com/workspace": &request.workspace_fingerprint[..40]
                    }
                }
            }
        }));
        claim.metadata.annotations = Some(BTreeMap::from([(
            WORKSPACE_FINGERPRINT_ANNOTATION.to_owned(),
            request.workspace_fingerprint.clone(),
        )]));
        Ok(claim)
    }

    fn validate_claim(
        &self,
        claim: &DynamicObject,
        request: &SandboxClaimRequest,
    ) -> Result<(), KubeControlPlaneError> {
        let fingerprint = claim
            .metadata
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.get(WORKSPACE_FINGERPRINT_ANNOTATION));
        let warm_pool = claim
            .data
            .pointer("/spec/warmPoolRef/name")
            .and_then(|v| v.as_str());
        if fingerprint == Some(&request.workspace_fingerprint)
            && warm_pool == Some(request.warm_pool.as_str())
        {
            Ok(())
        } else {
            Err(KubeControlPlaneError::ClaimConflict)
        }
    }

    async fn ready_claim(
        &self,
        request: &SandboxClaimRequest,
    ) -> Result<SandboxClaimHandle, KubeControlPlaneError> {
        let api = self.api(&request.namespace);
        let deadline = tokio::time::Instant::now() + self.config.ready_timeout;
        loop {
            let claim = api.get(&request.claim_name).await?;
            self.validate_claim(&claim, request)?;
            let ready = claim
                .data
                .pointer("/status/conditions")
                .and_then(|conditions| conditions.as_array())
                .is_some_and(|conditions| {
                    conditions.iter().any(|condition| {
                        condition.get("type").and_then(|v| v.as_str()) == Some("Ready")
                            && condition.get("status").and_then(|v| v.as_str()) == Some("True")
                    })
                });
            if ready {
                let sandbox_id = claim
                    .data
                    .pointer("/status/sandbox/name")
                    .and_then(|value| value.as_str())
                    .filter(|name| !name.is_empty())
                    .ok_or(KubeControlPlaneError::MissingSandboxIdentity)?;
                return Ok(SandboxClaimHandle {
                    namespace: request.namespace.clone(),
                    claim_name: request.claim_name.clone(),
                    sandbox_id: sandbox_id.to_owned(),
                });
            }
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return Err(KubeControlPlaneError::ReadyTimeout(
                    self.config.ready_timeout,
                ));
            }
            tokio::time::sleep(
                self.config
                    .poll_interval
                    .min(deadline.saturating_duration_since(now)),
            )
            .await;
        }
    }
}

impl AgentSandboxControlPlane for KubeAgentSandboxControlPlane {
    type Error = KubeControlPlaneError;

    fn create_or_get_claim(
        &self,
        request: SandboxClaimRequest,
    ) -> BoxFuture<'_, Result<SandboxClaimHandle, Self::Error>> {
        Box::pin(async move {
            let api = self.api(&request.namespace);
            match api.get_opt(&request.claim_name).await? {
                Some(existing) => self.validate_claim(&existing, &request)?,
                None => {
                    let claim = self.claim(&request)?;
                    match api.create(&PostParams::default(), &claim).await {
                        Ok(created) => self.validate_claim(&created, &request)?,
                        Err(kube::Error::Api(status)) if status.code == 409 => {
                            let raced = api.get(&request.claim_name).await?;
                            self.validate_claim(&raced, &request)?;
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            self.ready_claim(&request).await
        })
    }

    fn get_claim(
        &self,
        request: &SandboxClaimRequest,
    ) -> BoxFuture<'_, Result<Option<SandboxClaimHandle>, Self::Error>> {
        let request = request.clone();
        Box::pin(async move {
            let api = self.api(&request.namespace);
            let Some(claim) = api.get_opt(&request.claim_name).await? else {
                return Ok(None);
            };
            self.validate_claim(&claim, &request)?;
            self.ready_claim(&request).await.map(Some)
        })
    }

    fn delete_claim(
        &self,
        request: &SandboxClaimRequest,
    ) -> BoxFuture<'_, Result<(), Self::Error>> {
        let request = request.clone();
        Box::pin(async move {
            let api = self.api(&request.namespace);
            let Some(claim) = api.get_opt(&request.claim_name).await? else {
                return Ok(());
            };
            self.validate_claim(&claim, &request)?;
            api.delete(
                &request.claim_name,
                &DeleteParams {
                    propagation_policy: Some(PropagationPolicy::Foreground),
                    ..DeleteParams::default()
                },
            )
            .await?;
            Ok(())
        })
    }
}

#[derive(Clone)]
pub struct HttpSandboxSupervisorConfig {
    pub router_base_url: String,
    pub router_bearer_token: String,
    pub sandbox_port: u16,
    pub request_timeout: Duration,
}

#[derive(Debug, Error)]
pub enum SandboxSupervisorHttpError {
    #[error("sandbox router URL must use http or https and contain no query or fragment")]
    InvalidRouterUrl,
    #[error("sandbox execution or artifact ID is invalid")]
    InvalidId,
    #[error("sandbox router request failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("sandbox router returned HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("sandbox artifact payload is invalid base64: {0}")]
    ArtifactBase64(#[from] base64::DecodeError),
}

#[derive(Clone)]
pub struct HttpSandboxSupervisor {
    client: reqwest::Client,
    config: HttpSandboxSupervisorConfig,
    base_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ArtifactPayload {
    metadata: ArtifactRef,
    data_base64: String,
}

impl HttpSandboxSupervisor {
    pub fn new(config: HttpSandboxSupervisorConfig) -> Result<Self, SandboxSupervisorHttpError> {
        let parsed = reqwest::Url::parse(&config.router_base_url)
            .map_err(|_| SandboxSupervisorHttpError::InvalidRouterUrl)?;
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(SandboxSupervisorHttpError::InvalidRouterUrl);
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(config.request_timeout)
            .build()?;
        let base_url = config.router_base_url.trim_end_matches('/').to_owned();
        Ok(Self {
            client,
            config,
            base_url,
        })
    }

    fn request(
        &self,
        method: Method,
        sandbox: &SandboxClaimHandle,
        path: &str,
    ) -> reqwest::RequestBuilder {
        self.client
            .request(method, format!("{}{path}", self.base_url))
            .bearer_auth(&self.config.router_bearer_token)
            .header("X-Sandbox-ID", &sandbox.sandbox_id)
            .header("X-Sandbox-Namespace", &sandbox.namespace)
            .header("X-Sandbox-Port", self.config.sandbox_port)
            .header(
                "X-Sandbox-Timeout",
                self.config.request_timeout.as_secs().max(1),
            )
    }

    async fn decode_record(
        response: reqwest::Response,
    ) -> Result<ExecutionRecord, SandboxSupervisorHttpError> {
        if response.status().is_success() {
            return response.json().await.map_err(Into::into);
        }
        Err(http_error(response).await)
    }
}

impl SandboxSupervisor for HttpSandboxSupervisor {
    type Error = SandboxSupervisorHttpError;

    fn start(
        &self,
        sandbox: &SandboxClaimHandle,
        request: StartExecution,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        let sandbox = sandbox.clone();
        Box::pin(async move {
            let id = checked_id(&request.execution_id.0)?;
            let response = self
                .request(Method::PUT, &sandbox, &format!("/v1/executions/{id}"))
                .json(&request)
                .send()
                .await?;
            Self::decode_record(response).await
        })
    }

    fn lookup(
        &self,
        sandbox: &SandboxClaimHandle,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>> {
        let sandbox = sandbox.clone();
        let execution = execution.clone();
        Box::pin(async move {
            let id = checked_id(&execution.execution_id.0)?;
            let response = self
                .request(Method::GET, &sandbox, &format!("/v1/executions/{id}"))
                .send()
                .await?;
            if response.status() == StatusCode::NOT_FOUND {
                return Ok(None);
            }
            Self::decode_record(response).await.map(Some)
        })
    }

    fn cancel(
        &self,
        sandbox: &SandboxClaimHandle,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        let sandbox = sandbox.clone();
        let execution = execution.clone();
        Box::pin(async move {
            let id = checked_id(&execution.execution_id.0)?;
            let response = self
                .request(Method::DELETE, &sandbox, &format!("/v1/executions/{id}"))
                .send()
                .await?;
            Self::decode_record(response).await
        })
    }

    fn read_artifact(
        &self,
        sandbox: &SandboxClaimHandle,
        execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Artifact, Self::Error>> {
        let sandbox = sandbox.clone();
        let execution_id = execution.execution_id.0.clone();
        let artifact_id = artifact_id.to_owned();
        Box::pin(async move {
            let execution_id = checked_id(&execution_id)?;
            let artifact_id = checked_id(&artifact_id)?;
            let response = self
                .request(
                    Method::GET,
                    &sandbox,
                    &format!("/v1/executions/{execution_id}/artifacts/{artifact_id}"),
                )
                .send()
                .await?;
            if !response.status().is_success() {
                return Err(http_error(response).await);
            }
            let payload: ArtifactPayload = response.json().await?;
            let bytes = base64::engine::general_purpose::STANDARD.decode(payload.data_base64)?;
            Ok(Artifact {
                metadata: payload.metadata,
                bytes,
            })
        })
    }
}

fn checked_id(id: &str) -> Result<&str, SandboxSupervisorHttpError> {
    if !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(id)
    } else {
        Err(SandboxSupervisorHttpError::InvalidId)
    }
}

async fn http_error(response: reqwest::Response) -> SandboxSupervisorHttpError {
    let status = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    SandboxSupervisorHttpError::Http {
        status,
        body: body.chars().take(1024).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{HttpSandboxSupervisor, HttpSandboxSupervisorConfig, checked_id};
    use std::time::Duration;

    #[test]
    fn path_ids_are_strictly_bounded() {
        assert!(checked_id("tool_deadbeef-1").is_ok());
        assert!(checked_id("../escape").is_err());
        assert!(checked_id("").is_err());
    }

    #[test]
    fn router_redirects_and_url_injection_are_disabled() {
        let config = HttpSandboxSupervisorConfig {
            router_base_url: "https://sandbox-router.example.internal/base?target=evil".to_owned(),
            router_bearer_token: "secret".to_owned(),
            sandbox_port: 8080,
            request_timeout: Duration::from_secs(5),
        };
        assert!(HttpSandboxSupervisor::new(config).is_err());
    }
}
