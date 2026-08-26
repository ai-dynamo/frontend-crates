// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete Kubernetes Agent Sandbox control-plane and router clients.

use std::collections::BTreeMap;
use std::time::Duration;

use chrono::Utc;
use dynamo_agent_rt::BoxFuture;
use kube::api::{DeleteParams, PostParams, PropagationPolicy};
use kube::core::{ApiResource, DynamicObject, GroupVersionKind};
use kube::{Api, Client};
use serde_json::json;
use thiserror::Error;

use crate::{AgentSandboxControlPlane, SandboxClaimHandle, SandboxClaimRequest};

const WORKSPACE_FINGERPRINT_ANNOTATION: &str = "agent-rt.nvidia.com/workspace-fingerprint";
const WORKSPACE_LABEL: &str = "sandbox.users.io/workspace";

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
    #[error("ready sandbox claim does not contain status.sandbox.serviceFQDN")]
    MissingServiceEndpoint,
    #[error("sandbox workspace TTL is outside the supported timestamp range")]
    InvalidTtl,
}

#[derive(Clone)]
pub struct KubeAgentSandboxControlPlane {
    client: Client,
    config: KubeAgentSandboxControlPlaneConfig,
    claim_resource: ApiResource,
    sandbox_resource: ApiResource,
}

impl KubeAgentSandboxControlPlane {
    pub fn new(client: Client, config: KubeAgentSandboxControlPlaneConfig) -> Self {
        let claim_gvk =
            GroupVersionKind::gvk("extensions.agents.x-k8s.io", "v1beta1", "SandboxClaim");
        let sandbox_gvk = GroupVersionKind::gvk("agents.x-k8s.io", "v1beta1", "Sandbox");
        Self {
            client,
            config,
            claim_resource: ApiResource::from_gvk_with_plural(&claim_gvk, "sandboxclaims"),
            sandbox_resource: ApiResource::from_gvk_with_plural(&sandbox_gvk, "sandboxes"),
        }
    }

    fn claims(&self, namespace: &str) -> Api<DynamicObject> {
        Api::namespaced_with(self.client.clone(), namespace, &self.claim_resource)
    }

    fn sandboxes(&self, namespace: &str) -> Api<DynamicObject> {
        Api::namespaced_with(self.client.clone(), namespace, &self.sandbox_resource)
    }

    fn claim(&self, request: &SandboxClaimRequest) -> Result<DynamicObject, KubeControlPlaneError> {
        let ttl = chrono::Duration::from_std(request.expires_after)
            .map_err(|_| KubeControlPlaneError::InvalidTtl)?;
        let shutdown_time = (Utc::now() + ttl).to_rfc3339();
        let mut claim = DynamicObject::new(&request.claim_name, &self.claim_resource).data(json!({
            "spec": {
                "warmPoolRef": {"name": request.warm_pool},
                "lifecycle": {
                    "shutdownTime": shutdown_time,
                    "shutdownPolicy": "DeleteForeground",
                    "ttlSecondsAfterFinished": 0
                },
                "additionalPodMetadata": {
                    "labels": {
                        WORKSPACE_LABEL: &request.workspace_fingerprint[..40]
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
        let claims = self.claims(&request.namespace);
        let sandboxes = self.sandboxes(&request.namespace);
        let deadline = tokio::time::Instant::now() + self.config.ready_timeout;
        let mut ready_without_identity = false;
        let mut ready_without_endpoint = false;
        loop {
            let claim = claims.get(&request.claim_name).await?;
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
                let Some(sandbox_id) = claim
                    .data
                    .pointer("/status/sandbox/name")
                    .and_then(|value| value.as_str())
                    .filter(|name| !name.is_empty())
                else {
                    ready_without_identity = true;
                    continue_after_poll(&self.config, deadline).await?;
                    continue;
                };
                let claim_service_fqdn = claim
                    .data
                    .pointer("/status/sandbox/serviceFQDN")
                    .and_then(|value| value.as_str())
                    .filter(|name| !name.is_empty());
                let service_fqdn = if let Some(service_fqdn) = claim_service_fqdn {
                    Some(service_fqdn.to_owned())
                } else {
                    sandboxes.get_opt(sandbox_id).await?.and_then(|sandbox| {
                        sandbox
                            .data
                            .pointer("/status/serviceFQDN")
                            .and_then(|value| value.as_str())
                            .filter(|name| !name.is_empty())
                            .map(str::to_owned)
                    })
                };
                if let Some(service_fqdn) = service_fqdn {
                    return Ok(SandboxClaimHandle {
                        namespace: request.namespace.clone(),
                        claim_name: request.claim_name.clone(),
                        sandbox_id: sandbox_id.to_owned(),
                        service_fqdn,
                    });
                }
                ready_without_endpoint = true;
            }
            if let Err(KubeControlPlaneError::ReadyTimeout(_)) =
                continue_after_poll(&self.config, deadline).await
            {
                return if ready_without_endpoint {
                    Err(KubeControlPlaneError::MissingServiceEndpoint)
                } else if ready_without_identity {
                    Err(KubeControlPlaneError::MissingSandboxIdentity)
                } else {
                    Err(KubeControlPlaneError::ReadyTimeout(
                        self.config.ready_timeout,
                    ))
                };
            }
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
            let api = self.claims(&request.namespace);
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
            let api = self.claims(&request.namespace);
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
            let api = self.claims(&request.namespace);
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

async fn continue_after_poll(
    config: &KubeAgentSandboxControlPlaneConfig,
    deadline: tokio::time::Instant,
) -> Result<(), KubeControlPlaneError> {
    let now = tokio::time::Instant::now();
    if now >= deadline {
        return Err(KubeControlPlaneError::ReadyTimeout(config.ready_timeout));
    }
    tokio::time::sleep(
        config
            .poll_interval
            .min(deadline.saturating_duration_since(now)),
    )
    .await;
    Ok(())
}
