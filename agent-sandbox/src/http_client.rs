// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated client for an external sandbox executor service.

use std::time::Duration;

use base64::Engine;
use dynamo_agent_rt::BoxFuture;
use futures::StreamExt;
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{
    Artifact, ArtifactRef, ExecutionRecord, SandboxProvider, ScopedExecutionId, ScopedWorkspaceId,
    StartExecution,
};

pub const TENANT_HEADER: &str = "x-agent-sandbox-tenant-id";
pub const PRINCIPAL_HEADER: &str = "x-agent-sandbox-principal-id";

const START_PATH: &str = "/v1/executions";
const LOOKUP_PATH: &str = "/v1/executions:lookup";
const CANCEL_PATH: &str = "/v1/executions:cancel";
const ARTIFACT_PATH: &str = "/v1/artifacts:read";
const DELETE_WORKSPACE_PATH: &str = "/v1/workspaces:delete";

#[derive(Debug, Clone)]
pub struct HttpSandboxProviderConfig {
    pub endpoint: String,
    pub bearer_token: String,
    pub allow_http: bool,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub max_response_bytes: usize,
}

impl Default for HttpSandboxProviderConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8090".to_owned(),
            bearer_token: String::new(),
            allow_http: true,
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(90),
            max_response_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum HttpSandboxProviderConfigError {
    #[error("sandbox service endpoint is invalid: {0}")]
    InvalidEndpoint(String),
    #[error("sandbox service endpoint must use HTTPS")]
    HttpsRequired,
    #[error("sandbox service endpoint must contain only an origin")]
    EndpointMustBeOrigin,
    #[error("sandbox service bearer token must contain at least 32 bytes")]
    WeakBearerToken,
    #[error("sandbox service response limit must be nonzero")]
    ZeroResponseLimit,
    #[error("sandbox service connect and request timeouts must be nonzero")]
    ZeroTimeout,
    #[error("sandbox service HTTP client could not be built: {0}")]
    Client(String),
}

#[derive(Debug, Error)]
pub enum HttpSandboxProviderError {
    #[error("sandbox service request failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("sandbox service returned HTTP {status} ({code})")]
    Http { status: u16, code: String },
    #[error("sandbox service response exceeded {0} bytes")]
    ResponseTooLarge(usize),
    #[error("sandbox service response was malformed: {0}")]
    MalformedResponse(#[from] serde_json::Error),
    #[error("sandbox service artifact encoding is invalid")]
    InvalidArtifactEncoding,
}

#[derive(Clone)]
pub struct HttpSandboxProvider {
    endpoint: Url,
    bearer_token: String,
    max_response_bytes: usize,
    client: reqwest::Client,
}

impl HttpSandboxProvider {
    pub fn new(config: HttpSandboxProviderConfig) -> Result<Self, HttpSandboxProviderConfigError> {
        if config.bearer_token.len() < 32 {
            return Err(HttpSandboxProviderConfigError::WeakBearerToken);
        }
        if config.max_response_bytes == 0 {
            return Err(HttpSandboxProviderConfigError::ZeroResponseLimit);
        }
        if config.connect_timeout.is_zero() || config.request_timeout.is_zero() {
            return Err(HttpSandboxProviderConfigError::ZeroTimeout);
        }
        let endpoint = Url::parse(&config.endpoint)
            .map_err(|error| HttpSandboxProviderConfigError::InvalidEndpoint(error.to_string()))?;
        if endpoint.scheme() != "https" && !(config.allow_http && endpoint.scheme() == "http") {
            return Err(HttpSandboxProviderConfigError::HttpsRequired);
        }
        if endpoint.cannot_be_a_base()
            || endpoint.host_str().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
            || endpoint.path() != "/"
        {
            return Err(HttpSandboxProviderConfigError::EndpointMustBeOrigin);
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .build()
            .map_err(|error| HttpSandboxProviderConfigError::Client(error.to_string()))?;
        Ok(Self {
            endpoint,
            bearer_token: config.bearer_token,
            max_response_bytes: config.max_response_bytes,
            client,
        })
    }

    async fn post<T, R>(
        &self,
        path: &str,
        scope: &dynamo_agent_rt::AuthorizationScope,
        body: &T,
    ) -> Result<R, HttpSandboxProviderError>
    where
        T: Serialize + ?Sized,
        R: DeserializeOwned,
    {
        let response = self
            .client
            .post(self.request_url(path))
            .bearer_auth(&self.bearer_token)
            .header(TENANT_HEADER, &scope.tenant_id)
            .header(PRINCIPAL_HEADER, &scope.principal_id)
            .json(body)
            .send()
            .await?;
        let status = response.status();
        let bytes = bounded_body(response, self.max_response_bytes).await?;
        if !status.is_success() {
            let code = serde_json::from_slice::<ErrorEnvelope>(&bytes)
                .map(|body| body.error.code)
                .unwrap_or_else(|_| "sandbox_service_error".to_owned());
            return Err(HttpSandboxProviderError::Http {
                status: status.as_u16(),
                code,
            });
        }
        Ok(serde_json::from_slice(&bytes)?)
    }

    async fn post_empty<T: Serialize + ?Sized>(
        &self,
        path: &str,
        scope: &dynamo_agent_rt::AuthorizationScope,
        body: &T,
    ) -> Result<(), HttpSandboxProviderError> {
        let response = self
            .client
            .post(self.request_url(path))
            .bearer_auth(&self.bearer_token)
            .header(TENANT_HEADER, &scope.tenant_id)
            .header(PRINCIPAL_HEADER, &scope.principal_id)
            .json(body)
            .send()
            .await?;
        let status = response.status();
        let bytes = bounded_body(response, 4_096).await?;
        if status == StatusCode::NO_CONTENT {
            return Ok(());
        }
        let code = serde_json::from_slice::<ErrorEnvelope>(&bytes)
            .map(|body| body.error.code)
            .unwrap_or_else(|_| "sandbox_service_error".to_owned());
        Err(HttpSandboxProviderError::Http {
            status: status.as_u16(),
            code,
        })
    }

    fn request_url(&self, path: &str) -> Url {
        let mut url = self.endpoint.clone();
        url.set_path(path);
        url
    }
}

impl SandboxProvider for HttpSandboxProvider {
    type Error = HttpSandboxProviderError;

    fn start(
        &self,
        request: StartExecution,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        Box::pin(async move { self.post(START_PATH, &request.scope, &request).await })
    }

    fn lookup(
        &self,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>> {
        let execution = execution.clone();
        Box::pin(async move { self.post(LOOKUP_PATH, &execution.scope, &execution).await })
    }

    fn cancel(
        &self,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        let execution = execution.clone();
        Box::pin(async move { self.post(CANCEL_PATH, &execution.scope, &execution).await })
    }

    fn read_artifact(
        &self,
        execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Artifact, Self::Error>> {
        let request = ReadArtifactRequest {
            execution: execution.clone(),
            artifact_id: artifact_id.to_owned(),
        };
        Box::pin(async move {
            let artifact: ArtifactEnvelope = self
                .post(ARTIFACT_PATH, &request.execution.scope, &request)
                .await?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(artifact.bytes_base64)
                .map_err(|_| HttpSandboxProviderError::InvalidArtifactEncoding)?;
            if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != artifact.metadata.size_bytes {
                return Err(HttpSandboxProviderError::InvalidArtifactEncoding);
            }
            Ok(Artifact {
                metadata: artifact.metadata,
                bytes,
            })
        })
    }

    fn delete_workspace(
        &self,
        workspace: &ScopedWorkspaceId,
    ) -> BoxFuture<'_, Result<(), Self::Error>> {
        let workspace = workspace.clone();
        Box::pin(async move {
            self.post_empty(DELETE_WORKSPACE_PATH, &workspace.scope, &workspace)
                .await
        })
    }
}

async fn bounded_body(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, HttpSandboxProviderError> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(HttpSandboxProviderError::ResponseTooLarge(max_bytes));
    }
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(HttpSandboxProviderError::ResponseTooLarge(max_bytes));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[derive(Debug, Serialize)]
struct ReadArtifactRequest {
    execution: ScopedExecutionId,
    artifact_id: String,
}

#[derive(Debug, Deserialize)]
struct ArtifactEnvelope {
    metadata: ArtifactRef,
    bytes_base64: String,
}

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    code: String,
}
