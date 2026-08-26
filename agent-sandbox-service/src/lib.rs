// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use agent_rt_sandbox::{
    Artifact, ExecutionRecord, SandboxProvider, ScopedExecutionId, ScopedWorkspaceId,
    StartExecution,
};
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use dynamo_agent_rt::AuthorizationScope;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;

pub mod config;

const TENANT_HEADER: &str = "x-agent-sandbox-tenant-id";
const PRINCIPAL_HEADER: &str = "x-agent-sandbox-principal-id";

#[derive(Clone)]
pub struct TrustedProxyAuth {
    token_hash: [u8; 32],
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TrustedProxyAuthError {
    #[error("sandbox service trusted-proxy token must contain at least 32 bytes")]
    WeakToken,
}

impl TrustedProxyAuth {
    pub fn new(token: &str) -> Result<Self, TrustedProxyAuthError> {
        if token.len() < 32 {
            return Err(TrustedProxyAuthError::WeakToken);
        }
        Ok(Self {
            token_hash: *blake3::hash(token.as_bytes()).as_bytes(),
        })
    }

    fn authenticate(&self, headers: &HeaderMap) -> Result<AuthorizationScope, ApiError> {
        let presented = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .filter(|value| !value.is_empty())
            .ok_or(ApiError::unauthorized())?;
        let presented_hash = blake3::hash(presented.as_bytes());
        if presented_hash
            .as_bytes()
            .ct_eq(&self.token_hash)
            .unwrap_u8()
            != 1
        {
            return Err(ApiError::unauthorized());
        }
        Ok(AuthorizationScope {
            tenant_id: scope_header(headers, TENANT_HEADER)?,
            principal_id: scope_header(headers, PRINCIPAL_HEADER)?,
        })
    }
}

struct AppState<P> {
    provider: Arc<P>,
    auth: TrustedProxyAuth,
}

pub fn router<P>(provider: Arc<P>, auth: TrustedProxyAuth, max_request_bytes: usize) -> Router
where
    P: SandboxProvider,
{
    let state = Arc::new(AppState { provider, auth });
    Router::new()
        .route("/healthz", get(|| async { StatusCode::NO_CONTENT }))
        .route("/v1/executions", post(start::<P>))
        .route("/v1/executions:lookup", post(lookup::<P>))
        .route("/v1/executions:cancel", post(cancel::<P>))
        .route("/v1/artifacts:read", post(read_artifact::<P>))
        .route("/v1/workspaces:delete", post(delete_workspace::<P>))
        .layer(DefaultBodyLimit::max(max_request_bytes))
        .with_state(state)
}

async fn start<P>(
    State(state): State<Arc<AppState<P>>>,
    headers: HeaderMap,
    payload: Result<Json<StartExecution>, JsonRejection>,
) -> Result<Json<ExecutionRecord>, ApiError>
where
    P: SandboxProvider,
{
    let scope = state.auth.authenticate(&headers)?;
    let Json(request) = parse_json(payload)?;
    require_scope(&scope, &request.scope)?;
    state
        .provider
        .start(request)
        .await
        .map(Json)
        .map_err(provider_error)
}

async fn lookup<P>(
    State(state): State<Arc<AppState<P>>>,
    headers: HeaderMap,
    payload: Result<Json<ScopedExecutionId>, JsonRejection>,
) -> Result<Json<Option<ExecutionRecord>>, ApiError>
where
    P: SandboxProvider,
{
    let scope = state.auth.authenticate(&headers)?;
    let Json(execution) = parse_json(payload)?;
    require_scope(&scope, &execution.scope)?;
    state
        .provider
        .lookup(&execution)
        .await
        .map(Json)
        .map_err(provider_error)
}

async fn cancel<P>(
    State(state): State<Arc<AppState<P>>>,
    headers: HeaderMap,
    payload: Result<Json<ScopedExecutionId>, JsonRejection>,
) -> Result<Json<ExecutionRecord>, ApiError>
where
    P: SandboxProvider,
{
    let scope = state.auth.authenticate(&headers)?;
    let Json(execution) = parse_json(payload)?;
    require_scope(&scope, &execution.scope)?;
    state
        .provider
        .cancel(&execution)
        .await
        .map(Json)
        .map_err(provider_error)
}

async fn read_artifact<P>(
    State(state): State<Arc<AppState<P>>>,
    headers: HeaderMap,
    payload: Result<Json<ReadArtifactRequest>, JsonRejection>,
) -> Result<Json<ArtifactEnvelope>, ApiError>
where
    P: SandboxProvider,
{
    let scope = state.auth.authenticate(&headers)?;
    let Json(request) = parse_json(payload)?;
    require_scope(&scope, &request.execution.scope)?;
    let artifact = state
        .provider
        .read_artifact(&request.execution, &request.artifact_id)
        .await
        .map_err(provider_error)?;
    Ok(Json(ArtifactEnvelope::from(artifact)))
}

async fn delete_workspace<P>(
    State(state): State<Arc<AppState<P>>>,
    headers: HeaderMap,
    payload: Result<Json<ScopedWorkspaceId>, JsonRejection>,
) -> Result<StatusCode, ApiError>
where
    P: SandboxProvider,
{
    let scope = state.auth.authenticate(&headers)?;
    let Json(workspace) = parse_json(payload)?;
    require_scope(&scope, &workspace.scope)?;
    state
        .provider
        .delete_workspace(&workspace)
        .await
        .map_err(provider_error)?;
    Ok(StatusCode::NO_CONTENT)
}

fn parse_json<T>(payload: Result<Json<T>, JsonRejection>) -> Result<Json<T>, ApiError> {
    payload.map_err(|_| ApiError::bad_request("invalid_json"))
}

fn require_scope(
    authenticated: &AuthorizationScope,
    requested: &AuthorizationScope,
) -> Result<(), ApiError> {
    if authenticated == requested {
        Ok(())
    } else {
        Err(ApiError::forbidden("scope_mismatch"))
    }
}

fn scope_header(headers: &HeaderMap, name: &'static str) -> Result<String, ApiError> {
    let value = headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 128
                && value.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'@')
                })
        })
        .ok_or_else(|| ApiError::bad_request("invalid_scope_header"))?;
    Ok(value.to_owned())
}

fn provider_error<E>(error: E) -> ApiError
where
    E: std::error::Error,
{
    tracing::error!(error = %error, "sandbox provider request failed");
    ApiError::unavailable("sandbox_provider_unavailable")
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
}

impl ApiError {
    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
        }
    }

    fn forbidden(code: &'static str) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code,
        }
    }

    fn bad_request(code: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
        }
    }

    fn unavailable(code: &'static str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorEnvelope {
                error: ErrorBody { code: self.code },
            }),
        )
            .into_response()
    }
}

#[derive(Debug, Deserialize)]
struct ReadArtifactRequest {
    execution: ScopedExecutionId,
    artifact_id: String,
}

#[derive(Debug, Serialize)]
struct ArtifactEnvelope {
    metadata: agent_rt_sandbox::ArtifactRef,
    bytes_base64: String,
}

impl From<Artifact> for ArtifactEnvelope {
    fn from(artifact: Artifact) -> Self {
        Self {
            metadata: artifact.metadata,
            bytes_base64: base64::engine::general_purpose::STANDARD.encode(artifact.bytes),
        }
    }
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
}
