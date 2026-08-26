// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_rt_sandbox::{
    Artifact, ArtifactRef, ExecutionId, ExecutionRecord, ExecutionState, HttpSandboxProvider,
    HttpSandboxProviderConfig, HttpSandboxProviderError, SANDBOX_API_VERSION, SandboxCommand,
    SandboxLimits, SandboxProfile, SandboxProvider, ScopedExecutionId, ScopedWorkspaceId,
    StartExecution, WorkspaceId,
};
use agent_rt_sandbox_service::{TrustedProxyAuth, router};
use dynamo_agent_rt::{AuthorizationScope, BoxFuture};
use thiserror::Error;

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

#[derive(Debug, Error)]
#[error("fake provider error")]
struct FakeError;

#[derive(Default)]
struct FakeProvider {
    starts: AtomicUsize,
}

impl SandboxProvider for FakeProvider {
    type Error = FakeError;

    fn start(
        &self,
        request: StartExecution,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(record(&request)) })
    }

    fn lookup(
        &self,
        _execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>> {
        Box::pin(async { Ok(None) })
    }

    fn cancel(
        &self,
        _execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        Box::pin(async { Err(FakeError) })
    }

    fn read_artifact(
        &self,
        _execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Artifact, Self::Error>> {
        let artifact_id = artifact_id.to_owned();
        Box::pin(async move {
            Ok(Artifact {
                metadata: ArtifactRef {
                    artifact_id,
                    name: "result.txt".to_owned(),
                    size_bytes: 4,
                    media_type: Some("text/plain".to_owned()),
                },
                bytes: b"done".to_vec(),
            })
        })
    }

    fn delete_workspace(
        &self,
        _workspace: &ScopedWorkspaceId,
    ) -> BoxFuture<'_, Result<(), Self::Error>> {
        Box::pin(async { Ok(()) })
    }
}

fn request() -> StartExecution {
    StartExecution {
        api_version: SANDBOX_API_VERSION.to_owned(),
        scope: scope("tenant-a"),
        workspace_id: WorkspaceId("workspace-a".to_owned()),
        execution_id: ExecutionId("execution-a".to_owned()),
        profile: SandboxProfile("python-deny-egress".to_owned()),
        command: SandboxCommand {
            argv: vec!["python".to_owned(), "-c".to_owned(), "print(42)".to_owned()],
            cwd: Some("/workspace".to_owned()),
            env: BTreeMap::new(),
            stdin: Vec::new(),
            artifact_paths: Vec::new(),
        },
        limits: SandboxLimits {
            timeout_millis: 10_000,
            max_output_bytes: 1_024,
            max_artifact_bytes: 4_096,
        },
    }
}

fn scope(tenant: &str) -> AuthorizationScope {
    AuthorizationScope {
        tenant_id: tenant.to_owned(),
        principal_id: "principal-a".to_owned(),
    }
}

fn record(request: &StartExecution) -> ExecutionRecord {
    ExecutionRecord {
        request_fingerprint: request.fingerprint(),
        scope: request.scope.clone(),
        workspace_id: request.workspace_id.clone(),
        execution_id: request.execution_id.clone(),
        provider_sandbox_id: "sandbox-a".to_owned(),
        state: ExecutionState::Succeeded,
        exit_code: Some(0),
        stdout: b"42\n".to_vec(),
        stderr: Vec::new(),
        artifacts: Vec::new(),
        failure_code: None,
    }
}

#[tokio::test]
async fn authenticates_scope_before_dispatch_and_round_trips_artifacts() {
    let provider = Arc::new(FakeProvider::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = router(
        Arc::clone(&provider),
        TrustedProxyAuth::new(TOKEN).unwrap(),
        1024 * 1024,
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let client = HttpSandboxProvider::new(HttpSandboxProviderConfig {
        endpoint: format!("http://{address}"),
        bearer_token: TOKEN.to_owned(),
        ..HttpSandboxProviderConfig::default()
    })
    .unwrap();

    let execution = request();
    let result = client.start(execution.clone()).await.unwrap();
    assert_eq!(result.stdout, b"42\n");
    assert_eq!(provider.starts.load(Ordering::SeqCst), 1);

    let scoped = ScopedExecutionId {
        scope: execution.scope.clone(),
        workspace_id: execution.workspace_id.clone(),
        profile: execution.profile.clone(),
        execution_id: execution.execution_id.clone(),
    };
    let artifact = client.read_artifact(&scoped, "artifact-a").await.unwrap();
    assert_eq!(artifact.bytes, b"done");

    let unauthenticated = HttpSandboxProvider::new(HttpSandboxProviderConfig {
        endpoint: format!("http://{address}"),
        bearer_token: "badbadbadbadbadbadbadbadbadbadba".to_owned(),
        ..HttpSandboxProviderConfig::default()
    })
    .unwrap();
    assert!(matches!(
        unauthenticated.start(execution.clone()).await,
        Err(HttpSandboxProviderError::Http { status: 401, .. })
    ));
    assert_eq!(provider.starts.load(Ordering::SeqCst), 1);

    let response = reqwest::Client::new()
        .post(format!("http://{address}/v1/executions"))
        .bearer_auth(TOKEN)
        .header("x-agent-sandbox-tenant-id", "tenant-b")
        .header("x-agent-sandbox-principal-id", "principal-a")
        .json(&execution)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::FORBIDDEN);
    assert_eq!(provider.starts.load(Ordering::SeqCst), 1);

    server.abort();
}
