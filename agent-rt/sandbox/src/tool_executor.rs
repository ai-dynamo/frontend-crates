// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::time::Duration;

use dynamo_agent_rt::{
    BoxFuture, ToolExecutionFailure, ToolExecutionRequest, ToolExecutionResult, ToolExecutor,
    ToolFailureDisposition, ToolFailurePolicy,
};
use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

use crate::{
    ExecutionId, ExecutionRecord, ExecutionState, SANDBOX_API_VERSION, SandboxCommand,
    SandboxLimits, SandboxProfile, SandboxProvider, ScopedExecutionId, StartExecution, WorkspaceId,
};

#[derive(Debug, Clone)]
pub struct SandboxToolExecutorConfig {
    pub poll_interval: Duration,
    pub max_wait: Duration,
    pub max_output_bytes: u64,
    pub max_artifact_bytes: u64,
}

impl Default for SandboxToolExecutorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(100),
            max_wait: Duration::from_secs(60),
            max_output_bytes: 1024 * 1024,
            max_artifact_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SandboxToolExecutorConfigError {
    #[error("sandbox polling interval must be nonzero")]
    ZeroPollInterval,
    #[error("sandbox maximum wait must be nonzero")]
    ZeroMaxWait,
}

#[derive(Debug, Error)]
pub enum SandboxToolError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[error("sandbox connector does not support operation {0}")]
    UnsupportedOperation(String),
    #[error("invalid sandbox arguments: {0}")]
    InvalidArguments(#[source] serde_json::Error),
    #[error("sandbox command must contain at least one argv element")]
    EmptyCommand,
    #[error("sandbox provider failed: {0}")]
    Provider(E),
    #[error("sandbox execution failed with state {state:?}: {failure_code:?}")]
    KnownFailure {
        state: ExecutionState,
        failure_code: Option<String>,
    },
    #[error("sandbox execution outcome is unknown")]
    OutcomeUnknown,
    #[error("sandbox execution did not finish within {0:?}")]
    WaitTimedOut(Duration),
    #[error("sandbox provider returned a mismatched execution identity")]
    IdentityMismatch,
}

/// Conservative failure classification for the sandbox tool adapter.
#[derive(Debug, Clone, Copy, Default)]
pub struct SandboxFailurePolicy;

impl<E> ToolFailurePolicy<SandboxToolError<E>> for SandboxFailurePolicy
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn classify(&self, error: &SandboxToolError<E>) -> ToolFailureDisposition {
        match error {
            SandboxToolError::UnsupportedOperation(_)
            | SandboxToolError::InvalidArguments(_)
            | SandboxToolError::EmptyCommand
            | SandboxToolError::KnownFailure { .. } => {
                ToolFailureDisposition::Failed(ToolExecutionFailure {
                    code: "sandbox_execution_failed".to_owned(),
                    message: error.to_string(),
                    retryable: false,
                })
            }
            SandboxToolError::Provider(_)
            | SandboxToolError::OutcomeUnknown
            | SandboxToolError::WaitTimedOut(_)
            | SandboxToolError::IdentityMismatch => ToolFailureDisposition::OutcomeUnknown,
        }
    }
}

/// Adapts one external [`SandboxProvider`] to the runtime-owned tool seam.
pub struct SandboxProviderExecutor<P> {
    provider: P,
    config: SandboxToolExecutorConfig,
}

impl<P> SandboxProviderExecutor<P> {
    pub fn new(
        provider: P,
        config: SandboxToolExecutorConfig,
    ) -> Result<Self, SandboxToolExecutorConfigError> {
        if config.poll_interval.is_zero() {
            return Err(SandboxToolExecutorConfigError::ZeroPollInterval);
        }
        if config.max_wait.is_zero() {
            return Err(SandboxToolExecutorConfigError::ZeroMaxWait);
        }
        Ok(Self { provider, config })
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }
}

#[derive(Debug, Deserialize)]
struct PythonArguments {
    code: String,
    cwd: Option<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    stdin: String,
    #[serde(default)]
    artifact_paths: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CommandArguments {
    argv: Vec<String>,
    cwd: Option<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    stdin: String,
    #[serde(default)]
    artifact_paths: Vec<String>,
}

impl<P> SandboxProviderExecutor<P>
where
    P: SandboxProvider,
{
    fn provider_request(
        &self,
        request: &ToolExecutionRequest,
    ) -> Result<StartExecution, SandboxToolError<P::Error>> {
        let command = match request.operation.as_str() {
            "python" => {
                let args: PythonArguments = serde_json::from_value(request.arguments.clone())
                    .map_err(SandboxToolError::InvalidArguments)?;
                SandboxCommand {
                    argv: vec!["python".to_owned(), "-c".to_owned(), args.code],
                    cwd: args.cwd,
                    env: args.env,
                    stdin: args.stdin.into_bytes(),
                    artifact_paths: args.artifact_paths,
                }
            }
            "command" => {
                let args: CommandArguments = serde_json::from_value(request.arguments.clone())
                    .map_err(SandboxToolError::InvalidArguments)?;
                SandboxCommand {
                    argv: args.argv,
                    cwd: args.cwd,
                    env: args.env,
                    stdin: args.stdin.into_bytes(),
                    artifact_paths: args.artifact_paths,
                }
            }
            operation => {
                return Err(SandboxToolError::UnsupportedOperation(operation.to_owned()));
            }
        };
        if command.argv.is_empty() {
            return Err(SandboxToolError::EmptyCommand);
        }

        Ok(StartExecution {
            api_version: SANDBOX_API_VERSION.to_owned(),
            scope: request.scope.clone(),
            workspace_id: workspace_id(request),
            execution_id: ExecutionId(request.idempotency_key.to_string()),
            profile: SandboxProfile(request.profile.clone()),
            command,
            limits: SandboxLimits {
                timeout_millis: u64::try_from(self.config.max_wait.as_millis()).unwrap_or(u64::MAX),
                max_output_bytes: self.config.max_output_bytes,
                max_artifact_bytes: self.config.max_artifact_bytes,
            },
        })
    }

    async fn await_terminal(
        &self,
        expected: &StartExecution,
        mut record: ExecutionRecord,
    ) -> Result<ToolExecutionResult, SandboxToolError<P::Error>> {
        self.validate_identity(expected, &record)?;
        let deadline = tokio::time::Instant::now() + self.config.max_wait;
        while !record.state.is_terminal() {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return Err(SandboxToolError::WaitTimedOut(self.config.max_wait));
            }
            tokio::time::sleep(
                self.config
                    .poll_interval
                    .min(deadline.saturating_duration_since(now)),
            )
            .await;
            record = self
                .provider
                .lookup(&scoped_execution(expected))
                .await
                .map_err(SandboxToolError::Provider)?
                .ok_or(SandboxToolError::OutcomeUnknown)?;
            self.validate_identity(expected, &record)?;
        }

        match record.state {
            ExecutionState::Succeeded => Ok(ToolExecutionResult {
                output: json!({
                    "exit_code": record.exit_code,
                    "stdout": String::from_utf8_lossy(&record.stdout),
                    "stderr": String::from_utf8_lossy(&record.stderr),
                    "artifacts": record.artifacts,
                    "sandbox_id": record.provider_sandbox_id,
                }),
                is_error: false,
            }),
            ExecutionState::OutcomeUnknown => Err(SandboxToolError::OutcomeUnknown),
            state => Err(SandboxToolError::KnownFailure {
                state,
                failure_code: record.failure_code,
            }),
        }
    }

    fn validate_identity(
        &self,
        expected: &StartExecution,
        record: &ExecutionRecord,
    ) -> Result<(), SandboxToolError<P::Error>> {
        if record.scope == expected.scope
            && record.workspace_id == expected.workspace_id
            && record.execution_id == expected.execution_id
            && record.request_fingerprint == expected.fingerprint()
        {
            Ok(())
        } else {
            Err(SandboxToolError::IdentityMismatch)
        }
    }
}

impl<P> ToolExecutor for SandboxProviderExecutor<P>
where
    P: SandboxProvider,
{
    type Error = SandboxToolError<P::Error>;

    fn execute(
        &self,
        request: ToolExecutionRequest,
    ) -> BoxFuture<'_, Result<ToolExecutionResult, Self::Error>> {
        Box::pin(async move {
            let provider_request = self.provider_request(&request)?;
            let record = self
                .provider
                .start(provider_request.clone())
                .await
                .map_err(SandboxToolError::Provider)?;
            self.await_terminal(&provider_request, record).await
        })
    }

    fn lookup<'a>(
        &'a self,
        request: &'a ToolExecutionRequest,
    ) -> BoxFuture<'a, Result<Option<ToolExecutionResult>, Self::Error>> {
        Box::pin(async move {
            let provider_request = self.provider_request(request)?;
            let Some(record) = self
                .provider
                .lookup(&scoped_execution(&provider_request))
                .await
                .map_err(SandboxToolError::Provider)?
            else {
                return Ok(None);
            };
            self.await_terminal(&provider_request, record)
                .await
                .map(Some)
        })
    }
}

fn scoped_execution(request: &StartExecution) -> ScopedExecutionId {
    ScopedExecutionId {
        scope: request.scope.clone(),
        workspace_id: request.workspace_id.clone(),
        profile: request.profile.clone(),
        execution_id: request.execution_id.clone(),
    }
}

fn workspace_id(request: &ToolExecutionRequest) -> WorkspaceId {
    let mut hasher = blake3::Hasher::new();
    for value in [
        request.scope.tenant_id.as_str(),
        request.scope.principal_id.as_str(),
        request.response_id.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    WorkspaceId(format!("ws_{}", hasher.finalize().to_hex()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use dynamo_agent_rt::{
        AuthorizationScope, BoxFuture, IdempotencyKey, ResponseId, ToolExecutionRequest,
        ToolExecutor,
    };
    use serde_json::json;
    use thiserror::Error;

    use crate::{
        Artifact, ExecutionRecord, ExecutionState, SandboxProvider, ScopedExecutionId,
        StartExecution,
    };

    use super::{
        SandboxProviderExecutor, SandboxToolExecutorConfig, SandboxToolExecutorConfigError,
    };

    #[derive(Debug, Error)]
    #[error("fake provider failed")]
    struct FakeError;

    #[derive(Default)]
    struct FakeProvider {
        request: Mutex<Option<StartExecution>>,
    }

    impl SandboxProvider for FakeProvider {
        type Error = FakeError;

        fn start(
            &self,
            request: StartExecution,
        ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
            *self.request.lock().unwrap() = Some(request.clone());
            Box::pin(async move { Ok(success(&request)) })
        }

        fn lookup(
            &self,
            execution: &ScopedExecutionId,
        ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>> {
            let request = self.request.lock().unwrap().clone();
            let matches = request.filter(|request| {
                request.scope == execution.scope
                    && request.workspace_id == execution.workspace_id
                    && request.profile == execution.profile
                    && request.execution_id == execution.execution_id
            });
            Box::pin(async move { Ok(matches.as_ref().map(success)) })
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
            _artifact_id: &str,
        ) -> BoxFuture<'_, Result<Artifact, Self::Error>> {
            Box::pin(async { Err(FakeError) })
        }

        fn delete_workspace(
            &self,
            _workspace: &crate::ScopedWorkspaceId,
        ) -> BoxFuture<'_, Result<(), Self::Error>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn success(request: &StartExecution) -> ExecutionRecord {
        ExecutionRecord {
            request_fingerprint: request.fingerprint(),
            scope: request.scope.clone(),
            workspace_id: request.workspace_id.clone(),
            execution_id: request.execution_id.clone(),
            provider_sandbox_id: "claim-a".to_owned(),
            state: ExecutionState::Succeeded,
            exit_code: Some(0),
            stdout: b"42\n".to_vec(),
            stderr: Vec::new(),
            artifacts: Vec::new(),
            failure_code: None,
        }
    }

    fn request() -> ToolExecutionRequest {
        ToolExecutionRequest {
            response_id: ResponseId::from("resp-a"),
            call_id: "call-a".to_owned(),
            connector: "sandbox".to_owned(),
            operation: "python".to_owned(),
            profile: "python-deny-egress".to_owned(),
            arguments: json!({"code": "print(42)"}),
            scope: AuthorizationScope {
                tenant_id: "tenant-a".to_owned(),
                principal_id: "principal-a".to_owned(),
            },
            idempotency_key: IdempotencyKey::from("tool-a"),
            attempt: 0,
        }
    }

    #[tokio::test]
    async fn profile_is_server_selected_and_result_is_normalized() {
        let executor = SandboxProviderExecutor::new(
            FakeProvider::default(),
            SandboxToolExecutorConfig::default(),
        )
        .unwrap();

        let result = executor.execute(request()).await.unwrap();
        assert_eq!(result.output["stdout"], "42\n");
        let provider_request = executor.provider().request.lock().unwrap().clone().unwrap();
        assert_eq!(provider_request.profile.0, "python-deny-egress");
        assert_eq!(provider_request.command.argv[0], "python");
    }

    #[tokio::test]
    async fn recovery_reconstructs_the_same_workspace_and_execution() {
        let executor = SandboxProviderExecutor::new(
            FakeProvider::default(),
            SandboxToolExecutorConfig::default(),
        )
        .unwrap();
        let request = request();
        executor.execute(request.clone()).await.unwrap();

        let recovered = executor.lookup(&request).await.unwrap().unwrap();
        assert_eq!(recovered.output["exit_code"], 0);
    }

    #[test]
    fn rejects_a_zero_poll_interval() {
        let config = SandboxToolExecutorConfig {
            poll_interval: Duration::ZERO,
            ..SandboxToolExecutorConfig::default()
        };

        let result = SandboxProviderExecutor::new(FakeProvider::default(), config);
        assert!(matches!(
            result,
            Err(SandboxToolExecutorConfigError::ZeroPollInterval)
        ));
    }

    #[test]
    fn rejects_a_zero_maximum_wait() {
        let config = SandboxToolExecutorConfig {
            max_wait: Duration::ZERO,
            ..SandboxToolExecutorConfig::default()
        };

        let result = SandboxProviderExecutor::new(FakeProvider::default(), config);
        assert!(matches!(
            result,
            Err(SandboxToolExecutorConfigError::ZeroMaxWait)
        ));
    }
}
