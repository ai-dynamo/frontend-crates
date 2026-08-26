// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use dynamo_agent_rt::{AuthorizationScope, BoxFuture};
use thiserror::Error;

use crate::{
    Artifact, ExecutionRecord, SandboxProfile, SandboxProvider, ScopedExecutionId,
    ScopedWorkspaceId, StartExecution, WorkspaceId,
};

/// Model-independent command policy bound to one operator profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandPolicy {
    pub allowed_executables: BTreeSet<String>,
    pub allow_environment: bool,
    pub max_arguments: usize,
    pub max_argument_bytes: usize,
    pub max_environment_variables: usize,
    pub max_environment_bytes: usize,
    pub max_stdin_bytes: usize,
    pub max_artifacts: usize,
}

impl Default for CommandPolicy {
    fn default() -> Self {
        Self {
            allowed_executables: BTreeSet::new(),
            allow_environment: false,
            max_arguments: 64,
            max_argument_bytes: 1024 * 1024,
            max_environment_variables: 64,
            max_environment_bytes: 256 * 1024,
            max_stdin_bytes: 1024 * 1024,
            max_artifacts: 32,
        }
    }
}

/// Operator configuration for one Kubernetes Agent Sandbox warm pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KubernetesSandboxProfile {
    pub warm_pool: String,
    pub workspace_ttl: Duration,
    pub max_execution_timeout: Duration,
    pub max_output_bytes: u64,
    pub max_artifact_bytes: u64,
    pub command_policy: CommandPolicy,
}

/// Fail-closed tenant and profile catalog.
#[derive(Debug, Clone, Default)]
pub struct KubernetesSandboxConfig {
    pub tenant_namespaces: HashMap<String, String>,
    pub profiles: HashMap<String, KubernetesSandboxProfile>,
}

/// Desired `SandboxClaim`, independent of a Kubernetes client library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxClaimRequest {
    pub namespace: String,
    pub claim_name: String,
    pub warm_pool: String,
    pub workspace_fingerprint: String,
    pub expires_after: Duration,
}

/// Resolved stable sandbox identity returned by the Agent Sandbox controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxClaimHandle {
    pub namespace: String,
    pub claim_name: String,
    pub sandbox_id: String,
    /// In-cluster Service endpoint requested by the operator-owned template.
    pub service_fqdn: String,
}

/// Kubernetes Agent Sandbox CRD lifecycle seam.
pub trait AgentSandboxControlPlane: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn create_or_get_claim(
        &self,
        request: SandboxClaimRequest,
    ) -> BoxFuture<'_, Result<SandboxClaimHandle, Self::Error>>;

    fn get_claim(
        &self,
        request: &SandboxClaimRequest,
    ) -> BoxFuture<'_, Result<Option<SandboxClaimHandle>, Self::Error>>;

    fn delete_claim(&self, request: &SandboxClaimRequest)
    -> BoxFuture<'_, Result<(), Self::Error>>;
}

/// Durable command supervisor reached through the sandbox router data plane.
pub trait SandboxSupervisor: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn start(
        &self,
        sandbox: &SandboxClaimHandle,
        request: StartExecution,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>>;

    fn lookup(
        &self,
        sandbox: &SandboxClaimHandle,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>>;

    fn cancel(
        &self,
        sandbox: &SandboxClaimHandle,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>>;

    fn read_artifact(
        &self,
        execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Artifact, Self::Error>>;
}

#[derive(Debug, Error)]
pub enum KubernetesSandboxError<ControlError, SupervisorError>
where
    ControlError: std::error::Error + Send + Sync + 'static,
    SupervisorError: std::error::Error + Send + Sync + 'static,
{
    #[error("tenant is not assigned a sandbox namespace")]
    UnknownTenant,
    #[error("sandbox profile is not configured")]
    UnknownProfile,
    #[error("sandbox request API version is not supported")]
    UnsupportedApiVersion,
    #[error("sandbox command is empty")]
    EmptyCommand,
    #[error("executable is not allowed by the sandbox profile")]
    ExecutableDenied,
    #[error("sandbox command exceeds its argument count or byte limit")]
    ArgumentsTooLarge,
    #[error("sandbox command argument contains a NUL byte")]
    InvalidArgument,
    #[error("sandbox environment variables are not allowed by the profile")]
    EnvironmentDenied,
    #[error("sandbox environment variables exceed the profile limit")]
    EnvironmentTooLarge,
    #[error("sandbox environment variable name or value is invalid")]
    InvalidEnvironment,
    #[error("sandbox stdin exceeds the profile limit")]
    StdinTooLarge,
    #[error("sandbox working directory must remain beneath /workspace")]
    WorkingDirectoryDenied,
    #[error("sandbox artifact path must be a relative path beneath /workspace")]
    ArtifactPathDenied,
    #[error("sandbox artifact count exceeds the profile limit")]
    TooManyArtifacts,
    #[error("sandbox execution limits exceed the operator profile")]
    LimitsExceeded,
    #[error("sandbox control plane failed: {0}")]
    Control(ControlError),
    #[error("sandbox supervisor failed: {0}")]
    Supervisor(SupervisorError),
    #[error("sandbox supervisor returned a mismatched execution identity")]
    IdentityMismatch,
    #[error("sandbox workspace does not exist")]
    WorkspaceNotFound,
}

pub struct KubernetesSandboxProvider<C, S> {
    control_plane: C,
    supervisor: S,
    config: KubernetesSandboxConfig,
}

type ProviderError<C, S> =
    KubernetesSandboxError<<C as AgentSandboxControlPlane>::Error, <S as SandboxSupervisor>::Error>;

struct PreparedClaim<'a> {
    profile: &'a KubernetesSandboxProfile,
    request: SandboxClaimRequest,
}

impl<C, S> KubernetesSandboxProvider<C, S> {
    pub fn new(control_plane: C, supervisor: S, config: KubernetesSandboxConfig) -> Self {
        Self {
            control_plane,
            supervisor,
            config,
        }
    }
}

impl<C, S> KubernetesSandboxProvider<C, S>
where
    C: AgentSandboxControlPlane,
    S: SandboxSupervisor,
{
    fn prepare(
        &self,
        scope: &AuthorizationScope,
        workspace_id: &WorkspaceId,
        profile_name: &SandboxProfile,
    ) -> Result<PreparedClaim<'_>, ProviderError<C, S>> {
        let namespace = self
            .config
            .tenant_namespaces
            .get(&scope.tenant_id)
            .ok_or(KubernetesSandboxError::UnknownTenant)?;
        let profile = self
            .config
            .profiles
            .get(&profile_name.0)
            .ok_or(KubernetesSandboxError::UnknownProfile)?;
        let workspace_fingerprint = workspace_fingerprint(scope, workspace_id, profile_name);
        let claim_name = format!("art-{}", &workspace_fingerprint[..40]);
        Ok(PreparedClaim {
            profile,
            request: SandboxClaimRequest {
                namespace: namespace.clone(),
                claim_name,
                warm_pool: profile.warm_pool.clone(),
                workspace_fingerprint,
                expires_after: profile.workspace_ttl,
            },
        })
    }

    fn validate_request(
        &self,
        request: &StartExecution,
        profile: &KubernetesSandboxProfile,
    ) -> Result<(), KubernetesSandboxError<C::Error, S::Error>> {
        if request.api_version != crate::SANDBOX_API_VERSION {
            return Err(KubernetesSandboxError::UnsupportedApiVersion);
        }
        let Some(executable) = request.command.argv.first() else {
            return Err(KubernetesSandboxError::EmptyCommand);
        };
        if !profile
            .command_policy
            .allowed_executables
            .contains(executable)
        {
            return Err(KubernetesSandboxError::ExecutableDenied);
        }
        let argument_bytes = request
            .command
            .argv
            .iter()
            .map(String::len)
            .try_fold(0_usize, usize::checked_add)
            .unwrap_or(usize::MAX);
        if request.command.argv.len() > profile.command_policy.max_arguments
            || argument_bytes > profile.command_policy.max_argument_bytes
        {
            return Err(KubernetesSandboxError::ArgumentsTooLarge);
        }
        if request
            .command
            .argv
            .iter()
            .any(|argument| argument.as_bytes().contains(&0))
        {
            return Err(KubernetesSandboxError::InvalidArgument);
        }
        if !profile.command_policy.allow_environment && !request.command.env.is_empty() {
            return Err(KubernetesSandboxError::EnvironmentDenied);
        }
        let environment_bytes = request
            .command
            .env
            .iter()
            .map(|(name, value)| name.len().saturating_add(value.len()))
            .try_fold(0_usize, usize::checked_add)
            .unwrap_or(usize::MAX);
        if request.command.env.len() > profile.command_policy.max_environment_variables
            || environment_bytes > profile.command_policy.max_environment_bytes
        {
            return Err(KubernetesSandboxError::EnvironmentTooLarge);
        }
        if request.command.env.iter().any(|(name, value)| {
            name.is_empty()
                || name.bytes().any(|byte| matches!(byte, b'=' | 0))
                || value.as_bytes().contains(&0)
        }) {
            return Err(KubernetesSandboxError::InvalidEnvironment);
        }
        if request.command.stdin.len() > profile.command_policy.max_stdin_bytes {
            return Err(KubernetesSandboxError::StdinTooLarge);
        }
        if request
            .command
            .cwd
            .as_deref()
            .is_some_and(|path| !safe_workspace_path(path, true))
        {
            return Err(KubernetesSandboxError::WorkingDirectoryDenied);
        }
        if request.command.artifact_paths.len() > profile.command_policy.max_artifacts {
            return Err(KubernetesSandboxError::TooManyArtifacts);
        }
        if request
            .command
            .artifact_paths
            .iter()
            .any(|path| !safe_workspace_path(path, false))
        {
            return Err(KubernetesSandboxError::ArtifactPathDenied);
        }
        let max_timeout =
            u64::try_from(profile.max_execution_timeout.as_millis()).unwrap_or(u64::MAX);
        if request.limits.timeout_millis == 0
            || request.limits.timeout_millis > max_timeout
            || request.limits.max_output_bytes > profile.max_output_bytes
            || request.limits.max_artifact_bytes > profile.max_artifact_bytes
        {
            return Err(KubernetesSandboxError::LimitsExceeded);
        }
        Ok(())
    }

    fn validate_record(
        &self,
        execution: &ScopedExecutionId,
        record: &ExecutionRecord,
    ) -> Result<(), KubernetesSandboxError<C::Error, S::Error>> {
        if record.scope == execution.scope
            && record.workspace_id == execution.workspace_id
            && record.execution_id == execution.execution_id
        {
            Ok(())
        } else {
            Err(KubernetesSandboxError::IdentityMismatch)
        }
    }

    async fn existing_sandbox(
        &self,
        execution: &ScopedExecutionId,
    ) -> Result<SandboxClaimHandle, KubernetesSandboxError<C::Error, S::Error>> {
        let prepared = self.prepare(
            &execution.scope,
            &execution.workspace_id,
            &execution.profile,
        )?;
        self.control_plane
            .get_claim(&prepared.request)
            .await
            .map_err(KubernetesSandboxError::Control)?
            .ok_or(KubernetesSandboxError::WorkspaceNotFound)
    }
}

impl<C, S> SandboxProvider for KubernetesSandboxProvider<C, S>
where
    C: AgentSandboxControlPlane,
    S: SandboxSupervisor,
{
    type Error = KubernetesSandboxError<C::Error, S::Error>;

    fn start(
        &self,
        request: StartExecution,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        Box::pin(async move {
            let prepared = self.prepare(&request.scope, &request.workspace_id, &request.profile)?;
            self.validate_request(&request, prepared.profile)?;
            let sandbox = self
                .control_plane
                .create_or_get_claim(prepared.request)
                .await
                .map_err(KubernetesSandboxError::Control)?;
            let expected = ScopedExecutionId {
                scope: request.scope.clone(),
                workspace_id: request.workspace_id.clone(),
                profile: request.profile.clone(),
                execution_id: request.execution_id.clone(),
            };
            let record = self
                .supervisor
                .start(&sandbox, request)
                .await
                .map_err(KubernetesSandboxError::Supervisor)?;
            self.validate_record(&expected, &record)?;
            Ok(record)
        })
    }

    fn lookup(
        &self,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>> {
        let execution = execution.clone();
        Box::pin(async move {
            let sandbox = match self.existing_sandbox(&execution).await {
                Ok(sandbox) => sandbox,
                Err(KubernetesSandboxError::WorkspaceNotFound) => return Ok(None),
                Err(error) => return Err(error),
            };
            let record = self
                .supervisor
                .lookup(&sandbox, &execution)
                .await
                .map_err(KubernetesSandboxError::Supervisor)?;
            if let Some(record) = &record {
                self.validate_record(&execution, record)?;
            }
            Ok(record)
        })
    }

    fn cancel(
        &self,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        let execution = execution.clone();
        Box::pin(async move {
            let sandbox = self.existing_sandbox(&execution).await?;
            let record = self
                .supervisor
                .cancel(&sandbox, &execution)
                .await
                .map_err(KubernetesSandboxError::Supervisor)?;
            self.validate_record(&execution, &record)?;
            Ok(record)
        })
    }

    fn read_artifact(
        &self,
        execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Artifact, Self::Error>> {
        let execution = execution.clone();
        let artifact_id = artifact_id.to_owned();
        Box::pin(async move {
            self.prepare(
                &execution.scope,
                &execution.workspace_id,
                &execution.profile,
            )?;
            self.supervisor
                .read_artifact(&execution, &artifact_id)
                .await
                .map_err(KubernetesSandboxError::Supervisor)
        })
    }

    fn delete_workspace(
        &self,
        workspace: &ScopedWorkspaceId,
    ) -> BoxFuture<'_, Result<(), Self::Error>> {
        let workspace = workspace.clone();
        Box::pin(async move {
            let prepared = self.prepare(
                &workspace.scope,
                &workspace.workspace_id,
                &workspace.profile,
            )?;
            self.control_plane
                .delete_claim(&prepared.request)
                .await
                .map_err(KubernetesSandboxError::Control)
        })
    }
}

fn workspace_fingerprint(
    scope: &AuthorizationScope,
    workspace_id: &WorkspaceId,
    profile: &SandboxProfile,
) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in [
        scope.tenant_id.as_str(),
        scope.principal_id.as_str(),
        workspace_id.0.as_str(),
        profile.0.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn safe_workspace_path(path: &str, allow_absolute_workspace: bool) -> bool {
    if path.is_empty() || path.as_bytes().contains(&0) {
        return false;
    }
    let candidate = if let Some(relative) = path.strip_prefix("/workspace/") {
        if !allow_absolute_workspace {
            return false;
        }
        relative
    } else if path == "/workspace" {
        return allow_absolute_workspace;
    } else {
        if path.starts_with('/') {
            return false;
        }
        path
    };
    !candidate
        .split('/')
        .any(|component| component.is_empty() || component == "." || component == "..")
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use thiserror::Error;

    use super::*;
    use crate::{ExecutionId, ExecutionState, SandboxCommand, SandboxLimits};

    #[derive(Debug, Error)]
    #[error("fake error")]
    struct FakeError;

    #[derive(Default)]
    struct FakeControlPlane {
        created: Mutex<Vec<SandboxClaimRequest>>,
    }

    impl AgentSandboxControlPlane for FakeControlPlane {
        type Error = FakeError;

        fn create_or_get_claim(
            &self,
            request: SandboxClaimRequest,
        ) -> BoxFuture<'_, Result<SandboxClaimHandle, Self::Error>> {
            self.created.lock().unwrap().push(request.clone());
            Box::pin(async move { Ok(handle(&request)) })
        }

        fn get_claim(
            &self,
            request: &SandboxClaimRequest,
        ) -> BoxFuture<'_, Result<Option<SandboxClaimHandle>, Self::Error>> {
            let handle = handle(request);
            Box::pin(async move { Ok(Some(handle)) })
        }

        fn delete_claim(
            &self,
            _request: &SandboxClaimRequest,
        ) -> BoxFuture<'_, Result<(), Self::Error>> {
            Box::pin(async { Ok(()) })
        }
    }

    #[derive(Default)]
    struct FakeSupervisor;

    impl SandboxSupervisor for FakeSupervisor {
        type Error = FakeError;

        fn start(
            &self,
            sandbox: &SandboxClaimHandle,
            request: StartExecution,
        ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
            let sandbox_id = sandbox.sandbox_id.clone();
            Box::pin(async move { Ok(record(&request, sandbox_id)) })
        }

        fn lookup(
            &self,
            _sandbox: &SandboxClaimHandle,
            _execution: &ScopedExecutionId,
        ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>> {
            Box::pin(async { Ok(None) })
        }

        fn cancel(
            &self,
            _sandbox: &SandboxClaimHandle,
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
    }

    fn handle(request: &SandboxClaimRequest) -> SandboxClaimHandle {
        SandboxClaimHandle {
            namespace: request.namespace.clone(),
            claim_name: request.claim_name.clone(),
            sandbox_id: format!("{}-sandbox", request.claim_name),
            service_fqdn: format!(
                "{}.{}.svc.cluster.local",
                request.claim_name, request.namespace
            ),
        }
    }

    fn config() -> KubernetesSandboxConfig {
        KubernetesSandboxConfig {
            tenant_namespaces: HashMap::from([(
                "tenant-a".to_owned(),
                "tenant-a-sandboxes".to_owned(),
            )]),
            profiles: HashMap::from([(
                "python-deny-egress".to_owned(),
                KubernetesSandboxProfile {
                    warm_pool: "python-deny-egress".to_owned(),
                    workspace_ttl: Duration::from_secs(600),
                    max_execution_timeout: Duration::from_secs(60),
                    max_output_bytes: 1024,
                    max_artifact_bytes: 4096,
                    command_policy: CommandPolicy {
                        allowed_executables: BTreeSet::from(["python".to_owned()]),
                        ..CommandPolicy::default()
                    },
                },
            )]),
        }
    }

    fn request() -> StartExecution {
        StartExecution {
            api_version: crate::SANDBOX_API_VERSION.to_owned(),
            scope: AuthorizationScope {
                tenant_id: "tenant-a".to_owned(),
                principal_id: "principal-a".to_owned(),
            },
            workspace_id: WorkspaceId("workspace-a".to_owned()),
            execution_id: ExecutionId("execution-a".to_owned()),
            profile: SandboxProfile("python-deny-egress".to_owned()),
            command: SandboxCommand {
                argv: vec!["python".to_owned(), "-c".to_owned(), "print(42)".to_owned()],
                cwd: Some("/workspace".to_owned()),
                env: Default::default(),
                stdin: Vec::new(),
                artifact_paths: vec!["result.json".to_owned()],
            },
            limits: SandboxLimits {
                timeout_millis: 10_000,
                max_output_bytes: 1024,
                max_artifact_bytes: 4096,
            },
        }
    }

    fn record(request: &StartExecution, sandbox_id: String) -> ExecutionRecord {
        ExecutionRecord {
            request_fingerprint: request.fingerprint(),
            scope: request.scope.clone(),
            workspace_id: request.workspace_id.clone(),
            execution_id: request.execution_id.clone(),
            provider_sandbox_id: sandbox_id,
            state: ExecutionState::Succeeded,
            exit_code: Some(0),
            stdout: b"42\n".to_vec(),
            stderr: Vec::new(),
            artifacts: Vec::new(),
            failure_code: None,
        }
    }

    #[tokio::test]
    async fn resolves_only_operator_configured_namespace_profile_and_pool() {
        let provider =
            KubernetesSandboxProvider::new(FakeControlPlane::default(), FakeSupervisor, config());
        let record = provider.start(request()).await.unwrap();

        assert_eq!(record.state, ExecutionState::Succeeded);
        let claims = provider.control_plane.created.lock().unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].namespace, "tenant-a-sandboxes");
        assert_eq!(claims[0].warm_pool, "python-deny-egress");
    }

    #[tokio::test]
    async fn rejects_escape_paths_before_creating_a_claim() {
        let provider =
            KubernetesSandboxProvider::new(FakeControlPlane::default(), FakeSupervisor, config());
        let mut request = request();
        request.command.artifact_paths = vec!["../secret".to_owned()];

        assert!(matches!(
            provider.start(request).await,
            Err(KubernetesSandboxError::ArtifactPathDenied)
        ));
        assert!(provider.control_plane.created.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn rejects_client_selected_executables() {
        let provider =
            KubernetesSandboxProvider::new(FakeControlPlane::default(), FakeSupervisor, config());
        let mut request = request();
        request.command.argv[0] = "/bin/sh".to_owned();

        assert!(matches!(
            provider.start(request).await,
            Err(KubernetesSandboxError::ExecutableDenied)
        ));
    }

    #[tokio::test]
    async fn bounds_model_supplied_environment_variables() {
        let mut config = config();
        let policy = &mut config
            .profiles
            .get_mut("python-deny-egress")
            .unwrap()
            .command_policy;
        policy.allow_environment = true;
        policy.max_environment_bytes = 4;
        let provider =
            KubernetesSandboxProvider::new(FakeControlPlane::default(), FakeSupervisor, config);
        let mut request = request();
        request
            .command
            .env
            .insert("TOKEN".to_owned(), "secret".to_owned());

        assert!(matches!(
            provider.start(request).await,
            Err(KubernetesSandboxError::EnvironmentTooLarge)
        ));
    }
}
