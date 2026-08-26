// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! External sandbox execution contracts for `agent-rt`.
//!
//! The provider boundary is intentionally outside the agent runtime and the
//! inference frontend. Implementations own isolation, workspace lifecycle,
//! command execution, artifact storage, and durable outcome lookup.

use std::collections::BTreeMap;

use dynamo_agent_rt::{AuthorizationScope, BoxFuture};
use serde::{Deserialize, Serialize};

/// API version for persisted requests and service payloads.
pub const SANDBOX_API_VERSION: &str = "v1";

/// One provider-independent sandbox workspace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkspaceId(pub String);

/// Idempotency identity for one command execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExecutionId(pub String);

/// Operator-owned execution profile. It selects image, RuntimeClass, network,
/// resources, retention, and command policy; a model never supplies it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SandboxProfile(pub String);

/// An argv-native command. Shell evaluation is never implicit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxCommand {
    pub argv: Vec<String>,
    pub cwd: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub stdin: Vec<u8>,
    /// Exact workspace-relative files to expose as artifacts after completion.
    #[serde(default)]
    pub artifact_paths: Vec<String>,
}

/// Per-execution ceilings. Providers may enforce stricter profile limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxLimits {
    pub timeout_millis: u64,
    pub max_output_bytes: u64,
    pub max_artifact_bytes: u64,
}

/// Idempotent request to run one command in a scoped workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartExecution {
    pub api_version: String,
    pub scope: AuthorizationScope,
    pub workspace_id: WorkspaceId,
    pub execution_id: ExecutionId,
    pub profile: SandboxProfile,
    pub command: SandboxCommand,
    pub limits: SandboxLimits,
}

impl StartExecution {
    /// Stable request fingerprint used to reject execution-ID reuse with a
    /// different command, profile, scope, workspace, or limit.
    pub fn fingerprint(&self) -> String {
        let encoded =
            serde_json::to_vec(self).expect("serializing a sandbox execution request cannot fail");
        blake3::hash(&encoded).to_hex().to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionState {
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
    OutcomeUnknown,
}

impl ExecutionState {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Pending | Self::Running)
    }
}

/// Opaque artifact identity. Paths remain internal to the provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub artifact_id: String,
    pub name: String,
    pub size_bytes: u64,
    pub media_type: Option<String>,
}

/// Durable provider outcome for an execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub request_fingerprint: String,
    pub scope: AuthorizationScope,
    pub workspace_id: WorkspaceId,
    pub execution_id: ExecutionId,
    pub provider_sandbox_id: String,
    pub state: ExecutionState,
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub stdout: Vec<u8>,
    #[serde(default)]
    pub stderr: Vec<u8>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    pub failure_code: Option<String>,
}

/// Provider-neutral scoped lookup key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScopedExecutionId {
    pub scope: AuthorizationScope,
    pub workspace_id: WorkspaceId,
    pub profile: SandboxProfile,
    pub execution_id: ExecutionId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScopedWorkspaceId {
    pub scope: AuthorizationScope,
    pub workspace_id: WorkspaceId,
    pub profile: SandboxProfile,
}

/// Artifact bytes returned only after a scoped lookup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub metadata: ArtifactRef,
    pub bytes: Vec<u8>,
}

/// External sandbox lifecycle and execution seam.
///
/// `start` is create-or-get: the same execution ID and request fingerprint
/// returns the same record; a different fingerprint must fail. `lookup` is the
/// crash-recovery authority used before any redispatch.
pub trait SandboxProvider: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn start(&self, request: StartExecution)
    -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>>;

    fn lookup(
        &self,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>>;

    fn cancel(
        &self,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>>;

    fn read_artifact(
        &self,
        execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Artifact, Self::Error>>;

    fn delete_workspace(
        &self,
        workspace: &ScopedWorkspaceId,
    ) -> BoxFuture<'_, Result<(), Self::Error>>;
}

mod kubernetes;
#[cfg(feature = "kubernetes-client")]
mod kubernetes_client;
#[cfg(feature = "sandboxd-client")]
mod sandboxd;
mod tool_executor;

pub use kubernetes::{
    AgentSandboxControlPlane, CommandPolicy, KubernetesSandboxConfig, KubernetesSandboxError,
    KubernetesSandboxProfile, KubernetesSandboxProvider, SandboxClaimHandle, SandboxClaimRequest,
    SandboxSupervisor,
};
#[cfg(feature = "kubernetes-client")]
pub use kubernetes_client::{
    HttpSandboxSupervisor, HttpSandboxSupervisorConfig, KubeAgentSandboxControlPlane,
    KubeAgentSandboxControlPlaneConfig, KubeControlPlaneError, SandboxSupervisorHttpError,
};
#[cfg(feature = "sandboxd-client")]
pub use sandboxd::{SandboxdClient, SandboxdClientConfig, SandboxdClientError, SandboxdRunOutcome};
pub use tool_executor::{
    SandboxFailurePolicy, SandboxProviderExecutor, SandboxToolError, SandboxToolExecutorConfig,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use dynamo_agent_rt::AuthorizationScope;

    use super::{
        ExecutionId, ExecutionState, SANDBOX_API_VERSION, SandboxCommand, SandboxLimits,
        SandboxProfile, StartExecution, WorkspaceId,
    };

    fn request(profile: &str) -> StartExecution {
        StartExecution {
            api_version: SANDBOX_API_VERSION.to_owned(),
            scope: AuthorizationScope {
                tenant_id: "tenant-a".to_owned(),
                principal_id: "principal-a".to_owned(),
            },
            workspace_id: WorkspaceId("workspace-a".to_owned()),
            execution_id: ExecutionId("execution-a".to_owned()),
            profile: SandboxProfile(profile.to_owned()),
            command: SandboxCommand {
                argv: vec!["python".to_owned(), "-c".to_owned(), "print(42)".to_owned()],
                cwd: Some("/workspace".to_owned()),
                env: BTreeMap::new(),
                stdin: Vec::new(),
                artifact_paths: Vec::new(),
            },
            limits: SandboxLimits {
                timeout_millis: 10_000,
                max_output_bytes: 1024,
                max_artifact_bytes: 4096,
            },
        }
    }

    #[test]
    fn fingerprint_binds_the_operator_profile() {
        assert_ne!(
            request("python-deny-egress").fingerprint(),
            request("python-public-egress").fingerprint()
        );
    }

    #[test]
    fn only_pending_and_running_are_nonterminal() {
        assert!(!ExecutionState::Pending.is_terminal());
        assert!(!ExecutionState::Running.is_terminal());
        assert!(ExecutionState::Succeeded.is_terminal());
        assert!(ExecutionState::OutcomeUnknown.is_terminal());
    }
}
