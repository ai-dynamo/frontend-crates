// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fenced execution ownership for the external sandbox service.

use std::collections::HashMap;
use std::sync::Mutex;

use dynamo_agent_rt::BoxFuture;
use thiserror::Error;

use crate::{Artifact, ExecutionRecord, ExecutionState, ScopedExecutionId, StartExecution};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionLease {
    pub execution: ScopedExecutionId,
    pub owner_id: String,
    pub fence: u64,
    pub deadline_unix_millis: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredExecution {
    pub request: StartExecution,
    pub record: ExecutionRecord,
    pub lease: Option<ExecutionLease>,
    pub cancel_requested: bool,
}

#[derive(Debug, Clone)]
pub struct ClaimExecution {
    pub request: StartExecution,
    pub provider_sandbox_id: String,
    pub owner_id: String,
    pub now_unix_millis: u64,
    pub lease_millis: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionClaimResult {
    Acquired(ExecutionLease),
    Existing(Box<StoredExecution>),
}

#[derive(Debug, Clone)]
pub struct RenewExecution {
    pub lease: ExecutionLease,
    pub now_unix_millis: u64,
    pub lease_millis: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenewedExecutionLease {
    pub lease: ExecutionLease,
    pub cancel_requested: bool,
}

pub trait ExecutionStore: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn claim(
        &self,
        claim: ClaimExecution,
    ) -> BoxFuture<'_, Result<ExecutionClaimResult, Self::Error>>;

    fn mark_running(
        &self,
        lease: &ExecutionLease,
        now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<StoredExecution, Self::Error>>;

    fn renew(
        &self,
        renewal: RenewExecution,
    ) -> BoxFuture<'_, Result<RenewedExecutionLease, Self::Error>>;

    fn finish(
        &self,
        lease: &ExecutionLease,
        now_unix_millis: u64,
        record: ExecutionRecord,
        artifacts: Vec<Artifact>,
    ) -> BoxFuture<'_, Result<StoredExecution, Self::Error>>;

    fn load(
        &self,
        execution: &ScopedExecutionId,
        now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<Option<StoredExecution>, Self::Error>>;

    fn request_cancel(
        &self,
        execution: &ScopedExecutionId,
        now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<Option<StoredExecution>, Self::Error>>;

    fn read_artifact(
        &self,
        execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Option<Artifact>, Self::Error>>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum InMemoryExecutionStoreError {
    #[error("execution ID is already bound to a different request or sandbox")]
    Conflict,
    #[error("execution lease is stale or expired")]
    StaleLease,
    #[error("execution does not exist")]
    NotFound,
    #[error("execution record identity or fingerprint does not match")]
    IdentityMismatch,
    #[error("execution cannot transition from its current state")]
    InvalidTransition,
    #[error("terminal artifact metadata does not match the supplied snapshots")]
    ArtifactMismatch,
}

#[derive(Default)]
pub struct InMemoryExecutionStore {
    inner: Mutex<InMemoryState>,
}

#[derive(Default)]
struct InMemoryState {
    executions: HashMap<ScopedExecutionId, StoredExecution>,
    artifacts: HashMap<(ScopedExecutionId, String), Artifact>,
}

impl InMemoryExecutionStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ExecutionStore for InMemoryExecutionStore {
    type Error = InMemoryExecutionStoreError;

    fn claim(
        &self,
        claim: ClaimExecution,
    ) -> BoxFuture<'_, Result<ExecutionClaimResult, Self::Error>> {
        Box::pin(async move {
            let execution = scoped(&claim.request);
            let fingerprint = claim.request.fingerprint();
            let mut state = self.inner.lock().expect("execution store lock poisoned");
            if let Some(existing) = state.executions.get_mut(&execution) {
                if existing.record.request_fingerprint != fingerprint
                    || existing.record.provider_sandbox_id != claim.provider_sandbox_id
                {
                    return Err(InMemoryExecutionStoreError::Conflict);
                }
                if existing.record.state == ExecutionState::Running
                    && lease_expired(existing.lease.as_ref(), claim.now_unix_millis)
                {
                    existing.record.state = ExecutionState::OutcomeUnknown;
                    existing.record.failure_code = Some("execution_owner_lost".to_owned());
                    existing.lease = None;
                    return Ok(ExecutionClaimResult::Existing(Box::new(existing.clone())));
                }
                if existing.record.state == ExecutionState::Pending
                    && lease_expired(existing.lease.as_ref(), claim.now_unix_millis)
                {
                    let fence = existing
                        .lease
                        .as_ref()
                        .map_or(1, |lease| lease.fence.saturating_add(1));
                    let lease = new_lease(&execution, &claim, fence);
                    existing.lease = Some(lease.clone());
                    return Ok(ExecutionClaimResult::Acquired(lease));
                }
                return Ok(ExecutionClaimResult::Existing(Box::new(existing.clone())));
            }

            let lease = new_lease(&execution, &claim, 1);
            let record = ExecutionRecord {
                request_fingerprint: fingerprint,
                scope: claim.request.scope.clone(),
                workspace_id: claim.request.workspace_id.clone(),
                execution_id: claim.request.execution_id.clone(),
                provider_sandbox_id: claim.provider_sandbox_id,
                state: ExecutionState::Pending,
                exit_code: None,
                stdout: Vec::new(),
                stderr: Vec::new(),
                artifacts: Vec::new(),
                failure_code: None,
            };
            state.executions.insert(
                execution,
                StoredExecution {
                    request: claim.request,
                    record,
                    lease: Some(lease.clone()),
                    cancel_requested: false,
                },
            );
            Ok(ExecutionClaimResult::Acquired(lease))
        })
    }

    fn mark_running(
        &self,
        lease: &ExecutionLease,
        now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<StoredExecution, Self::Error>> {
        let lease = lease.clone();
        Box::pin(async move {
            let mut state = self.inner.lock().expect("execution store lock poisoned");
            let stored = state
                .executions
                .get_mut(&lease.execution)
                .ok_or(InMemoryExecutionStoreError::NotFound)?;
            validate_lease(stored, &lease, now_unix_millis)?;
            if stored.record.state != ExecutionState::Pending {
                return Err(InMemoryExecutionStoreError::InvalidTransition);
            }
            stored.record.state = ExecutionState::Running;
            Ok(stored.clone())
        })
    }

    fn renew(
        &self,
        renewal: RenewExecution,
    ) -> BoxFuture<'_, Result<RenewedExecutionLease, Self::Error>> {
        Box::pin(async move {
            let mut state = self.inner.lock().expect("execution store lock poisoned");
            let stored = state
                .executions
                .get_mut(&renewal.lease.execution)
                .ok_or(InMemoryExecutionStoreError::NotFound)?;
            validate_lease(stored, &renewal.lease, renewal.now_unix_millis)?;
            if stored.record.state.is_terminal() {
                return Err(InMemoryExecutionStoreError::InvalidTransition);
            }
            let mut lease = renewal.lease;
            lease.deadline_unix_millis =
                renewal.now_unix_millis.saturating_add(renewal.lease_millis);
            stored.lease = Some(lease.clone());
            Ok(RenewedExecutionLease {
                lease,
                cancel_requested: stored.cancel_requested,
            })
        })
    }

    fn finish(
        &self,
        lease: &ExecutionLease,
        now_unix_millis: u64,
        record: ExecutionRecord,
        artifacts: Vec<Artifact>,
    ) -> BoxFuture<'_, Result<StoredExecution, Self::Error>> {
        let lease = lease.clone();
        Box::pin(async move {
            let mut state = self.inner.lock().expect("execution store lock poisoned");
            let stored = state
                .executions
                .get(&lease.execution)
                .cloned()
                .ok_or(InMemoryExecutionStoreError::NotFound)?;
            validate_lease(&stored, &lease, now_unix_millis)?;
            if stored.record.state != ExecutionState::Running || !record.state.is_terminal() {
                return Err(InMemoryExecutionStoreError::InvalidTransition);
            }
            if record.request_fingerprint != stored.record.request_fingerprint
                || record.scope != stored.record.scope
                || record.workspace_id != stored.record.workspace_id
                || record.execution_id != stored.record.execution_id
                || record.provider_sandbox_id != stored.record.provider_sandbox_id
            {
                return Err(InMemoryExecutionStoreError::IdentityMismatch);
            }
            let supplied_metadata = artifacts
                .iter()
                .map(|artifact| artifact.metadata.clone())
                .collect::<Vec<_>>();
            if supplied_metadata != record.artifacts {
                return Err(InMemoryExecutionStoreError::ArtifactMismatch);
            }
            for artifact in artifacts {
                state.artifacts.insert(
                    (
                        lease.execution.clone(),
                        artifact.metadata.artifact_id.clone(),
                    ),
                    artifact,
                );
            }
            let finished = StoredExecution {
                request: stored.request,
                record,
                lease: None,
                cancel_requested: stored.cancel_requested,
            };
            state
                .executions
                .insert(lease.execution.clone(), finished.clone());
            Ok(finished)
        })
    }

    fn load(
        &self,
        execution: &ScopedExecutionId,
        now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<Option<StoredExecution>, Self::Error>> {
        let execution = execution.clone();
        Box::pin(async move {
            let mut state = self.inner.lock().expect("execution store lock poisoned");
            let Some(stored) = state.executions.get_mut(&execution) else {
                return Ok(None);
            };
            expire_running(stored, now_unix_millis);
            Ok(Some(stored.clone()))
        })
    }

    fn request_cancel(
        &self,
        execution: &ScopedExecutionId,
        now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<Option<StoredExecution>, Self::Error>> {
        let execution = execution.clone();
        Box::pin(async move {
            let mut state = self.inner.lock().expect("execution store lock poisoned");
            let Some(stored) = state.executions.get_mut(&execution) else {
                return Ok(None);
            };
            expire_running(stored, now_unix_millis);
            if !stored.record.state.is_terminal() {
                stored.cancel_requested = true;
            }
            Ok(Some(stored.clone()))
        })
    }

    fn read_artifact(
        &self,
        execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Option<Artifact>, Self::Error>> {
        let key = (execution.clone(), artifact_id.to_owned());
        Box::pin(async move {
            let state = self.inner.lock().expect("execution store lock poisoned");
            Ok(state.artifacts.get(&key).cloned())
        })
    }
}

fn scoped(request: &StartExecution) -> ScopedExecutionId {
    ScopedExecutionId {
        scope: request.scope.clone(),
        workspace_id: request.workspace_id.clone(),
        profile: request.profile.clone(),
        execution_id: request.execution_id.clone(),
    }
}

fn new_lease(execution: &ScopedExecutionId, claim: &ClaimExecution, fence: u64) -> ExecutionLease {
    ExecutionLease {
        execution: execution.clone(),
        owner_id: claim.owner_id.clone(),
        fence,
        deadline_unix_millis: claim.now_unix_millis.saturating_add(claim.lease_millis),
    }
}

fn validate_lease(
    stored: &StoredExecution,
    lease: &ExecutionLease,
    now_unix_millis: u64,
) -> Result<(), InMemoryExecutionStoreError> {
    let matches = stored.lease.as_ref().is_some_and(|current| {
        current.execution == lease.execution
            && current.owner_id == lease.owner_id
            && current.fence == lease.fence
            && current.deadline_unix_millis == lease.deadline_unix_millis
    });
    if matches && lease.deadline_unix_millis > now_unix_millis {
        Ok(())
    } else {
        Err(InMemoryExecutionStoreError::StaleLease)
    }
}

fn lease_expired(lease: Option<&ExecutionLease>, now_unix_millis: u64) -> bool {
    lease.is_none_or(|lease| lease.deadline_unix_millis <= now_unix_millis)
}

fn expire_running(stored: &mut StoredExecution, now_unix_millis: u64) {
    if stored.record.state == ExecutionState::Running
        && lease_expired(stored.lease.as_ref(), now_unix_millis)
    {
        stored.record.state = ExecutionState::OutcomeUnknown;
        stored.record.failure_code = Some("execution_owner_lost".to_owned());
        stored.lease = None;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use dynamo_agent_rt::AuthorizationScope;

    use super::*;
    use crate::{
        ExecutionId, SANDBOX_API_VERSION, SandboxCommand, SandboxLimits, SandboxProfile,
        WorkspaceId,
    };

    fn request(code: &str) -> StartExecution {
        StartExecution {
            api_version: SANDBOX_API_VERSION.to_owned(),
            scope: AuthorizationScope {
                tenant_id: "tenant-a".to_owned(),
                principal_id: "principal-a".to_owned(),
            },
            workspace_id: WorkspaceId("workspace-a".to_owned()),
            execution_id: ExecutionId("execution-a".to_owned()),
            profile: SandboxProfile("python-deny-egress".to_owned()),
            command: SandboxCommand {
                argv: vec!["python".to_owned(), "-c".to_owned(), code.to_owned()],
                cwd: Some("/workspace".to_owned()),
                env: BTreeMap::new(),
                stdin: Vec::new(),
                artifact_paths: Vec::new(),
            },
            limits: SandboxLimits {
                timeout_millis: 1_000,
                max_output_bytes: 1_024,
                max_artifact_bytes: 1_024,
            },
        }
    }

    fn claim(request: StartExecution, owner: &str, now: u64) -> ClaimExecution {
        ClaimExecution {
            request,
            provider_sandbox_id: "sandbox-a".to_owned(),
            owner_id: owner.to_owned(),
            now_unix_millis: now,
            lease_millis: 10,
        }
    }

    #[tokio::test]
    async fn expired_pending_execution_can_be_taken_over_with_a_new_fence() {
        let store = InMemoryExecutionStore::new();
        let ExecutionClaimResult::Acquired(first) = store
            .claim(claim(request("print(1)"), "owner-a", 100))
            .await
            .unwrap()
        else {
            panic!("first claim was not acquired")
        };
        let ExecutionClaimResult::Acquired(second) = store
            .claim(claim(request("print(1)"), "owner-b", 110))
            .await
            .unwrap()
        else {
            panic!("expired pending claim was not acquired")
        };
        assert_eq!(first.fence, 1);
        assert_eq!(second.fence, 2);
    }

    #[tokio::test]
    async fn expired_running_execution_becomes_unknown_and_is_never_redispatched() {
        let store = InMemoryExecutionStore::new();
        let ExecutionClaimResult::Acquired(lease) = store
            .claim(claim(request("print(1)"), "owner-a", 100))
            .await
            .unwrap()
        else {
            panic!("claim was not acquired")
        };
        store.mark_running(&lease, 101).await.unwrap();
        let ExecutionClaimResult::Existing(existing) = store
            .claim(claim(request("print(1)"), "owner-b", 110))
            .await
            .unwrap()
        else {
            panic!("running execution was incorrectly reacquired")
        };
        assert_eq!(existing.record.state, ExecutionState::OutcomeUnknown);
        assert_eq!(
            existing.record.failure_code.as_deref(),
            Some("execution_owner_lost")
        );
    }

    #[tokio::test]
    async fn stale_owner_cannot_commit_after_pending_takeover() {
        let store = InMemoryExecutionStore::new();
        let request = request("print(1)");
        let ExecutionClaimResult::Acquired(first) = store
            .claim(claim(request.clone(), "owner-a", 100))
            .await
            .unwrap()
        else {
            panic!("claim was not acquired")
        };
        let ExecutionClaimResult::Acquired(second) = store
            .claim(claim(request.clone(), "owner-b", 110))
            .await
            .unwrap()
        else {
            panic!("takeover was not acquired")
        };
        store.mark_running(&second, 111).await.unwrap();
        let mut record = store
            .load(&second.execution, 111)
            .await
            .unwrap()
            .unwrap()
            .record;
        record.state = ExecutionState::Succeeded;
        assert_eq!(
            store
                .finish(&first, 111, record, Vec::new())
                .await
                .unwrap_err(),
            InMemoryExecutionStoreError::StaleLease
        );
    }

    #[tokio::test]
    async fn terminal_commit_after_lease_expiry_is_rejected_without_a_takeover() {
        let store = InMemoryExecutionStore::new();
        let ExecutionClaimResult::Acquired(lease) = store
            .claim(claim(request("print(1)"), "owner-a", 100))
            .await
            .unwrap()
        else {
            panic!("claim was not acquired")
        };
        store.mark_running(&lease, 101).await.unwrap();
        let mut record = store
            .load(&lease.execution, 105)
            .await
            .unwrap()
            .unwrap()
            .record;
        record.state = ExecutionState::Succeeded;
        assert_eq!(
            store
                .finish(&lease, 110, record, Vec::new())
                .await
                .unwrap_err(),
            InMemoryExecutionStoreError::StaleLease
        );
    }

    #[tokio::test]
    async fn changed_request_cannot_reuse_an_execution_id() {
        let store = InMemoryExecutionStore::new();
        store
            .claim(claim(request("print(1)"), "owner-a", 100))
            .await
            .unwrap();
        assert_eq!(
            store
                .claim(claim(request("print(2)"), "owner-a", 101))
                .await
                .unwrap_err(),
            InMemoryExecutionStoreError::Conflict
        );
    }
}
