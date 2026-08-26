// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable supervision for commands executed through an isolated data plane.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dynamo_agent_rt::BoxFuture;
use thiserror::Error;
use tokio::sync::watch;

use crate::{
    Artifact, ArtifactRef, ClaimExecution, ExecutionClaimResult, ExecutionLease, ExecutionRecord,
    ExecutionState, ExecutionStore, RenewExecution, SandboxClaimHandle, SandboxCommand,
    SandboxLimits, SandboxSupervisor, ScopedExecutionId, StartExecution,
};

/// Process and file operations exposed by an isolated sandbox runtime.
pub trait SandboxDataPlane: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn execute(
        &self,
        sandbox: &SandboxClaimHandle,
        command: &SandboxCommand,
        limits: &SandboxLimits,
        cancellation: watch::Receiver<bool>,
    ) -> BoxFuture<'_, Result<SandboxRunOutcome, Self::Error>>;

    fn read_file(
        &self,
        sandbox: &SandboxClaimHandle,
        path: &str,
        max_bytes: u64,
    ) -> BoxFuture<'_, Result<Vec<u8>, Self::Error>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxRunOutcome {
    pub state: ExecutionState,
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub failure_code: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DurableSandboxSupervisorConfig {
    pub owner_id: String,
    pub lease_duration: Duration,
    pub renew_interval: Duration,
    pub cancellation_grace: Duration,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DurableSandboxSupervisorConfigError {
    #[error("sandbox supervisor owner ID must not be empty")]
    EmptyOwner,
    #[error("lease duration, renewal interval, and cancellation grace must be nonzero")]
    ZeroDuration,
    #[error("renewal interval must be shorter than the lease duration")]
    RenewalAfterExpiry,
    #[error("lease duration is too large")]
    LeaseTooLarge,
}

#[derive(Debug, Error)]
pub enum DurableSandboxSupervisorError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[error("sandbox execution store failed: {0}")]
    Store(E),
    #[error("sandbox execution does not exist")]
    NotFound,
    #[error("sandbox execution is bound to a different sandbox or profile")]
    IdentityMismatch,
}

pub struct DurableSandboxSupervisor<D, S> {
    data_plane: Arc<D>,
    store: Arc<S>,
    config: DurableSandboxSupervisorConfig,
    lease_millis: u64,
    active: Arc<Mutex<HashMap<ScopedExecutionId, watch::Sender<bool>>>>,
}

impl<D, S> DurableSandboxSupervisor<D, S> {
    pub fn new(
        data_plane: Arc<D>,
        store: Arc<S>,
        config: DurableSandboxSupervisorConfig,
    ) -> Result<Self, DurableSandboxSupervisorConfigError> {
        if config.owner_id.is_empty() {
            return Err(DurableSandboxSupervisorConfigError::EmptyOwner);
        }
        if config.lease_duration.is_zero()
            || config.renew_interval.is_zero()
            || config.cancellation_grace.is_zero()
        {
            return Err(DurableSandboxSupervisorConfigError::ZeroDuration);
        }
        if config.renew_interval >= config.lease_duration {
            return Err(DurableSandboxSupervisorConfigError::RenewalAfterExpiry);
        }
        let lease_millis = u64::try_from(config.lease_duration.as_millis())
            .map_err(|_| DurableSandboxSupervisorConfigError::LeaseTooLarge)?;
        Ok(Self {
            data_plane,
            store,
            config,
            lease_millis,
            active: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

impl<D, S> DurableSandboxSupervisor<D, S>
where
    D: SandboxDataPlane,
    S: ExecutionStore,
{
    async fn claim(
        &self,
        sandbox: &SandboxClaimHandle,
        request: StartExecution,
    ) -> Result<ExecutionRecord, DurableSandboxSupervisorError<S::Error>> {
        let result = self
            .store
            .claim(ClaimExecution {
                request,
                provider_sandbox_id: sandbox.sandbox_id.clone(),
                owner_id: self.config.owner_id.clone(),
                now_unix_millis: unix_millis(),
                lease_millis: self.lease_millis,
            })
            .await
            .map_err(DurableSandboxSupervisorError::Store)?;
        match result {
            ExecutionClaimResult::Acquired(lease) => self.dispatch(sandbox.clone(), lease).await,
            ExecutionClaimResult::Existing(stored) => {
                validate_stored(sandbox, &stored)?;
                Ok(stored.record)
            }
        }
    }

    async fn dispatch(
        &self,
        sandbox: SandboxClaimHandle,
        lease: ExecutionLease,
    ) -> Result<ExecutionRecord, DurableSandboxSupervisorError<S::Error>> {
        let stored = self
            .store
            .mark_running(&lease, unix_millis())
            .await
            .map_err(DurableSandboxSupervisorError::Store)?;
        validate_stored(&sandbox, &stored)?;

        let (cancel, cancellation) = watch::channel(stored.cancel_requested);
        self.active
            .lock()
            .expect("sandbox active execution lock poisoned")
            .insert(lease.execution.clone(), cancel.clone());

        let data_plane = Arc::clone(&self.data_plane);
        let store = Arc::clone(&self.store);
        let active = Arc::clone(&self.active);
        let renew_interval = self.config.renew_interval;
        let cancellation_grace = self.config.cancellation_grace;
        let lease_millis = self.lease_millis;
        let execution = lease.execution.clone();
        let request = stored.request;
        tokio::spawn(async move {
            if let Err(error) = run_execution(
                data_plane,
                store,
                sandbox,
                request,
                lease,
                cancel,
                cancellation,
                renew_interval,
                lease_millis,
                cancellation_grace,
            )
            .await
            {
                tracing::error!(
                    execution_id = %execution.execution_id.0,
                    error = %error,
                    "sandbox execution supervisor stopped"
                );
            }
            active
                .lock()
                .expect("sandbox active execution lock poisoned")
                .remove(&execution);
        });
        Ok(stored.record)
    }

    async fn resume_expired_pending(
        &self,
        sandbox: &SandboxClaimHandle,
        stored: crate::StoredExecution,
    ) -> Result<ExecutionRecord, DurableSandboxSupervisorError<S::Error>> {
        let expired = stored.record.state == ExecutionState::Pending
            && stored
                .lease
                .as_ref()
                .is_none_or(|lease| lease.deadline_unix_millis <= unix_millis());
        if expired {
            self.claim(sandbox, stored.request).await
        } else {
            Ok(stored.record)
        }
    }
}

impl<D, S> SandboxSupervisor for DurableSandboxSupervisor<D, S>
where
    D: SandboxDataPlane,
    S: ExecutionStore,
{
    type Error = DurableSandboxSupervisorError<S::Error>;

    fn start(
        &self,
        sandbox: &SandboxClaimHandle,
        request: StartExecution,
    ) -> BoxFuture<'_, Result<ExecutionRecord, Self::Error>> {
        let sandbox = sandbox.clone();
        Box::pin(async move { self.claim(&sandbox, request).await })
    }

    fn lookup(
        &self,
        sandbox: &SandboxClaimHandle,
        execution: &ScopedExecutionId,
    ) -> BoxFuture<'_, Result<Option<ExecutionRecord>, Self::Error>> {
        let sandbox = sandbox.clone();
        let execution = execution.clone();
        Box::pin(async move {
            let Some(stored) = self
                .store
                .load(&execution, unix_millis())
                .await
                .map_err(DurableSandboxSupervisorError::Store)?
            else {
                return Ok(None);
            };
            validate_stored(&sandbox, &stored)?;
            self.resume_expired_pending(&sandbox, stored)
                .await
                .map(Some)
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
            let stored = self
                .store
                .request_cancel(&execution, unix_millis())
                .await
                .map_err(DurableSandboxSupervisorError::Store)?
                .ok_or(DurableSandboxSupervisorError::NotFound)?;
            validate_stored(&sandbox, &stored)?;
            if let Some(sender) = self
                .active
                .lock()
                .expect("sandbox active execution lock poisoned")
                .get(&execution)
            {
                let _ = sender.send(true);
            }
            Ok(stored.record)
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
            self.store
                .read_artifact(&execution, &artifact_id)
                .await
                .map_err(DurableSandboxSupervisorError::Store)?
                .ok_or(DurableSandboxSupervisorError::NotFound)
        })
    }
}

#[derive(Debug, Error)]
enum BackgroundError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[error("execution store failed: {0}")]
    Store(E),
}

#[allow(clippy::too_many_arguments)]
async fn run_execution<D, S>(
    data_plane: Arc<D>,
    store: Arc<S>,
    sandbox: SandboxClaimHandle,
    request: StartExecution,
    mut lease: ExecutionLease,
    cancel: watch::Sender<bool>,
    cancellation: watch::Receiver<bool>,
    renew_interval: Duration,
    lease_millis: u64,
    cancellation_grace: Duration,
) -> Result<(), BackgroundError<S::Error>>
where
    D: SandboxDataPlane,
    S: ExecutionStore,
{
    if *cancellation.borrow() {
        let record = terminal_record(
            &request,
            &sandbox,
            SandboxRunOutcome {
                state: ExecutionState::Cancelled,
                exit_code: None,
                stdout: Vec::new(),
                stderr: Vec::new(),
                failure_code: Some("cancelled_before_dispatch".to_owned()),
            },
            Vec::new(),
        );
        store
            .finish(&lease, unix_millis(), record, Vec::new())
            .await
            .map_err(BackgroundError::Store)?;
        return Ok(());
    }

    let run = data_plane.execute(&sandbox, &request.command, &request.limits, cancellation);
    tokio::pin!(run);
    let mut interval =
        tokio::time::interval_at(tokio::time::Instant::now() + renew_interval, renew_interval);
    let outcome = loop {
        tokio::select! {
            result = &mut run => {
                break match result {
                    Ok(outcome) => outcome,
                    Err(_) => SandboxRunOutcome {
                        state: ExecutionState::OutcomeUnknown,
                        exit_code: None,
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                        failure_code: Some("sandbox_data_plane_error".to_owned()),
                    },
                };
            }
            _ = interval.tick() => {
                let renewed = match store.renew(RenewExecution {
                    lease: lease.clone(),
                    now_unix_millis: unix_millis(),
                    lease_millis,
                }).await {
                    Ok(renewed) => renewed,
                    Err(error) => {
                        let _ = cancel.send(true);
                        let _ = tokio::time::timeout(cancellation_grace, &mut run).await;
                        return Err(BackgroundError::Store(error));
                    }
                };
                lease = renewed.lease;
                if renewed.cancel_requested {
                    let _ = cancel.send(true);
                }
            }
        }
    };
    let outcome = normalize_outcome(outcome, request.limits.max_output_bytes);
    lease = renew_lease(store.as_ref(), lease, lease_millis).await?;

    let (outcome, artifacts) = if outcome.state == ExecutionState::Succeeded {
        let capture = capture_artifacts(data_plane.as_ref(), &sandbox, &request);
        tokio::pin!(capture);
        let captured = loop {
            tokio::select! {
                result = &mut capture => break result,
                _ = interval.tick() => {
                    lease = renew_lease(store.as_ref(), lease, lease_millis).await?;
                }
            }
        };
        match captured {
            Ok(artifacts) => (outcome, artifacts),
            Err(()) => (
                SandboxRunOutcome {
                    state: ExecutionState::Failed,
                    exit_code: outcome.exit_code,
                    stdout: outcome.stdout,
                    stderr: outcome.stderr,
                    failure_code: Some("artifact_capture_failed".to_owned()),
                },
                Vec::new(),
            ),
        }
    } else {
        (outcome, Vec::new())
    };
    lease = renew_lease(store.as_ref(), lease, lease_millis).await?;
    let metadata = artifacts
        .iter()
        .map(|artifact| artifact.metadata.clone())
        .collect();
    let record = terminal_record(&request, &sandbox, outcome, metadata);
    store
        .finish(&lease, unix_millis(), record, artifacts)
        .await
        .map_err(BackgroundError::Store)?;
    Ok(())
}

async fn renew_lease<S: ExecutionStore>(
    store: &S,
    lease: ExecutionLease,
    lease_millis: u64,
) -> Result<ExecutionLease, BackgroundError<S::Error>> {
    store
        .renew(RenewExecution {
            lease,
            now_unix_millis: unix_millis(),
            lease_millis,
        })
        .await
        .map(|renewed| renewed.lease)
        .map_err(BackgroundError::Store)
}

async fn capture_artifacts<D: SandboxDataPlane>(
    data_plane: &D,
    sandbox: &SandboxClaimHandle,
    request: &StartExecution,
) -> Result<Vec<Artifact>, ()> {
    let mut remaining = request.limits.max_artifact_bytes;
    let mut artifacts = Vec::with_capacity(request.command.artifact_paths.len());
    for name in &request.command.artifact_paths {
        let path = format!("/workspace/{name}");
        let bytes = data_plane
            .read_file(sandbox, &path, remaining)
            .await
            .map_err(|_| ())?;
        let size_bytes = u64::try_from(bytes.len()).map_err(|_| ())?;
        remaining = remaining.checked_sub(size_bytes).ok_or(())?;
        let artifact_id = format!("art_{}", blake3::hash(&bytes).to_hex());
        artifacts.push(Artifact {
            metadata: ArtifactRef {
                artifact_id,
                name: name.clone(),
                size_bytes,
                media_type: None,
            },
            bytes,
        });
    }
    Ok(artifacts)
}

fn terminal_record(
    request: &StartExecution,
    sandbox: &SandboxClaimHandle,
    outcome: SandboxRunOutcome,
    artifacts: Vec<ArtifactRef>,
) -> ExecutionRecord {
    ExecutionRecord {
        request_fingerprint: request.fingerprint(),
        scope: request.scope.clone(),
        workspace_id: request.workspace_id.clone(),
        execution_id: request.execution_id.clone(),
        provider_sandbox_id: sandbox.sandbox_id.clone(),
        state: outcome.state,
        exit_code: outcome.exit_code,
        stdout: outcome.stdout,
        stderr: outcome.stderr,
        artifacts,
        failure_code: outcome.failure_code,
    }
}

fn normalize_outcome(mut outcome: SandboxRunOutcome, max_output_bytes: u64) -> SandboxRunOutcome {
    if !outcome.state.is_terminal() {
        return SandboxRunOutcome {
            state: ExecutionState::OutcomeUnknown,
            exit_code: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            failure_code: Some("invalid_data_plane_outcome".to_owned()),
        };
    }
    let max = usize::try_from(max_output_bytes).unwrap_or(usize::MAX);
    if outcome.stdout.len().saturating_add(outcome.stderr.len()) > max {
        let stdout_len = outcome.stdout.len().min(max);
        outcome.stdout.truncate(stdout_len);
        outcome.stderr.truncate(max.saturating_sub(stdout_len));
        outcome.state = ExecutionState::Failed;
        outcome.exit_code = None;
        outcome.failure_code = Some("output_limit_exceeded".to_owned());
    }
    outcome
}

fn validate_stored<E>(
    sandbox: &SandboxClaimHandle,
    stored: &crate::StoredExecution,
) -> Result<(), DurableSandboxSupervisorError<E>>
where
    E: std::error::Error + Send + Sync + 'static,
{
    if stored.record.provider_sandbox_id == sandbox.sandbox_id
        && stored.record.request_fingerprint == stored.request.fingerprint()
        && stored.record.scope == stored.request.scope
        && stored.record.workspace_id == stored.request.workspace_id
        && stored.record.execution_id == stored.request.execution_id
    {
        Ok(())
    } else {
        Err(DurableSandboxSupervisorError::IdentityMismatch)
    }
}

fn unix_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}

#[cfg(feature = "sandboxd-client")]
impl SandboxDataPlane for crate::SandboxdClient {
    type Error = crate::SandboxdClientError;

    fn execute(
        &self,
        sandbox: &SandboxClaimHandle,
        command: &SandboxCommand,
        limits: &SandboxLimits,
        cancellation: watch::Receiver<bool>,
    ) -> BoxFuture<'_, Result<SandboxRunOutcome, Self::Error>> {
        let sandbox = sandbox.clone();
        let command = command.clone();
        let limits = limits.clone();
        Box::pin(async move {
            let outcome = self
                .execute(&sandbox, &command, &limits, cancellation)
                .await?;
            Ok(SandboxRunOutcome {
                state: outcome.state,
                exit_code: outcome.exit_code,
                stdout: outcome.stdout,
                stderr: outcome.stderr,
                failure_code: outcome.failure_code,
            })
        })
    }

    fn read_file(
        &self,
        sandbox: &SandboxClaimHandle,
        path: &str,
        max_bytes: u64,
    ) -> BoxFuture<'_, Result<Vec<u8>, Self::Error>> {
        let sandbox = sandbox.clone();
        let path = path.to_owned();
        Box::pin(async move { self.read_file(&sandbox, &path, max_bytes).await })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dynamo_agent_rt::AuthorizationScope;
    use thiserror::Error;

    use super::*;
    use crate::{
        ExecutionId, InMemoryExecutionStore, SANDBOX_API_VERSION, SandboxProfile, WorkspaceId,
    };

    #[derive(Clone, Copy)]
    enum Mode {
        Success,
        AwaitCancellation,
    }

    struct FakeDataPlane {
        mode: Mode,
        calls: AtomicUsize,
        files: HashMap<String, Vec<u8>>,
    }

    #[derive(Debug, Error)]
    #[error("fake data plane error")]
    struct FakeError;

    impl SandboxDataPlane for FakeDataPlane {
        type Error = FakeError;

        fn execute(
            &self,
            _sandbox: &SandboxClaimHandle,
            _command: &SandboxCommand,
            _limits: &SandboxLimits,
            mut cancellation: watch::Receiver<bool>,
        ) -> BoxFuture<'_, Result<SandboxRunOutcome, Self::Error>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                match self.mode {
                    Mode::Success => Ok(SandboxRunOutcome {
                        state: ExecutionState::Succeeded,
                        exit_code: Some(0),
                        stdout: b"42\n".to_vec(),
                        stderr: Vec::new(),
                        failure_code: None,
                    }),
                    Mode::AwaitCancellation => {
                        while !*cancellation.borrow() {
                            cancellation.changed().await.map_err(|_| FakeError)?;
                        }
                        Ok(SandboxRunOutcome {
                            state: ExecutionState::Cancelled,
                            exit_code: None,
                            stdout: Vec::new(),
                            stderr: Vec::new(),
                            failure_code: Some("cancelled".to_owned()),
                        })
                    }
                }
            })
        }

        fn read_file(
            &self,
            _sandbox: &SandboxClaimHandle,
            path: &str,
            max_bytes: u64,
        ) -> BoxFuture<'_, Result<Vec<u8>, Self::Error>> {
            let value = self.files.get(path).cloned();
            Box::pin(async move {
                let bytes = value.ok_or(FakeError)?;
                if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
                    return Err(FakeError);
                }
                Ok(bytes)
            })
        }
    }

    type Supervisor = DurableSandboxSupervisor<FakeDataPlane, InMemoryExecutionStore>;

    fn supervisor(mode: Mode) -> (Supervisor, Arc<FakeDataPlane>) {
        let data_plane = Arc::new(FakeDataPlane {
            mode,
            calls: AtomicUsize::new(0),
            files: HashMap::from([("/workspace/result.txt".to_owned(), b"done".to_vec())]),
        });
        let supervisor = DurableSandboxSupervisor::new(
            Arc::clone(&data_plane),
            Arc::new(InMemoryExecutionStore::new()),
            DurableSandboxSupervisorConfig {
                owner_id: "replica-a".to_owned(),
                lease_duration: Duration::from_secs(1),
                renew_interval: Duration::from_millis(100),
                cancellation_grace: Duration::from_millis(100),
            },
        )
        .unwrap();
        (supervisor, data_plane)
    }

    fn sandbox() -> SandboxClaimHandle {
        SandboxClaimHandle {
            namespace: "tenant-a-sandboxes".to_owned(),
            claim_name: "claim-a".to_owned(),
            sandbox_id: "sandbox-a".to_owned(),
            service_fqdn: "claim-a.tenant-a-sandboxes.svc.cluster.local".to_owned(),
        }
    }

    fn request() -> StartExecution {
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
                argv: vec!["python".to_owned(), "-c".to_owned(), "print(42)".to_owned()],
                cwd: Some("/workspace".to_owned()),
                env: BTreeMap::new(),
                stdin: Vec::new(),
                artifact_paths: vec!["result.txt".to_owned()],
            },
            limits: SandboxLimits {
                timeout_millis: 5_000,
                max_output_bytes: 1_024,
                max_artifact_bytes: 1_024,
            },
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

    async fn terminal(supervisor: &Supervisor, execution: &ScopedExecutionId) -> ExecutionRecord {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let record = supervisor
                    .lookup(&sandbox(), execution)
                    .await
                    .unwrap()
                    .unwrap();
                if record.state.is_terminal() {
                    return record;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("execution did not become terminal")
    }

    #[tokio::test]
    async fn persists_terminal_output_and_artifacts_before_publishing_success() {
        let (supervisor, _) = supervisor(Mode::Success);
        let request = request();
        let execution = scoped(&request);
        supervisor.start(&sandbox(), request).await.unwrap();
        let record = terminal(&supervisor, &execution).await;
        assert_eq!(record.state, ExecutionState::Succeeded);
        assert_eq!(record.stdout, b"42\n");
        let artifact = supervisor
            .read_artifact(&execution, &record.artifacts[0].artifact_id)
            .await
            .unwrap();
        assert_eq!(artifact.bytes, b"done");
    }

    #[tokio::test]
    async fn repeated_start_returns_the_durable_record_without_redispatch() {
        let (supervisor, data_plane) = supervisor(Mode::Success);
        let request = request();
        let execution = scoped(&request);
        supervisor.start(&sandbox(), request.clone()).await.unwrap();
        terminal(&supervisor, &execution).await;
        let replay = supervisor.start(&sandbox(), request).await.unwrap();
        assert_eq!(replay.state, ExecutionState::Succeeded);
        assert_eq!(data_plane.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancellation_reaches_the_data_plane_and_is_committed() {
        let (supervisor, _) = supervisor(Mode::AwaitCancellation);
        let request = request();
        let execution = scoped(&request);
        supervisor.start(&sandbox(), request).await.unwrap();
        supervisor.cancel(&sandbox(), &execution).await.unwrap();
        let record = terminal(&supervisor, &execution).await;
        assert_eq!(record.state, ExecutionState::Cancelled);
    }
}
