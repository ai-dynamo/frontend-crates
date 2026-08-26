// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-replica execution storage using PostgreSQL row locks and database time.

use std::collections::HashSet;

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod, Runtime};
use dynamo_agent_rt::BoxFuture;
use thiserror::Error;
use tokio_postgres::{NoTls, Row, Transaction};

use crate::{
    Artifact, ClaimExecution, ExecutionClaimResult, ExecutionLease, ExecutionRecord,
    ExecutionState, ExecutionStore, RenewExecution, RenewedExecutionLease, ScopedExecutionId,
    StoredExecution,
};

const MIGRATION: &str = include_str!("../migrations/postgres/0001_sandbox_executions.sql");

#[derive(Debug, Error)]
pub enum PostgresExecutionStoreError {
    #[error("PostgreSQL failed: {0}")]
    Database(#[from] tokio_postgres::Error),
    #[error("PostgreSQL pool failed: {0}")]
    Pool(#[from] deadpool_postgres::PoolError),
    #[error("PostgreSQL pool configuration failed: {0}")]
    PoolBuild(#[from] deadpool_postgres::BuildError),
    #[error("persisted sandbox JSON failed: {0}")]
    Json(#[from] serde_json::Error),
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
    #[error("persisted sandbox execution is invalid: {0}")]
    InvalidPersisted(&'static str),
    #[error("sandbox execution numeric value exceeds the PostgreSQL representation")]
    NumericOverflow,
}

impl ExecutionStore for PostgresExecutionStore {
    type Error = PostgresExecutionStoreError;

    fn claim(
        &self,
        claim: ClaimExecution,
    ) -> BoxFuture<'_, Result<ExecutionClaimResult, Self::Error>> {
        Box::pin(async move {
            let mut client = self.pool.get().await?;
            let transaction = client.transaction().await?;
            let execution = scoped(&claim.request);
            lock_execution(&transaction, &execution).await?;
            let now = database_now(&transaction).await?;
            let deadline = add_millis(now, claim.lease_millis)?;
            let existing = load_locked(&transaction, &execution).await?;

            if let Some(mut existing) = existing {
                if existing.record.request_fingerprint != claim.request.fingerprint()
                    || existing.record.provider_sandbox_id != claim.provider_sandbox_id
                {
                    return Err(PostgresExecutionStoreError::Conflict);
                }
                if existing.record.state == ExecutionState::Running
                    && lease_expired(existing.lease.as_ref(), now)
                {
                    existing.record.state = ExecutionState::OutcomeUnknown;
                    existing.record.failure_code = Some("execution_owner_lost".to_owned());
                    existing.lease = None;
                    persist_record(&transaction, &execution, &existing).await?;
                    transaction.commit().await?;
                    return Ok(ExecutionClaimResult::Existing(Box::new(existing)));
                }
                if existing.record.state == ExecutionState::Pending
                    && lease_expired(existing.lease.as_ref(), now)
                {
                    let fence = existing
                        .lease
                        .as_ref()
                        .map_or(1, |lease| lease.fence.saturating_add(1));
                    let lease = ExecutionLease {
                        execution: execution.clone(),
                        owner_id: claim.owner_id,
                        fence,
                        deadline_unix_millis: deadline,
                    };
                    existing.lease = Some(lease.clone());
                    persist_record(&transaction, &execution, &existing).await?;
                    transaction.commit().await?;
                    return Ok(ExecutionClaimResult::Acquired(lease));
                }
                transaction.commit().await?;
                return Ok(ExecutionClaimResult::Existing(Box::new(existing)));
            }

            let request_fingerprint = claim.request.fingerprint();
            let lease = ExecutionLease {
                execution: execution.clone(),
                owner_id: claim.owner_id,
                fence: 1,
                deadline_unix_millis: deadline,
            };
            let record = ExecutionRecord {
                request_fingerprint,
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
            let stored = StoredExecution {
                request: claim.request,
                record,
                lease: Some(lease.clone()),
                cancel_requested: false,
            };
            insert_execution(&transaction, &execution, &stored).await?;
            transaction.commit().await?;
            Ok(ExecutionClaimResult::Acquired(lease))
        })
    }

    fn mark_running(
        &self,
        lease: &ExecutionLease,
        _now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<StoredExecution, Self::Error>> {
        let lease = lease.clone();
        Box::pin(async move {
            let mut client = self.pool.get().await?;
            let transaction = client.transaction().await?;
            let now = database_now(&transaction).await?;
            let mut stored = load_locked(&transaction, &lease.execution)
                .await?
                .ok_or(PostgresExecutionStoreError::NotFound)?;
            validate_lease(&stored, &lease, now)?;
            if stored.record.state != ExecutionState::Pending {
                return Err(PostgresExecutionStoreError::InvalidTransition);
            }
            stored.record.state = ExecutionState::Running;
            persist_record(&transaction, &lease.execution, &stored).await?;
            transaction.commit().await?;
            Ok(stored)
        })
    }

    fn renew(
        &self,
        renewal: RenewExecution,
    ) -> BoxFuture<'_, Result<RenewedExecutionLease, Self::Error>> {
        Box::pin(async move {
            let mut client = self.pool.get().await?;
            let transaction = client.transaction().await?;
            let now = database_now(&transaction).await?;
            let mut stored = load_locked(&transaction, &renewal.lease.execution)
                .await?
                .ok_or(PostgresExecutionStoreError::NotFound)?;
            validate_lease(&stored, &renewal.lease, now)?;
            if stored.record.state.is_terminal() {
                return Err(PostgresExecutionStoreError::InvalidTransition);
            }
            let mut lease = renewal.lease;
            lease.deadline_unix_millis = add_millis(now, renewal.lease_millis)?;
            stored.lease = Some(lease.clone());
            persist_record(&transaction, &lease.execution, &stored).await?;
            transaction.commit().await?;
            Ok(RenewedExecutionLease {
                lease,
                cancel_requested: stored.cancel_requested,
            })
        })
    }

    fn finish(
        &self,
        lease: &ExecutionLease,
        _now_unix_millis: u64,
        record: ExecutionRecord,
        artifacts: Vec<Artifact>,
    ) -> BoxFuture<'_, Result<StoredExecution, Self::Error>> {
        let lease = lease.clone();
        Box::pin(async move {
            let mut client = self.pool.get().await?;
            let transaction = client.transaction().await?;
            let now = database_now(&transaction).await?;
            let stored = load_locked(&transaction, &lease.execution)
                .await?
                .ok_or(PostgresExecutionStoreError::NotFound)?;
            validate_lease(&stored, &lease, now)?;
            if stored.record.state != ExecutionState::Running || !record.state.is_terminal() {
                return Err(PostgresExecutionStoreError::InvalidTransition);
            }
            if record.request_fingerprint != stored.record.request_fingerprint
                || record.scope != stored.record.scope
                || record.workspace_id != stored.record.workspace_id
                || record.execution_id != stored.record.execution_id
                || record.provider_sandbox_id != stored.record.provider_sandbox_id
            {
                return Err(PostgresExecutionStoreError::IdentityMismatch);
            }
            let supplied_metadata = artifacts
                .iter()
                .map(|artifact| artifact.metadata.clone())
                .collect::<Vec<_>>();
            let unique_ids = artifacts
                .iter()
                .map(|artifact| artifact.metadata.artifact_id.as_str())
                .collect::<HashSet<_>>();
            if supplied_metadata != record.artifacts || unique_ids.len() != artifacts.len() {
                return Err(PostgresExecutionStoreError::ArtifactMismatch);
            }
            for artifact in &artifacts {
                insert_artifact(&transaction, &lease.execution, artifact).await?;
            }
            let finished = StoredExecution {
                request: stored.request,
                record,
                lease: None,
                cancel_requested: stored.cancel_requested,
            };
            persist_record(&transaction, &lease.execution, &finished).await?;
            transaction.commit().await?;
            Ok(finished)
        })
    }

    fn load(
        &self,
        execution: &ScopedExecutionId,
        _now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<Option<StoredExecution>, Self::Error>> {
        let execution = execution.clone();
        Box::pin(async move {
            let mut client = self.pool.get().await?;
            let transaction = client.transaction().await?;
            let Some(mut stored) = load_locked(&transaction, &execution).await? else {
                transaction.commit().await?;
                return Ok(None);
            };
            let now = database_now(&transaction).await?;
            if expire_running(&mut stored, now) {
                persist_record(&transaction, &execution, &stored).await?;
            }
            transaction.commit().await?;
            Ok(Some(stored))
        })
    }

    fn request_cancel(
        &self,
        execution: &ScopedExecutionId,
        _now_unix_millis: u64,
    ) -> BoxFuture<'_, Result<Option<StoredExecution>, Self::Error>> {
        let execution = execution.clone();
        Box::pin(async move {
            let mut client = self.pool.get().await?;
            let transaction = client.transaction().await?;
            let Some(mut stored) = load_locked(&transaction, &execution).await? else {
                transaction.commit().await?;
                return Ok(None);
            };
            let now = database_now(&transaction).await?;
            expire_running(&mut stored, now);
            if !stored.record.state.is_terminal() {
                stored.cancel_requested = true;
            }
            persist_record(&transaction, &execution, &stored).await?;
            transaction.commit().await?;
            Ok(Some(stored))
        })
    }

    fn read_artifact(
        &self,
        execution: &ScopedExecutionId,
        artifact_id: &str,
    ) -> BoxFuture<'_, Result<Option<Artifact>, Self::Error>> {
        let execution = execution.clone();
        let artifact_id = artifact_id.to_owned();
        Box::pin(async move {
            let client = self.pool.get().await?;
            let row = client
                .query_opt(
            "SELECT metadata_json::TEXT AS metadata_json, bytes FROM agent_sandbox_artifacts
                     WHERE tenant_id = $1 AND principal_id = $2 AND workspace_id = $3
                       AND profile = $4 AND execution_id = $5 AND artifact_id = $6",
                    &[
                        &execution.scope.tenant_id,
                        &execution.scope.principal_id,
                        &execution.workspace_id.0,
                        &execution.profile.0,
                        &execution.execution_id.0,
                        &artifact_id,
                    ],
                )
                .await?;
            row.map(decode_artifact).transpose()
        })
    }
}

fn scoped(request: &crate::StartExecution) -> ScopedExecutionId {
    ScopedExecutionId {
        scope: request.scope.clone(),
        workspace_id: request.workspace_id.clone(),
        profile: request.profile.clone(),
        execution_id: request.execution_id.clone(),
    }
}

async fn lock_execution(
    transaction: &Transaction<'_>,
    execution: &ScopedExecutionId,
) -> Result<(), PostgresExecutionStoreError> {
    let key = format!(
        "{}\0{}\0{}\0{}\0{}",
        execution.scope.tenant_id,
        execution.scope.principal_id,
        execution.workspace_id.0,
        execution.profile.0,
        execution.execution_id.0
    );
    transaction
        .query_one(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            &[&key],
        )
        .await?;
    Ok(())
}

async fn load_locked(
    transaction: &Transaction<'_>,
    execution: &ScopedExecutionId,
) -> Result<Option<StoredExecution>, PostgresExecutionStoreError> {
    transaction
        .query_opt(
            "SELECT request_json::TEXT AS request_json, record_json::TEXT AS record_json,
                    state, lease_owner_id, lease_fence,
                    lease_deadline_unix_millis, cancel_requested
             FROM agent_sandbox_executions
             WHERE tenant_id = $1 AND principal_id = $2 AND workspace_id = $3
               AND profile = $4 AND execution_id = $5
             FOR UPDATE",
            &[
                &execution.scope.tenant_id,
                &execution.scope.principal_id,
                &execution.workspace_id.0,
                &execution.profile.0,
                &execution.execution_id.0,
            ],
        )
        .await?
        .map(|row| decode_stored(execution, row))
        .transpose()
}

fn decode_stored(
    execution: &ScopedExecutionId,
    row: Row,
) -> Result<StoredExecution, PostgresExecutionStoreError> {
    let request = serde_json::from_str(row.get("request_json"))?;
    let record: ExecutionRecord = serde_json::from_str(row.get("record_json"))?;
    let state: String = row.get("state");
    if state_name(record.state) != state {
        return Err(PostgresExecutionStoreError::InvalidPersisted(
            "record state does not match indexed state",
        ));
    }
    let owner_id: Option<String> = row.get("lease_owner_id");
    let fence: Option<i64> = row.get("lease_fence");
    let deadline: Option<i64> = row.get("lease_deadline_unix_millis");
    let lease = match (owner_id, fence, deadline) {
        (None, None, None) => None,
        (Some(owner_id), Some(fence), Some(deadline)) => Some(ExecutionLease {
            execution: execution.clone(),
            owner_id,
            fence: to_u64(fence)?,
            deadline_unix_millis: to_u64(deadline)?,
        }),
        _ => {
            return Err(PostgresExecutionStoreError::InvalidPersisted(
                "partial execution lease",
            ));
        }
    };
    Ok(StoredExecution {
        request,
        record,
        lease,
        cancel_requested: row.get("cancel_requested"),
    })
}

async fn insert_execution(
    transaction: &Transaction<'_>,
    execution: &ScopedExecutionId,
    stored: &StoredExecution,
) -> Result<(), PostgresExecutionStoreError> {
    let request_json = serde_json::to_string(&stored.request)?;
    let record_json = serde_json::to_string(&stored.record)?;
    let lease = lease_columns(stored.lease.as_ref())?;
    transaction
        .execute(
            "INSERT INTO agent_sandbox_executions (
                tenant_id, principal_id, workspace_id, profile, execution_id,
                request_fingerprint, provider_sandbox_id, request_json, record_json, state,
                lease_owner_id, lease_fence, lease_deadline_unix_millis, cancel_requested
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::JSONB, $9::JSONB, $10, $11, $12, $13, $14)",
            &[
                &execution.scope.tenant_id,
                &execution.scope.principal_id,
                &execution.workspace_id.0,
                &execution.profile.0,
                &execution.execution_id.0,
                &stored.record.request_fingerprint,
                &stored.record.provider_sandbox_id,
                &request_json,
                &record_json,
                &state_name(stored.record.state),
                &lease.owner,
                &lease.fence,
                &lease.deadline,
                &stored.cancel_requested,
            ],
        )
        .await?;
    Ok(())
}

async fn persist_record(
    transaction: &Transaction<'_>,
    execution: &ScopedExecutionId,
    stored: &StoredExecution,
) -> Result<(), PostgresExecutionStoreError> {
    let record_json = serde_json::to_string(&stored.record)?;
    let lease = lease_columns(stored.lease.as_ref())?;
    let updated = transaction
        .execute(
            "UPDATE agent_sandbox_executions
             SET record_json = $1::JSONB, state = $2, lease_owner_id = $3, lease_fence = $4,
                 lease_deadline_unix_millis = $5, cancel_requested = $6,
                 updated_at = clock_timestamp()
             WHERE tenant_id = $7 AND principal_id = $8 AND workspace_id = $9
               AND profile = $10 AND execution_id = $11",
            &[
                &record_json,
                &state_name(stored.record.state),
                &lease.owner,
                &lease.fence,
                &lease.deadline,
                &stored.cancel_requested,
                &execution.scope.tenant_id,
                &execution.scope.principal_id,
                &execution.workspace_id.0,
                &execution.profile.0,
                &execution.execution_id.0,
            ],
        )
        .await?;
    if updated != 1 {
        return Err(PostgresExecutionStoreError::NotFound);
    }
    Ok(())
}

async fn insert_artifact(
    transaction: &Transaction<'_>,
    execution: &ScopedExecutionId,
    artifact: &Artifact,
) -> Result<(), PostgresExecutionStoreError> {
    let metadata_json = serde_json::to_string(&artifact.metadata)?;
    transaction
        .execute(
            "INSERT INTO agent_sandbox_artifacts (
                tenant_id, principal_id, workspace_id, profile, execution_id,
                artifact_id, metadata_json, bytes
             ) VALUES ($1, $2, $3, $4, $5, $6, $7::JSONB, $8)",
            &[
                &execution.scope.tenant_id,
                &execution.scope.principal_id,
                &execution.workspace_id.0,
                &execution.profile.0,
                &execution.execution_id.0,
                &artifact.metadata.artifact_id,
                &metadata_json,
                &artifact.bytes,
            ],
        )
        .await?;
    Ok(())
}

fn decode_artifact(row: Row) -> Result<Artifact, PostgresExecutionStoreError> {
    Ok(Artifact {
        metadata: serde_json::from_str(row.get("metadata_json"))?,
        bytes: row.get("bytes"),
    })
}

fn validate_lease(
    stored: &StoredExecution,
    lease: &ExecutionLease,
    now: u64,
) -> Result<(), PostgresExecutionStoreError> {
    let matches = stored.lease.as_ref().is_some_and(|current| {
        current.execution == lease.execution
            && current.owner_id == lease.owner_id
            && current.fence == lease.fence
            && current.deadline_unix_millis == lease.deadline_unix_millis
    });
    if matches && lease.deadline_unix_millis > now {
        Ok(())
    } else {
        Err(PostgresExecutionStoreError::StaleLease)
    }
}

fn lease_expired(lease: Option<&ExecutionLease>, now: u64) -> bool {
    lease.is_none_or(|lease| lease.deadline_unix_millis <= now)
}

fn expire_running(stored: &mut StoredExecution, now: u64) -> bool {
    if stored.record.state == ExecutionState::Running && lease_expired(stored.lease.as_ref(), now) {
        stored.record.state = ExecutionState::OutcomeUnknown;
        stored.record.failure_code = Some("execution_owner_lost".to_owned());
        stored.lease = None;
        true
    } else {
        false
    }
}

struct LeaseColumns {
    owner: Option<String>,
    fence: Option<i64>,
    deadline: Option<i64>,
}

fn lease_columns(
    lease: Option<&ExecutionLease>,
) -> Result<LeaseColumns, PostgresExecutionStoreError> {
    match lease {
        Some(lease) => Ok(LeaseColumns {
            owner: Some(lease.owner_id.clone()),
            fence: Some(to_i64(lease.fence)?),
            deadline: Some(to_i64(lease.deadline_unix_millis)?),
        }),
        None => Ok(LeaseColumns {
            owner: None,
            fence: None,
            deadline: None,
        }),
    }
}

async fn database_now(transaction: &Transaction<'_>) -> Result<u64, PostgresExecutionStoreError> {
    let millis: i64 = transaction
        .query_one(
            "SELECT (EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT",
            &[],
        )
        .await?
        .get(0);
    to_u64(millis)
}

fn add_millis(now: u64, duration: u64) -> Result<u64, PostgresExecutionStoreError> {
    if duration == 0 {
        return Err(PostgresExecutionStoreError::InvalidTransition);
    }
    now.checked_add(duration)
        .ok_or(PostgresExecutionStoreError::NumericOverflow)
}

fn to_i64(value: u64) -> Result<i64, PostgresExecutionStoreError> {
    i64::try_from(value).map_err(|_| PostgresExecutionStoreError::NumericOverflow)
}

fn to_u64(value: i64) -> Result<u64, PostgresExecutionStoreError> {
    u64::try_from(value)
        .map_err(|_| PostgresExecutionStoreError::InvalidPersisted("negative value"))
}

fn state_name(state: ExecutionState) -> &'static str {
    match state {
        ExecutionState::Pending => "pending",
        ExecutionState::Running => "running",
        ExecutionState::Succeeded => "succeeded",
        ExecutionState::Failed => "failed",
        ExecutionState::Cancelled => "cancelled",
        ExecutionState::TimedOut => "timed_out",
        ExecutionState::OutcomeUnknown => "outcome_unknown",
    }
}

#[derive(Clone)]
pub struct PostgresExecutionStore {
    pool: Pool,
}

impl PostgresExecutionStore {
    /// Builds a pool without transport TLS. Production deployments that need
    /// TLS should construct their connector and use [`Self::from_pool`].
    pub async fn connect_no_tls(
        database_url: &str,
        max_pool_size: usize,
    ) -> Result<Self, PostgresExecutionStoreError> {
        let postgres_config = database_url.parse::<tokio_postgres::Config>()?;
        let manager = Manager::from_config(
            postgres_config,
            NoTls,
            ManagerConfig {
                recycling_method: RecyclingMethod::Fast,
            },
        );
        let pool = Pool::builder(manager)
            .runtime(Runtime::Tokio1)
            .max_size(max_pool_size)
            .build()?;
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    pub fn from_pool(pool: Pool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub async fn migrate(&self) -> Result<(), PostgresExecutionStoreError> {
        let mut client = self.pool.get().await?;
        let transaction = client.transaction().await?;
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended('agent-rt-sandbox-migrations', 0))",
                &[],
            )
            .await?;
        transaction.batch_execute(MIGRATION).await?;
        transaction.commit().await?;
        Ok(())
    }
}
