// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::marker::PhantomData;

use deadpool_postgres::{GenericClient, Manager, ManagerConfig, Pool, RecyclingMethod, Runtime};
use dynamo_agent_rt::{
    AgentProtocol, AuthorizationScope, BeginTurn, BeginTurnResult, BoxFuture, CheckpointRecord,
    CheckpointStore, CheckpointVersion, CommitTurn, CommitTurnResult, IdempotencyKey,
    LeaseDeadline, LoadChain, RenewLease, RequestFingerprint, ResponseId, ToolClaimResult,
    ToolExecutionFailure, ToolExecutionRequest, ToolExecutionResult, ToolJournal, ToolJournalKey,
    ToolJournalOutcome, ToolJournalRecord, ToolJournalState, TurnId, TurnLease, TurnState,
};
use thiserror::Error;
use tokio_postgres::{NoTls, Row};

use crate::StoreInvariantError;

const MIGRATION: &str = include_str!("../migrations/postgres/0001_agent_rt.sql");

#[derive(Debug, Error)]
pub enum PostgresStoreError {
    #[error(transparent)]
    Invariant(#[from] StoreInvariantError),
    #[error("PostgreSQL failed: {0}")]
    Database(#[from] tokio_postgres::Error),
    #[error("PostgreSQL pool failed: {0}")]
    Pool(#[from] deadpool_postgres::PoolError),
    #[error("PostgreSQL pool configuration failed: {0}")]
    PoolBuild(#[from] deadpool_postgres::BuildError),
    #[error("persisted JSON failed: {0}")]
    Json(#[from] serde_json::Error),
}

/// Shared multi-replica store backed by PostgreSQL row locks and database time.
#[derive(Clone)]
pub struct PostgresStore<P>
where
    P: AgentProtocol,
{
    pool: Pool,
    protocol: PhantomData<fn() -> P>,
}

impl<P> PostgresStore<P>
where
    P: AgentProtocol,
{
    /// Builds a pool without transport TLS. Production deployments that
    /// require TLS should construct a `deadpool_postgres::Pool` with their
    /// chosen connector and use [`Self::from_pool`].
    pub async fn connect_no_tls(
        database_url: &str,
        max_pool_size: usize,
    ) -> Result<Self, PostgresStoreError> {
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
        let store = Self {
            pool,
            protocol: PhantomData,
        };
        store.migrate().await?;
        Ok(store)
    }

    pub fn from_pool(pool: Pool) -> Self {
        Self {
            pool,
            protocol: PhantomData,
        }
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub async fn migrate(&self) -> Result<(), PostgresStoreError> {
        let mut client = self.pool.get().await?;
        let transaction = client.transaction().await?;
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended('dynamo-agent-rt-store-migrations', 0))",
                &[],
            )
            .await?;
        transaction.batch_execute(MIGRATION).await?;
        transaction.commit().await?;
        Ok(())
    }
}

impl<P> CheckpointStore<P> for PostgresStore<P>
where
    P: AgentProtocol,
{
    type Error = PostgresStoreError;

    fn begin_turn(
        &self,
        command: BeginTurn<P>,
    ) -> BoxFuture<'_, Result<BeginTurnResult<P>, Self::Error>> {
        Box::pin(async move { begin_turn::<P>(&self.pool, command).await })
    }

    fn load_chain(
        &self,
        query: LoadChain,
    ) -> BoxFuture<'_, Result<Vec<CheckpointRecord<P>>, Self::Error>> {
        Box::pin(async move { load_chain::<P>(&self.pool, query).await })
    }

    fn commit_turn(
        &self,
        command: CommitTurn<P>,
    ) -> BoxFuture<'_, Result<CommitTurnResult<P>, Self::Error>> {
        Box::pin(async move { commit_turn::<P>(&self.pool, command).await })
    }

    fn renew_lease(&self, command: RenewLease) -> BoxFuture<'_, Result<TurnLease, Self::Error>> {
        Box::pin(async move { renew_lease::<P>(&self.pool, command).await })
    }
}

impl<P> ToolJournal for PostgresStore<P>
where
    P: AgentProtocol,
{
    type Error = PostgresStoreError;

    fn claim(
        &self,
        request: ToolExecutionRequest,
    ) -> BoxFuture<'_, Result<ToolClaimResult, Self::Error>> {
        Box::pin(async move { claim_tool(&self.pool, request).await })
    }

    fn load(
        &self,
        key: &ToolJournalKey,
    ) -> BoxFuture<'_, Result<Option<ToolJournalRecord>, Self::Error>> {
        let key = key.clone();
        Box::pin(async move {
            let client = self.pool.get().await?;
            load_tool(&client, &key, false).await
        })
    }

    fn finish(
        &self,
        key: ToolJournalKey,
        outcome: ToolJournalOutcome,
    ) -> BoxFuture<'_, Result<ToolJournalRecord, Self::Error>> {
        Box::pin(async move { finish_tool(&self.pool, key, outcome).await })
    }
}

struct StoredCheckpoint<P>
where
    P: AgentProtocol,
{
    record: CheckpointRecord<P>,
    lease: Option<TurnLease>,
}

async fn begin_turn<P: AgentProtocol>(
    pool: &Pool,
    command: BeginTurn<P>,
) -> Result<BeginTurnResult<P>, PostgresStoreError> {
    let mut client = pool.get().await?;
    let transaction = client.transaction().await?;
    lock_idempotency::<P>(&transaction, &command).await?;
    let now = database_now(&transaction).await?;
    if command.lease_deadline.0 <= now {
        return Err(StoreInvariantError::InvalidLeaseDeadline.into());
    }
    let deadline = to_i64(command.lease_deadline.0)?;
    let scope = &command.authorization.scope;
    let existing = transaction
        .query_opt(
            "SELECT response_id FROM agent_rt_checkpoints
             WHERE protocol = $1 AND tenant_id = $2 AND principal_id = $3 AND idempotency_key = $4",
            &[
                &P::STORAGE_KEY,
                &scope.tenant_id,
                &scope.principal_id,
                &command.idempotency_key.as_str(),
            ],
        )
        .await?;
    if let Some(existing) = existing {
        let existing_id: String = existing.get(0);
        let mut stored = load_checkpoint::<P, _>(&transaction, &existing_id, None, true).await?;
        if stored.record.parent_response_id != command.parent_response_id
            || stored.record.request_fingerprint != command.request_fingerprint
        {
            return Err(StoreInvariantError::IdempotencyConflict.into());
        }
        if let Some(lease) = &stored.lease
            && lease.deadline.0 <= now
        {
            if stored.record.state == TurnState::InFlight {
                let version = increment_version(stored.record.version)?;
                let updated = transaction
                    .execute(
                        "UPDATE agent_rt_checkpoints
                         SET version = $1, lease_turn_id = $2, lease_deadline = $3
                         WHERE protocol = $4 AND response_id = $5 AND version = $6",
                        &[
                            &to_i64(version.0)?,
                            &command.turn_id.as_str(),
                            &deadline,
                            &P::STORAGE_KEY,
                            &existing_id,
                            &to_i64(stored.record.version.0)?,
                        ],
                    )
                    .await?;
                if updated != 1 {
                    return Err(StoreInvariantError::VersionConflict.into());
                }
                transaction.commit().await?;
                return Ok(BeginTurnResult::Acquired(TurnLease {
                    response_id: ResponseId::from(existing_id),
                    turn_id: command.turn_id,
                    version,
                    deadline: command.lease_deadline,
                }));
            }
            if stored.record.state == TurnState::ToolStarted {
                let version = increment_version(stored.record.version)?;
                let updated = transaction
                    .execute(
                        "UPDATE agent_rt_checkpoints
                         SET state = 'outcome_unknown', version = $1,
                             lease_turn_id = NULL, lease_deadline = NULL
                         WHERE protocol = $2 AND response_id = $3 AND version = $4",
                        &[
                            &to_i64(version.0)?,
                            &P::STORAGE_KEY,
                            &existing_id,
                            &to_i64(stored.record.version.0)?,
                        ],
                    )
                    .await?;
                if updated != 1 {
                    return Err(StoreInvariantError::VersionConflict.into());
                }
                stored.record.state = TurnState::OutcomeUnknown;
                stored.record.version = version;
                transaction.commit().await?;
                return Ok(BeginTurnResult::Existing(Box::new(stored.record)));
            }
        }
        transaction.commit().await?;
        return Ok(BeginTurnResult::Existing(Box::new(stored.record)));
    }

    if transaction
        .query_opt(
            "SELECT 1 FROM agent_rt_checkpoints WHERE protocol = $1 AND response_id = $2",
            &[&P::STORAGE_KEY, &command.response_id.as_str()],
        )
        .await?
        .is_some()
    {
        return Err(StoreInvariantError::ResponseAlreadyExists(command.response_id).into());
    }
    if let Some(parent_id) = &command.parent_response_id {
        let parent =
            load_checkpoint::<P, _>(&transaction, parent_id.as_str(), Some(scope), false).await?;
        if !matches!(
            parent.record.state,
            TurnState::Completed | TurnState::AwaitingClientToolOutput
        ) {
            return Err(StoreInvariantError::ParentNotReplayable(parent.record.state).into());
        }
    }
    let request_json = serde_json::to_string(&command.request)?;
    let parent = command.parent_response_id.as_ref().map(ResponseId::as_str);
    let fingerprint = command.request_fingerprint.as_bytes().as_slice();
    transaction
        .execute(
            "INSERT INTO agent_rt_checkpoints (
                protocol, response_id, parent_response_id, tenant_id, principal_id,
                idempotency_key, request_fingerprint, state, version, request_json,
                response_json, lease_turn_id, lease_deadline
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'in_flight', 0, $8, NULL, $9, $10)",
            &[
                &P::STORAGE_KEY,
                &command.response_id.as_str(),
                &parent,
                &scope.tenant_id,
                &scope.principal_id,
                &command.idempotency_key.as_str(),
                &fingerprint,
                &request_json,
                &command.turn_id.as_str(),
                &deadline,
            ],
        )
        .await?;
    let lease = TurnLease {
        response_id: command.response_id,
        turn_id: command.turn_id,
        version: CheckpointVersion(0),
        deadline: command.lease_deadline,
    };
    transaction.commit().await?;
    Ok(BeginTurnResult::Acquired(lease))
}

async fn load_chain<P: AgentProtocol>(
    pool: &Pool,
    query: LoadChain,
) -> Result<Vec<CheckpointRecord<P>>, PostgresStoreError> {
    let client = pool.get().await?;
    let mut current = Some(query.response_id);
    let mut seen = HashSet::new();
    let mut reversed = Vec::new();
    while let Some(response_id) = current {
        if !seen.insert(response_id.clone()) {
            return Err(StoreInvariantError::Corrupt.into());
        }
        let stored =
            load_checkpoint::<P, _>(&client, response_id.as_str(), Some(&query.scope), false)
                .await?;
        current = stored.record.parent_response_id.clone();
        reversed.push(stored.record);
    }
    reversed.reverse();
    Ok(reversed)
}

async fn commit_turn<P: AgentProtocol>(
    pool: &Pool,
    command: CommitTurn<P>,
) -> Result<CommitTurnResult<P>, PostgresStoreError> {
    let mut client = pool.get().await?;
    let transaction = client.transaction().await?;
    let now = database_now(&transaction).await?;
    let stored =
        load_checkpoint::<P, _>(&transaction, command.lease.response_id.as_str(), None, true)
            .await?;
    validate_lease(&stored, &command.lease, now)?;
    if !stored
        .record
        .state
        .permits_transition_to(&command.next_state)
    {
        return Err(StoreInvariantError::InvalidTransition {
            from: stored.record.state,
            to: command.next_state,
        }
        .into());
    }
    let version = increment_version(stored.record.version)?;
    let response_json = command
        .response
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;
    let retains_lease = matches!(
        command.next_state,
        TurnState::InFlight | TurnState::ToolStarted
    );
    let lease_turn_id = retains_lease.then_some(command.lease.turn_id.as_str());
    let lease_deadline = retains_lease
        .then(|| to_i64(command.lease.deadline.0))
        .transpose()?;
    let updated = transaction
        .execute(
            "UPDATE agent_rt_checkpoints
             SET state = $1, version = $2, response_json = COALESCE($3, response_json),
                 lease_turn_id = $4, lease_deadline = $5
             WHERE protocol = $6 AND response_id = $7 AND version = $8
               AND lease_turn_id = $9 AND lease_deadline = $10",
            &[
                &state_name(&command.next_state),
                &to_i64(version.0)?,
                &response_json,
                &lease_turn_id,
                &lease_deadline,
                &P::STORAGE_KEY,
                &command.lease.response_id.as_str(),
                &to_i64(command.lease.version.0)?,
                &command.lease.turn_id.as_str(),
                &to_i64(command.lease.deadline.0)?,
            ],
        )
        .await?;
    if updated != 1 {
        return Err(StoreInvariantError::VersionConflict.into());
    }

    if !command.append_output_items.is_empty() {
        let row = transaction
            .query_one(
                "SELECT COALESCE(MAX(sequence), -1) + 1
                 FROM agent_rt_checkpoint_output_items
                 WHERE protocol = $1 AND response_id = $2",
                &[&P::STORAGE_KEY, &command.lease.response_id.as_str()],
            )
            .await?;
        let first_sequence: i64 = row.get(0);
        let sequences = (0..command.append_output_items.len())
            .map(|offset| {
                first_sequence
                    .checked_add(i64::try_from(offset).map_err(|_| StoreInvariantError::Corrupt)?)
                    .ok_or(StoreInvariantError::Corrupt)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let items = command
            .append_output_items
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()?;
        transaction
            .execute(
                "INSERT INTO agent_rt_checkpoint_output_items
                    (protocol, response_id, sequence, item_json)
                 SELECT $1, $2, values.sequence, values.item_json
                 FROM UNNEST($3::BIGINT[], $4::TEXT[]) AS values(sequence, item_json)",
                &[
                    &P::STORAGE_KEY,
                    &command.lease.response_id.as_str(),
                    &sequences,
                    &items,
                ],
            )
            .await?;
    }
    let record = load_checkpoint::<P, _>(
        &transaction,
        command.lease.response_id.as_str(),
        None,
        false,
    )
    .await?
    .record;
    let lease = retains_lease.then(|| TurnLease {
        response_id: record.response_id.clone(),
        turn_id: command.lease.turn_id,
        version,
        deadline: command.lease.deadline,
    });
    transaction.commit().await?;
    Ok(CommitTurnResult { record, lease })
}

async fn renew_lease<P: AgentProtocol>(
    pool: &Pool,
    command: RenewLease,
) -> Result<TurnLease, PostgresStoreError> {
    let mut client = pool.get().await?;
    let transaction = client.transaction().await?;
    let now = database_now(&transaction).await?;
    if command.new_deadline.0 <= now {
        return Err(StoreInvariantError::InvalidLeaseDeadline.into());
    }
    let stored =
        load_checkpoint::<P, _>(&transaction, command.lease.response_id.as_str(), None, true)
            .await?;
    let current = validate_lease(&stored, &command.lease, now)?;
    if command.new_deadline <= current.deadline {
        return Err(StoreInvariantError::LeaseDeadlineNotExtended.into());
    }
    let updated = transaction
        .execute(
            "UPDATE agent_rt_checkpoints SET lease_deadline = $1
             WHERE protocol = $2 AND response_id = $3 AND version = $4
               AND lease_turn_id = $5 AND lease_deadline = $6",
            &[
                &to_i64(command.new_deadline.0)?,
                &P::STORAGE_KEY,
                &current.response_id.as_str(),
                &to_i64(current.version.0)?,
                &current.turn_id.as_str(),
                &to_i64(current.deadline.0)?,
            ],
        )
        .await?;
    if updated != 1 {
        return Err(StoreInvariantError::VersionConflict.into());
    }
    let renewed = TurnLease {
        deadline: command.new_deadline,
        ..current
    };
    transaction.commit().await?;
    Ok(renewed)
}

async fn claim_tool(
    pool: &Pool,
    request: ToolExecutionRequest,
) -> Result<ToolClaimResult, PostgresStoreError> {
    let mut client = pool.get().await?;
    let transaction = client.transaction().await?;
    let key = request.journal_key();
    advisory_lock(
        &transaction,
        &lock_key(&[
            "tool_journal",
            key.scope.tenant_id.as_str(),
            key.scope.principal_id.as_str(),
            key.idempotency_key.as_str(),
        ]),
    )
    .await?;
    if let Some(existing) = load_tool(&transaction, &key, false).await? {
        if existing.request != request {
            return Err(StoreInvariantError::IdempotencyConflict.into());
        }
        transaction.commit().await?;
        return Ok(ToolClaimResult::Existing(Box::new(existing)));
    }
    let request_json = serde_json::to_string(&request)?;
    transaction
        .execute(
            "INSERT INTO agent_rt_tool_journal
             (tenant_id, principal_id, idempotency_key, request_json, state, result_json, failure_json)
             VALUES ($1, $2, $3, $4, 'started', NULL, NULL)",
            &[
                &request.scope.tenant_id,
                &request.scope.principal_id,
                &request.idempotency_key.as_str(),
                &request_json,
            ],
        )
        .await?;
    let record = ToolJournalRecord {
        request,
        state: ToolJournalState::Started,
        result: None,
        failure: None,
    };
    transaction.commit().await?;
    Ok(ToolClaimResult::Acquired(Box::new(record)))
}

async fn finish_tool(
    pool: &Pool,
    key: ToolJournalKey,
    outcome: ToolJournalOutcome,
) -> Result<ToolJournalRecord, PostgresStoreError> {
    let mut client = pool.get().await?;
    let transaction = client.transaction().await?;
    let existing = load_tool(&transaction, &key, true)
        .await?
        .ok_or(StoreInvariantError::NotFound)?;
    if existing.state != ToolJournalState::Started {
        return Err(StoreInvariantError::ToolAlreadyFinished(existing.state).into());
    }
    let (state, result, failure) = match outcome {
        ToolJournalOutcome::Completed(result) => (
            ToolJournalState::Completed,
            Some(serde_json::to_string(&result)?),
            None,
        ),
        ToolJournalOutcome::Failed(failure) => (
            ToolJournalState::Failed,
            None,
            Some(serde_json::to_string(&failure)?),
        ),
        ToolJournalOutcome::OutcomeUnknown => (ToolJournalState::OutcomeUnknown, None, None),
    };
    let updated = transaction
        .execute(
            "UPDATE agent_rt_tool_journal
             SET state = $1, result_json = $2, failure_json = $3
             WHERE tenant_id = $4 AND principal_id = $5 AND idempotency_key = $6
               AND state = 'started'",
            &[
                &tool_state_name(&state),
                &result,
                &failure,
                &key.scope.tenant_id,
                &key.scope.principal_id,
                &key.idempotency_key.as_str(),
            ],
        )
        .await?;
    if updated != 1 {
        return Err(StoreInvariantError::ToolAlreadyFinished(existing.state).into());
    }
    let record = load_tool(&transaction, &key, false)
        .await?
        .ok_or(StoreInvariantError::Corrupt)?;
    transaction.commit().await?;
    Ok(record)
}

async fn load_tool<C: GenericClient + Sync>(
    client: &C,
    key: &ToolJournalKey,
    for_update: bool,
) -> Result<Option<ToolJournalRecord>, PostgresStoreError> {
    let suffix = if for_update { " FOR UPDATE" } else { "" };
    let statement = format!(
        "SELECT request_json, state, result_json, failure_json
         FROM agent_rt_tool_journal
         WHERE tenant_id = $1 AND principal_id = $2 AND idempotency_key = $3{suffix}"
    );
    let row = client
        .query_opt(
            &statement,
            &[
                &key.scope.tenant_id,
                &key.scope.principal_id,
                &key.idempotency_key.as_str(),
            ],
        )
        .await?;
    row.map(decode_tool).transpose()
}

fn decode_tool(row: Row) -> Result<ToolJournalRecord, PostgresStoreError> {
    let result_json: Option<&str> = row.get(2);
    let failure_json: Option<&str> = row.get(3);
    Ok(ToolJournalRecord {
        request: serde_json::from_str(row.get(0))?,
        state: parse_tool_state(row.get(1))?,
        result: result_json
            .map(serde_json::from_str::<ToolExecutionResult>)
            .transpose()?,
        failure: failure_json
            .map(serde_json::from_str::<ToolExecutionFailure>)
            .transpose()?,
    })
}

async fn lock_idempotency<P: AgentProtocol>(
    transaction: &deadpool_postgres::Transaction<'_>,
    command: &BeginTurn<P>,
) -> Result<(), PostgresStoreError> {
    let scope = &command.authorization.scope;
    let key = lock_key(&[
        P::STORAGE_KEY,
        scope.tenant_id.as_str(),
        scope.principal_id.as_str(),
        command.idempotency_key.as_str(),
    ]);
    advisory_lock(transaction, &key).await
}

fn lock_key(values: &[&str]) -> String {
    let mut key = String::new();
    for value in values {
        key.push_str(&value.len().to_string());
        key.push(':');
        key.push_str(value);
    }
    key
}

async fn advisory_lock(
    transaction: &deadpool_postgres::Transaction<'_>,
    key: &str,
) -> Result<(), PostgresStoreError> {
    transaction
        .query_one(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            &[&key],
        )
        .await?;
    Ok(())
}

async fn database_now<C: GenericClient + Sync>(client: &C) -> Result<u64, PostgresStoreError> {
    let row = client
        .query_one(
            "SELECT FLOOR(EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT",
            &[],
        )
        .await?;
    Ok(from_i64(row.get(0))?)
}

async fn load_checkpoint<P, C>(
    client: &C,
    response_id: &str,
    expected_scope: Option<&AuthorizationScope>,
    for_update: bool,
) -> Result<StoredCheckpoint<P>, PostgresStoreError>
where
    P: AgentProtocol,
    C: GenericClient + Sync,
{
    let suffix = if for_update { " FOR UPDATE" } else { "" };
    let statement = format!(
        "SELECT parent_response_id, tenant_id, principal_id, idempotency_key,
                request_fingerprint, state, version, request_json, response_json,
                lease_turn_id, lease_deadline
         FROM agent_rt_checkpoints WHERE protocol = $1 AND response_id = $2{suffix}"
    );
    let row = client
        .query_opt(&statement, &[&P::STORAGE_KEY, &response_id])
        .await?
        .ok_or(StoreInvariantError::NotFound)?;
    decode_checkpoint::<P, C>(client, response_id, expected_scope, row).await
}

async fn decode_checkpoint<P, C>(
    client: &C,
    response_id: &str,
    expected_scope: Option<&AuthorizationScope>,
    row: Row,
) -> Result<StoredCheckpoint<P>, PostgresStoreError>
where
    P: AgentProtocol,
    C: GenericClient + Sync,
{
    let scope = AuthorizationScope {
        tenant_id: row.get(1),
        principal_id: row.get(2),
    };
    if expected_scope.is_some_and(|expected| expected != &scope) {
        return Err(StoreInvariantError::NotFound.into());
    }
    let version = CheckpointVersion(from_i64(row.get(6))?);
    let fingerprint: Vec<u8> = row.get(4);
    let fingerprint: [u8; 32] = fingerprint
        .try_into()
        .map_err(|_| StoreInvariantError::Corrupt)?;
    let item_rows = client
        .query(
            "SELECT item_json FROM agent_rt_checkpoint_output_items
             WHERE protocol = $1 AND response_id = $2 ORDER BY sequence",
            &[&P::STORAGE_KEY, &response_id],
        )
        .await?;
    let output_items = item_rows
        .into_iter()
        .map(|row| serde_json::from_str::<P::ReplayItem>(row.get::<_, &str>(0)))
        .collect::<Result<Vec<_>, _>>()?;
    let response_json: Option<&str> = row.get(8);
    let record = CheckpointRecord {
        response_id: ResponseId::from(response_id),
        parent_response_id: row.get::<_, Option<String>>(0).map(ResponseId::from),
        scope,
        idempotency_key: IdempotencyKey::from(row.get::<_, String>(3)),
        request_fingerprint: RequestFingerprint::new(fingerprint),
        state: parse_state(row.get(5))?,
        version,
        request: serde_json::from_str(row.get(7))?,
        output_items,
        response: response_json.map(serde_json::from_str).transpose()?,
    };
    let lease = match (
        row.get::<_, Option<String>>(9),
        row.get::<_, Option<i64>>(10),
    ) {
        (Some(turn_id), Some(deadline)) => Some(TurnLease {
            response_id: record.response_id.clone(),
            turn_id: TurnId::from(turn_id),
            version,
            deadline: LeaseDeadline(from_i64(deadline)?),
        }),
        (None, None) => None,
        _ => return Err(StoreInvariantError::Corrupt.into()),
    };
    Ok(StoredCheckpoint { record, lease })
}

fn validate_lease<P: AgentProtocol>(
    stored: &StoredCheckpoint<P>,
    supplied: &TurnLease,
    now: u64,
) -> Result<TurnLease, StoreInvariantError> {
    let current = stored
        .lease
        .as_ref()
        .ok_or(StoreInvariantError::LeaseNotFound)?;
    if current.turn_id != supplied.turn_id || current.deadline != supplied.deadline {
        return Err(StoreInvariantError::LeaseMismatch);
    }
    if current.version != supplied.version {
        return Err(StoreInvariantError::VersionConflict);
    }
    if current.deadline.0 <= now {
        return Err(StoreInvariantError::LeaseExpired);
    }
    Ok(current.clone())
}

fn state_name(state: &TurnState) -> &'static str {
    match state {
        TurnState::InFlight => "in_flight",
        TurnState::AwaitingClientToolOutput => "awaiting_client_tool_output",
        TurnState::ToolStarted => "tool_started",
        TurnState::OutcomeUnknown => "outcome_unknown",
        TurnState::Completed => "completed",
        TurnState::Failed => "failed",
    }
}

fn parse_state(value: &str) -> Result<TurnState, StoreInvariantError> {
    match value {
        "in_flight" => Ok(TurnState::InFlight),
        "awaiting_client_tool_output" => Ok(TurnState::AwaitingClientToolOutput),
        "tool_started" => Ok(TurnState::ToolStarted),
        "outcome_unknown" => Ok(TurnState::OutcomeUnknown),
        "completed" => Ok(TurnState::Completed),
        "failed" => Ok(TurnState::Failed),
        _ => Err(StoreInvariantError::Corrupt),
    }
}

fn tool_state_name(state: &ToolJournalState) -> &'static str {
    match state {
        ToolJournalState::Started => "started",
        ToolJournalState::Completed => "completed",
        ToolJournalState::Failed => "failed",
        ToolJournalState::OutcomeUnknown => "outcome_unknown",
    }
}

fn parse_tool_state(value: &str) -> Result<ToolJournalState, StoreInvariantError> {
    match value {
        "started" => Ok(ToolJournalState::Started),
        "completed" => Ok(ToolJournalState::Completed),
        "failed" => Ok(ToolJournalState::Failed),
        "outcome_unknown" => Ok(ToolJournalState::OutcomeUnknown),
        _ => Err(StoreInvariantError::Corrupt),
    }
}

fn increment_version(version: CheckpointVersion) -> Result<CheckpointVersion, StoreInvariantError> {
    version
        .0
        .checked_add(1)
        .map(CheckpointVersion)
        .ok_or(StoreInvariantError::VersionOverflow)
}

fn to_i64(value: u64) -> Result<i64, StoreInvariantError> {
    i64::try_from(value).map_err(|_| StoreInvariantError::Corrupt)
}

fn from_i64(value: i64) -> Result<u64, StoreInvariantError> {
    u64::try_from(value).map_err(|_| StoreInvariantError::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::{parse_state, parse_tool_state, state_name, tool_state_name};
    use dynamo_agent_rt::{ToolJournalState, TurnState};

    #[test]
    fn checkpoint_states_round_trip() {
        for state in [
            TurnState::InFlight,
            TurnState::AwaitingClientToolOutput,
            TurnState::ToolStarted,
            TurnState::OutcomeUnknown,
            TurnState::Completed,
            TurnState::Failed,
        ] {
            assert_eq!(parse_state(state_name(&state)).unwrap(), state);
        }
    }

    #[test]
    fn tool_states_round_trip() {
        for state in [
            ToolJournalState::Started,
            ToolJournalState::Completed,
            ToolJournalState::Failed,
            ToolJournalState::OutcomeUnknown,
        ] {
            assert_eq!(parse_tool_state(tool_state_name(&state)).unwrap(), state);
        }
    }
}
