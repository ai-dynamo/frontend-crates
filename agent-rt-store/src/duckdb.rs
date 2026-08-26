// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ::duckdb::{Connection, OptionalExt, params};
use dynamo_agent_rt::{
    AgentProtocol, AuthorizationScope, BeginTurn, BeginTurnResult, BoxFuture, CheckpointRecord,
    CheckpointStore, CheckpointVersion, Clock, CommitTurn, CommitTurnResult, IdempotencyKey,
    LeaseDeadline, LoadChain, RenewLease, RequestFingerprint, ResponseId, SystemClock, TurnId,
    TurnLease, TurnState,
};
use thiserror::Error;

use crate::StoreInvariantError;

const MIGRATION: &str = include_str!("../migrations/duckdb/0001_agent_rt.sql");

#[derive(Debug, Error)]
pub enum DuckDbStoreError {
    #[error(transparent)]
    Invariant(#[from] StoreInvariantError),
    #[error("DuckDB failed: {0}")]
    Database(#[from] ::duckdb::Error),
    #[error("persisted JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("DuckDB connection mutex is poisoned")]
    Poisoned,
    #[error("DuckDB blocking task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Embedded, serialized store for local and single-process deployments.
///
/// Every operation runs on Tokio's blocking pool and one connection mutex
/// serializes transactions. This type is intentionally not a multi-replica
/// coordination mechanism.
pub struct DuckDbStore<P, C = SystemClock>
where
    P: AgentProtocol,
{
    connection: Arc<Mutex<Connection>>,
    clock: C,
    protocol: PhantomData<fn() -> P>,
}

impl<P, C> Clone for DuckDbStore<P, C>
where
    P: AgentProtocol,
    C: Clone,
{
    fn clone(&self) -> Self {
        Self {
            connection: Arc::clone(&self.connection),
            clock: self.clock.clone(),
            protocol: PhantomData,
        }
    }
}

impl<P> DuckDbStore<P, SystemClock>
where
    P: AgentProtocol,
{
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DuckDbStoreError> {
        Self::open_with_clock(path, SystemClock)
    }

    pub fn open_in_memory() -> Result<Self, DuckDbStoreError> {
        Self::from_connection(Connection::open_in_memory()?, SystemClock)
    }
}

impl<P, C> DuckDbStore<P, C>
where
    P: AgentProtocol,
{
    pub fn open_with_clock(path: impl AsRef<Path>, clock: C) -> Result<Self, DuckDbStoreError> {
        Self::from_connection(Connection::open(path)?, clock)
    }

    pub fn from_connection(connection: Connection, clock: C) -> Result<Self, DuckDbStoreError> {
        connection.execute_batch(MIGRATION)?;
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
            clock,
            protocol: PhantomData,
        })
    }

    async fn blocking<R, F>(&self, operation: F) -> Result<R, DuckDbStoreError>
    where
        R: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<R, DuckDbStoreError> + Send + 'static,
    {
        let connection = Arc::clone(&self.connection);
        tokio::task::spawn_blocking(move || {
            let mut connection = connection.lock().map_err(|_| DuckDbStoreError::Poisoned)?;
            operation(&mut connection)
        })
        .await?
    }
}

impl<P, C> CheckpointStore<P> for DuckDbStore<P, C>
where
    P: AgentProtocol,
    C: Clock + Clone,
{
    type Error = DuckDbStoreError;

    fn begin_turn(
        &self,
        command: BeginTurn<P>,
    ) -> BoxFuture<'_, Result<BeginTurnResult<P>, Self::Error>> {
        let now = self.clock.now_millis();
        Box::pin(async move {
            self.blocking(move |connection| begin_turn::<P>(connection, command, now))
                .await
        })
    }

    fn load_chain(
        &self,
        query: LoadChain,
    ) -> BoxFuture<'_, Result<Vec<CheckpointRecord<P>>, Self::Error>> {
        Box::pin(async move {
            self.blocking(move |connection| load_chain::<P>(connection, query))
                .await
        })
    }

    fn commit_turn(
        &self,
        command: CommitTurn<P>,
    ) -> BoxFuture<'_, Result<CommitTurnResult<P>, Self::Error>> {
        let now = self.clock.now_millis();
        Box::pin(async move {
            self.blocking(move |connection| commit_turn::<P>(connection, command, now))
                .await
        })
    }

    fn renew_lease(&self, command: RenewLease) -> BoxFuture<'_, Result<TurnLease, Self::Error>> {
        let now = self.clock.now_millis();
        Box::pin(async move {
            self.blocking(move |connection| renew_lease::<P>(connection, command, now))
                .await
        })
    }
}

struct StoredCheckpoint<P>
where
    P: AgentProtocol,
{
    record: CheckpointRecord<P>,
    lease: Option<TurnLease>,
}

#[allow(clippy::too_many_arguments)]
fn begin_turn<P: AgentProtocol>(
    connection: &mut Connection,
    command: BeginTurn<P>,
    now: u64,
) -> Result<BeginTurnResult<P>, DuckDbStoreError> {
    if command.lease_deadline.0 <= now {
        return Err(StoreInvariantError::InvalidLeaseDeadline.into());
    }
    let lease_deadline = to_i64(command.lease_deadline.0)?;
    let transaction = connection.transaction()?;
    let existing_id = transaction
        .query_row(
            "SELECT response_id FROM agent_rt_checkpoints
             WHERE protocol = ?1 AND tenant_id = ?2 AND principal_id = ?3 AND idempotency_key = ?4",
            params![
                P::STORAGE_KEY,
                command.authorization.scope.tenant_id,
                command.authorization.scope.principal_id,
                command.idempotency_key.as_str(),
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()?;

    if let Some(existing_id) = existing_id {
        let mut stored = load_checkpoint::<P>(&transaction, &existing_id, None)?;
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
                let updated = transaction.execute(
                    "UPDATE agent_rt_checkpoints
                     SET version = ?1, lease_turn_id = ?2, lease_deadline = ?3
                     WHERE protocol = ?4 AND response_id = ?5 AND version = ?6",
                    params![
                        to_i64(version.0)?,
                        command.turn_id.as_str(),
                        lease_deadline,
                        P::STORAGE_KEY,
                        existing_id,
                        to_i64(stored.record.version.0)?,
                    ],
                )?;
                if updated != 1 {
                    return Err(StoreInvariantError::VersionConflict.into());
                }
                transaction.commit()?;
                return Ok(BeginTurnResult::Acquired(TurnLease {
                    response_id: ResponseId::from(existing_id),
                    turn_id: command.turn_id,
                    version,
                    deadline: command.lease_deadline,
                }));
            }
            if stored.record.state == TurnState::ToolStarted {
                let version = increment_version(stored.record.version)?;
                let updated = transaction.execute(
                    "UPDATE agent_rt_checkpoints
                     SET state = 'outcome_unknown', version = ?1,
                         lease_turn_id = NULL, lease_deadline = NULL
                     WHERE protocol = ?2 AND response_id = ?3 AND version = ?4",
                    params![
                        to_i64(version.0)?,
                        P::STORAGE_KEY,
                        existing_id,
                        to_i64(stored.record.version.0)?,
                    ],
                )?;
                if updated != 1 {
                    return Err(StoreInvariantError::VersionConflict.into());
                }
                stored.record.version = version;
                stored.record.state = TurnState::OutcomeUnknown;
                transaction.commit()?;
                return Ok(BeginTurnResult::Existing(Box::new(stored.record)));
            }
        }
        transaction.commit()?;
        return Ok(BeginTurnResult::Existing(Box::new(stored.record)));
    }

    if checkpoint_exists::<P>(&transaction, command.response_id.as_str())? {
        return Err(StoreInvariantError::ResponseAlreadyExists(command.response_id).into());
    }
    if let Some(parent_id) = &command.parent_response_id {
        let parent = load_checkpoint::<P>(
            &transaction,
            parent_id.as_str(),
            Some(&command.authorization.scope),
        )?;
        if !matches!(
            parent.record.state,
            TurnState::Completed | TurnState::AwaitingClientToolOutput
        ) {
            return Err(StoreInvariantError::ParentNotReplayable(parent.record.state).into());
        }
    }

    let request_json = serde_json::to_string(&command.request)?;
    transaction.execute(
        "INSERT INTO agent_rt_checkpoints (
            protocol, response_id, parent_response_id, tenant_id, principal_id,
            idempotency_key, request_fingerprint, state, version, request_json,
            response_json, lease_turn_id, lease_deadline
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'in_flight', 0, ?8, NULL, ?9, ?10)",
        params![
            P::STORAGE_KEY,
            command.response_id.as_str(),
            command.parent_response_id.as_ref().map(ResponseId::as_str),
            command.authorization.scope.tenant_id,
            command.authorization.scope.principal_id,
            command.idempotency_key.as_str(),
            command.request_fingerprint.as_bytes().as_slice(),
            request_json,
            command.turn_id.as_str(),
            lease_deadline,
        ],
    )?;
    let lease = TurnLease {
        response_id: command.response_id,
        turn_id: command.turn_id,
        version: CheckpointVersion(0),
        deadline: command.lease_deadline,
    };
    transaction.commit()?;
    Ok(BeginTurnResult::Acquired(lease))
}

fn load_chain<P: AgentProtocol>(
    connection: &mut Connection,
    query: LoadChain,
) -> Result<Vec<CheckpointRecord<P>>, DuckDbStoreError> {
    let mut current = Some(query.response_id);
    let mut seen = HashSet::new();
    let mut reversed = Vec::new();
    while let Some(response_id) = current {
        if !seen.insert(response_id.clone()) {
            return Err(StoreInvariantError::Corrupt.into());
        }
        let stored = load_checkpoint::<P>(connection, response_id.as_str(), Some(&query.scope))?;
        current = stored.record.parent_response_id.clone();
        reversed.push(stored.record);
    }
    reversed.reverse();
    Ok(reversed)
}

fn commit_turn<P: AgentProtocol>(
    connection: &mut Connection,
    command: CommitTurn<P>,
    now: u64,
) -> Result<CommitTurnResult<P>, DuckDbStoreError> {
    let transaction = connection.transaction()?;
    let stored = load_checkpoint::<P>(&transaction, command.lease.response_id.as_str(), None)?;
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
    let updated = transaction.execute(
        "UPDATE agent_rt_checkpoints
         SET state = ?1, version = ?2, response_json = COALESCE(?3, response_json),
             lease_turn_id = ?4, lease_deadline = ?5
         WHERE protocol = ?6 AND response_id = ?7 AND version = ?8
           AND lease_turn_id = ?9 AND lease_deadline = ?10",
        params![
            state_name(&command.next_state),
            to_i64(version.0)?,
            response_json,
            retains_lease.then_some(command.lease.turn_id.as_str()),
            retains_lease.then_some(to_i64(command.lease.deadline.0)?),
            P::STORAGE_KEY,
            command.lease.response_id.as_str(),
            to_i64(command.lease.version.0)?,
            command.lease.turn_id.as_str(),
            to_i64(command.lease.deadline.0)?,
        ],
    )?;
    if updated != 1 {
        return Err(StoreInvariantError::VersionConflict.into());
    }

    let mut next_sequence: i64 = transaction.query_row(
        "SELECT COALESCE(MAX(sequence), -1) + 1
         FROM agent_rt_checkpoint_output_items
         WHERE protocol = ?1 AND response_id = ?2",
        params![P::STORAGE_KEY, command.lease.response_id.as_str()],
        |row| row.get(0),
    )?;
    for item in command.append_output_items {
        transaction.execute(
            "INSERT INTO agent_rt_checkpoint_output_items
             (protocol, response_id, sequence, item_json) VALUES (?1, ?2, ?3, ?4)",
            params![
                P::STORAGE_KEY,
                command.lease.response_id.as_str(),
                next_sequence,
                serde_json::to_string(&item)?,
            ],
        )?;
        next_sequence = next_sequence
            .checked_add(1)
            .ok_or(StoreInvariantError::Corrupt)?;
    }

    let record =
        load_checkpoint::<P>(&transaction, command.lease.response_id.as_str(), None)?.record;
    let lease = retains_lease.then(|| TurnLease {
        response_id: record.response_id.clone(),
        turn_id: command.lease.turn_id,
        version,
        deadline: command.lease.deadline,
    });
    transaction.commit()?;
    Ok(CommitTurnResult { record, lease })
}

fn renew_lease<P: AgentProtocol>(
    connection: &mut Connection,
    command: RenewLease,
    now: u64,
) -> Result<TurnLease, DuckDbStoreError> {
    if command.new_deadline.0 <= now {
        return Err(StoreInvariantError::InvalidLeaseDeadline.into());
    }
    let transaction = connection.transaction()?;
    let stored = load_checkpoint::<P>(&transaction, command.lease.response_id.as_str(), None)?;
    let current = validate_lease(&stored, &command.lease, now)?;
    if command.new_deadline <= current.deadline {
        return Err(StoreInvariantError::LeaseDeadlineNotExtended.into());
    }
    let updated = transaction.execute(
        "UPDATE agent_rt_checkpoints SET lease_deadline = ?1
         WHERE protocol = ?2 AND response_id = ?3 AND version = ?4
           AND lease_turn_id = ?5 AND lease_deadline = ?6",
        params![
            to_i64(command.new_deadline.0)?,
            P::STORAGE_KEY,
            current.response_id.as_str(),
            to_i64(current.version.0)?,
            current.turn_id.as_str(),
            to_i64(current.deadline.0)?,
        ],
    )?;
    if updated != 1 {
        return Err(StoreInvariantError::VersionConflict.into());
    }
    let renewed = TurnLease {
        deadline: command.new_deadline,
        ..current
    };
    transaction.commit()?;
    Ok(renewed)
}

fn checkpoint_exists<P: AgentProtocol>(
    connection: &Connection,
    response_id: &str,
) -> Result<bool, DuckDbStoreError> {
    connection
        .query_row(
            "SELECT 1 FROM agent_rt_checkpoints WHERE protocol = ?1 AND response_id = ?2",
            params![P::STORAGE_KEY, response_id],
            |_| Ok(()),
        )
        .optional()
        .map(|value| value.is_some())
        .map_err(Into::into)
}

fn load_checkpoint<P: AgentProtocol>(
    connection: &Connection,
    response_id: &str,
    expected_scope: Option<&AuthorizationScope>,
) -> Result<StoredCheckpoint<P>, DuckDbStoreError> {
    let raw = connection
        .query_row(
            "SELECT parent_response_id, tenant_id, principal_id, idempotency_key,
                    request_fingerprint, state, version, request_json, response_json,
                    lease_turn_id, lease_deadline
             FROM agent_rt_checkpoints WHERE protocol = ?1 AND response_id = ?2",
            params![P::STORAGE_KEY, response_id],
            |row| {
                Ok(RawCheckpoint {
                    parent_response_id: row.get(0)?,
                    tenant_id: row.get(1)?,
                    principal_id: row.get(2)?,
                    idempotency_key: row.get(3)?,
                    request_fingerprint: row.get(4)?,
                    state: row.get(5)?,
                    version: row.get(6)?,
                    request_json: row.get(7)?,
                    response_json: row.get(8)?,
                    lease_turn_id: row.get(9)?,
                    lease_deadline: row.get(10)?,
                })
            },
        )
        .optional()?
        .ok_or(StoreInvariantError::NotFound)?;
    let scope = AuthorizationScope {
        tenant_id: raw.tenant_id,
        principal_id: raw.principal_id,
    };
    if expected_scope.is_some_and(|expected| expected != &scope) {
        return Err(StoreInvariantError::NotFound.into());
    }
    let version = CheckpointVersion(from_i64(raw.version)?);
    let fingerprint: [u8; 32] = raw
        .request_fingerprint
        .try_into()
        .map_err(|_| StoreInvariantError::Corrupt)?;
    let mut statement = connection.prepare(
        "SELECT item_json FROM agent_rt_checkpoint_output_items
         WHERE protocol = ?1 AND response_id = ?2 ORDER BY sequence",
    )?;
    let output_rows = statement.query_map(params![P::STORAGE_KEY, response_id], |row| {
        row.get::<_, String>(0)
    })?;
    let mut output_items = Vec::new();
    for row in output_rows {
        output_items.push(serde_json::from_str(&row?)?);
    }
    drop(statement);

    let response = raw
        .response_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()?;
    let record = CheckpointRecord {
        response_id: ResponseId::from(response_id),
        parent_response_id: raw.parent_response_id.map(ResponseId::from),
        scope,
        idempotency_key: IdempotencyKey::from(raw.idempotency_key),
        request_fingerprint: RequestFingerprint::new(fingerprint),
        state: parse_state(&raw.state)?,
        version,
        request: serde_json::from_str(&raw.request_json)?,
        output_items,
        response,
    };
    let lease = match (raw.lease_turn_id, raw.lease_deadline) {
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
    if current.turn_id != supplied.turn_id {
        return Err(StoreInvariantError::LeaseMismatch);
    }
    if current.version != supplied.version {
        return Err(StoreInvariantError::VersionConflict);
    }
    if current.deadline != supplied.deadline {
        return Err(StoreInvariantError::LeaseMismatch);
    }
    if current.deadline.0 <= now {
        return Err(StoreInvariantError::LeaseExpired);
    }
    Ok(current.clone())
}

struct RawCheckpoint {
    parent_response_id: Option<String>,
    tenant_id: String,
    principal_id: String,
    idempotency_key: String,
    request_fingerprint: Vec<u8>,
    state: String,
    version: i64,
    request_json: String,
    response_json: Option<String>,
    lease_turn_id: Option<String>,
    lease_deadline: Option<i64>,
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
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dynamo_agent_rt::{
        AuthorizationScope, Clock, IdempotencyKey, RequestFingerprint, ResponseId, TurnId,
        TurnState,
    };
    use dynamo_agent_rt::{
        BeginTurn, BeginTurnResult, CheckpointStore, CommitTurn, LeaseDeadline, LoadChain,
        OpenAiResponses, RuntimeAuthorization, RuntimeLimits,
    };

    use super::*;

    #[derive(Clone)]
    struct TestClock(Arc<AtomicU64>);

    impl TestClock {
        fn new(now: u64) -> Self {
            Self(Arc::new(AtomicU64::new(now)))
        }

        fn set(&self, now: u64) {
            self.0.store(now, Ordering::Relaxed);
        }
    }

    impl Clock for TestClock {
        fn now_millis(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    fn scope(tenant: &str) -> AuthorizationScope {
        AuthorizationScope {
            tenant_id: tenant.to_owned(),
            principal_id: "principal-a".to_owned(),
        }
    }

    fn begin(response_id: &str, parent: Option<&str>, deadline: u64) -> BeginTurn<OpenAiResponses> {
        BeginTurn {
            response_id: ResponseId::from(response_id),
            turn_id: TurnId::from(format!("turn-{response_id}")),
            parent_response_id: parent.map(ResponseId::from),
            authorization: RuntimeAuthorization {
                scope: scope("tenant-a"),
                permitted_connectors: BTreeSet::new(),
                limits: RuntimeLimits::default(),
            },
            idempotency_key: IdempotencyKey::from(format!("idem-{response_id}")),
            request_fingerprint: RequestFingerprint::new([response_id.len() as u8; 32]),
            request: Default::default(),
            lease_deadline: LeaseDeadline(deadline),
        }
    }

    #[tokio::test]
    async fn survives_restart_and_loads_parent_first() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("agent.duckdb");
        let clock = TestClock::new(1_000);
        let store =
            DuckDbStore::<OpenAiResponses, _>::open_with_clock(&path, clock.clone()).unwrap();
        let BeginTurnResult::Acquired(parent) = store
            .begin_turn(begin("parent", None, 2_000))
            .await
            .unwrap()
        else {
            panic!("expected parent lease");
        };
        store
            .commit_turn(CommitTurn {
                lease: parent,
                next_state: TurnState::Completed,
                append_output_items: Vec::new(),
                response: None,
            })
            .await
            .unwrap();
        store
            .begin_turn(begin("child", Some("parent"), 2_000))
            .await
            .unwrap();
        drop(store);

        let reopened = DuckDbStore::<OpenAiResponses, _>::open_with_clock(&path, clock).unwrap();
        let chain = reopened
            .load_chain(LoadChain {
                scope: scope("tenant-a"),
                response_id: ResponseId::from("child"),
            })
            .await
            .unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].response_id.as_str(), "parent");
        assert_eq!(chain[1].response_id.as_str(), "child");
    }

    #[tokio::test]
    async fn expired_owner_is_replaced_and_stale_commit_is_fenced() {
        let clock = TestClock::new(1_000);
        let store = DuckDbStore::<OpenAiResponses, _>::from_connection(
            Connection::open_in_memory().unwrap(),
            clock.clone(),
        )
        .unwrap();
        let original = begin("response", None, 2_000);
        let BeginTurnResult::Acquired(old) = store.begin_turn(original.clone()).await.unwrap()
        else {
            panic!("expected original lease");
        };
        clock.set(2_000);
        let replacement = BeginTurn {
            response_id: ResponseId::from("ignored"),
            turn_id: TurnId::from("replacement"),
            lease_deadline: LeaseDeadline(3_000),
            ..original
        };
        let BeginTurnResult::Acquired(new) = store.begin_turn(replacement).await.unwrap() else {
            panic!("expected replacement lease");
        };
        assert_eq!(new.response_id.as_str(), "response");
        assert_eq!(new.version.0, 1);
        let error = store
            .commit_turn(CommitTurn {
                lease: old,
                next_state: TurnState::Failed,
                append_output_items: Vec::new(),
                response: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            DuckDbStoreError::Invariant(
                StoreInvariantError::LeaseMismatch | StoreInvariantError::VersionConflict
            )
        ));
    }

    #[tokio::test]
    async fn cross_scope_chain_reads_are_denied() {
        let store = DuckDbStore::<OpenAiResponses>::open_in_memory().unwrap();
        store
            .begin_turn(begin("response", None, u64::MAX / 2))
            .await
            .unwrap();
        let error = store
            .load_chain(LoadChain {
                scope: scope("tenant-b"),
                response_id: ResponseId::from("response"),
            })
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            DuckDbStoreError::Invariant(StoreInvariantError::NotFound)
        ));
    }
}
