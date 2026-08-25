// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use thiserror::Error;

use crate::{
    AgentProtocol, AuthorizationScope, BeginTurn, BeginTurnResult, BoxFuture, CheckpointRecord,
    CheckpointStore, CheckpointVersion, Clock, CommitTurn, CommitTurnResult, IdempotencyKey,
    LoadChain, OpenAiResponses, RenewLease, ResponseId, SystemClock, TurnLease, TurnState,
};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InMemoryStoreError {
    #[error("checkpoint store mutex is poisoned")]
    Poisoned,
    #[error("response {0} already exists")]
    ResponseAlreadyExists(ResponseId),
    #[error("idempotency key was reused with a different request")]
    IdempotencyConflict,
    #[error("response checkpoint was not found")]
    NotFound,
    #[error("parent response is not replayable in state {0:?}")]
    ParentNotReplayable(TurnState),
    #[error("lease deadline must be in the future")]
    InvalidLeaseDeadline,
    #[error("renewed lease deadline must extend the current lease")]
    LeaseDeadlineNotExtended,
    #[error("turn lease was not found")]
    LeaseNotFound,
    #[error("turn lease does not own the checkpoint")]
    LeaseMismatch,
    #[error("turn lease expired")]
    LeaseExpired,
    #[error("checkpoint version conflict")]
    VersionConflict,
    #[error("invalid state transition from {from:?} to {to:?}")]
    InvalidTransition { from: TurnState, to: TurnState },
    #[error("checkpoint version overflow")]
    VersionOverflow,
    #[error("response chain is corrupt")]
    CorruptChain,
}

#[derive(Debug)]
struct StoreState<P>
where
    P: AgentProtocol,
{
    records: HashMap<ResponseId, CheckpointRecord<P>>,
    idempotency: HashMap<(AuthorizationScope, IdempotencyKey), ResponseId>,
    leases: HashMap<ResponseId, TurnLease>,
}

impl<P> Default for StoreState<P>
where
    P: AgentProtocol,
{
    fn default() -> Self {
        Self {
            records: HashMap::new(),
            idempotency: HashMap::new(),
            leases: HashMap::new(),
        }
    }
}

/// Single-process checkpoint store for tests and local proofs of concept.
///
/// A mutex protects each multi-index operation so turn creation, idempotency
/// lookup, and lease fencing are atomic with respect to each other.
#[derive(Debug)]
pub struct InMemoryCheckpointStore<P = OpenAiResponses, C = SystemClock>
where
    P: AgentProtocol,
{
    clock: C,
    state: Mutex<StoreState<P>>,
}

impl<P> Default for InMemoryCheckpointStore<P, SystemClock>
where
    P: AgentProtocol,
{
    fn default() -> Self {
        Self::new(SystemClock)
    }
}

impl<P, C> InMemoryCheckpointStore<P, C>
where
    P: AgentProtocol,
{
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            state: Mutex::new(StoreState::default()),
        }
    }
}

impl<P, C> InMemoryCheckpointStore<P, C>
where
    P: AgentProtocol,
    C: Clock,
{
    fn validate_lease(
        &self,
        state: &StoreState<P>,
        supplied: &TurnLease,
    ) -> Result<TurnLease, InMemoryStoreError> {
        let current = state
            .leases
            .get(&supplied.response_id)
            .ok_or(InMemoryStoreError::LeaseNotFound)?;
        if current.turn_id != supplied.turn_id {
            return Err(InMemoryStoreError::LeaseMismatch);
        }
        if current.version != supplied.version {
            return Err(InMemoryStoreError::VersionConflict);
        }
        if current.deadline.0 <= self.clock.now_millis() {
            return Err(InMemoryStoreError::LeaseExpired);
        }
        Ok(current.clone())
    }
}

impl<P, C> CheckpointStore<P> for InMemoryCheckpointStore<P, C>
where
    P: AgentProtocol,
    C: Clock,
{
    type Error = InMemoryStoreError;

    fn begin_turn(
        &self,
        command: BeginTurn<P>,
    ) -> BoxFuture<'_, Result<BeginTurnResult<P>, Self::Error>> {
        Box::pin(async move {
            let now = self.clock.now_millis();
            if command.lease_deadline.0 <= now {
                return Err(InMemoryStoreError::InvalidLeaseDeadline);
            }

            let mut state = self
                .state
                .lock()
                .map_err(|_| InMemoryStoreError::Poisoned)?;
            let idempotency = (
                command.authorization.scope.clone(),
                command.idempotency_key.clone(),
            );
            if let Some(existing_id) = state.idempotency.get(&idempotency) {
                let existing = state
                    .records
                    .get(existing_id)
                    .ok_or(InMemoryStoreError::CorruptChain)?;
                if existing.parent_response_id != command.parent_response_id
                    || existing.request_fingerprint != command.request_fingerprint
                {
                    return Err(InMemoryStoreError::IdempotencyConflict);
                }
                return Ok(BeginTurnResult::Existing(Box::new(existing.clone())));
            }

            if state.records.contains_key(&command.response_id) {
                return Err(InMemoryStoreError::ResponseAlreadyExists(
                    command.response_id,
                ));
            }

            if let Some(parent_id) = &command.parent_response_id {
                let parent = state
                    .records
                    .get(parent_id)
                    .filter(|record| record.scope == command.authorization.scope)
                    .ok_or(InMemoryStoreError::NotFound)?;
                if !matches!(
                    parent.state,
                    TurnState::Completed | TurnState::AwaitingClientToolOutput
                ) {
                    return Err(InMemoryStoreError::ParentNotReplayable(
                        parent.state.clone(),
                    ));
                }
            }

            let record = CheckpointRecord {
                response_id: command.response_id.clone(),
                parent_response_id: command.parent_response_id,
                scope: command.authorization.scope,
                idempotency_key: command.idempotency_key,
                request_fingerprint: command.request_fingerprint,
                state: TurnState::InFlight,
                version: CheckpointVersion(0),
                request: command.request,
                output_items: Vec::new(),
            };
            let lease = TurnLease {
                response_id: record.response_id.clone(),
                turn_id: command.turn_id,
                version: record.version,
                deadline: command.lease_deadline,
            };

            state
                .idempotency
                .insert(idempotency, record.response_id.clone());
            state
                .leases
                .insert(record.response_id.clone(), lease.clone());
            state.records.insert(record.response_id.clone(), record);
            Ok(BeginTurnResult::Acquired(lease))
        })
    }

    fn load_chain(
        &self,
        query: LoadChain,
    ) -> BoxFuture<'_, Result<Vec<CheckpointRecord<P>>, Self::Error>> {
        Box::pin(async move {
            let state = self
                .state
                .lock()
                .map_err(|_| InMemoryStoreError::Poisoned)?;
            let mut current = Some(query.response_id);
            let mut seen = HashSet::new();
            let mut reversed = Vec::new();

            while let Some(response_id) = current {
                if !seen.insert(response_id.clone()) {
                    return Err(InMemoryStoreError::CorruptChain);
                }
                let record = state
                    .records
                    .get(&response_id)
                    .filter(|record| record.scope == query.scope)
                    .ok_or(InMemoryStoreError::NotFound)?;
                current = record.parent_response_id.clone();
                reversed.push(record.clone());
            }

            reversed.reverse();
            Ok(reversed)
        })
    }

    fn commit_turn(
        &self,
        command: CommitTurn<P>,
    ) -> BoxFuture<'_, Result<CommitTurnResult<P>, Self::Error>> {
        Box::pin(async move {
            let mut state = self
                .state
                .lock()
                .map_err(|_| InMemoryStoreError::Poisoned)?;
            let current_lease = self.validate_lease(&state, &command.lease)?;
            let record = state
                .records
                .get_mut(&command.lease.response_id)
                .ok_or(InMemoryStoreError::NotFound)?;
            if record.version != current_lease.version {
                return Err(InMemoryStoreError::VersionConflict);
            }
            if !record.state.permits_transition_to(&command.next_state) {
                return Err(InMemoryStoreError::InvalidTransition {
                    from: record.state.clone(),
                    to: command.next_state,
                });
            }

            record.version = CheckpointVersion(
                record
                    .version
                    .0
                    .checked_add(1)
                    .ok_or(InMemoryStoreError::VersionOverflow)?,
            );
            record.state = command.next_state;
            record.output_items.extend(command.append_output_items);
            let record = record.clone();

            let lease = if matches!(record.state, TurnState::InFlight | TurnState::ToolStarted) {
                let updated = TurnLease {
                    version: record.version,
                    ..current_lease
                };
                state
                    .leases
                    .insert(record.response_id.clone(), updated.clone());
                Some(updated)
            } else {
                state.leases.remove(&record.response_id);
                None
            };

            Ok(CommitTurnResult { record, lease })
        })
    }

    fn renew_lease(&self, command: RenewLease) -> BoxFuture<'_, Result<TurnLease, Self::Error>> {
        Box::pin(async move {
            if command.new_deadline.0 <= self.clock.now_millis() {
                return Err(InMemoryStoreError::InvalidLeaseDeadline);
            }
            let mut state = self
                .state
                .lock()
                .map_err(|_| InMemoryStoreError::Poisoned)?;
            let current = self.validate_lease(&state, &command.lease)?;
            if command.new_deadline <= current.deadline {
                return Err(InMemoryStoreError::LeaseDeadlineNotExtended);
            }
            let updated = TurnLease {
                deadline: command.new_deadline,
                ..current
            };
            state
                .leases
                .insert(updated.response_id.clone(), updated.clone());
            Ok(updated)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dynamo_protocols::types::responses::CreateResponse;

    use super::{InMemoryCheckpointStore, InMemoryStoreError};
    use crate::{
        AuthorizationScope, BeginTurn, BeginTurnResult, CheckpointStore, Clock, CommitTurn,
        IdempotencyKey, LeaseDeadline, LoadChain, OpenAiResponses, RenewLease, RequestFingerprint,
        ResponseId, RuntimeAuthorization, RuntimeLimits, TurnId, TurnState,
    };

    #[derive(Debug, Clone)]
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

    fn authorization(tenant: &str) -> RuntimeAuthorization {
        RuntimeAuthorization {
            scope: AuthorizationScope {
                tenant_id: tenant.to_owned(),
                principal_id: "principal".to_owned(),
            },
            permitted_connectors: BTreeSet::new(),
            limits: RuntimeLimits::default(),
        }
    }

    fn begin(
        response_id: &str,
        parent_response_id: Option<&str>,
        tenant: &str,
        idempotency_key: &str,
    ) -> BeginTurn<OpenAiResponses> {
        BeginTurn {
            response_id: ResponseId::from(response_id),
            turn_id: TurnId::from(format!("turn_{response_id}")),
            parent_response_id: parent_response_id.map(ResponseId::from),
            authorization: authorization(tenant),
            idempotency_key: IdempotencyKey::from(idempotency_key),
            request_fingerprint: RequestFingerprint::new([response_id.len() as u8; 32]),
            request: CreateResponse::default(),
            lease_deadline: LeaseDeadline(2_000),
        }
    }

    #[tokio::test]
    async fn duplicate_idempotency_key_returns_existing_turn() {
        let store = InMemoryCheckpointStore::new(TestClock::new(1_000));
        let first = begin("resp_one", None, "tenant", "idem");
        assert!(matches!(
            store.begin_turn(first.clone()).await.unwrap(),
            BeginTurnResult::Acquired(_)
        ));

        let duplicate = BeginTurn {
            response_id: ResponseId::from("resp_ignored"),
            turn_id: TurnId::from("turn_ignored"),
            ..first
        };
        let BeginTurnResult::Existing(existing) = store.begin_turn(duplicate).await.unwrap() else {
            panic!("expected existing turn");
        };
        assert_eq!(existing.response_id.as_str(), "resp_one");
    }

    #[tokio::test]
    async fn duplicate_idempotency_key_rejects_a_different_fingerprint() {
        let store = InMemoryCheckpointStore::new(TestClock::new(1_000));
        store
            .begin_turn(begin("resp_one", None, "tenant", "idem"))
            .await
            .unwrap();
        let mut conflicting = begin("resp_two", None, "tenant", "idem");
        conflicting.request_fingerprint = RequestFingerprint::new([99; 32]);

        assert_eq!(
            store.begin_turn(conflicting).await.unwrap_err(),
            InMemoryStoreError::IdempotencyConflict
        );
    }

    #[tokio::test]
    async fn concurrent_duplicate_claims_have_one_owner() {
        let store = InMemoryCheckpointStore::new(TestClock::new(1_000));
        let first = begin("resp_one", None, "tenant", "idem");
        let second = BeginTurn {
            response_id: ResponseId::from("resp_two"),
            turn_id: TurnId::from("turn_two"),
            ..first.clone()
        };

        let (left, right) = tokio::join!(store.begin_turn(first), store.begin_turn(second));
        let outcomes = [left.unwrap(), right.unwrap()];
        assert_eq!(
            outcomes
                .iter()
                .filter(|result| matches!(result, BeginTurnResult::Acquired(_)))
                .count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|result| matches!(result, BeginTurnResult::Existing(_)))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn commit_is_version_fenced_and_returns_updated_lease() {
        let store = InMemoryCheckpointStore::new(TestClock::new(1_000));
        let BeginTurnResult::Acquired(first_lease) = store
            .begin_turn(begin("resp_one", None, "tenant", "idem"))
            .await
            .unwrap()
        else {
            panic!("expected lease");
        };

        let started = store
            .commit_turn(CommitTurn {
                lease: first_lease.clone(),
                next_state: TurnState::ToolStarted,
                append_output_items: Vec::new(),
            })
            .await
            .unwrap();
        let updated_lease = started.lease.expect("tool work retains lease");
        assert_eq!(updated_lease.version.0, 1);

        let stale = store
            .commit_turn(CommitTurn {
                lease: first_lease,
                next_state: TurnState::Failed,
                append_output_items: Vec::new(),
            })
            .await;
        assert_eq!(stale.unwrap_err(), InMemoryStoreError::VersionConflict);

        let completed = store
            .commit_turn(CommitTurn {
                lease: updated_lease,
                next_state: TurnState::InFlight,
                append_output_items: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(completed.record.version.0, 2);
        assert!(completed.lease.is_some());
    }

    #[tokio::test]
    async fn load_chain_is_parent_first_and_scope_isolated() {
        let store = InMemoryCheckpointStore::new(TestClock::new(1_000));
        let BeginTurnResult::Acquired(parent_lease) = store
            .begin_turn(begin("resp_parent", None, "tenant", "parent"))
            .await
            .unwrap()
        else {
            panic!("expected parent lease");
        };
        store
            .commit_turn(CommitTurn {
                lease: parent_lease,
                next_state: TurnState::Completed,
                append_output_items: Vec::new(),
            })
            .await
            .unwrap();
        store
            .begin_turn(begin("resp_child", Some("resp_parent"), "tenant", "child"))
            .await
            .unwrap();

        let chain = store
            .load_chain(LoadChain {
                scope: authorization("tenant").scope,
                response_id: ResponseId::from("resp_child"),
            })
            .await
            .unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].response_id.as_str(), "resp_parent");
        assert_eq!(chain[1].response_id.as_str(), "resp_child");

        let wrong_scope = store
            .load_chain(LoadChain {
                scope: authorization("other").scope,
                response_id: ResponseId::from("resp_child"),
            })
            .await;
        assert_eq!(wrong_scope.unwrap_err(), InMemoryStoreError::NotFound);
    }

    #[tokio::test]
    async fn expired_lease_cannot_commit() {
        let clock = TestClock::new(1_000);
        let store = InMemoryCheckpointStore::new(clock.clone());
        let BeginTurnResult::Acquired(lease) = store
            .begin_turn(begin("resp_one", None, "tenant", "idem"))
            .await
            .unwrap()
        else {
            panic!("expected lease");
        };
        clock.set(2_000);

        let result = store
            .commit_turn(CommitTurn {
                lease,
                next_state: TurnState::Failed,
                append_output_items: Vec::new(),
            })
            .await;
        assert_eq!(result.unwrap_err(), InMemoryStoreError::LeaseExpired);
    }

    #[tokio::test]
    async fn renewal_must_extend_a_live_lease() {
        let store = InMemoryCheckpointStore::new(TestClock::new(1_000));
        let BeginTurnResult::Acquired(lease) = store
            .begin_turn(begin("resp_one", None, "tenant", "idem"))
            .await
            .unwrap()
        else {
            panic!("expected lease");
        };

        let unchanged = store
            .renew_lease(RenewLease {
                lease: lease.clone(),
                new_deadline: lease.deadline,
            })
            .await;
        assert_eq!(
            unchanged.unwrap_err(),
            InMemoryStoreError::LeaseDeadlineNotExtended
        );

        let renewed = store
            .renew_lease(RenewLease {
                lease,
                new_deadline: LeaseDeadline(3_000),
            })
            .await
            .unwrap();
        assert_eq!(renewed.deadline, LeaseDeadline(3_000));
    }
}
