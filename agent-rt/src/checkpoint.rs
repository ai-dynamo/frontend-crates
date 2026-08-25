// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::responses::{CreateResponse, InputItem};
use serde::{Deserialize, Serialize};

use crate::{
    AuthorizationScope, BoxFuture, IdempotencyKey, ResponseId, RuntimeAuthorization, TurnId,
};

/// Monotonic checkpoint version used for fenced state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CheckpointVersion(pub u64);

/// Absolute lease deadline expressed as Unix epoch milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LeaseDeadline(pub u64);

/// Durable state of one public Responses turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnState {
    InFlight,
    AwaitingClientToolOutput,
    ToolStarted,
    OutcomeUnknown,
    Completed,
    Failed,
}

impl TurnState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::OutcomeUnknown | Self::Completed | Self::Failed)
    }

    pub fn permits_transition_to(&self, next: &Self) -> bool {
        matches!(
            (self, next),
            (Self::InFlight, Self::AwaitingClientToolOutput)
                | (Self::InFlight, Self::ToolStarted)
                | (Self::InFlight, Self::Completed)
                | (Self::InFlight, Self::Failed)
                | (Self::AwaitingClientToolOutput, Self::InFlight)
                | (Self::ToolStarted, Self::InFlight)
                | (Self::ToolStarted, Self::OutcomeUnknown)
                | (Self::ToolStarted, Self::Failed)
        )
    }
}

/// Fenced ownership of an in-flight response turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnLease {
    pub response_id: ResponseId,
    pub turn_id: TurnId,
    pub version: CheckpointVersion,
    pub deadline: LeaseDeadline,
}

/// Append-oriented checkpoint record for one response in a response chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRecord {
    pub response_id: ResponseId,
    pub parent_response_id: Option<ResponseId>,
    pub scope: AuthorizationScope,
    pub idempotency_key: IdempotencyKey,
    pub state: TurnState,
    pub version: CheckpointVersion,
    /// Original input and effective per-turn request controls. Implementations
    /// must not place bearer credentials or arbitrary headers in this value.
    pub request: CreateResponse,
    /// Model-visible items produced by this turn and replayable by descendants.
    pub output_items: Vec<InputItem>,
}

/// Atomically creates and claims a new response turn.
#[derive(Debug, Clone, PartialEq)]
pub struct BeginTurn {
    pub response_id: ResponseId,
    pub turn_id: TurnId,
    pub parent_response_id: Option<ResponseId>,
    pub authorization: RuntimeAuthorization,
    pub idempotency_key: IdempotencyKey,
    pub request: CreateResponse,
    pub lease_deadline: LeaseDeadline,
}

/// Result of an idempotent turn claim.
#[derive(Debug, Clone, PartialEq)]
pub enum BeginTurnResult {
    Acquired(TurnLease),
    Existing(Box<CheckpointRecord>),
}

/// Loads a complete parent-first response chain within an authenticated scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadChain {
    pub scope: AuthorizationScope,
    pub response_id: ResponseId,
}

/// Applies one fenced durable state transition.
#[derive(Debug, Clone, PartialEq)]
pub struct CommitTurn {
    pub lease: TurnLease,
    pub next_state: TurnState,
    pub append_output_items: Vec<InputItem>,
}

/// Result of a durable state transition.
#[derive(Debug, Clone, PartialEq)]
pub struct CommitTurnResult {
    pub record: CheckpointRecord,
    /// Updated lease for another nonterminal transition. `None` means the
    /// transition released ownership of this turn.
    pub lease: Option<TurnLease>,
}

/// Extends a live lease without changing checkpoint state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenewLease {
    pub lease: TurnLease,
    pub new_deadline: LeaseDeadline,
}

/// Durable checkpoint storage boundary.
///
/// Implementations must make `begin_turn` idempotent for `(scope,
/// idempotency_key)`, validate parent access before acquiring a lease, and
/// fence every commit by turn id and checkpoint version.
pub trait CheckpointStore: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn begin_turn(&self, command: BeginTurn)
    -> BoxFuture<'_, Result<BeginTurnResult, Self::Error>>;

    fn load_chain(
        &self,
        query: LoadChain,
    ) -> BoxFuture<'_, Result<Vec<CheckpointRecord>, Self::Error>>;

    fn commit_turn(
        &self,
        command: CommitTurn,
    ) -> BoxFuture<'_, Result<CommitTurnResult, Self::Error>>;

    fn renew_lease(&self, command: RenewLease) -> BoxFuture<'_, Result<TurnLease, Self::Error>>;
}

#[cfg(test)]
mod tests {
    use super::TurnState;

    #[test]
    fn terminal_states_cannot_transition() {
        for terminal in [
            TurnState::OutcomeUnknown,
            TurnState::Completed,
            TurnState::Failed,
        ] {
            assert!(terminal.is_terminal());
            assert!(!terminal.permits_transition_to(&TurnState::InFlight));
        }
    }

    #[test]
    fn tool_started_can_resume_or_stop_with_unknown_outcome() {
        assert!(TurnState::ToolStarted.permits_transition_to(&TurnState::InFlight));
        assert!(TurnState::ToolStarted.permits_transition_to(&TurnState::OutcomeUnknown));
    }
}
