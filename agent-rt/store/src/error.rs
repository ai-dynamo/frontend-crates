// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_agent_rt::{ResponseId, ToolJournalState, TurnState};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StoreInvariantError {
    #[error("response {0} already exists")]
    ResponseAlreadyExists(ResponseId),
    #[error("idempotency key was reused with a different request")]
    IdempotencyConflict,
    #[error("record was not found")]
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
    #[error("tool journal record is already in terminal state {0:?}")]
    ToolAlreadyFinished(ToolJournalState),
    #[error("persisted state is corrupt")]
    Corrupt,
}
