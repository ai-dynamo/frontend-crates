// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::{AuthorizationScope, BoxFuture, IdempotencyKey, ResponseId};

/// Durable lookup key for one external tool execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolJournalKey {
    pub scope: AuthorizationScope,
    pub idempotency_key: IdempotencyKey,
}

/// Already-authorized request sent to a runtime-owned tool worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionRequest {
    pub response_id: ResponseId,
    pub call_id: String,
    pub connector: String,
    pub operation: String,
    pub arguments: serde_json::Value,
    pub scope: AuthorizationScope,
    pub idempotency_key: IdempotencyKey,
    pub attempt: u32,
}

impl ToolExecutionRequest {
    pub fn journal_key(&self) -> ToolJournalKey {
        ToolJournalKey {
            scope: self.scope.clone(),
            idempotency_key: self.idempotency_key.clone(),
        }
    }
}

/// Normalized result safe to persist and append to the next model request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub output: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolJournalState {
    Started,
    Completed,
    Failed,
    OutcomeUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolExecutionFailure {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

/// Credential-free durable record used to recover external side effects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolJournalRecord {
    pub request: ToolExecutionRequest,
    pub state: ToolJournalState,
    pub result: Option<ToolExecutionResult>,
    pub failure: Option<ToolExecutionFailure>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolClaimResult {
    Acquired(Box<ToolJournalRecord>),
    Existing(Box<ToolJournalRecord>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolJournalOutcome {
    Completed(ToolExecutionResult),
    Failed(ToolExecutionFailure),
    OutcomeUnknown,
}

/// Durable idempotency and outcome boundary for runtime-owned tools.
pub trait ToolJournal: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn claim(
        &self,
        request: ToolExecutionRequest,
    ) -> BoxFuture<'_, Result<ToolClaimResult, Self::Error>>;

    fn load(
        &self,
        key: &ToolJournalKey,
    ) -> BoxFuture<'_, Result<Option<ToolJournalRecord>, Self::Error>>;

    fn finish(
        &self,
        key: ToolJournalKey,
        outcome: ToolJournalOutcome,
    ) -> BoxFuture<'_, Result<ToolJournalRecord, Self::Error>>;
}

/// External runtime-owned tool execution boundary.
pub trait ToolExecutor: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn execute(
        &self,
        request: ToolExecutionRequest,
    ) -> BoxFuture<'_, Result<ToolExecutionResult, Self::Error>>;

    /// Resolves a previously dispatched idempotency key after process failure.
    /// `None` means the executor cannot prove whether the side effect occurred.
    fn lookup(
        &self,
        scope: &AuthorizationScope,
        idempotency_key: &IdempotencyKey,
    ) -> BoxFuture<'_, Result<Option<ToolExecutionResult>, Self::Error>>;
}
