// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::{AgentProtocol, AuthorizationScope, BoxFuture, IdempotencyKey, ResponseId};

/// Credential-free external call selected by trusted deployment routing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeToolCall {
    pub call_id: String,
    pub connector: String,
    pub operation: String,
    pub arguments: serde_json::Value,
}

/// Result paired with its normalized call for deterministic protocol replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeToolResult {
    pub call: RuntimeToolCall,
    pub result: ToolExecutionResult,
}

/// Protocol-specific extraction and continuation mutation for runtime tools.
pub trait ToolLoopAdapter<P>: Send + Sync + 'static
where
    P: AgentProtocol,
{
    type Error: std::error::Error + Send + Sync + 'static;

    /// Extract calls owned by the runtime according to trusted server policy.
    fn runtime_calls(&self, response: &P::Response) -> Result<Vec<RuntimeToolCall>, Self::Error>;

    /// Append the model response and ordered tool results to the next native
    /// inference request. Returns only the new result replay items that must be
    /// appended to the active checkpoint.
    fn append_results(
        &self,
        request: &mut P::Request,
        response: &P::Response,
        results: &[RuntimeToolResult],
    ) -> Result<Vec<P::ReplayItem>, Self::Error>;
}

/// Trusted server-side mapping from a model-visible tool name to an executor.
pub trait ToolRouter: Send + Sync + 'static {
    fn route(&self, tool_name: &str) -> Option<ToolRoute>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRoute {
    pub connector: String,
    pub operation: String,
}

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
