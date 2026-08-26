// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{AgentProtocol, AuthorizationScope, BoxFuture, IdempotencyKey, ResponseId};

/// Credential-free external call selected by trusted deployment routing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeToolCall {
    pub call_id: String,
    pub connector: String,
    pub operation: String,
    /// Trusted deployment profile selected by [`ToolRouter`]. This is never
    /// taken from model-generated arguments.
    pub profile: String,
    pub arguments: serde_json::Value,
}

/// Result paired with its normalized call for deterministic protocol replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeToolResult {
    pub call: RuntimeToolCall,
    pub result: ToolExecutionResult,
}

/// Derives a stable executor idempotency key for one tool-call attempt.
pub trait ToolIdempotencyKeyProvider: Send + Sync + 'static {
    fn idempotency_key(
        &self,
        response_id: &ResponseId,
        call: &RuntimeToolCall,
        attempt: u32,
    ) -> IdempotencyKey;
}

/// BLAKE3-backed deterministic tool idempotency keys.
#[derive(Debug, Clone, Copy, Default)]
pub struct Blake3ToolIdempotencyKeys;

impl ToolIdempotencyKeyProvider for Blake3ToolIdempotencyKeys {
    fn idempotency_key(
        &self,
        response_id: &ResponseId,
        call: &RuntimeToolCall,
        attempt: u32,
    ) -> IdempotencyKey {
        let mut hasher = blake3::Hasher::new();
        for value in [
            response_id.as_str(),
            call.call_id.as_str(),
            call.connector.as_str(),
            call.operation.as_str(),
            call.profile.as_str(),
        ] {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update(&attempt.to_le_bytes());
        IdempotencyKey::new(format!("tool_{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolFailureDisposition {
    Failed(ToolExecutionFailure),
    OutcomeUnknown,
}

/// Classifies whether an executor error proves that no unknown side effect is
/// outstanding. Implementations must be conservative for transport failures.
pub trait ToolFailurePolicy<E>: Send + Sync + 'static
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn classify(&self, error: &E) -> ToolFailureDisposition;
}

/// Safe default for executors without an explicit side-effect contract.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConservativeToolFailurePolicy;

impl<E> ToolFailurePolicy<E> for ConservativeToolFailurePolicy
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn classify(&self, _error: &E) -> ToolFailureDisposition {
        ToolFailureDisposition::OutcomeUnknown
    }
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
    pub profile: String,
}

impl ToolRoute {
    pub fn new(connector: impl Into<String>, operation: impl Into<String>) -> Self {
        Self {
            connector: connector.into(),
            operation: operation.into(),
            profile: "default".to_owned(),
        }
    }

    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = profile.into();
        self
    }
}

/// Immutable trusted routing table built by frontend deployment policy.
#[derive(Debug, Clone, Default)]
pub struct ConfiguredToolRouter {
    routes: HashMap<String, ToolRoute>,
}

impl ConfiguredToolRouter {
    pub fn new(routes: impl IntoIterator<Item = (String, ToolRoute)>) -> Self {
        Self {
            routes: routes.into_iter().collect(),
        }
    }
}

impl ToolRouter for ConfiguredToolRouter {
    fn route(&self, tool_name: &str) -> Option<ToolRoute> {
        self.routes.get(tool_name).cloned()
    }
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
    pub profile: String,
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
    /// A completed, model-visible tool error. This is not an executor,
    /// transport, or protocol failure and is therefore journaled as completed.
    #[serde(default)]
    pub is_error: bool,
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
    /// An executor for an explicitly side-effect-free operation may safely
    /// re-execute the request and return that result from this method.
    /// `None` means the executor cannot prove whether the side effect occurred.
    fn lookup<'a>(
        &'a self,
        request: &'a ToolExecutionRequest,
    ) -> BoxFuture<'a, Result<Option<ToolExecutionResult>, Self::Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_tool_results_default_to_success() {
        let result: ToolExecutionResult =
            serde_json::from_value(serde_json::json!({"output": {"answer": 42}})).unwrap();

        assert_eq!(result.output["answer"], 42);
        assert!(!result.is_error);
    }
}
