// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::{AuthorizationScope, BoxFuture, IdempotencyKey, ResponseId};

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

/// Normalized result safe to persist and append to the next model request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub output: serde_json::Value,
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
