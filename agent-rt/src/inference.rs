// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::responses::{CreateResponse, Response, ResponseStreamEvent};

use crate::{BoxFuture, BoxStream};

/// Marker for frontend-owned invocation metadata.
///
/// The runtime only clones and forwards this value. The invoker implementation
/// owns its contents and conversion into backend-specific request metadata.
pub trait InvocationContext: Clone + Send + Sync + 'static {}

impl<T> InvocationContext for T where T: Clone + Send + Sync + 'static {}

/// Semantic origin of one inference step in a public response turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStepKind {
    Initial,
    ClientToolContinuation,
    RuntimeToolContinuation,
}

/// Per-step lifecycle intent supplied to an inference implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceIntent {
    pub step_kind: ModelStepKind,
    /// True only when the runtime knows no additional model step follows.
    pub session_final: bool,
}

/// Fully materialized native Responses invocation.
#[derive(Debug, Clone)]
pub struct InferenceRequest<C> {
    pub request: CreateResponse,
    pub context: C,
    pub intent: InferenceIntent,
}

/// Unary or streaming inference result normalized at the invocation boundary.
pub enum InferenceOutput<'a, E> {
    Unary(Box<Response>),
    Streaming(BoxStream<'a, Result<ResponseStreamEvent, E>>),
}

/// Invokes inference without exposing engine or routing details to the runtime.
pub trait InferenceInvoker: Send + Sync + 'static {
    type Context: InvocationContext;
    type Error: std::error::Error + Send + Sync + 'static;

    fn invoke(
        &self,
        request: InferenceRequest<Self::Context>,
    ) -> BoxFuture<'_, Result<InferenceOutput<'_, Self::Error>, Self::Error>>;
}
