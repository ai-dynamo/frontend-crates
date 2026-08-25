// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{AgentProtocol, BoxFuture, BoxStream};

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
pub struct InferenceRequest<P, C>
where
    P: AgentProtocol,
{
    pub request: P::Request,
    pub context: C,
    pub intent: InferenceIntent,
}

/// Unary or streaming inference result normalized at the invocation boundary.
pub enum InferenceOutput<'a, P, E>
where
    P: AgentProtocol,
{
    Unary(Box<P::Response>),
    Streaming(BoxStream<'a, Result<P::StreamEvent, E>>),
}

/// Future returned by a native protocol inference invocation.
pub type InferenceFuture<'a, P, E> = BoxFuture<'a, Result<InferenceOutput<'a, P, E>, E>>;

/// Invokes inference without exposing engine or routing details to the runtime.
pub trait InferenceInvoker<P>: Send + Sync + 'static
where
    P: AgentProtocol,
{
    type Context: InvocationContext;
    type Error: std::error::Error + Send + Sync + 'static;

    fn invoke(
        &self,
        request: InferenceRequest<P, Self::Context>,
    ) -> InferenceFuture<'_, P, Self::Error>;
}
