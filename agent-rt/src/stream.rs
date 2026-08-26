// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{AgentProtocol, ModelStepKind, OutputIdentity};

/// Result of observing one native inference event.
///
/// The runtime emits ordinary events immediately. It retains terminal events
/// until the corresponding checkpoint transition commits.
#[derive(Debug)]
pub enum StreamEventAction<P>
where
    P: AgentProtocol,
{
    Emit(P::StreamEvent),
    Suppress,
    Terminal {
        event: P::StreamEvent,
        response: P::Response,
    },
}

/// Protocol-specific stream semantics used by the generic runtime.
///
/// Implementations operate on native DTOs. They may hide lifecycle events for
/// internal model steps and rewrite public identity or ordering, but they do
/// not serialize events or own transport buffering.
pub trait StreamEventInterpreter<P>: Send + 'static
where
    P: AgentProtocol,
{
    type Error: std::error::Error + Send + Sync + 'static;

    fn begin_step(&mut self, step_kind: ModelStepKind);

    fn observe(
        &mut self,
        event: P::StreamEvent,
        identity: &OutputIdentity,
    ) -> Result<StreamEventAction<P>, Self::Error>;

    /// Applies public ordering immediately before an event is yielded.
    /// Suppressed terminal events therefore consume no public sequence value.
    fn prepare_emit(&mut self, event: &mut P::StreamEvent);
}
