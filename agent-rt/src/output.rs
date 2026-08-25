// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{AgentProtocol, TurnState};

/// Native response plus the durable model-visible result of one inference step.
#[derive(Debug, Clone)]
pub struct InterpretedOutput<P>
where
    P: AgentProtocol,
{
    pub response: P::Response,
    pub replay_items: Vec<P::ReplayItem>,
    /// Must be a legal transition from `TurnState::InFlight`.
    pub next_state: TurnState,
}

/// Converts a native protocol response into checkpoint state.
///
/// Tool ownership and stop semantics remain protocol/deployment policy. The
/// coordinator only applies the returned fenced transition.
pub trait OutputInterpreter<P>: Send + Sync + 'static
where
    P: AgentProtocol,
{
    type Error: std::error::Error + Send + Sync + 'static;

    fn interpret(&self, response: P::Response) -> Result<InterpretedOutput<P>, Self::Error>;
}
