// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;

use dynamo_protocols::types::anthropic::{
    AnthropicCreateMessageRequest, AnthropicMessage, AnthropicMessageResponse, AnthropicStopReason,
};
use thiserror::Error;

use crate::{
    AnthropicMessages, CheckpointRecord, InterpretedOutput, MaterializedTurn, OutputIdentity,
    OutputInterpreter, RequestMaterializer, TurnState,
};

/// Materializes Anthropic Messages requests without introducing a shared IR.
///
/// Claude Code already submits its complete message history. External
/// continuation chains therefore remain unsupported; runtime-owned tool rounds
/// append native Anthropic messages inside the active public turn.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicRequestMaterializer;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicMaterializationError {
    #[error("Anthropic Messages requests cannot hydrate an external continuation chain")]
    ExternalContinuationUnsupported,
}

impl RequestMaterializer<AnthropicMessages> for AnthropicRequestMaterializer {
    type Error = AnthropicMaterializationError;

    fn materialize(
        &self,
        current: AnthropicCreateMessageRequest,
        chain: &[CheckpointRecord<AnthropicMessages>],
    ) -> Result<MaterializedTurn<AnthropicMessages>, Self::Error> {
        if !chain.is_empty() {
            return Err(AnthropicMaterializationError::ExternalContinuationUnsupported);
        }

        Ok(MaterializedTurn {
            checkpoint_request: current.clone(),
            inference_request: current,
        })
    }
}

/// Selects the durable transition for one native Anthropic Messages result.
pub trait AnthropicOutcomePolicy: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn next_state(&self, response: &AnthropicMessageResponse) -> Result<TurnState, Self::Error>;
}

/// Conservative default for client-executed Anthropic tool calls.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClientToolAnthropicPolicy;

impl AnthropicOutcomePolicy for ClientToolAnthropicPolicy {
    type Error = Infallible;

    fn next_state(&self, response: &AnthropicMessageResponse) -> Result<TurnState, Self::Error> {
        if response.stop_reason == Some(AnthropicStopReason::ToolUse) {
            Ok(TurnState::AwaitingClientToolOutput)
        } else {
            Ok(TurnState::Completed)
        }
    }
}

/// Native Anthropic output interpreter composed with deployment outcome policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolicyAnthropicOutputInterpreter<P> {
    policy: P,
}

impl<P> PolicyAnthropicOutputInterpreter<P> {
    pub fn new(policy: P) -> Self {
        Self { policy }
    }
}

pub type AnthropicOutputInterpreter = PolicyAnthropicOutputInterpreter<ClientToolAnthropicPolicy>;

impl<P> OutputInterpreter<AnthropicMessages> for PolicyAnthropicOutputInterpreter<P>
where
    P: AnthropicOutcomePolicy,
{
    type Error = P::Error;

    fn interpret(
        &self,
        response: AnthropicMessageResponse,
        _identity: &OutputIdentity,
    ) -> Result<InterpretedOutput<AnthropicMessages>, Self::Error> {
        let next_state = self.policy.next_state(&response)?;
        let replay_items = vec![AnthropicMessage::from(&response)];
        Ok(InterpretedOutput {
            response,
            replay_items,
            next_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::anthropic::AnthropicCreateMessageRequest;

    use crate::{
        AgentProtocol, AnthropicMessages, AuthorizationScope, CheckpointRecord, CheckpointVersion,
        IdempotencyKey, OutputIdentity, OutputInterpreter, RequestFingerprint, ResponseId,
        TurnState,
    };

    use super::{
        AnthropicMaterializationError, AnthropicOutputInterpreter, AnthropicRequestMaterializer,
        RequestMaterializer,
    };

    fn request() -> AnthropicCreateMessageRequest {
        serde_json::from_value(serde_json::json!({
            "model": "claude-sonnet",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .unwrap()
    }

    #[test]
    fn preserves_the_native_complete_message_request() {
        let request = request();
        let expected = serde_json::to_value(&request).unwrap();

        let turn = AnthropicRequestMaterializer
            .materialize(request, &[])
            .unwrap();

        assert_eq!(
            serde_json::to_value(turn.checkpoint_request).unwrap(),
            expected
        );
        assert_eq!(
            serde_json::to_value(turn.inference_request).unwrap(),
            expected
        );
    }

    #[test]
    fn rejects_a_responses_style_external_chain() {
        let record = CheckpointRecord::<AnthropicMessages> {
            response_id: ResponseId::from("msg-parent"),
            parent_response_id: None,
            scope: AuthorizationScope {
                tenant_id: "tenant".to_owned(),
                principal_id: "principal".to_owned(),
            },
            idempotency_key: IdempotencyKey::from("idem-parent"),
            request_fingerprint: RequestFingerprint::new([1; 32]),
            state: TurnState::Completed,
            version: CheckpointVersion(1),
            request: request(),
            output_items: Vec::<<AnthropicMessages as AgentProtocol>::ReplayItem>::new(),
            response: None,
        };

        assert_eq!(
            AnthropicRequestMaterializer
                .materialize(request(), &[record])
                .unwrap_err(),
            AnthropicMaterializationError::ExternalContinuationUnsupported
        );
    }

    fn response(stop_reason: &str) -> dynamo_protocols::types::anthropic::AnthropicMessageResponse {
        serde_json::from_value(serde_json::json!({
            "id": "msg_backend",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "tool_1",
                "name": "lookup",
                "input": {}
            }],
            "model": "claude",
            "stop_reason": stop_reason,
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 4}
        }))
        .unwrap()
    }

    #[test]
    fn tool_use_response_waits_for_client_and_replays_native_message() {
        let identity = OutputIdentity {
            response_id: ResponseId::from("internal-turn"),
            parent_response_id: None,
        };
        let interpreted = AnthropicOutputInterpreter::default()
            .interpret(response("tool_use"), &identity)
            .unwrap();

        assert_eq!(interpreted.next_state, TurnState::AwaitingClientToolOutput);
        assert_eq!(interpreted.response.id, "msg_backend");
        assert_eq!(interpreted.replay_items.len(), 1);
        assert_eq!(
            interpreted.replay_items[0].role,
            dynamo_protocols::types::anthropic::AnthropicRole::Assistant
        );
    }

    #[test]
    fn end_turn_response_completes() {
        let identity = OutputIdentity {
            response_id: ResponseId::from("internal-turn"),
            parent_response_id: None,
        };
        let interpreted = AnthropicOutputInterpreter::default()
            .interpret(response("end_turn"), &identity)
            .unwrap();
        assert_eq!(interpreted.next_state, TurnState::Completed);
    }
}
