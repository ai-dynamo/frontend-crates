// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;

use dynamo_protocols::types::anthropic::{
    AnthropicContentBlock, AnthropicCreateMessageRequest, AnthropicMessage,
    AnthropicMessageContent, AnthropicMessageResponse, AnthropicResponseContentBlock,
    AnthropicRole, AnthropicStopReason, ToolResultContent,
};
use thiserror::Error;

use crate::{
    AnthropicMessages, CheckpointRecord, InterpretedOutput, MaterializedTurn, OutputIdentity,
    OutputInterpreter, RequestMaterializer, RuntimeToolCall, RuntimeToolResult, ToolLoopAdapter,
    ToolRouter, TurnState,
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

/// Outcome policy that promotes configured Anthropic tools to runtime work.
#[derive(Debug, Clone)]
pub struct RoutedAnthropicOutcomePolicy<R> {
    router: R,
}

impl<R> RoutedAnthropicOutcomePolicy<R> {
    pub fn new(router: R) -> Self {
        Self { router }
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum RoutedAnthropicOutcomeError {
    #[error("one model step mixed runtime-owned and client-owned tool calls")]
    MixedToolOwnership,
}

impl<R> AnthropicOutcomePolicy for RoutedAnthropicOutcomePolicy<R>
where
    R: ToolRouter,
{
    type Error = RoutedAnthropicOutcomeError;

    fn next_state(&self, response: &AnthropicMessageResponse) -> Result<TurnState, Self::Error> {
        if response.stop_reason != Some(AnthropicStopReason::ToolUse) {
            return Ok(TurnState::Completed);
        }

        let (runtime_calls, client_calls) =
            response
                .content
                .iter()
                .fold(
                    (0_u32, 0_u32),
                    |(runtime_calls, client_calls), block| match block {
                        AnthropicResponseContentBlock::ToolUse { name, .. }
                            if self.router.route(name).is_some() =>
                        {
                            (runtime_calls + 1, client_calls)
                        }
                        AnthropicResponseContentBlock::ToolUse { .. } => {
                            (runtime_calls, client_calls + 1)
                        }
                        _ => (runtime_calls, client_calls),
                    },
                );
        match (runtime_calls > 0, client_calls > 0) {
            (true, true) => Err(RoutedAnthropicOutcomeError::MixedToolOwnership),
            (true, false) => Ok(TurnState::ToolStarted),
            (false, _) => Ok(TurnState::AwaitingClientToolOutput),
        }
    }
}

/// Anthropic custom-tool adapter backed by trusted server routing.
#[derive(Debug, Clone)]
pub struct AnthropicToolLoopAdapter<R> {
    router: R,
}

impl<R> AnthropicToolLoopAdapter<R> {
    pub fn new(router: R) -> Self {
        Self { router }
    }
}

impl<R> ToolLoopAdapter<AnthropicMessages> for AnthropicToolLoopAdapter<R>
where
    R: ToolRouter,
{
    type Error = Infallible;

    fn runtime_calls(
        &self,
        response: &AnthropicMessageResponse,
    ) -> Result<Vec<RuntimeToolCall>, Self::Error> {
        Ok(response
            .content
            .iter()
            .filter_map(|block| match block {
                AnthropicResponseContentBlock::ToolUse { id, name, input } => {
                    self.router.route(name).map(|route| RuntimeToolCall {
                        call_id: id.clone(),
                        connector: route.connector,
                        operation: route.operation,
                        arguments: input.clone(),
                    })
                }
                _ => None,
            })
            .collect())
    }

    fn append_results(
        &self,
        request: &mut AnthropicCreateMessageRequest,
        response: &AnthropicMessageResponse,
        results: &[RuntimeToolResult],
    ) -> Result<Vec<AnthropicMessage>, Self::Error> {
        request.messages.push(AnthropicMessage::from(response));
        let content = results
            .iter()
            .map(|result| {
                let output = result
                    .result
                    .output
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| result.result.output.to_string());
                AnthropicContentBlock::ToolResult {
                    tool_use_id: result.call.call_id.clone(),
                    content: Some(ToolResultContent::Text(output)),
                    is_error: Some(false),
                    cache_control: None,
                }
            })
            .collect();
        let result_message = AnthropicMessage {
            role: AnthropicRole::User,
            content: AnthropicMessageContent::Blocks { content },
        };
        request.messages.push(result_message.clone());
        Ok(vec![result_message])
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
        ConfiguredToolRouter, IdempotencyKey, OutputIdentity, OutputInterpreter,
        RequestFingerprint, ResponseId, RuntimeToolResult, ToolExecutionResult, ToolLoopAdapter,
        ToolRoute, TurnState,
    };

    use super::{
        AnthropicMaterializationError, AnthropicOutputInterpreter, AnthropicRequestMaterializer,
        AnthropicToolLoopAdapter, PolicyAnthropicOutputInterpreter, RequestMaterializer,
        RoutedAnthropicOutcomePolicy,
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

    fn router() -> ConfiguredToolRouter {
        ConfiguredToolRouter::new([("lookup".to_owned(), ToolRoute::new("search", "query"))])
    }

    #[test]
    fn routed_tool_use_starts_runtime_work() {
        let interpreter =
            PolicyAnthropicOutputInterpreter::new(RoutedAnthropicOutcomePolicy::new(router()));
        let identity = OutputIdentity {
            response_id: ResponseId::from("internal-turn"),
            parent_response_id: None,
        };

        let interpreted = interpreter
            .interpret(response("tool_use"), &identity)
            .unwrap();
        assert_eq!(interpreted.next_state, TurnState::ToolStarted);
    }

    #[test]
    fn tool_adapter_appends_native_assistant_and_result_messages() {
        let adapter = AnthropicToolLoopAdapter::new(router());
        let response = response("tool_use");
        let calls = adapter.runtime_calls(&response).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].connector, "search");
        assert_eq!(calls[0].arguments, serde_json::json!({}));

        let mut request = request();
        let replay = adapter
            .append_results(
                &mut request,
                &response,
                &[RuntimeToolResult {
                    call: calls[0].clone(),
                    result: ToolExecutionResult {
                        output: serde_json::json!({"answer": 42}),
                    },
                }],
            )
            .unwrap();

        assert_eq!(request.messages.len(), 3);
        assert_eq!(replay.len(), 1);
        assert_eq!(
            replay[0].role,
            dynamo_protocols::types::anthropic::AnthropicRole::User
        );
        let dynamo_protocols::types::anthropic::AnthropicMessageContent::Blocks { content } =
            &replay[0].content
        else {
            panic!("tool result must use structured native blocks")
        };
        assert!(matches!(
            content.as_slice(),
            [dynamo_protocols::types::anthropic::AnthropicContentBlock::ToolResult {
                tool_use_id,
                ..
            }] if tool_use_id == "tool_1"
        ));
    }
}
