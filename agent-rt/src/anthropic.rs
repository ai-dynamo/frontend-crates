// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::convert::Infallible;

use dynamo_protocols::types::anthropic::{
    AnthropicContentBlock, AnthropicCreateMessageRequest, AnthropicDelta, AnthropicMessage,
    AnthropicMessageContent, AnthropicMessageResponse, AnthropicResponseContentBlock,
    AnthropicRole, AnthropicStopReason, AnthropicStreamEvent, ToolResultContent,
};
use thiserror::Error;

use crate::{
    AnthropicMessages, CheckpointRecord, InterpretedOutput, MaterializedTurn, OutputIdentity,
    OutputInterpreter, RequestMaterializer, RuntimeToolCall, RuntimeToolResult, StreamEventAction,
    StreamEventInterpreter, ToolLoopAdapter, ToolRouter, TurnState,
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
                        profile: route.profile,
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

/// Reconstructs one native Anthropic response while preserving Dynamo's typed stream.
#[derive(Debug, Default)]
pub struct AnthropicStreamEventInterpreter {
    response: Option<AnthropicMessageResponse>,
    blocks: BTreeMap<u32, AnthropicResponseContentBlock>,
    tool_input_fragments: BTreeMap<u32, String>,
    expose_message_start: bool,
    public_stream_started: bool,
    stage_step_output: bool,
}

impl AnthropicStreamEventInterpreter {
    /// Stages model output until the step reveals whether a tool call is runtime-owned.
    pub fn stage_runtime_tool_rounds() -> Self {
        Self {
            stage_step_output: true,
            ..Self::default()
        }
    }

    fn output_action(&self, event: AnthropicStreamEvent) -> StreamEventAction<AnthropicMessages> {
        if self.stage_step_output {
            StreamEventAction::Stage(event)
        } else {
            StreamEventAction::Emit(event)
        }
    }

    fn apply_delta(
        &mut self,
        index: u32,
        delta: &AnthropicDelta,
    ) -> Result<(), AnthropicStreamEventError> {
        let block = self
            .blocks
            .get_mut(&index)
            .ok_or(AnthropicStreamEventError::UnknownContentBlock(index))?;
        match (block, delta) {
            (
                AnthropicResponseContentBlock::Thinking { thinking, .. },
                AnthropicDelta::ThinkingDelta { thinking: fragment },
            ) => thinking.push_str(fragment),
            (
                AnthropicResponseContentBlock::Thinking { signature, .. },
                AnthropicDelta::SignatureDelta {
                    signature: fragment,
                },
            ) => signature.push_str(fragment),
            (
                AnthropicResponseContentBlock::Text { text, .. },
                AnthropicDelta::TextDelta { text: fragment },
            ) => text.push_str(fragment),
            (
                AnthropicResponseContentBlock::Text { citations, .. },
                AnthropicDelta::CitationsDelta { citation },
            ) => citations
                .get_or_insert_with(Vec::new)
                .push(citation.clone()),
            (
                AnthropicResponseContentBlock::ToolUse { .. },
                AnthropicDelta::InputJsonDelta { partial_json },
            ) => self
                .tool_input_fragments
                .entry(index)
                .or_default()
                .push_str(partial_json),
            _ => return Err(AnthropicStreamEventError::MismatchedDelta(index)),
        }
        Ok(())
    }

    fn finish_block(&mut self, index: u32) -> Result<(), AnthropicStreamEventError> {
        let Some(input) = self.tool_input_fragments.remove(&index) else {
            return Ok(());
        };
        let block = self
            .blocks
            .get_mut(&index)
            .ok_or(AnthropicStreamEventError::UnknownContentBlock(index))?;
        let AnthropicResponseContentBlock::ToolUse {
            input: tool_input, ..
        } = block
        else {
            return Err(AnthropicStreamEventError::MismatchedDelta(index));
        };
        *tool_input = serde_json::from_str(&input)
            .map_err(|source| AnthropicStreamEventError::InvalidToolInput { index, source })?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AnthropicStreamEventError {
    #[error("Anthropic stream emitted more than one message_start event")]
    DuplicateMessageStart,
    #[error("Anthropic stream emitted content before message_start")]
    MissingMessageStart,
    #[error("Anthropic stream emitted duplicate content block {0}")]
    DuplicateContentBlock(u32),
    #[error("Anthropic stream referenced unknown content block {0}")]
    UnknownContentBlock(u32),
    #[error("Anthropic stream emitted a mismatched delta for content block {0}")]
    MismatchedDelta(u32),
    #[error("Anthropic stream emitted invalid JSON for tool block {index}: {source}")]
    InvalidToolInput {
        index: u32,
        #[source]
        source: serde_json::Error,
    },
    #[error("Anthropic inference stream failed ({error_type}): {message}")]
    Backend { error_type: String, message: String },
}

impl StreamEventInterpreter<AnthropicMessages> for AnthropicStreamEventInterpreter {
    type Error = AnthropicStreamEventError;

    fn begin_step(&mut self, _step_kind: crate::ModelStepKind) {
        self.response = None;
        self.blocks.clear();
        self.tool_input_fragments.clear();
        self.expose_message_start = !self.public_stream_started;
        self.public_stream_started = true;
    }

    fn observe(
        &mut self,
        mut event: AnthropicStreamEvent,
        identity: &OutputIdentity,
    ) -> Result<StreamEventAction<AnthropicMessages>, Self::Error> {
        match &mut event {
            AnthropicStreamEvent::MessageStart { message } => {
                if self.response.is_some() {
                    return Err(AnthropicStreamEventError::DuplicateMessageStart);
                }
                message.id = identity.response_id.to_string();
                self.blocks
                    .extend((0_u32..).zip(message.content.iter().cloned()));
                self.response = Some(message.clone());
                if self.expose_message_start {
                    Ok(StreamEventAction::Emit(event))
                } else {
                    Ok(StreamEventAction::Suppress)
                }
            }
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if self.response.is_none() {
                    return Err(AnthropicStreamEventError::MissingMessageStart);
                }
                if self.blocks.insert(*index, content_block.clone()).is_some() {
                    return Err(AnthropicStreamEventError::DuplicateContentBlock(*index));
                }
                Ok(self.output_action(event))
            }
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                self.apply_delta(*index, delta)?;
                Ok(self.output_action(event))
            }
            AnthropicStreamEvent::ContentBlockStop { index } => {
                self.finish_block(*index)?;
                Ok(self.output_action(event))
            }
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                let response = self
                    .response
                    .as_mut()
                    .ok_or(AnthropicStreamEventError::MissingMessageStart)?;
                response.stop_reason.clone_from(&delta.stop_reason);
                response.stop_sequence.clone_from(&delta.stop_sequence);
                response.usage = usage.clone();
                Ok(self.output_action(event))
            }
            AnthropicStreamEvent::MessageStop {} => {
                while let Some(index) = self.tool_input_fragments.keys().next().copied() {
                    self.finish_block(index)?;
                }
                let mut response = self
                    .response
                    .take()
                    .ok_or(AnthropicStreamEventError::MissingMessageStart)?;
                response.content = std::mem::take(&mut self.blocks).into_values().collect();
                Ok(StreamEventAction::Terminal { event, response })
            }
            AnthropicStreamEvent::Ping {} => Ok(StreamEventAction::Emit(event)),
            AnthropicStreamEvent::Error { error } => Err(AnthropicStreamEventError::Backend {
                error_type: error.error_type.clone(),
                message: error.message.clone(),
            }),
        }
    }

    fn prepare_emit(&mut self, _event: &mut AnthropicStreamEvent) {}
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
        mut response: AnthropicMessageResponse,
        identity: &OutputIdentity,
    ) -> Result<InterpretedOutput<AnthropicMessages>, Self::Error> {
        response.id = identity.response_id.to_string();
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
    use dynamo_protocols::types::anthropic::{
        AnthropicCreateMessageRequest, AnthropicDelta, AnthropicMessageDeltaBody,
        AnthropicResponseContentBlock, AnthropicStopReason, AnthropicStreamEvent, AnthropicUsage,
    };

    use crate::{
        AgentProtocol, AnthropicMessages, AuthorizationScope, CheckpointRecord, CheckpointVersion,
        ConfiguredToolRouter, IdempotencyKey, ModelStepKind, OutputIdentity, OutputInterpreter,
        RequestFingerprint, ResponseId, RuntimeToolResult, StreamEventAction,
        StreamEventInterpreter, ToolExecutionResult, ToolLoopAdapter, ToolRoute, TurnState,
    };

    use super::{
        AnthropicMaterializationError, AnthropicOutputInterpreter, AnthropicRequestMaterializer,
        AnthropicStreamEventInterpreter, AnthropicToolLoopAdapter,
        PolicyAnthropicOutputInterpreter, RequestMaterializer, RoutedAnthropicOutcomePolicy,
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
        assert_eq!(interpreted.response.id, "internal-turn");
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

    fn stream_start() -> AnthropicStreamEvent {
        serde_json::from_value(serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_backend",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude",
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }))
        .unwrap()
    }

    #[test]
    fn native_stream_reconstructs_and_rewrites_the_public_message() {
        let identity = OutputIdentity {
            response_id: ResponseId::from("msg_public"),
            parent_response_id: None,
        };
        let mut interpreter = AnthropicStreamEventInterpreter::default();
        interpreter.begin_step(ModelStepKind::Initial);

        let StreamEventAction::Emit(AnthropicStreamEvent::MessageStart { message }) =
            interpreter.observe(stream_start(), &identity).unwrap()
        else {
            panic!("message_start must be public")
        };
        assert_eq!(message.id, "msg_public");

        let start = AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: AnthropicResponseContentBlock::Text {
                text: String::new(),
                citations: None,
            },
        };
        assert!(matches!(
            interpreter.observe(start, &identity).unwrap(),
            StreamEventAction::Emit(_)
        ));
        let delta = AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: AnthropicDelta::TextDelta {
                text: "hello".to_owned(),
            },
        };
        assert!(matches!(
            interpreter.observe(delta, &identity).unwrap(),
            StreamEventAction::Emit(_)
        ));
        let message_delta = AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: Some(AnthropicStopReason::EndTurn),
                stop_sequence: None,
            },
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 1,
                ..Default::default()
            },
        };
        interpreter.observe(message_delta, &identity).unwrap();

        let StreamEventAction::Terminal { response, .. } = interpreter
            .observe(AnthropicStreamEvent::MessageStop {}, &identity)
            .unwrap()
        else {
            panic!("message_stop must be terminal")
        };
        assert_eq!(response.id, "msg_public");
        assert_eq!(response.stop_reason, Some(AnthropicStopReason::EndTurn));
        assert_eq!(response.usage.output_tokens, 1);
        assert!(matches!(
            response.content.as_slice(),
            [AnthropicResponseContentBlock::Text { text, .. }] if text == "hello"
        ));
    }

    #[test]
    fn native_stream_reassembles_tool_input_and_stages_internal_output() {
        let identity = OutputIdentity {
            response_id: ResponseId::from("msg_public"),
            parent_response_id: None,
        };
        let mut interpreter = AnthropicStreamEventInterpreter::stage_runtime_tool_rounds();
        interpreter.begin_step(ModelStepKind::Initial);
        assert!(matches!(
            interpreter.observe(stream_start(), &identity).unwrap(),
            StreamEventAction::Emit(_)
        ));
        assert!(matches!(
            interpreter
                .observe(
                    AnthropicStreamEvent::ContentBlockStart {
                        index: 0,
                        content_block: AnthropicResponseContentBlock::ToolUse {
                            id: "tool_1".to_owned(),
                            name: "lookup".to_owned(),
                            input: serde_json::json!({}),
                        },
                    },
                    &identity,
                )
                .unwrap(),
            StreamEventAction::Stage(_)
        ));
        for partial_json in ["{\"query\":", "\"rust\"}"] {
            interpreter
                .observe(
                    AnthropicStreamEvent::ContentBlockDelta {
                        index: 0,
                        delta: AnthropicDelta::InputJsonDelta {
                            partial_json: partial_json.to_owned(),
                        },
                    },
                    &identity,
                )
                .unwrap();
        }
        interpreter
            .observe(
                AnthropicStreamEvent::ContentBlockStop { index: 0 },
                &identity,
            )
            .unwrap();
        interpreter
            .observe(
                AnthropicStreamEvent::MessageDelta {
                    delta: AnthropicMessageDeltaBody {
                        stop_reason: Some(AnthropicStopReason::ToolUse),
                        stop_sequence: None,
                    },
                    usage: AnthropicUsage::default(),
                },
                &identity,
            )
            .unwrap();
        let StreamEventAction::Terminal { response, .. } = interpreter
            .observe(AnthropicStreamEvent::MessageStop {}, &identity)
            .unwrap()
        else {
            panic!("message_stop must be terminal")
        };
        assert!(matches!(
            response.content.as_slice(),
            [AnthropicResponseContentBlock::ToolUse { input, .. }]
                if input == &serde_json::json!({"query": "rust"})
        ));

        interpreter.begin_step(ModelStepKind::RuntimeToolContinuation);
        assert!(matches!(
            interpreter.observe(stream_start(), &identity).unwrap(),
            StreamEventAction::Suppress
        ));
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
        assert_eq!(calls[0].profile, "default");
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
