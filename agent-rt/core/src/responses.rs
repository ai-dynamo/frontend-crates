// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;

use dynamo_protocols::types::responses::{
    FunctionCallOutput, FunctionCallOutputItemParam, InputItem, InputParam, Item, OutputItem,
    OutputStatus, Response, ResponseStreamEvent, Status,
};
use thiserror::Error;

use crate::{
    InterpretedOutput, OpenAiResponses, OutputIdentity, OutputInterpreter, RuntimeToolCall,
    RuntimeToolResult, StreamEventAction, StreamEventInterpreter, ToolLoopAdapter, ToolRouter,
    TurnState,
};

/// Rewrites backend Responses events into one public response stream.
#[derive(Debug, Default)]
pub struct ResponsesStreamEventInterpreter {
    next_sequence_number: u64,
    expose_step_lifecycle: bool,
    public_stream_started: bool,
    stage_step_output: bool,
}

impl ResponsesStreamEventInterpreter {
    /// Stages each model step's output until its terminal response reveals
    /// whether the step is an internal runtime-tool transition.
    pub fn stage_runtime_tool_rounds() -> Self {
        Self {
            stage_step_output: true,
            ..Self::default()
        }
    }
}

impl StreamEventInterpreter<OpenAiResponses> for ResponsesStreamEventInterpreter {
    type Error = Infallible;

    fn begin_step(&mut self, _step_kind: crate::ModelStepKind) {
        self.expose_step_lifecycle = !self.public_stream_started;
        self.public_stream_started = true;
    }

    fn observe(
        &mut self,
        mut event: ResponseStreamEvent,
        identity: &OutputIdentity,
    ) -> Result<StreamEventAction<OpenAiResponses>, Self::Error> {
        let expose_lifecycle = self.expose_step_lifecycle;
        let action = match &mut event {
            ResponseStreamEvent::ResponseCreated(inner) => {
                apply_response_identity(&mut inner.response, identity);
                if expose_lifecycle {
                    StreamEventAction::Emit(event)
                } else {
                    StreamEventAction::Suppress
                }
            }
            ResponseStreamEvent::ResponseInProgress(inner) => {
                apply_response_identity(&mut inner.response, identity);
                if expose_lifecycle {
                    StreamEventAction::Emit(event)
                } else {
                    StreamEventAction::Suppress
                }
            }
            ResponseStreamEvent::ResponseQueued(inner) => {
                apply_response_identity(&mut inner.response, identity);
                if expose_lifecycle {
                    StreamEventAction::Emit(event)
                } else {
                    StreamEventAction::Suppress
                }
            }
            ResponseStreamEvent::ResponseCompleted(inner) => {
                apply_response_identity(&mut inner.response, identity);
                StreamEventAction::Terminal {
                    response: inner.response.clone(),
                    event,
                }
            }
            ResponseStreamEvent::ResponseFailed(inner) => {
                apply_response_identity(&mut inner.response, identity);
                StreamEventAction::Terminal {
                    response: inner.response.clone(),
                    event,
                }
            }
            ResponseStreamEvent::ResponseIncomplete(inner) => {
                apply_response_identity(&mut inner.response, identity);
                StreamEventAction::Terminal {
                    response: inner.response.clone(),
                    event,
                }
            }
            _ if self.stage_step_output => StreamEventAction::Stage(event),
            _ => StreamEventAction::Emit(event),
        };
        Ok(action)
    }

    fn prepare_emit(&mut self, event: &mut ResponseStreamEvent) {
        set_sequence_number(event, self.next_sequence_number);
        self.next_sequence_number += 1;
    }
}

fn apply_response_identity(response: &mut Response, identity: &OutputIdentity) {
    response.id = identity.response_id.to_string();
    response.previous_response_id = identity
        .parent_response_id
        .as_ref()
        .map(ToString::to_string);
}

macro_rules! set_event_sequence_number {
    ($event:expr, $sequence_number:expr, $($variant:ident),+ $(,)?) => {
        match $event {
            $(ResponseStreamEvent::$variant(inner) => {
                inner.sequence_number = $sequence_number;
            })+
        }
    };
}

fn set_sequence_number(event: &mut ResponseStreamEvent, sequence_number: u64) {
    set_event_sequence_number!(
        event,
        sequence_number,
        ResponseCreated,
        ResponseInProgress,
        ResponseCompleted,
        ResponseFailed,
        ResponseIncomplete,
        ResponseOutputItemAdded,
        ResponseOutputItemDone,
        ResponseContentPartAdded,
        ResponseContentPartDone,
        ResponseOutputTextDelta,
        ResponseOutputTextDone,
        ResponseRefusalDelta,
        ResponseRefusalDone,
        ResponseFunctionCallArgumentsDelta,
        ResponseFunctionCallArgumentsDone,
        ResponseFileSearchCallInProgress,
        ResponseFileSearchCallSearching,
        ResponseFileSearchCallCompleted,
        ResponseWebSearchCallInProgress,
        ResponseWebSearchCallSearching,
        ResponseWebSearchCallCompleted,
        ResponseReasoningSummaryPartAdded,
        ResponseReasoningSummaryPartDone,
        ResponseReasoningSummaryTextDelta,
        ResponseReasoningSummaryTextDone,
        ResponseReasoningTextDelta,
        ResponseReasoningTextDone,
        ResponseImageGenerationCallCompleted,
        ResponseImageGenerationCallGenerating,
        ResponseImageGenerationCallInProgress,
        ResponseImageGenerationCallPartialImage,
        ResponseMCPCallArgumentsDelta,
        ResponseMCPCallArgumentsDone,
        ResponseMCPCallCompleted,
        ResponseMCPCallFailed,
        ResponseMCPCallInProgress,
        ResponseMCPListToolsCompleted,
        ResponseMCPListToolsFailed,
        ResponseMCPListToolsInProgress,
        ResponseCodeInterpreterCallInProgress,
        ResponseCodeInterpreterCallInterpreting,
        ResponseCodeInterpreterCallCompleted,
        ResponseCodeInterpreterCallCodeDelta,
        ResponseCodeInterpreterCallCodeDone,
        ResponseOutputTextAnnotationAdded,
        ResponseQueued,
        ResponseCustomToolCallInputDelta,
        ResponseCustomToolCallInputDone,
        ResponseError,
    );
}

/// Selects the durable transition for one native Responses result.
pub trait ResponsesOutcomePolicy: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn next_state(&self, response: &Response) -> Result<TurnState, Self::Error>;
}

/// Conservative default: client-executable calls wait for client output.
/// Backend-owned calls that already carry results remain part of a completed
/// response. A runtime-tool policy can replace this and return `ToolStarted`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClientToolResponsesPolicy;

impl ResponsesOutcomePolicy for ClientToolResponsesPolicy {
    type Error = ResponsesOutputError;

    fn next_state(&self, response: &Response) -> Result<TurnState, Self::Error> {
        match response.status {
            Status::Failed | Status::Cancelled => return Ok(TurnState::Failed),
            Status::InProgress | Status::Queued => {
                return Err(ResponsesOutputError::NonterminalUnaryStatus(
                    response.status.clone(),
                ));
            }
            Status::Completed | Status::Incomplete => {}
        }

        if response.output.iter().any(is_client_action) {
            Ok(TurnState::AwaitingClientToolOutput)
        } else {
            Ok(TurnState::Completed)
        }
    }
}

fn is_client_action(item: &OutputItem) -> bool {
    matches!(
        item,
        OutputItem::FunctionCall(_)
            | OutputItem::ComputerCall(_)
            | OutputItem::LocalShellCall(_)
            | OutputItem::ShellCall(_)
            | OutputItem::ApplyPatchCall(_)
            | OutputItem::McpApprovalRequest(_)
            | OutputItem::CustomToolCall(_)
    )
}

/// Outcome policy that promotes configured function calls to runtime work.
#[derive(Debug, Clone)]
pub struct RoutedResponsesOutcomePolicy<R> {
    router: R,
}

impl<R> RoutedResponsesOutcomePolicy<R> {
    pub fn new(router: R) -> Self {
        Self { router }
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum RoutedResponsesOutcomeError {
    #[error(transparent)]
    Base(#[from] ResponsesOutputError),
    #[error("one model step mixed runtime-owned and client-owned tool calls")]
    MixedToolOwnership,
}

impl<R> ResponsesOutcomePolicy for RoutedResponsesOutcomePolicy<R>
where
    R: ToolRouter,
{
    type Error = RoutedResponsesOutcomeError;

    fn next_state(&self, response: &Response) -> Result<TurnState, Self::Error> {
        let base = ClientToolResponsesPolicy.next_state(response)?;
        if base == TurnState::Failed {
            return Ok(base);
        }

        let has_runtime_call = response.output.iter().any(|item| match item {
            OutputItem::FunctionCall(call) => self.router.route(&call.name).is_some(),
            _ => false,
        });
        let has_client_call = response.output.iter().any(|item| match item {
            OutputItem::FunctionCall(call) => self.router.route(&call.name).is_none(),
            other => is_client_action(other),
        });
        match (has_runtime_call, has_client_call) {
            (true, true) => Err(RoutedResponsesOutcomeError::MixedToolOwnership),
            (true, false) => Ok(TurnState::ToolStarted),
            (false, _) => Ok(base),
        }
    }
}

/// Responses function-tool adapter backed by trusted server routing.
#[derive(Debug, Clone)]
pub struct ResponsesToolLoopAdapter<R> {
    router: R,
}

impl<R> ResponsesToolLoopAdapter<R> {
    pub fn new(router: R) -> Self {
        Self { router }
    }
}

#[derive(Debug, Error)]
pub enum ResponsesToolAdapterError {
    #[error("function call {call_id} has invalid JSON arguments: {source}")]
    InvalidArguments {
        call_id: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("Responses tool continuation requires materialized item input")]
    UnmaterializedInput,
}

impl<R> ToolLoopAdapter<OpenAiResponses> for ResponsesToolLoopAdapter<R>
where
    R: ToolRouter,
{
    type Error = ResponsesToolAdapterError;

    fn runtime_calls(&self, response: &Response) -> Result<Vec<RuntimeToolCall>, Self::Error> {
        response
            .output
            .iter()
            .filter_map(|item| match item {
                OutputItem::FunctionCall(call) => {
                    self.router.route(&call.name).map(|route| (call, route))
                }
                _ => None,
            })
            .map(|(call, route)| {
                let arguments = serde_json::from_str(&call.arguments).map_err(|source| {
                    ResponsesToolAdapterError::InvalidArguments {
                        call_id: call.call_id.clone(),
                        source,
                    }
                })?;
                Ok(RuntimeToolCall {
                    call_id: call.call_id.clone(),
                    connector: route.connector,
                    operation: route.operation,
                    profile: route.profile,
                    arguments,
                })
            })
            .collect()
    }

    fn append_results(
        &self,
        request: &mut <OpenAiResponses as crate::AgentProtocol>::Request,
        response: &Response,
        results: &[RuntimeToolResult],
    ) -> Result<Vec<InputItem>, Self::Error> {
        let InputParam::Items(items) = &mut request.input else {
            return Err(ResponsesToolAdapterError::UnmaterializedInput);
        };
        items.extend(response.output.iter().cloned().map(Into::into));

        let result_items = results
            .iter()
            .map(|result| {
                let output = if result.result.is_error {
                    serde_json::json!({
                        "is_error": true,
                        "output": &result.result.output,
                    })
                    .to_string()
                } else {
                    result
                        .result
                        .output
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| result.result.output.to_string())
                };
                InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                    call_id: result.call.call_id.clone(),
                    output: FunctionCallOutput::Text(output),
                    id: None,
                    status: Some(OutputStatus::Completed),
                }))
            })
            .collect::<Vec<_>>();
        items.extend(result_items.iter().cloned());
        Ok(result_items)
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ResponsesOutputError {
    #[error("unary inference returned nonterminal Responses status {0:?}")]
    NonterminalUnaryStatus(Status),
}

/// Native Responses output interpreter composed with deployment outcome policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolicyResponsesOutputInterpreter<P> {
    policy: P,
}

impl<P> PolicyResponsesOutputInterpreter<P> {
    pub fn new(policy: P) -> Self {
        Self { policy }
    }
}

pub type ResponsesOutputInterpreter = PolicyResponsesOutputInterpreter<ClientToolResponsesPolicy>;

impl<P> OutputInterpreter<OpenAiResponses> for PolicyResponsesOutputInterpreter<P>
where
    P: ResponsesOutcomePolicy,
{
    type Error = P::Error;

    fn interpret(
        &self,
        mut response: Response,
        identity: &OutputIdentity,
    ) -> Result<InterpretedOutput<OpenAiResponses>, Self::Error> {
        response.id = identity.response_id.to_string();
        response.previous_response_id = identity
            .parent_response_id
            .as_ref()
            .map(ToString::to_string);
        let next_state = self.policy.next_state(&response)?;
        let replay_items = response.output.iter().cloned().map(Into::into).collect();

        Ok(InterpretedOutput {
            response,
            replay_items,
            next_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::responses::{
        CreateResponse, FunctionCallOutput, InputItem, InputParam, Item, OutputStatus, Response,
        ResponseCompletedEvent, ResponseCreatedEvent, ResponseStreamEvent, ResponseTextDeltaEvent,
    };

    use crate::{
        ConfiguredToolRouter, ModelStepKind, OutputIdentity, OutputInterpreter, ResponseId,
        RuntimeToolResult, StreamEventAction, StreamEventInterpreter, ToolExecutionResult,
        ToolLoopAdapter, ToolRoute, TurnState,
    };

    use super::{
        PolicyResponsesOutputInterpreter, ResponsesOutputError, ResponsesOutputInterpreter,
        ResponsesStreamEventInterpreter, ResponsesToolLoopAdapter, RoutedResponsesOutcomePolicy,
    };

    fn response(status: &str, output: serde_json::Value) -> Response {
        serde_json::from_value(serde_json::json!({
            "created_at": 1,
            "id": "backend-id",
            "model": "model",
            "object": "response",
            "output": output,
            "status": status
        }))
        .unwrap()
    }

    fn identity() -> OutputIdentity {
        OutputIdentity {
            response_id: ResponseId::from("resp-public"),
            parent_response_id: Some(ResponseId::from("resp-parent")),
        }
    }

    #[test]
    fn completed_text_is_replayable_and_uses_runtime_identity() {
        let output = serde_json::json!([{
            "type": "message",
            "id": "msg-1",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "hello",
                "annotations": [],
                "logprobs": null
            }]
        }]);

        let interpreted = ResponsesOutputInterpreter::default()
            .interpret(response("completed", output), &identity())
            .unwrap();

        assert_eq!(interpreted.response.id, "resp-public");
        assert_eq!(
            interpreted.response.previous_response_id.as_deref(),
            Some("resp-parent")
        );
        assert_eq!(interpreted.next_state, TurnState::Completed);
        assert!(matches!(
            interpreted.replay_items.as_slice(),
            [InputItem::Item(Item::Message(_))]
        ));
    }

    #[test]
    fn stream_interpreter_rewrites_identity_and_public_sequence() {
        let mut interpreter = ResponsesStreamEventInterpreter::default();
        interpreter.begin_step(ModelStepKind::Initial);

        let created = ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: 9,
            response: response("in_progress", serde_json::json!([])),
        });
        let StreamEventAction::Emit(mut created) =
            interpreter.observe(created, &identity()).unwrap()
        else {
            panic!("created event must be emitted")
        };
        interpreter.prepare_emit(&mut created);
        let ResponseStreamEvent::ResponseCreated(created) = created else {
            unreachable!()
        };
        assert_eq!(created.sequence_number, 0);
        assert_eq!(created.response.id, "resp-public");
        assert_eq!(
            created.response.previous_response_id.as_deref(),
            Some("resp-parent")
        );

        let terminal = ResponseStreamEvent::ResponseCompleted(ResponseCompletedEvent {
            sequence_number: 99,
            response: response("completed", serde_json::json!([])),
        });
        let StreamEventAction::Terminal {
            mut event,
            response,
        } = interpreter.observe(terminal, &identity()).unwrap()
        else {
            panic!("completed event must be retained as terminal")
        };
        assert_eq!(response.id, "resp-public");
        interpreter.prepare_emit(&mut event);
        let ResponseStreamEvent::ResponseCompleted(completed) = event else {
            unreachable!()
        };
        assert_eq!(completed.sequence_number, 1);
    }

    #[test]
    fn internal_step_lifecycle_is_suppressed_without_sequence_gap() {
        let mut interpreter = ResponsesStreamEventInterpreter::default();
        interpreter.begin_step(ModelStepKind::Initial);
        interpreter.begin_step(ModelStepKind::RuntimeToolContinuation);

        let created = ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: 0,
            response: response("in_progress", serde_json::json!([])),
        });
        assert!(matches!(
            interpreter.observe(created, &identity()).unwrap(),
            StreamEventAction::Suppress
        ));

        let delta = ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
            sequence_number: 42,
            item_id: "msg-1".to_owned(),
            output_index: 0,
            content_index: 0,
            delta: "hello".to_owned(),
            logprobs: None,
        });
        let StreamEventAction::Emit(mut delta) = interpreter.observe(delta, &identity()).unwrap()
        else {
            panic!("content delta must be emitted")
        };
        interpreter.prepare_emit(&mut delta);
        let ResponseStreamEvent::ResponseOutputTextDelta(delta) = delta else {
            unreachable!()
        };
        assert_eq!(delta.sequence_number, 0);
    }

    #[test]
    fn runtime_tool_mode_stages_output_events() {
        let mut interpreter = ResponsesStreamEventInterpreter::stage_runtime_tool_rounds();
        interpreter.begin_step(ModelStepKind::Initial);

        let delta = ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
            sequence_number: 42,
            item_id: "msg-1".to_owned(),
            output_index: 0,
            content_index: 0,
            delta: "internal".to_owned(),
            logprobs: None,
        });
        assert!(matches!(
            interpreter.observe(delta, &identity()).unwrap(),
            StreamEventAction::Stage(_)
        ));
    }

    #[test]
    fn function_call_waits_for_client_output() {
        let output = serde_json::json!([{
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": "{}",
            "status": "completed"
        }]);

        let interpreted = ResponsesOutputInterpreter::default()
            .interpret(response("completed", output), &identity())
            .unwrap();
        assert_eq!(interpreted.next_state, TurnState::AwaitingClientToolOutput);
    }

    #[test]
    fn unary_in_progress_response_is_rejected() {
        assert_eq!(
            ResponsesOutputInterpreter::default()
                .interpret(response("in_progress", serde_json::json!([])), &identity())
                .unwrap_err(),
            ResponsesOutputError::NonterminalUnaryStatus(
                dynamo_protocols::types::responses::Status::InProgress
            )
        );
    }

    fn routed_tool_response() -> Response {
        response(
            "completed",
            serde_json::json!([{
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{\"query\":\"rust\"}",
                "status": "completed"
            }]),
        )
    }

    fn router() -> ConfiguredToolRouter {
        ConfiguredToolRouter::new([("lookup".to_owned(), ToolRoute::new("search", "query"))])
    }

    #[test]
    fn routed_function_call_starts_runtime_tool_work() {
        let interpreter =
            PolicyResponsesOutputInterpreter::new(RoutedResponsesOutcomePolicy::new(router()));
        let interpreted = interpreter
            .interpret(routed_tool_response(), &identity())
            .unwrap();
        assert_eq!(interpreted.next_state, TurnState::ToolStarted);
    }

    #[test]
    fn tool_adapter_extracts_and_appends_native_results() {
        let adapter = ResponsesToolLoopAdapter::new(router());
        let response = routed_tool_response();
        let calls = adapter.runtime_calls(&response).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].connector, "search");
        assert_eq!(calls[0].operation, "query");
        assert_eq!(calls[0].profile, "default");
        assert_eq!(calls[0].arguments["query"], "rust");

        let mut request = CreateResponse {
            input: InputParam::Items(Vec::new()),
            ..Default::default()
        };
        let replay_items = adapter
            .append_results(
                &mut request,
                &response,
                &[RuntimeToolResult {
                    call: calls[0].clone(),
                    result: ToolExecutionResult {
                        output: serde_json::json!({"answer": 42}),
                        is_error: false,
                    },
                }],
            )
            .unwrap();

        assert_eq!(replay_items.len(), 1);
        let InputParam::Items(items) = request.input else {
            panic!("tool continuation must remain item input")
        };
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[1],
            InputItem::Item(Item::FunctionCallOutput(output))
                if matches!(&output.output, FunctionCallOutput::Text(text) if text == "{\"answer\":42}")
        ));
    }

    #[test]
    fn tool_adapter_preserves_model_visible_errors_as_completed_outputs() {
        let adapter = ResponsesToolLoopAdapter::new(router());
        let response = routed_tool_response();
        let call = adapter.runtime_calls(&response).unwrap().remove(0);
        let mut request = CreateResponse {
            input: InputParam::Items(Vec::new()),
            ..Default::default()
        };

        adapter
            .append_results(
                &mut request,
                &response,
                &[RuntimeToolResult {
                    call,
                    result: ToolExecutionResult {
                        output: serde_json::json!({"message": "refused"}),
                        is_error: true,
                    },
                }],
            )
            .unwrap();

        let InputParam::Items(items) = request.input else {
            panic!("tool continuation must remain item input")
        };
        assert!(matches!(
            &items[1],
            InputItem::Item(Item::FunctionCallOutput(output))
                if output.status == Some(OutputStatus::Completed)
                    && matches!(
                        &output.output,
                        FunctionCallOutput::Text(text)
                            if text == "{\"is_error\":true,\"output\":{\"message\":\"refused\"}}"
                    )
        ));
    }
}
