// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::future::join_all;
use thiserror::Error;

use crate::runtime::{PrepareTurnResult, PreparedTurn};
use crate::{
    AgentProtocol, AgentRuntime, AgentRuntimeError, AgentStreamRuntimeError, CheckpointStore,
    Clock, CommitTurn, IdGenerator, InferenceIntent, InferenceInvoker, InferenceOutput,
    InferenceRequest, ModelStepKind, OutputInterpreter, RequestFingerprinter, RequestMaterializer,
    RunStreamResult, RunTurn, RunTurnResult, StreamEventAction, StreamEventInterpreter,
    StreamRuntimeErrorFor, ToolExecutor, ToolFailurePolicy, ToolIdempotencyKeyProvider,
    ToolJournal, ToolLoopAdapter, ToolRunError, ToolRunner, TurnLease, TurnState,
};

#[derive(Debug, Error)]
pub enum AgentToolRuntimeError<RuntimeError, AdapterError, ToolError, StoreError>
where
    RuntimeError: std::error::Error + Send + Sync + 'static,
    AdapterError: std::error::Error + Send + Sync + 'static,
    ToolError: std::error::Error + Send + Sync + 'static,
    StoreError: std::error::Error + Send + Sync + 'static,
{
    #[error("agent turn failed: {0}")]
    Runtime(RuntimeError),
    #[error("tool protocol adapter failed: {error}; terminal-state commit: {checkpoint_error:?}")]
    Adapter {
        error: AdapterError,
        checkpoint_error: Option<StoreError>,
    },
    #[error(
        "runtime tool batch failed with {errors_len} error(s); terminal-state commit: {checkpoint_error:?}"
    )]
    ToolBatch {
        errors_len: usize,
        errors: Vec<ToolError>,
        checkpoint_error: Option<StoreError>,
    },
    #[error(
        "runtime tool response contained no configured calls; failed-state commit: {checkpoint_error:?}"
    )]
    MissingCalls {
        checkpoint_error: Option<StoreError>,
    },
    #[error(
        "runtime tool round limit {limit} was reached; failed-state commit: {checkpoint_error:?}"
    )]
    ToolRoundLimit {
        limit: u32,
        checkpoint_error: Option<StoreError>,
    },
    #[error(
        "runtime tool batch contains {actual} calls, exceeding limit {limit}; failed-state commit: {checkpoint_error:?}"
    )]
    ParallelToolLimit {
        actual: usize,
        limit: u32,
        checkpoint_error: Option<StoreError>,
    },
    #[error("tool-started checkpoint did not retain its turn lease")]
    MissingLease,
    #[error("tool-started checkpoint did not retain its native response")]
    MissingResponse,
}

pub type ToolRuntimeErrorFor<P, S, M, F, I, O, A, J, E> = AgentToolRuntimeError<
    AgentRuntimeError<
        <S as CheckpointStore<P>>::Error,
        <M as RequestMaterializer<P>>::Error,
        <F as RequestFingerprinter<P>>::Error,
        <I as InferenceInvoker<P>>::Error,
        <O as OutputInterpreter<P>>::Error,
    >,
    <A as ToolLoopAdapter<P>>::Error,
    ToolRunError<<J as ToolJournal>::Error, <E as ToolExecutor>::Error>,
    <S as CheckpointStore<P>>::Error,
>;

pub type ToolStreamRuntimeErrorFor<P, S, M, F, I, O, V, A, J, E> = AgentToolRuntimeError<
    StreamRuntimeErrorFor<P, S, M, F, I, O, V>,
    <A as ToolLoopAdapter<P>>::Error,
    ToolRunError<<J as ToolJournal>::Error, <E as ToolExecutor>::Error>,
    <S as CheckpointStore<P>>::Error,
>;

impl<P, S, M, F, I, O, G, C> AgentRuntime<P, S, M, F, I, O, G, C>
where
    P: AgentProtocol,
    S: CheckpointStore<P>,
    M: RequestMaterializer<P>,
    F: RequestFingerprinter<P>,
    I: InferenceInvoker<P>,
    O: OutputInterpreter<P>,
    G: IdGenerator,
    C: Clock,
{
    /// Runs a unary public turn through zero or more runtime-owned tool rounds.
    ///
    /// All model and tool results are durably committed before the next step.
    /// Duplicate public idempotency keys return the existing checkpoint and do
    /// not take over in-progress work.
    pub async fn run_unary_with_tools<A, J, E, K, H>(
        &self,
        command: RunTurn<P, I::Context>,
        adapter: &A,
        tool_runner: &ToolRunner<J, E, K, H>,
    ) -> Result<RunTurnResult<P>, ToolRuntimeErrorFor<P, S, M, F, I, O, A, J, E>>
    where
        A: ToolLoopAdapter<P>,
        J: ToolJournal,
        E: ToolExecutor,
        K: ToolIdempotencyKeyProvider,
        H: ToolFailurePolicy<E::Error>,
    {
        let prepared = match self
            .prepare_turn(command)
            .await
            .map_err(AgentToolRuntimeError::Runtime)?
        {
            PrepareTurnResult::Acquired(prepared) => *prepared,
            PrepareTurnResult::Existing(record) => return Ok(RunTurnResult::Existing(record)),
        };
        let PreparedTurn {
            inference_request,
            authorization,
            invocation_context,
            inference_intent,
            identity,
            lease,
        } = prepared;
        let mut request = InferenceRequest {
            request: inference_request,
            context: invocation_context,
            intent: inference_intent,
        };
        let mut lease = lease;
        let mut round = 0_u32;

        loop {
            let committed = self
                .invoke_step(&request, &identity, lease)
                .await
                .map_err(AgentToolRuntimeError::Runtime)?;
            let record = committed.record;
            if record.state != TurnState::ToolStarted {
                return Ok(RunTurnResult::Committed {
                    record: Box::new(record),
                    lease: committed.lease,
                });
            }

            lease = committed.lease.ok_or(AgentToolRuntimeError::MissingLease)?;
            let response = record
                .response
                .as_ref()
                .ok_or(AgentToolRuntimeError::MissingResponse)?;
            if round >= authorization.limits.max_tool_rounds {
                let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                return Err(AgentToolRuntimeError::ToolRoundLimit {
                    limit: authorization.limits.max_tool_rounds,
                    checkpoint_error,
                });
            }

            let calls = match adapter.runtime_calls(response) {
                Ok(calls) => calls,
                Err(error) => {
                    let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                    return Err(AgentToolRuntimeError::Adapter {
                        error,
                        checkpoint_error,
                    });
                }
            };
            if calls.is_empty() {
                let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                return Err(AgentToolRuntimeError::MissingCalls { checkpoint_error });
            }
            if calls.len() > authorization.limits.max_parallel_tools as usize {
                let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                return Err(AgentToolRuntimeError::ParallelToolLimit {
                    actual: calls.len(),
                    limit: authorization.limits.max_parallel_tools,
                    checkpoint_error,
                });
            }

            let outcomes =
                join_all(calls.into_iter().map(|call| {
                    tool_runner.run(&identity.response_id, call, &authorization, round)
                }))
                .await;
            let mut results = Vec::with_capacity(outcomes.len());
            let mut errors = Vec::new();
            let mut outcome_unknown = false;
            for outcome in outcomes {
                match outcome {
                    Ok(result) => results.push(result),
                    Err(error) => {
                        outcome_unknown |= error.requires_unknown_outcome();
                        errors.push(error);
                    }
                }
            }
            if !errors.is_empty() {
                let next_state = if outcome_unknown {
                    TurnState::OutcomeUnknown
                } else {
                    TurnState::Failed
                };
                let checkpoint_error = self.finish_tool_turn(lease, next_state).await;
                return Err(AgentToolRuntimeError::ToolBatch {
                    errors_len: errors.len(),
                    errors,
                    checkpoint_error,
                });
            }

            let replay_items =
                match adapter.append_results(&mut request.request, response, &results) {
                    Ok(items) => items,
                    Err(error) => {
                        let checkpoint_error =
                            self.finish_tool_turn(lease, TurnState::Failed).await;
                        return Err(AgentToolRuntimeError::Adapter {
                            error,
                            checkpoint_error,
                        });
                    }
                };
            let resumed = self
                .store()
                .commit_turn(CommitTurn {
                    lease,
                    next_state: TurnState::InFlight,
                    append_output_items: replay_items,
                    response: None,
                })
                .await
                .map_err(|error| AgentToolRuntimeError::Runtime(AgentRuntimeError::Store(error)))?;
            lease = resumed.lease.ok_or(AgentToolRuntimeError::MissingLease)?;
            round += 1;
            request.intent = InferenceIntent {
                step_kind: ModelStepKind::RuntimeToolContinuation,
            };
        }
    }

    /// Runs one public native stream through zero or more runtime-owned tool rounds.
    ///
    /// Intermediate model-step terminal events are retained internally. The
    /// sole public terminal event is emitted after the final fenced commit.
    pub async fn run_stream_with_tools<'a, V, A, J, E, K, H>(
        &'a self,
        command: RunTurn<P, I::Context>,
        mut stream_interpreter: V,
        adapter: &'a A,
        tool_runner: &'a ToolRunner<J, E, K, H>,
    ) -> Result<
        RunStreamResult<'a, P, ToolStreamRuntimeErrorFor<P, S, M, F, I, O, V, A, J, E>>,
        ToolStreamRuntimeErrorFor<P, S, M, F, I, O, V, A, J, E>,
    >
    where
        V: StreamEventInterpreter<P> + 'a,
        A: ToolLoopAdapter<P>,
        J: ToolJournal,
        E: ToolExecutor,
        K: ToolIdempotencyKeyProvider,
        H: ToolFailurePolicy<E::Error>,
    {
        let prepared = match self.prepare_turn(command).await.map_err(|error| {
            AgentToolRuntimeError::Runtime(AgentStreamRuntimeError::Runtime(error))
        })? {
            PrepareTurnResult::Acquired(prepared) => *prepared,
            PrepareTurnResult::Existing(record) => return Ok(RunStreamResult::Existing(record)),
        };
        let PreparedTurn {
            inference_request,
            authorization,
            invocation_context,
            inference_intent,
            identity,
            lease,
        } = prepared;
        let mut request = InferenceRequest {
            request: inference_request,
            context: invocation_context,
            intent: inference_intent,
        };

        let output = async_stream::stream! {
            let mut lease = lease;
            let mut round = 0_u32;
            loop {
                stream_interpreter.begin_step(request.intent.step_kind);
                let mut inference = match self.invoker().invoke(&request).await {
                    Ok(InferenceOutput::Streaming(stream)) => stream,
                    Ok(InferenceOutput::Unary(_)) => {
                        let checkpoint_error = self.mark_failed(lease).await;
                        yield Err(AgentToolRuntimeError::Runtime(
                            AgentStreamRuntimeError::ExpectedStreaming { checkpoint_error },
                        ));
                        return;
                    }
                    Err(error) => {
                        let checkpoint_error = self.mark_failed(lease).await;
                        yield Err(AgentToolRuntimeError::Runtime(
                            AgentStreamRuntimeError::Runtime(AgentRuntimeError::Inference {
                                error,
                                checkpoint_error,
                            }),
                        ));
                        return;
                    }
                };

                let terminal = loop {
                    let Some(item) = futures::StreamExt::next(&mut inference).await else {
                        let checkpoint_error = self.mark_failed(lease).await;
                        yield Err(AgentToolRuntimeError::Runtime(
                            AgentStreamRuntimeError::MissingTerminal { checkpoint_error },
                        ));
                        return;
                    };
                    let event = match item {
                        Ok(event) => event,
                        Err(error) => {
                            let checkpoint_error = self.mark_failed(lease).await;
                            yield Err(AgentToolRuntimeError::Runtime(
                                AgentStreamRuntimeError::Runtime(AgentRuntimeError::Inference {
                                    error,
                                    checkpoint_error,
                                }),
                            ));
                            return;
                        }
                    };
                    let action = match stream_interpreter.observe(event, &identity) {
                        Ok(action) => action,
                        Err(error) => {
                            let checkpoint_error = self.mark_failed(lease).await;
                            yield Err(AgentToolRuntimeError::Runtime(
                                AgentStreamRuntimeError::Interpreter {
                                    error,
                                    checkpoint_error,
                                },
                            ));
                            return;
                        }
                    };
                    match action {
                        StreamEventAction::Emit(mut event) => {
                            stream_interpreter.prepare_emit(&mut event);
                            yield Ok(event);
                        }
                        StreamEventAction::Suppress => {}
                        StreamEventAction::Terminal { event, response } => {
                            break (event, response);
                        }
                    }
                };

                let (mut terminal_event, response) = terminal;
                let committed = match self.commit_response(response, &identity, lease).await {
                    Ok(committed) => committed,
                    Err(error) => {
                        yield Err(AgentToolRuntimeError::Runtime(
                            AgentStreamRuntimeError::Runtime(error),
                        ));
                        return;
                    }
                };
                let record = committed.record;
                if record.state != TurnState::ToolStarted {
                    stream_interpreter.prepare_emit(&mut terminal_event);
                    yield Ok(terminal_event);
                    return;
                }

                lease = match committed.lease {
                    Some(lease) => lease,
                    None => {
                        yield Err(AgentToolRuntimeError::MissingLease);
                        return;
                    }
                };
                let response = match record.response.as_ref() {
                    Some(response) => response,
                    None => {
                        yield Err(AgentToolRuntimeError::MissingResponse);
                        return;
                    }
                };
                if round >= authorization.limits.max_tool_rounds {
                    let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                    yield Err(AgentToolRuntimeError::ToolRoundLimit {
                        limit: authorization.limits.max_tool_rounds,
                        checkpoint_error,
                    });
                    return;
                }

                let calls = match adapter.runtime_calls(response) {
                    Ok(calls) => calls,
                    Err(error) => {
                        let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                        yield Err(AgentToolRuntimeError::Adapter {
                            error,
                            checkpoint_error,
                        });
                        return;
                    }
                };
                if calls.is_empty() {
                    let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                    yield Err(AgentToolRuntimeError::MissingCalls { checkpoint_error });
                    return;
                }
                if calls.len() > authorization.limits.max_parallel_tools as usize {
                    let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                    yield Err(AgentToolRuntimeError::ParallelToolLimit {
                        actual: calls.len(),
                        limit: authorization.limits.max_parallel_tools,
                        checkpoint_error,
                    });
                    return;
                }

                let outcomes = join_all(calls.into_iter().map(|call| {
                    tool_runner.run(&identity.response_id, call, &authorization, round)
                }))
                .await;
                let mut results = Vec::with_capacity(outcomes.len());
                let mut errors = Vec::new();
                let mut outcome_unknown = false;
                for outcome in outcomes {
                    match outcome {
                        Ok(result) => results.push(result),
                        Err(error) => {
                            outcome_unknown |= error.requires_unknown_outcome();
                            errors.push(error);
                        }
                    }
                }
                if !errors.is_empty() {
                    let next_state = if outcome_unknown {
                        TurnState::OutcomeUnknown
                    } else {
                        TurnState::Failed
                    };
                    let checkpoint_error = self.finish_tool_turn(lease, next_state).await;
                    yield Err(AgentToolRuntimeError::ToolBatch {
                        errors_len: errors.len(),
                        errors,
                        checkpoint_error,
                    });
                    return;
                }

                let replay_items = match adapter.append_results(
                    &mut request.request,
                    response,
                    &results,
                ) {
                    Ok(items) => items,
                    Err(error) => {
                        let checkpoint_error = self.finish_tool_turn(lease, TurnState::Failed).await;
                        yield Err(AgentToolRuntimeError::Adapter {
                            error,
                            checkpoint_error,
                        });
                        return;
                    }
                };
                let resumed = match self.store().commit_turn(CommitTurn {
                    lease,
                    next_state: TurnState::InFlight,
                    append_output_items: replay_items,
                    response: None,
                }).await {
                    Ok(resumed) => resumed,
                    Err(error) => {
                        yield Err(AgentToolRuntimeError::Runtime(
                            AgentStreamRuntimeError::Runtime(AgentRuntimeError::Store(error)),
                        ));
                        return;
                    }
                };
                lease = match resumed.lease {
                    Some(lease) => lease,
                    None => {
                        yield Err(AgentToolRuntimeError::MissingLease);
                        return;
                    }
                };
                round += 1;
                request.intent = InferenceIntent {
                    step_kind: ModelStepKind::RuntimeToolContinuation,
                };
            }
        };
        Ok(RunStreamResult::Live(Box::pin(output)))
    }

    async fn finish_tool_turn(&self, lease: TurnLease, next_state: TurnState) -> Option<S::Error> {
        self.store()
            .commit_turn(CommitTurn {
                lease,
                next_state,
                append_output_items: Vec::new(),
                response: None,
            })
            .await
            .err()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use dynamo_protocols::types::responses::{
        CreateResponse, InputParam, Item, Response, ResponseCompletedEvent, ResponseCreatedEvent,
        ResponseStreamEvent, ResponseTextDeltaEvent, Status,
    };
    use futures::StreamExt;
    use serde_json::json;

    use crate::{
        AgentRuntime, AuthorizationScope, Blake3ToolIdempotencyKeys, BoxFuture,
        CanonicalJsonFingerprinter, CheckpointStore, ConfiguredToolRouter,
        ConservativeToolFailurePolicy, IdGenerator, IdempotencyKey, InMemoryCheckpointStore,
        InMemoryToolJournal, InferenceFuture, InferenceIntent, InferenceInvoker, InferenceOutput,
        InferenceRequest, ModelStepKind, OpenAiResponses, PolicyResponsesOutputInterpreter,
        ResponseId, ResponsesRequestMaterializer, ResponsesStreamEventInterpreter,
        ResponsesToolLoopAdapter, RoutedResponsesOutcomePolicy, RunStreamResult, RunTurn,
        RunTurnResult, RuntimeAuthorization, RuntimeLimits, ToolExecutionRequest,
        ToolExecutionResult, ToolExecutor, ToolRoute, ToolRunner, TurnId, TurnState,
    };

    #[derive(Debug, Clone, Copy)]
    struct FixedClock;

    impl crate::Clock for FixedClock {
        fn now_millis(&self) -> u64 {
            1_000
        }
    }

    #[derive(Debug, Default)]
    struct SequentialIds {
        response: AtomicU64,
        turn: AtomicU64,
    }

    impl IdGenerator for SequentialIds {
        fn response_id(&self) -> ResponseId {
            ResponseId::new(format!(
                "resp-{}",
                self.response.fetch_add(1, Ordering::Relaxed) + 1
            ))
        }

        fn turn_id(&self) -> TurnId {
            TurnId::new(format!(
                "turn-{}",
                self.turn.fetch_add(1, Ordering::Relaxed) + 1
            ))
        }
    }

    struct SequenceInvoker {
        requests: Arc<Mutex<Vec<(CreateResponse, InferenceIntent)>>>,
        responses: Mutex<VecDeque<Response>>,
    }

    impl InferenceInvoker<OpenAiResponses> for SequenceInvoker {
        type Context = ();
        type Error = Infallible;

        fn invoke<'a>(
            &'a self,
            request: &'a InferenceRequest<OpenAiResponses, Self::Context>,
        ) -> InferenceFuture<'a, OpenAiResponses, Self::Error> {
            self.requests
                .lock()
                .unwrap()
                .push((request.request.clone(), request.intent));
            let response = self.responses.lock().unwrap().pop_front().unwrap();
            Box::pin(async move { Ok(InferenceOutput::Unary(Box::new(response))) })
        }
    }

    struct StreamingSequenceInvoker {
        requests: Arc<Mutex<Vec<(CreateResponse, InferenceIntent)>>>,
        responses: Mutex<VecDeque<Vec<ResponseStreamEvent>>>,
    }

    impl InferenceInvoker<OpenAiResponses> for StreamingSequenceInvoker {
        type Context = ();
        type Error = Infallible;

        fn invoke<'a>(
            &'a self,
            request: &'a InferenceRequest<OpenAiResponses, Self::Context>,
        ) -> InferenceFuture<'a, OpenAiResponses, Self::Error> {
            self.requests
                .lock()
                .unwrap()
                .push((request.request.clone(), request.intent));
            let events = self.responses.lock().unwrap().pop_front().unwrap();
            Box::pin(async move {
                Ok(InferenceOutput::Streaming(Box::pin(futures::stream::iter(
                    events.into_iter().map(Ok),
                ))))
            })
        }
    }

    struct CountingExecutor {
        calls: Arc<AtomicUsize>,
    }

    impl ToolExecutor for CountingExecutor {
        type Error = Infallible;

        fn execute(
            &self,
            request: ToolExecutionRequest,
        ) -> BoxFuture<'_, Result<ToolExecutionResult, Self::Error>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                assert_eq!(request.connector, "search");
                assert_eq!(request.operation, "query");
                Ok(ToolExecutionResult {
                    output: json!({"answer": 42}),
                })
            })
        }

        fn lookup(
            &self,
            _scope: &AuthorizationScope,
            _idempotency_key: &crate::IdempotencyKey,
        ) -> BoxFuture<'_, Result<Option<ToolExecutionResult>, Self::Error>> {
            Box::pin(async { Ok(None) })
        }
    }

    fn response(output: serde_json::Value) -> Response {
        serde_json::from_value(json!({
            "created_at": 1,
            "id": "backend-id",
            "model": "model",
            "object": "response",
            "output": output,
            "status": "completed"
        }))
        .unwrap()
    }

    fn tool_response() -> Response {
        response(json!([{
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": "{\"query\":\"rust\"}",
            "status": "completed"
        }]))
    }

    fn final_response() -> Response {
        response(json!([{
            "type": "message",
            "id": "msg-1",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "42",
                "annotations": [],
                "logprobs": null
            }]
        }]))
    }

    fn authorization() -> RuntimeAuthorization {
        RuntimeAuthorization {
            scope: AuthorizationScope {
                tenant_id: "tenant".to_owned(),
                principal_id: "principal".to_owned(),
            },
            permitted_connectors: BTreeSet::from(["search".to_owned()]),
            limits: RuntimeLimits::default(),
        }
    }

    fn command(idempotency_key: &str) -> RunTurn<OpenAiResponses, ()> {
        RunTurn {
            request: CreateResponse {
                input: InputParam::Text("find the answer".to_owned()),
                ..Default::default()
            },
            parent_response_id: None,
            authorization: authorization(),
            idempotency_key: IdempotencyKey::from(idempotency_key),
            invocation_context: (),
            inference_intent: InferenceIntent {
                step_kind: ModelStepKind::Initial,
            },
            lease_duration_millis: 30_000,
        }
    }

    fn stream_events(response: Response, text_delta: Option<&str>) -> Vec<ResponseStreamEvent> {
        let created_response = Response {
            status: Status::InProgress,
            output: Vec::new(),
            ..response.clone()
        };
        let mut events = vec![ResponseStreamEvent::ResponseCreated(ResponseCreatedEvent {
            sequence_number: 0,
            response: created_response,
        })];
        if let Some(delta) = text_delta {
            events.push(ResponseStreamEvent::ResponseOutputTextDelta(
                ResponseTextDeltaEvent {
                    sequence_number: 1,
                    item_id: "msg-1".to_owned(),
                    output_index: 0,
                    content_index: 0,
                    delta: delta.to_owned(),
                    logprobs: None,
                },
            ));
        }
        events.push(ResponseStreamEvent::ResponseCompleted(
            ResponseCompletedEvent {
                sequence_number: 2,
                response,
            },
        ));
        events
    }

    #[tokio::test]
    async fn responses_tool_round_is_native_durable_and_idempotent() {
        let router =
            ConfiguredToolRouter::new([("lookup".to_owned(), ToolRoute::new("search", "query"))]);
        let inference_requests = Arc::new(Mutex::new(Vec::new()));
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let runtime = AgentRuntime::new(
            InMemoryCheckpointStore::<OpenAiResponses, _>::new(FixedClock),
            ResponsesRequestMaterializer::default(),
            CanonicalJsonFingerprinter,
            SequenceInvoker {
                requests: inference_requests.clone(),
                responses: Mutex::new(VecDeque::from([tool_response(), final_response()])),
            },
            PolicyResponsesOutputInterpreter::new(RoutedResponsesOutcomePolicy::new(
                router.clone(),
            )),
            SequentialIds::default(),
            FixedClock,
        );
        let adapter = ResponsesToolLoopAdapter::new(router);
        let tool_runner = ToolRunner::new(
            InMemoryToolJournal::default(),
            CountingExecutor {
                calls: tool_calls.clone(),
            },
            Blake3ToolIdempotencyKeys,
            ConservativeToolFailurePolicy,
        );
        let command = command("idem-1");

        let first = runtime
            .run_unary_with_tools(command.clone(), &adapter, &tool_runner)
            .await
            .unwrap();
        assert_eq!(first.record().state, TurnState::Completed);
        assert_eq!(first.record().response.as_ref().unwrap().id, "resp-1");
        assert_eq!(first.record().output_items.len(), 3);
        assert_eq!(tool_calls.load(Ordering::Relaxed), 1);

        {
            let requests = inference_requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            assert_eq!(requests[0].1.step_kind, ModelStepKind::Initial);
            assert_eq!(
                requests[1].1.step_kind,
                ModelStepKind::RuntimeToolContinuation
            );
            let InputParam::Items(items) = &requests[1].0.input else {
                panic!("tool continuation must use native item input")
            };
            assert_eq!(items.len(), 3);
            assert!(matches!(
                &items[2],
                dynamo_protocols::types::responses::InputItem::Item(Item::FunctionCallOutput(_))
            ));
        }

        let duplicate = runtime
            .run_unary_with_tools(command, &adapter, &tool_runner)
            .await
            .unwrap();
        assert!(matches!(duplicate, RunTurnResult::Existing(_)));
        assert_eq!(inference_requests.lock().unwrap().len(), 2);
        assert_eq!(tool_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn streamed_tool_round_has_one_public_lifecycle_and_terminal() {
        let router =
            ConfiguredToolRouter::new([("lookup".to_owned(), ToolRoute::new("search", "query"))]);
        let inference_requests = Arc::new(Mutex::new(Vec::new()));
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let runtime = AgentRuntime::new(
            InMemoryCheckpointStore::<OpenAiResponses, _>::new(FixedClock),
            ResponsesRequestMaterializer::default(),
            CanonicalJsonFingerprinter,
            StreamingSequenceInvoker {
                requests: inference_requests.clone(),
                responses: Mutex::new(VecDeque::from([
                    stream_events(tool_response(), None),
                    stream_events(final_response(), Some("42")),
                ])),
            },
            PolicyResponsesOutputInterpreter::new(RoutedResponsesOutcomePolicy::new(
                router.clone(),
            )),
            SequentialIds::default(),
            FixedClock,
        );
        let adapter = ResponsesToolLoopAdapter::new(router);
        let tool_runner = ToolRunner::new(
            InMemoryToolJournal::default(),
            CountingExecutor {
                calls: tool_calls.clone(),
            },
            Blake3ToolIdempotencyKeys,
            ConservativeToolFailurePolicy,
        );
        let command = command("idem-stream-tools");
        let result = runtime
            .run_stream_with_tools(
                command.clone(),
                ResponsesStreamEventInterpreter::default(),
                &adapter,
                &tool_runner,
            )
            .await
            .unwrap();
        let RunStreamResult::Live(mut stream) = result else {
            panic!("expected a newly acquired stream")
        };

        let ResponseStreamEvent::ResponseCreated(created) = stream.next().await.unwrap().unwrap()
        else {
            panic!("expected one public created event")
        };
        assert_eq!(created.sequence_number, 0);
        assert_eq!(created.response.id, "resp-1");
        let ResponseStreamEvent::ResponseOutputTextDelta(delta) =
            stream.next().await.unwrap().unwrap()
        else {
            panic!("expected final-step text delta")
        };
        assert_eq!(delta.sequence_number, 1);
        assert_eq!(delta.delta, "42");
        let ResponseStreamEvent::ResponseCompleted(completed) =
            stream.next().await.unwrap().unwrap()
        else {
            panic!("expected one public terminal event")
        };
        assert_eq!(completed.sequence_number, 2);
        assert_eq!(completed.response.id, "resp-1");
        assert!(stream.next().await.is_none());

        assert_eq!(tool_calls.load(Ordering::Relaxed), 1);
        {
            let requests = inference_requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            assert_eq!(requests[0].1.step_kind, ModelStepKind::Initial);
            assert_eq!(
                requests[1].1.step_kind,
                ModelStepKind::RuntimeToolContinuation
            );
        }
        assert_eq!(
            runtime
                .store()
                .load_chain(crate::LoadChain {
                    scope: authorization().scope,
                    response_id: ResponseId::from("resp-1"),
                })
                .await
                .unwrap()
                .pop()
                .unwrap()
                .state,
            TurnState::Completed
        );

        let duplicate = runtime
            .run_stream_with_tools(
                command,
                ResponsesStreamEventInterpreter::default(),
                &adapter,
                &tool_runner,
            )
            .await
            .unwrap();
        assert!(matches!(duplicate, RunStreamResult::Existing(_)));
        assert_eq!(inference_requests.lock().unwrap().len(), 2);
        assert_eq!(tool_calls.load(Ordering::Relaxed), 1);
    }
}
