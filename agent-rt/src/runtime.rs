// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;

use thiserror::Error;

use crate::{
    AgentProtocol, BeginTurn, BeginTurnResult, CheckpointRecord, CheckpointStore, Clock,
    CommitTurn, IdGenerator, IdempotencyKey, InferenceIntent, InferenceInvoker, InferenceOutput,
    InferenceRequest, LeaseDeadline, LoadChain, OutputIdentity, OutputInterpreter,
    RequestFingerprinter, RequestMaterializer, ResponseId, RuntimeAuthorization, TurnLease,
    TurnState,
};

/// Inputs supplied by authenticated frontend policy for one public turn.
#[derive(Debug, Clone)]
pub struct RunTurn<P, C>
where
    P: AgentProtocol,
{
    pub request: P::Request,
    pub parent_response_id: Option<ResponseId>,
    pub authorization: RuntimeAuthorization,
    pub idempotency_key: IdempotencyKey,
    pub invocation_context: C,
    pub inference_intent: InferenceIntent,
    pub lease_duration_millis: u64,
}

/// Durable result of a new or idempotently repeated turn.
#[derive(Debug, Clone)]
pub enum RunTurnResult<P>
where
    P: AgentProtocol,
{
    Committed {
        record: Box<CheckpointRecord<P>>,
        /// Present only when the committed state retains fenced ownership for
        /// immediate runtime work such as tool execution.
        lease: Option<TurnLease>,
    },
    Existing(Box<CheckpointRecord<P>>),
}

impl<P> RunTurnResult<P>
where
    P: AgentProtocol,
{
    pub fn record(&self) -> &CheckpointRecord<P> {
        match self {
            Self::Committed { record, .. } | Self::Existing(record) => record,
        }
    }

    pub fn is_existing(&self) -> bool {
        matches!(self, Self::Existing(_))
    }
}

#[derive(Debug, Error)]
pub enum AgentRuntimeError<
    StoreError,
    MaterializerError,
    FingerprintError,
    InferenceError,
    OutputError,
> where
    StoreError: std::error::Error + Send + Sync + 'static,
    MaterializerError: std::error::Error + Send + Sync + 'static,
    FingerprintError: std::error::Error + Send + Sync + 'static,
    InferenceError: std::error::Error + Send + Sync + 'static,
    OutputError: std::error::Error + Send + Sync + 'static,
{
    #[error("checkpoint store failed: {0}")]
    Store(StoreError),
    #[error("request materialization failed: {0}")]
    Materialize(MaterializerError),
    #[error("request fingerprinting failed: {0}")]
    Fingerprint(FingerprintError),
    #[error("lease deadline overflow")]
    LeaseDeadlineOverflow,
    #[error("inference failed: {error}; failed-state commit: {checkpoint_error:?}")]
    Inference {
        error: InferenceError,
        checkpoint_error: Option<StoreError>,
    },
    #[error(
        "unary runtime received a streaming inference result; failed-state commit: {checkpoint_error:?}"
    )]
    StreamingUnsupported {
        checkpoint_error: Option<StoreError>,
    },
    #[error("output interpretation failed: {error}; failed-state commit: {checkpoint_error:?}")]
    Output {
        error: OutputError,
        checkpoint_error: Option<StoreError>,
    },
    #[error(
        "output requested invalid in-flight transition to {state:?}; failed-state commit: {checkpoint_error:?}"
    )]
    InvalidOutputState {
        state: TurnState,
        checkpoint_error: Option<StoreError>,
    },
}

pub type RuntimeErrorFor<P, S, M, F, I, O> = AgentRuntimeError<
    <S as CheckpointStore<P>>::Error,
    <M as RequestMaterializer<P>>::Error,
    <F as RequestFingerprinter<P>>::Error,
    <I as InferenceInvoker<P>>::Error,
    <O as OutputInterpreter<P>>::Error,
>;

struct PreparedTurn<P, C>
where
    P: AgentProtocol,
{
    inference_request: P::Request,
    invocation_context: C,
    inference_intent: InferenceIntent,
    identity: OutputIdentity,
    lease: TurnLease,
}

enum PrepareTurnResult<P, C>
where
    P: AgentProtocol,
{
    Acquired(Box<PreparedTurn<P, C>>),
    Existing(Box<CheckpointRecord<P>>),
}

/// Protocol-generic coordinator composed entirely from replaceable boundaries.
pub struct AgentRuntime<P, S, M, F, I, O, G, C>
where
    P: AgentProtocol,
{
    store: S,
    materializer: M,
    fingerprinter: F,
    invoker: I,
    output_interpreter: O,
    ids: G,
    clock: C,
    protocol: PhantomData<fn() -> P>,
}

impl<P, S, M, F, I, O, G, C> AgentRuntime<P, S, M, F, I, O, G, C>
where
    P: AgentProtocol,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: S,
        materializer: M,
        fingerprinter: F,
        invoker: I,
        output_interpreter: O,
        ids: G,
        clock: C,
    ) -> Self {
        Self {
            store,
            materializer,
            fingerprinter,
            invoker,
            output_interpreter,
            ids,
            clock,
            protocol: PhantomData,
        }
    }

    pub fn store(&self) -> &S {
        &self.store
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use std::collections::BTreeSet;
    use std::convert::Infallible;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dynamo_protocols::types::{
        anthropic::{AnthropicCreateMessageRequest, AnthropicMessageResponse},
        responses::{CreateResponse, InputParam, Response},
    };
    use thiserror::Error;

    use crate::{
        AgentRuntime, AgentRuntimeError, AnthropicMessages, AnthropicOutputInterpreter,
        AnthropicRequestMaterializer, AuthorizationScope, CanonicalJsonFingerprinter,
        CheckpointStore, Clock, IdGenerator, IdempotencyKey, InMemoryCheckpointStore,
        InferenceFuture, InferenceIntent, InferenceInvoker, InferenceOutput, InferenceRequest,
        LoadChain, ModelStepKind, OpenAiResponses, ResponseId, ResponsesOutputInterpreter,
        ResponsesRequestMaterializer, RunTurn, RunTurnResult, RuntimeAuthorization, RuntimeLimits,
        TurnId, TurnState,
    };

    #[derive(Debug, Clone, Copy)]
    struct FixedClock(u64);

    impl Clock for FixedClock {
        fn now_millis(&self) -> u64 {
            self.0
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

    #[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
    #[error("mock inference failure")]
    struct MockInferenceError;

    #[derive(Debug)]
    struct MockInvoker {
        requests: Mutex<Vec<CreateResponse>>,
        response: Response,
        fail: bool,
    }

    impl MockInvoker {
        fn new(fail: bool) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: serde_json::from_value(serde_json::json!({
                    "created_at": 1,
                    "id": "backend-id",
                    "model": "model",
                    "object": "response",
                    "output": [{
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": "answer",
                            "annotations": [],
                            "logprobs": null
                        }]
                    }],
                    "status": "completed"
                }))
                .unwrap(),
                fail,
            }
        }

        fn requests(&self) -> Vec<CreateResponse> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl InferenceInvoker<OpenAiResponses> for MockInvoker {
        type Context = ();
        type Error = MockInferenceError;

        fn invoke(
            &self,
            request: InferenceRequest<OpenAiResponses, Self::Context>,
        ) -> InferenceFuture<'_, OpenAiResponses, Self::Error> {
            self.requests.lock().unwrap().push(request.request);
            let fail = self.fail;
            let response = self.response.clone();
            Box::pin(async move {
                if fail {
                    Err(MockInferenceError)
                } else {
                    Ok(InferenceOutput::Unary(Box::new(response)))
                }
            })
        }
    }

    #[derive(Debug)]
    struct AnthropicInvoker {
        requests: Mutex<Vec<AnthropicCreateMessageRequest>>,
        response: AnthropicMessageResponse,
    }

    impl AnthropicInvoker {
        fn new() -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: serde_json::from_value(serde_json::json!({
                    "id": "msg_backend",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer"}],
                    "model": "claude",
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {"input_tokens": 10, "output_tokens": 4}
                }))
                .unwrap(),
            }
        }
    }

    impl InferenceInvoker<AnthropicMessages> for AnthropicInvoker {
        type Context = ();
        type Error = Infallible;

        fn invoke(
            &self,
            request: InferenceRequest<AnthropicMessages, Self::Context>,
        ) -> InferenceFuture<'_, AnthropicMessages, Self::Error> {
            self.requests.lock().unwrap().push(request.request);
            let response = self.response.clone();
            Box::pin(async move { Ok(InferenceOutput::Unary(Box::new(response))) })
        }
    }

    type TestRuntime = AgentRuntime<
        OpenAiResponses,
        InMemoryCheckpointStore<OpenAiResponses, FixedClock>,
        ResponsesRequestMaterializer,
        CanonicalJsonFingerprinter,
        MockInvoker,
        ResponsesOutputInterpreter,
        SequentialIds,
        FixedClock,
    >;

    fn runtime(fail: bool) -> TestRuntime {
        let clock = FixedClock(1_000);
        AgentRuntime::new(
            InMemoryCheckpointStore::new(clock),
            ResponsesRequestMaterializer::default(),
            CanonicalJsonFingerprinter,
            MockInvoker::new(fail),
            ResponsesOutputInterpreter::default(),
            SequentialIds::default(),
            clock,
        )
    }

    fn authorization() -> RuntimeAuthorization {
        RuntimeAuthorization {
            scope: AuthorizationScope {
                tenant_id: "tenant".to_owned(),
                principal_id: "principal".to_owned(),
            },
            permitted_connectors: BTreeSet::new(),
            limits: RuntimeLimits::default(),
        }
    }

    fn command(
        input: &str,
        previous_response_id: Option<&str>,
        idempotency_key: &str,
    ) -> RunTurn<OpenAiResponses, ()> {
        RunTurn {
            request: CreateResponse {
                input: InputParam::Text(input.to_owned()),
                previous_response_id: previous_response_id.map(str::to_owned),
                store: Some(true),
                ..Default::default()
            },
            parent_response_id: previous_response_id.map(ResponseId::from),
            authorization: authorization(),
            idempotency_key: IdempotencyKey::from(idempotency_key),
            invocation_context: (),
            inference_intent: InferenceIntent {
                step_kind: ModelStepKind::Initial,
                session_final: true,
            },
            lease_duration_millis: 30_000,
        }
    }

    #[tokio::test]
    async fn commits_a_materialized_native_response() {
        let runtime = runtime(false);
        let result = runtime
            .run_unary(command("hello", None, "idem-1"))
            .await
            .unwrap();

        let RunTurnResult::Committed { record, lease } = result else {
            panic!("expected a newly committed turn")
        };
        assert!(lease.is_none());
        assert_eq!(record.response_id.as_str(), "resp-1");
        assert_eq!(record.state, TurnState::Completed);
        assert_eq!(
            record
                .response
                .as_ref()
                .map(|response| response.id.as_str()),
            Some("resp-1")
        );
        assert_eq!(record.output_items.len(), 1);

        let requests = runtime.invoker.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].store, Some(false));
        assert_eq!(requests[0].previous_response_id, None);
    }

    #[tokio::test]
    async fn idempotent_duplicate_does_not_invoke_inference_twice() {
        let runtime = runtime(false);
        runtime
            .run_unary(command("hello", None, "idem-1"))
            .await
            .unwrap();
        let duplicate = runtime
            .run_unary(command("hello", None, "idem-1"))
            .await
            .unwrap();

        assert!(duplicate.is_existing());
        assert_eq!(duplicate.record().response_id.as_str(), "resp-1");
        assert_eq!(runtime.invoker.requests().len(), 1);
    }

    #[tokio::test]
    async fn continuation_hydrates_parent_input_and_output() {
        let runtime = runtime(false);
        runtime
            .run_unary(command("first", None, "idem-1"))
            .await
            .unwrap();
        let second = runtime
            .run_unary(command("second", Some("resp-1"), "idem-2"))
            .await
            .unwrap();

        assert_eq!(
            second
                .record()
                .parent_response_id
                .as_ref()
                .map(ResponseId::as_str),
            Some("resp-1")
        );
        let requests = runtime.invoker.requests();
        let InputParam::Items(items) = &requests[1].input else {
            panic!("continuation must be fully hydrated")
        };
        assert_eq!(items.len(), 3);
        assert_eq!(
            second
                .record()
                .response
                .as_ref()
                .and_then(|response| response.previous_response_id.as_deref()),
            Some("resp-1")
        );
    }

    #[tokio::test]
    async fn inference_failure_is_durably_marked_failed() {
        let runtime = runtime(true);
        let error = runtime
            .run_unary(command("hello", None, "idem-1"))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            AgentRuntimeError::Inference {
                error: MockInferenceError,
                checkpoint_error: None
            }
        ));

        let record = runtime
            .store()
            .load_chain(LoadChain {
                scope: authorization().scope,
                response_id: ResponseId::from("resp-1"),
            })
            .await
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(record.state, TurnState::Failed);
    }

    #[tokio::test]
    async fn anthropic_turn_stays_native_end_to_end() {
        let clock = FixedClock(1_000);
        let runtime = AgentRuntime::<AnthropicMessages, _, _, _, _, _, _, _>::new(
            InMemoryCheckpointStore::new(clock),
            AnthropicRequestMaterializer,
            CanonicalJsonFingerprinter,
            AnthropicInvoker::new(),
            AnthropicOutputInterpreter::default(),
            SequentialIds::default(),
            clock,
        );
        let request: AnthropicCreateMessageRequest = serde_json::from_value(serde_json::json!({
            "model": "claude",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let result = runtime
            .run_unary(RunTurn {
                request,
                parent_response_id: None,
                authorization: authorization(),
                idempotency_key: IdempotencyKey::from("idem-anthropic"),
                invocation_context: (),
                inference_intent: InferenceIntent {
                    step_kind: ModelStepKind::Initial,
                    session_final: true,
                },
                lease_duration_millis: 30_000,
            })
            .await
            .unwrap();

        assert_eq!(result.record().state, TurnState::Completed);
        assert_eq!(
            result
                .record()
                .response
                .as_ref()
                .map(|response| response.id.as_str()),
            Some("msg_backend")
        );
        assert_eq!(result.record().output_items.len(), 1);
        assert_eq!(runtime.invoker.requests.lock().unwrap().len(), 1);
    }
}

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
    pub async fn run_unary(
        &self,
        command: RunTurn<P, I::Context>,
    ) -> Result<RunTurnResult<P>, RuntimeErrorFor<P, S, M, F, I, O>> {
        let prepared = match self.prepare_turn(command).await? {
            PrepareTurnResult::Acquired(prepared) => *prepared,
            PrepareTurnResult::Existing(record) => return Ok(RunTurnResult::Existing(record)),
        };
        let PreparedTurn {
            inference_request,
            invocation_context,
            inference_intent,
            identity,
            lease,
            ..
        } = prepared;

        let inference = self
            .invoker
            .invoke(InferenceRequest {
                request: inference_request,
                context: invocation_context,
                intent: inference_intent,
            })
            .await;
        let response = match inference {
            Ok(InferenceOutput::Unary(response)) => *response,
            Ok(InferenceOutput::Streaming(_)) => {
                let checkpoint_error = self.mark_failed(lease).await;
                return Err(AgentRuntimeError::StreamingUnsupported { checkpoint_error });
            }
            Err(error) => {
                let checkpoint_error = self.mark_failed(lease).await;
                return Err(AgentRuntimeError::Inference {
                    error,
                    checkpoint_error,
                });
            }
        };

        let output = match self.output_interpreter.interpret(response, &identity) {
            Ok(output) => output,
            Err(error) => {
                let checkpoint_error = self.mark_failed(lease).await;
                return Err(AgentRuntimeError::Output {
                    error,
                    checkpoint_error,
                });
            }
        };
        if !TurnState::InFlight.permits_transition_to(&output.next_state) {
            let state = output.next_state;
            let checkpoint_error = self.mark_failed(lease).await;
            return Err(AgentRuntimeError::InvalidOutputState {
                state,
                checkpoint_error,
            });
        }

        let committed = self
            .store
            .commit_turn(CommitTurn {
                lease,
                next_state: output.next_state,
                append_output_items: output.replay_items,
                response: Some(output.response),
            })
            .await
            .map_err(AgentRuntimeError::Store)?;
        Ok(RunTurnResult::Committed {
            record: Box::new(committed.record),
            lease: committed.lease,
        })
    }

    async fn prepare_turn(
        &self,
        command: RunTurn<P, I::Context>,
    ) -> Result<PrepareTurnResult<P, I::Context>, RuntimeErrorFor<P, S, M, F, I, O>> {
        let chain = match &command.parent_response_id {
            Some(parent_response_id) => self
                .store
                .load_chain(LoadChain {
                    scope: command.authorization.scope.clone(),
                    response_id: parent_response_id.clone(),
                })
                .await
                .map_err(AgentRuntimeError::Store)?,
            None => Vec::new(),
        };
        let materialized = self
            .materializer
            .materialize(command.request, &chain)
            .map_err(AgentRuntimeError::Materialize)?;
        let request_fingerprint = self
            .fingerprinter
            .fingerprint(&materialized.checkpoint_request)
            .map_err(AgentRuntimeError::Fingerprint)?;
        let response_id = self.ids.response_id();
        let turn_id = self.ids.turn_id();
        let lease_deadline = LeaseDeadline(
            self.clock
                .now_millis()
                .checked_add(command.lease_duration_millis)
                .ok_or(AgentRuntimeError::LeaseDeadlineOverflow)?,
        );

        let lease = match self
            .store
            .begin_turn(BeginTurn {
                response_id: response_id.clone(),
                turn_id,
                parent_response_id: command.parent_response_id.clone(),
                authorization: command.authorization,
                idempotency_key: command.idempotency_key,
                request_fingerprint,
                request: materialized.checkpoint_request,
                lease_deadline,
            })
            .await
            .map_err(AgentRuntimeError::Store)?
        {
            BeginTurnResult::Acquired(lease) => lease,
            BeginTurnResult::Existing(record) => return Ok(PrepareTurnResult::Existing(record)),
        };

        Ok(PrepareTurnResult::Acquired(Box::new(PreparedTurn {
            inference_request: materialized.inference_request,
            invocation_context: command.invocation_context,
            inference_intent: command.inference_intent,
            identity: OutputIdentity {
                response_id,
                parent_response_id: command.parent_response_id,
            },
            lease,
        })))
    }

    async fn mark_failed(&self, lease: TurnLease) -> Option<S::Error> {
        self.store
            .commit_turn(CommitTurn {
                lease,
                next_state: TurnState::Failed,
                append_output_items: Vec::new(),
                response: None,
            })
            .await
            .err()
    }
}
