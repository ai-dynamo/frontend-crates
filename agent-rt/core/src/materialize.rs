// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;

use dynamo_protocols::types::responses::{
    CreateResponse, EasyInputContent, EasyInputMessage, InputItem, InputParam, MessageType, Role,
};
use thiserror::Error;

use crate::{
    AgentProtocol, CheckpointRecord, OpenAiResponses, ResponseId, ResponsesCheckpointRecord,
    TurnState,
};

/// Resolves controls omitted by a continuation request.
///
/// Implementations can preserve OpenAI-compatible behavior, enforce a
/// deployment-specific reset policy, or reject unsupported combinations.
pub trait ContinuationPolicy: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn resolve(
        &self,
        current: &mut CreateResponse,
        parent: Option<&ResponsesCheckpointRecord>,
    ) -> Result<(), Self::Error>;
}

/// Inherits tool controls from the immediately preceding turn.
///
/// Explicit values on the current request always win. Checkpoints hold the
/// effective values, so resolving a continuation does not scan the full chain.
/// Instructions deliberately do not carry across `previous_response_id`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InheritContinuationControls;

impl ContinuationPolicy for InheritContinuationControls {
    type Error = Infallible;

    fn resolve(
        &self,
        current: &mut CreateResponse,
        parent: Option<&ResponsesCheckpointRecord>,
    ) -> Result<(), Self::Error> {
        let Some(parent) = parent else {
            return Ok(());
        };

        if current.tools.is_none() {
            current.tools.clone_from(&parent.request.tools);
        }
        if current.tool_choice.is_none() {
            current.tool_choice.clone_from(&parent.request.tool_choice);
        }
        Ok(())
    }
}

/// The two request forms produced for one stateful turn.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedTurn<P>
where
    P: AgentProtocol,
{
    /// Append-only current-turn request persisted in the checkpoint.
    pub checkpoint_request: P::Request,
    /// Complete model-visible request sent through the inference frontend.
    pub inference_request: P::Request,
}

/// Hydrates a protocol request from an authorized checkpoint chain.
pub trait RequestMaterializer<P>: Send + Sync + 'static
where
    P: AgentProtocol,
{
    type Error: std::error::Error + Send + Sync + 'static;

    fn materialize(
        &self,
        current: P::Request,
        chain: &[CheckpointRecord<P>],
    ) -> Result<MaterializedTurn<P>, Self::Error>;
}

#[derive(Debug, Error)]
pub enum MaterializationError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[error("request references a previous response but no checkpoint chain was loaded")]
    MissingChain,
    #[error("a checkpoint chain was supplied for a request without previous_response_id")]
    UnexpectedChain,
    #[error("request references {requested}, but the checkpoint chain ends at {actual}")]
    ParentMismatch {
        requested: ResponseId,
        actual: ResponseId,
    },
    #[error("checkpoint chain has a broken parent link at response {0}")]
    BrokenChain(ResponseId),
    #[error("checkpoint chain crosses authorization scopes at response {0}")]
    ScopeMismatch(ResponseId),
    #[error("response {response_id} is not replayable in state {state:?}")]
    NonReplayable {
        response_id: ResponseId,
        state: TurnState,
    },
    #[error("continuation policy rejected the request: {0}")]
    Policy(E),
}

/// Responses materializer composed with a replaceable continuation policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolicyRequestMaterializer<P> {
    policy: P,
}

impl<P> PolicyRequestMaterializer<P> {
    pub fn new(policy: P) -> Self {
        Self { policy }
    }
}

/// Default Responses materializer with inherited continuation controls.
pub type ResponsesRequestMaterializer = PolicyRequestMaterializer<InheritContinuationControls>;

impl<P> RequestMaterializer<OpenAiResponses> for PolicyRequestMaterializer<P>
where
    P: ContinuationPolicy,
{
    type Error = MaterializationError<P::Error>;

    fn materialize(
        &self,
        current: CreateResponse,
        chain: &[ResponsesCheckpointRecord],
    ) -> Result<MaterializedTurn<OpenAiResponses>, Self::Error> {
        validate_chain(&current, chain)?;

        let mut checkpoint_request = current;
        checkpoint_request.previous_response_id = None;
        self.policy
            .resolve(&mut checkpoint_request, chain.last())
            .map_err(MaterializationError::Policy)?;

        let mut inference_request = checkpoint_request.clone();
        inference_request.input = InputParam::Items(flatten_inputs(chain, &checkpoint_request));
        // The runtime is the single owner of durable response state. Backends
        // receive a self-contained request and must not create a second chain.
        inference_request.store = Some(false);

        Ok(MaterializedTurn {
            checkpoint_request,
            inference_request,
        })
    }
}

fn validate_chain<E>(
    current: &CreateResponse,
    chain: &[ResponsesCheckpointRecord],
) -> Result<(), MaterializationError<E>>
where
    E: std::error::Error + Send + Sync + 'static,
{
    match (current.previous_response_id.as_deref(), chain.last()) {
        (Some(_), None) => return Err(MaterializationError::MissingChain),
        (None, Some(_)) => return Err(MaterializationError::UnexpectedChain),
        (None, None) => return Ok(()),
        (Some(requested), Some(actual)) if requested != actual.response_id.as_str() => {
            return Err(MaterializationError::ParentMismatch {
                requested: requested.into(),
                actual: actual.response_id.clone(),
            });
        }
        (Some(_), Some(_)) => {}
    }

    let scope = &chain[0].scope;
    for (index, record) in chain.iter().enumerate() {
        let expected_parent = index
            .checked_sub(1)
            .map(|parent_index| &chain[parent_index].response_id);
        if record.parent_response_id.as_ref() != expected_parent {
            return Err(MaterializationError::BrokenChain(
                record.response_id.clone(),
            ));
        }
        if &record.scope != scope {
            return Err(MaterializationError::ScopeMismatch(
                record.response_id.clone(),
            ));
        }
        if !matches!(
            record.state,
            TurnState::Completed | TurnState::AwaitingClientToolOutput
        ) {
            return Err(MaterializationError::NonReplayable {
                response_id: record.response_id.clone(),
                state: record.state.clone(),
            });
        }
    }
    Ok(())
}

fn flatten_inputs(chain: &[ResponsesCheckpointRecord], current: &CreateResponse) -> Vec<InputItem> {
    let item_count = chain
        .iter()
        .map(|record| input_len(&record.request.input) + record.output_items.len())
        .sum::<usize>()
        + input_len(&current.input);
    let mut items = Vec::with_capacity(item_count);

    for record in chain {
        append_input(&mut items, &record.request.input);
        items.extend(record.output_items.iter().cloned());
    }
    append_input(&mut items, &current.input);
    items
}

fn input_len(input: &InputParam) -> usize {
    match input {
        InputParam::Text(_) => 1,
        InputParam::Items(items) => items.len(),
    }
}

fn append_input(items: &mut Vec<InputItem>, input: &InputParam) {
    match input {
        InputParam::Text(text) => items.push(InputItem::EasyMessage(EasyInputMessage {
            r#type: MessageType::Message,
            role: Role::User,
            content: EasyInputContent::Text(text.clone()),
            phase: None,
        })),
        InputParam::Items(input_items) => items.extend(input_items.iter().cloned()),
    }
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::responses::{
        CreateResponse, EasyInputContent, EasyInputMessage, InputItem, InputParam, MessageType,
        Role, Tool, ToolChoiceOptions, ToolChoiceParam,
    };

    use crate::{
        AuthorizationScope, CheckpointVersion, IdempotencyKey, RequestFingerprint, ResponseId,
        ResponsesCheckpointRecord, TurnState,
    };

    use super::{MaterializationError, RequestMaterializer, ResponsesRequestMaterializer};

    fn message(role: Role, text: &str) -> InputItem {
        InputItem::EasyMessage(EasyInputMessage {
            r#type: MessageType::Message,
            role,
            content: EasyInputContent::Text(text.to_owned()),
            phase: None,
        })
    }

    fn scope(principal: &str) -> AuthorizationScope {
        AuthorizationScope {
            tenant_id: "tenant-a".to_owned(),
            principal_id: principal.to_owned(),
        }
    }

    fn record(
        id: &str,
        parent: Option<&str>,
        input: &str,
        output: &str,
    ) -> ResponsesCheckpointRecord {
        ResponsesCheckpointRecord {
            response_id: ResponseId::from(id),
            parent_response_id: parent.map(ResponseId::from),
            scope: scope("principal-a"),
            idempotency_key: IdempotencyKey::new(format!("idem-{id}")),
            request_fingerprint: RequestFingerprint::new([id.len() as u8; 32]),
            state: TurnState::Completed,
            version: CheckpointVersion(1),
            request: CreateResponse {
                input: InputParam::Text(input.to_owned()),
                ..Default::default()
            },
            output_items: vec![message(Role::Assistant, output)],
            response: None,
        }
    }

    fn text(item: &InputItem) -> &str {
        let InputItem::EasyMessage(message) = item else {
            panic!("expected an easy message")
        };
        let EasyInputContent::Text(text) = &message.content else {
            panic!("expected text content")
        };
        text
    }

    #[test]
    fn standalone_request_has_distinct_checkpoint_and_inference_forms() {
        let request = CreateResponse {
            input: InputParam::Text("hello".to_owned()),
            store: Some(true),
            ..Default::default()
        };

        let turn = ResponsesRequestMaterializer::default()
            .materialize(request, &[])
            .unwrap();

        assert_eq!(
            turn.checkpoint_request.input,
            InputParam::Text("hello".to_owned())
        );
        assert_eq!(turn.checkpoint_request.store, Some(true));
        assert_eq!(turn.inference_request.store, Some(false));
        let InputParam::Items(items) = turn.inference_request.input else {
            panic!("inference input must be materialized")
        };
        assert_eq!(items.len(), 1);
        assert_eq!(text(&items[0]), "hello");
    }

    #[test]
    fn continuation_flattens_each_append_only_turn_once() {
        let chain = vec![
            record("resp-1", None, "one", "answer-one"),
            record("resp-2", Some("resp-1"), "two", "answer-two"),
        ];
        let request = CreateResponse {
            input: InputParam::Text("three".to_owned()),
            previous_response_id: Some("resp-2".to_owned()),
            store: Some(true),
            ..Default::default()
        };

        let turn = ResponsesRequestMaterializer::default()
            .materialize(request, &chain)
            .unwrap();

        assert_eq!(
            turn.checkpoint_request.input,
            InputParam::Text("three".to_owned())
        );
        assert_eq!(turn.checkpoint_request.previous_response_id, None);
        let InputParam::Items(items) = turn.inference_request.input else {
            panic!("inference input must be materialized")
        };
        assert_eq!(
            items.iter().map(text).collect::<Vec<_>>(),
            ["one", "answer-one", "two", "answer-two", "three"]
        );
    }

    #[test]
    fn inherits_tool_controls_but_not_instructions() {
        let mut parent = record("resp-1", None, "one", "answer-one");
        parent.request.instructions = Some("parent instructions".to_owned());
        parent.request.tools = Some(vec![Tool::LocalShell]);
        parent.request.tool_choice = Some(ToolChoiceParam::Mode(ToolChoiceOptions::Required));

        let inherited = ResponsesRequestMaterializer::default()
            .materialize(
                CreateResponse {
                    input: InputParam::Text("two".to_owned()),
                    previous_response_id: Some("resp-1".to_owned()),
                    ..Default::default()
                },
                std::slice::from_ref(&parent),
            )
            .unwrap();
        assert_eq!(inherited.checkpoint_request.instructions, None);
        assert_eq!(inherited.checkpoint_request.tools, parent.request.tools);
        assert_eq!(
            inherited.checkpoint_request.tool_choice,
            parent.request.tool_choice
        );

        let overridden = ResponsesRequestMaterializer::default()
            .materialize(
                CreateResponse {
                    input: InputParam::Text("two".to_owned()),
                    instructions: Some("new instructions".to_owned()),
                    tools: Some(Vec::new()),
                    tool_choice: Some(ToolChoiceParam::Mode(ToolChoiceOptions::None)),
                    previous_response_id: Some("resp-1".to_owned()),
                    ..Default::default()
                },
                &[parent],
            )
            .unwrap();
        assert_eq!(
            overridden.checkpoint_request.instructions.as_deref(),
            Some("new instructions")
        );
        assert_eq!(overridden.checkpoint_request.tools, Some(Vec::new()));
        assert_eq!(
            overridden.checkpoint_request.tool_choice,
            Some(ToolChoiceParam::Mode(ToolChoiceOptions::None))
        );
    }

    #[test]
    fn rejects_malformed_or_non_replayable_chains() {
        let request = CreateResponse {
            previous_response_id: Some("resp-2".to_owned()),
            ..Default::default()
        };
        let broken = vec![
            record("resp-1", None, "one", "answer-one"),
            record("resp-2", None, "two", "answer-two"),
        ];
        assert!(matches!(
            ResponsesRequestMaterializer::default().materialize(request.clone(), &broken),
            Err(MaterializationError::BrokenChain(id)) if id == ResponseId::from("resp-2")
        ));

        let mut in_flight = record("resp-2", None, "two", "answer-two");
        in_flight.state = TurnState::InFlight;
        assert!(matches!(
            ResponsesRequestMaterializer::default().materialize(request, &[in_flight]),
            Err(MaterializationError::NonReplayable { response_id, state: TurnState::InFlight })
                if response_id == ResponseId::from("resp-2")
        ));
    }

    #[test]
    fn rejects_cross_scope_chains() {
        let mut second = record("resp-2", Some("resp-1"), "two", "answer-two");
        second.scope = scope("principal-b");
        let request = CreateResponse {
            previous_response_id: Some("resp-2".to_owned()),
            ..Default::default()
        };

        assert!(matches!(
            ResponsesRequestMaterializer::default()
                .materialize(request, &[record("resp-1", None, "one", "answer-one"), second]),
            Err(MaterializationError::ScopeMismatch(id)) if id == ResponseId::from("resp-2")
        ));
    }
}
