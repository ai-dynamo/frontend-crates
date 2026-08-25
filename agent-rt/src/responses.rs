// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::responses::{OutputItem, Response, Status};
use thiserror::Error;

use crate::{InterpretedOutput, OpenAiResponses, OutputIdentity, OutputInterpreter, TurnState};

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
    use dynamo_protocols::types::responses::{InputItem, Item, Response};

    use crate::{OutputIdentity, OutputInterpreter, ResponseId, TurnState};

    use super::{ResponsesOutputError, ResponsesOutputInterpreter};

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
}
