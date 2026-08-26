// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::responses::CreateResponse;

use crate::{AgentProtocol, AnthropicMessages, OpenAiResponses};

/// Trusted frontend policy inputs that are not inferred from client payloads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeSelectionContext {
    /// At least one tool was classified as runtime-owned by server policy.
    pub has_runtime_owned_tools: bool,
    /// The deployment requires a durable checkpoint for this request.
    pub requires_durable_state: bool,
}

/// Frontend policy deciding whether a native request needs agent orchestration.
pub trait RuntimeSelector<P>: Send + Sync + 'static
where
    P: AgentProtocol,
{
    fn requires_runtime(&self, request: &P::Request, context: RuntimeSelectionContext) -> bool;
}

/// Selects stateful Responses requests while preserving stateless passthrough.
#[derive(Debug, Clone, Copy, Default)]
pub struct StatefulRequestSelector;

impl RuntimeSelector<OpenAiResponses> for StatefulRequestSelector {
    fn requires_runtime(&self, request: &CreateResponse, context: RuntimeSelectionContext) -> bool {
        request.store == Some(true)
            || request.previous_response_id.is_some()
            || context.has_runtime_owned_tools
            || context.requires_durable_state
    }
}

/// Selects Anthropic Messages only when trusted frontend policy needs runtime
/// ownership. Claude Code's ordinary client-tool loop remains passthrough.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicRequestSelector;

impl RuntimeSelector<AnthropicMessages> for AnthropicRequestSelector {
    fn requires_runtime(
        &self,
        _request: &<AnthropicMessages as AgentProtocol>::Request,
        context: RuntimeSelectionContext,
    ) -> bool {
        context.has_runtime_owned_tools || context.requires_durable_state
    }
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::{
        anthropic::AnthropicCreateMessageRequest, responses::CreateResponse,
    };

    use super::{
        AnthropicRequestSelector, RuntimeSelectionContext, RuntimeSelector, StatefulRequestSelector,
    };

    #[test]
    fn responses_selector_uses_payload_and_trusted_policy() {
        let selector = StatefulRequestSelector;
        assert!(!selector.requires_runtime(
            &CreateResponse::default(),
            RuntimeSelectionContext::default()
        ));

        let stored = CreateResponse {
            store: Some(true),
            ..Default::default()
        };
        assert!(selector.requires_runtime(&stored, RuntimeSelectionContext::default()));

        let continuation = CreateResponse {
            previous_response_id: Some("resp_parent".to_owned()),
            ..Default::default()
        };
        assert!(selector.requires_runtime(&continuation, RuntimeSelectionContext::default()));

        assert!(selector.requires_runtime(
            &CreateResponse::default(),
            RuntimeSelectionContext {
                has_runtime_owned_tools: true,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn anthropic_client_loop_is_passthrough_unless_policy_selects_runtime() {
        let selector = AnthropicRequestSelector;
        let request: AnthropicCreateMessageRequest = serde_json::from_value(serde_json::json!({
            "model": "claude",
            "max_tokens": 1024,
            "messages": []
        }))
        .unwrap();

        assert!(!selector.requires_runtime(&request, RuntimeSelectionContext::default()));
        assert!(selector.requires_runtime(
            &request,
            RuntimeSelectionContext {
                has_runtime_owned_tools: true,
                ..Default::default()
            }
        ));
    }
}
