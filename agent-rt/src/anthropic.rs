// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::anthropic::AnthropicCreateMessageRequest;
use thiserror::Error;

use crate::{AnthropicMessages, CheckpointRecord, MaterializedTurn, RequestMaterializer};

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

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::anthropic::AnthropicCreateMessageRequest;

    use crate::{
        AgentProtocol, AnthropicMessages, AuthorizationScope, CheckpointRecord, CheckpointVersion,
        IdempotencyKey, RequestFingerprint, ResponseId, TurnState,
    };

    use super::{AnthropicMaterializationError, AnthropicRequestMaterializer, RequestMaterializer};

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
        };

        assert_eq!(
            AnthropicRequestMaterializer
                .materialize(request(), &[record])
                .unwrap_err(),
            AnthropicMaterializationError::ExternalContinuationUnsupported
        );
    }
}
