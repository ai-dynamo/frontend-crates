// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::{
    anthropic::{
        AnthropicCreateMessageRequest, AnthropicMessage, AnthropicMessageResponse,
        AnthropicStreamEvent,
    },
    responses::{CreateResponse, InputItem, Response, ResponseStreamEvent},
};
use serde::{Serialize, de::DeserializeOwned};

/// Native protocol types used by one runtime instantiation.
///
/// This trait bundles concrete DTOs; it is not a universal request or message
/// representation. Materializers continue to operate on the native protocol.
pub trait AgentProtocol: std::fmt::Debug + Clone + Send + Sync + 'static {
    /// Stable discriminator used by shared durable stores. This value is part
    /// of the persisted schema and must not be renamed after deployment.
    const STORAGE_KEY: &'static str;

    type Request: std::fmt::Debug + Clone + Serialize + DeserializeOwned + Send + Sync + 'static;
    type ReplayItem: std::fmt::Debug + Clone + Serialize + DeserializeOwned + Send + Sync + 'static;
    type Response: std::fmt::Debug + Clone + Serialize + DeserializeOwned + Send + Sync + 'static;
    type StreamEvent: std::fmt::Debug + Send + 'static;
}

/// OpenAI Responses protocol family.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiResponses;

impl AgentProtocol for OpenAiResponses {
    const STORAGE_KEY: &'static str = "openai_responses_v1";

    type Request = CreateResponse;
    type ReplayItem = InputItem;
    type Response = Response;
    type StreamEvent = ResponseStreamEvent;
}

/// Anthropic Messages protocol family used by clients such as Claude Code.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicMessages;

impl AgentProtocol for AnthropicMessages {
    const STORAGE_KEY: &'static str = "anthropic_messages_v1";

    type Request = AnthropicCreateMessageRequest;
    type ReplayItem = AnthropicMessage;
    type Response = AnthropicMessageResponse;
    type StreamEvent = AnthropicStreamEvent;
}
