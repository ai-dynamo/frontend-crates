// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
#[cfg(feature = "protocol-schema")]
use utoipa::{PartialSchema, ToSchema, openapi::RefOr};

use super::{ChatCompletionToolChoiceOption, CreateChatCompletionRequest, FunctionName};

/// Public Chat Completions input with support for `allowed_tools`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CreateChatCompletionInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionInputToolChoice>,
    #[serde(flatten)]
    request: CreateChatCompletionRequest,
}

impl CreateChatCompletionInput {
    /// Restore standard choices to the request and return any restriction for
    /// the serving layer to validate and apply before dispatch.
    pub fn into_parts(
        self,
    ) -> (
        CreateChatCompletionRequest,
        Option<ChatCompletionAllowedToolsSelection>,
    ) {
        let Self {
            tool_choice,
            mut request,
        } = self;
        match tool_choice {
            Some(ChatCompletionInputToolChoice::Allowed(
                ChatCompletionAllowedToolsChoice::AllowedTools { allowed_tools },
            )) => (request, Some(allowed_tools)),
            Some(ChatCompletionInputToolChoice::Standard(choice)) => {
                request.tool_choice = Some(choice);
                (request, None)
            }
            None => (request, None),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionInputToolChoice))]
pub enum ChatCompletionInputToolChoice {
    Standard(ChatCompletionToolChoiceOption),
    Allowed(ChatCompletionAllowedToolsChoice),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionAllowedToolsChoice))]
pub enum ChatCompletionAllowedToolsChoice {
    AllowedTools {
        allowed_tools: ChatCompletionAllowedToolsSelection,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionAllowedToolsSelection))]
pub struct ChatCompletionAllowedToolsSelection {
    pub mode: ChatCompletionAllowedToolsMode,
    pub tools: Vec<ChatCompletionAllowedTool>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionAllowedToolsMode))]
pub enum ChatCompletionAllowedToolsMode {
    Auto,
    Required,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = dynamo_protocols::chat::ChatCompletionAllowedTool))]
pub enum ChatCompletionAllowedTool {
    Function { function: FunctionName },
}

#[cfg(feature = "protocol-schema")]
impl PartialSchema for CreateChatCompletionInput {
    fn schema() -> RefOr<utoipa::openapi::Schema> {
        use utoipa::openapi::schema::{AnyOfBuilder, ObjectBuilder, Ref, Schema, Type};

        // The flattened request supplies the common fields. Replace its
        // tool_choice property because an intersection would keep the old restriction.
        let RefOr::T(Schema::Object(mut request)) = CreateChatCompletionRequest::schema() else {
            unreachable!("CreateChatCompletionRequest has an object schema");
        };
        request.properties.insert(
            "tool_choice".into(),
            AnyOfBuilder::new()
                .item(Ref::from_schema_name(ChatCompletionInputToolChoice::name()))
                .item(ObjectBuilder::new().schema_type(Type::Null))
                .into(),
        );
        Schema::Object(request).into()
    }
}

#[cfg(feature = "protocol-schema")]
impl ToSchema for CreateChatCompletionInput {
    fn name() -> std::borrow::Cow<'static, str> {
        "dynamo_protocols.chat.CreateChatCompletionInput".into()
    }

    fn schemas(schemas: &mut Vec<(String, RefOr<utoipa::openapi::Schema>)>) {
        CreateChatCompletionRequest::schemas(schemas);
        schemas.push((
            ChatCompletionInputToolChoice::name().into_owned(),
            ChatCompletionInputToolChoice::schema(),
        ));
        ChatCompletionInputToolChoice::schemas(schemas);
    }
}
