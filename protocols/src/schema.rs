// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-only placeholders for async-openai types that lack schema support.
//!
//! These placeholders let Dynamo-owned types derive `ToSchema` without
//! requiring `async-openai/protocol-schema`. Field-level `schema(value_type = ...)`
//! annotations select them for schema generation only; runtime field types and
//! Serde behavior remain unchanged.
//!
//! Each placeholder preserves the upstream type's identity through an
//! `x-dynamo-schema-import` marker, but does not describe its detailed shape.
//! Consumers must resolve these markers against version-aligned upstream
//! schemas before claiming complete coverage. This module does not perform
//! that resolution or merge specifications.
//!
//! Once native upstream schema support is integrated, the corresponding
//! placeholders and field overrides can be removed.

macro_rules! imports {
    ($($name:ident),* $(,)?) => {$(
        pub struct $name;

        impl utoipa::PartialSchema for $name {
            fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::Schema> {
                use utoipa::openapi::schema::{ObjectBuilder, SchemaType, Type};
                ObjectBuilder::new()
                    // These imported Rust types are non-Option. Excluding null
                    // prevents overlap with utoipa's Option<T> oneOf null arm.
                    .schema_type(SchemaType::Array(vec![
                        Type::Object, Type::Array, Type::String,
                        Type::Number, Type::Boolean,
                    ]))
                    .extensions(Some([(
                        "x-dynamo-schema-import",
                        serde_json::json!({"crate": "async-openai", "type": stringify!($name)}),
                    )].into_iter().collect()))
                    .into()
            }
        }

        impl utoipa::ToSchema for $name {
            fn name() -> std::borrow::Cow<'static, str> {
                concat!("async_openai.", stringify!($name)).into()
            }
        }
    )*};
}

imports!(
    FunctionType,
    ImageDetail,
    TopLogprobs,
    FunctionObject,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartAudio,
    ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageAudio,
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestFunctionMessage,
    Role,
    ChatCompletionResponseMessageAudio,
    ResponseModalities,
    PredictionContent,
    ChatCompletionAudio,
    ResponseFormat,
    ServiceTier,
    ChatCompletionFunctionCall,
    ChatCompletionFunctions,
    WebSearchOptions,
    FinishReason,
    CompletionUsage,
    Prompt,
);
