// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Decisions types, pinned to openai-python 4e152cdefe1844c2d5d78653310e9b9c0195c44e.
use serde::{Deserialize, Serialize};

pub use crate::types::decisions as openai;
pub use crate::types::systemone;

/// Choice values retain their JSON type; `true` differs from `"true"`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionSharedChoiceValue))]
pub enum ChoiceValue {
    String(String),
    Bool(bool),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiRequest))]
pub struct Request {
    pub model: String,
    pub input: Input,
    pub questions: Vec<Question>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiInput))]
pub enum Input {
    Text(String),
    Messages(Vec<InputMessage>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiInputMessage))]
pub struct InputMessage {
    pub role: UserRole,
    pub content: Content,
    #[serde(
        default,
        rename = "type",
        deserialize_with = "optional_nonnull",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "protocol-schema", schema(nullable = false))]
    pub kind: Option<MessageType>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiUserRole))]
pub enum UserRole {
    User,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiMessageType))]
pub enum MessageType {
    Message,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiImageDetail))]
pub enum ImageDetail {
    Low,
    High,
    Auto,
    Original,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiContent))]
pub enum Content {
    Text(String),
    Parts(Vec<InputPart>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiInputPart))]
pub enum InputPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(default)]
        detail: Option<ImageDetail>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiQuestion))]
pub enum Question {
    Predicate {
        instructions: String,
        #[serde(
            default,
            deserialize_with = "optional_nonnull",
            skip_serializing_if = "Option::is_none"
        )]
        #[cfg_attr(feature = "protocol-schema", schema(nullable = false))]
        name: Option<String>,
    },
    Choice {
        instructions: String,
        #[serde(
            default,
            deserialize_with = "optional_nonnull",
            skip_serializing_if = "Option::is_none"
        )]
        #[cfg_attr(feature = "protocol-schema", schema(nullable = false))]
        name: Option<String>,
        choices: Vec<ChoiceOption>,
    },
    Score {
        instructions: String,
        #[serde(
            default,
            deserialize_with = "optional_nonnull",
            skip_serializing_if = "Option::is_none"
        )]
        #[cfg_attr(feature = "protocol-schema", schema(nullable = false))]
        name: Option<String>,
        levels: Vec<ScoreLevel>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiChoiceOption))]
pub struct ChoiceOption {
    pub value: ChoiceValue,
    #[serde(
        default,
        deserialize_with = "optional_nonnull",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "protocol-schema", schema(nullable = false))]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiScoreLevel))]
pub struct ScoreLevel {
    pub label: String,
    #[serde(
        default,
        deserialize_with = "optional_nonnull",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "protocol-schema", schema(nullable = false))]
    pub description: Option<String>,
}

fn optional_nonnull<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiResponse))]
pub struct Response {
    pub model: String,
    pub answers: Vec<Answer>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiAnswer))]
pub enum Answer {
    Predicate {
        name: Option<String>,
        probability: f64,
    },
    Choice {
        name: Option<String>,
        choice: ChoiceValue,
        probabilities: Vec<ChoiceProbability>,
        confidence: f64,
    },
    Score {
        name: Option<String>,
        score: f64,
        probabilities: Vec<ScoreProbability>,
        confidence: f64,
    },
    Refusal {
        name: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiChoiceProbability))]
pub struct ChoiceProbability {
    pub value: ChoiceValue,
    pub probability: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiScoreProbability))]
pub struct ScoreProbability {
    pub value: usize,
    pub label: String,
    pub probability: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiUsage))]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub input_tokens_details: InputTokenDetails,
    pub output_tokens_details: OutputTokenDetails,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiInputTokenDetails))]
pub struct InputTokenDetails {
    pub cached_tokens: u64,
    pub cache_write_tokens: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionOpenAiOutputTokenDetails))]
pub struct OutputTokenDetails {
    pub reasoning_tokens: u64,
}
