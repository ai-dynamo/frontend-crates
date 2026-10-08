// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang native Decisions types, pinned to 0b635266d4a09f8db2d12bdcb793b085199faa8f.
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

fn default_model() -> String {
    "default".into()
}
fn default_temperature() -> f64 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSglangRequest))]
pub struct Request {
    #[serde(default = "default_model")]
    pub model: String,
    pub input: Value,
    pub questions: Vec<Question>,
    #[serde(default)]
    pub images: Vec<Value>,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub prompt_format_version: Option<u32>,
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
    #[serde(default)]
    pub return_prompt_token_ids: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSglangQuestion))]
pub enum Question {
    Choice {
        id: String,
        question: Value,
        options: Vec<ChoiceOption>,
    },
    Score {
        id: String,
        question: Value,
        levels: Vec<Value>,
    },
    YesNo {
        id: String,
        question: Value,
        #[serde(default)]
        yes: Option<Value>,
        #[serde(default)]
        no: Option<Value>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSglangChoiceOption))]
pub struct ChoiceOption {
    pub name: String,
    #[serde(default)]
    pub description: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSglangResponse))]
pub struct Response {
    pub object: String,
    pub model: String,
    pub prompt_format_version: u32,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSglangAnswer))]
pub struct Answer {
    #[serde(rename = "type")]
    pub kind: AnswerKind,
    pub probabilities: IndexMap<String, f64>,
    pub label_mass: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_token_ids: Option<Vec<u32>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSglangAnswerKind))]
pub enum AnswerKind {
    Choice,
    Score,
    YesNo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSglangUsage))]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub reasoning_tokens: u64,
}
