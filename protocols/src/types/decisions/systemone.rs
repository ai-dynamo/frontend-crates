// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Jev/System One evaluation wire types, OpenAPI 0.2.0.
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSystemOneRequest))]
pub struct Request {
    pub model: String,
    pub state: Value,
    pub questions: IndexMap<String, Question>,
    #[serde(default)]
    pub images: Vec<Value>,
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSystemOneQuestion))]
pub enum Question {
    Noul {
        #[serde(default)]
        instructions: Option<Value>,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: IndexMap<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: Vec<Value>,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSystemOneNoulCriteria))]
pub struct NoulCriteria {
    #[serde(default, rename = "true")]
    pub positive: Option<Value>,
    #[serde(default, rename = "false")]
    pub negative: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSystemOneResponse))]
pub struct Response {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSystemOneAnswer))]
pub enum Answer {
    Noul {
        noul: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        x_label_mass: Option<f64>,
    },
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        x_label_mass: Option<f64>,
    },
    Score {
        score: f64,
        probabilities: IndexMap<String, f64>,
        legend: IndexMap<String, Value>,
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        x_label_mass: Option<f64>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSystemOneUsage))]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}
