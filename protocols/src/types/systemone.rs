// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Jev/System One evaluation wire types, OpenAPI 0.2.0.
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Text or structured JSON evidence; scalars remain valid inside objects and arrays.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionSystemOneStructuredText))]
pub enum StructuredText {
    String(String),
    Object(Map<String, Value>),
    Array(Vec<Value>),
}

impl From<StructuredText> for Value {
    fn from(text: StructuredText) -> Self {
        match text {
            StructuredText::String(text) => Self::String(text),
            StructuredText::Object(object) => Self::Object(object),
            StructuredText::Array(array) => Self::Array(array),
        }
    }
}

impl TryFrom<Value> for StructuredText {
    type Error = &'static str;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(text) => Ok(Self::String(text)),
            Value::Object(object) => Ok(Self::Object(object)),
            Value::Array(array) => Ok(Self::Array(array)),
            _ => Err("structured text must be a string, object, or array"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionSystemOneRequest))]
pub struct Request {
    pub model: String,
    pub state: StructuredText,
    pub questions: IndexMap<String, Question>,
    /// SGLang extension outside TypeSafe OpenAPI 0.2.0: images accompanying the
    /// state for vision-capable execution. Text-only profiles reject nonempty lists.
    #[serde(default)]
    pub images: Vec<Value>,
    /// SGLang extension outside TypeSafe OpenAPI 0.2.0: backend-validated chat
    /// template options, such as a reasoning toggle, honored during prompt rendering.
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionSystemOneQuestion))]
pub enum Question {
    Noul {
        #[serde(default)]
        instructions: Option<StructuredText>,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        #[serde(default)]
        instructions: Option<StructuredText>,
        #[cfg_attr(feature = "protocol-schema", schema(schema_with = choice_criteria_schema))]
        criteria: IndexMap<String, Option<StructuredText>>,
    },
    Score {
        #[serde(default)]
        instructions: Option<StructuredText>,
        criteria: Vec<StructuredText>,
    },
}

#[cfg(feature = "protocol-schema")]
fn choice_criteria_schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
    use utoipa::{
        ToSchema,
        openapi::{
            Ref,
            schema::{ObjectBuilder, OneOfBuilder, Schema, Type},
        },
    };

    ObjectBuilder::new()
        .additional_properties(Some(Schema::OneOf(
            OneOfBuilder::new()
                .item(Ref::from_schema_name(StructuredText::name()))
                .item(ObjectBuilder::new().schema_type(Type::Null))
                .build(),
        )))
        .into()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionSystemOneNoulCriteria))]
pub struct NoulCriteria {
    #[serde(default, rename = "true")]
    pub positive: Option<StructuredText>,
    #[serde(default, rename = "false")]
    pub negative: Option<StructuredText>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionSystemOneResponse))]
pub struct Response {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionSystemOneAnswer))]
pub enum Answer {
    Noul {
        noul: f64,
        /// SGLang extension outside TypeSafe OpenAPI 0.2.0: full-vocabulary
        /// answer-label probability mass, emitted only when the executor supplies it.
        #[serde(skip_serializing_if = "Option::is_none")]
        x_label_mass: Option<f64>,
    },
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        /// SGLang extension outside TypeSafe OpenAPI 0.2.0: full-vocabulary
        /// answer-label probability mass, emitted only when the executor supplies it.
        #[serde(skip_serializing_if = "Option::is_none")]
        x_label_mass: Option<f64>,
    },
    Score {
        score: f64,
        probabilities: IndexMap<String, f64>,
        legend: IndexMap<String, StructuredText>,
        confidence: f64,
        /// SGLang extension outside TypeSafe OpenAPI 0.2.0: full-vocabulary
        /// answer-label probability mass, emitted only when the executor supplies it.
        #[serde(skip_serializing_if = "Option::is_none")]
        x_label_mass: Option<f64>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "protocol-schema", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "protocol-schema", schema(as = DecisionSystemOneUsage))]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}
