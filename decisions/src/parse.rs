// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::text::is_blank;
use crate::*;
use protocols::{openai, systemone};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use std::collections::HashSet;

/// HTTP body limits remain transport-owned; these bound semantic fanout.
#[derive(Clone, Copy, Debug)]
pub struct ParseOptions {
    pub max_questions: usize,
}
impl Default for ParseOptions {
    fn default() -> Self {
        Self { max_questions: 128 }
    }
}

pub fn parse_request(bytes: &[u8], route: Route) -> Result<CanonicalRequest, DecisionError> {
    parse_request_with_options(bytes, route, ParseOptions::default())
}

pub fn parse_request_with_options(
    bytes: &[u8],
    route: Route,
    options: ParseOptions,
) -> Result<CanonicalRequest, DecisionError> {
    let dialect = if route == Route::SystemOne {
        Dialect::Jev
    } else {
        Dialect::OpenAi
    };
    let body = crate::strict_json::parse(bytes).map_err(|_| {
        DecisionError::new(
            dialect,
            400,
            "invalid_json",
            "body must be valid JSON with unique object keys",
        )
    })?;
    let object = body
        .as_object()
        .ok_or_else(|| DecisionError::validation(dialect, "body must be a JSON object"))?;
    if object.contains_key("nvext") {
        return Err(DecisionError::validation(
            dialect,
            "nvext is not supported by decision APIs",
        ));
    }
    let request = match dialect {
        Dialect::OpenAi => from_openai(decode(body, dialect)?)?,
        Dialect::Jev => from_jev(decode(body, dialect)?)?,
    };
    if request.model.trim().is_empty() {
        return Err(DecisionError::validation(
            dialect,
            "model must not be blank",
        ));
    }
    if request.questions.is_empty() || request.questions.len() > options.max_questions {
        return Err(DecisionError::validation(
            dialect,
            "question count exceeds the configured decision request limit",
        ));
    }
    validate_controls(&request)?;
    Ok(request)
}

fn decode<T: DeserializeOwned>(body: Value, dialect: Dialect) -> Result<T, DecisionError> {
    serde_json::from_value(body).map_err(|_| {
        DecisionError::validation(
            dialect,
            "request does not match the selected decision schema",
        )
    })
}

fn base(
    dialect: Dialect,
    model: String,
    input: Value,
    questions: Vec<CanonicalQuestion>,
) -> CanonicalRequest {
    CanonicalRequest {
        dialect,
        model,
        input,
        questions,
        temperature: 1.0,
        prompt_format_version: 1,
        chat_template_kwargs: Map::new(),
        safety_identifier: None,
    }
}
fn candidate(value: ChoiceValue, description: Option<Value>) -> Candidate {
    Candidate {
        value,
        description,
        label: None,
    }
}
fn predicate_candidates(positive: Option<Value>, negative: Option<Value>) -> Vec<Candidate> {
    vec![
        candidate(ChoiceValue::Bool(true), positive),
        candidate(ChoiceValue::Bool(false), negative),
    ]
}
fn cardinality(
    count: usize,
    min: usize,
    max: usize,
    dialect: Dialect,
) -> Result<(), DecisionError> {
    if !(min..=max).contains(&count) {
        return Err(DecisionError::validation(
            dialect,
            "candidate count is outside the selected schema's bounds",
        ));
    }
    Ok(())
}

fn from_openai(request: openai::Request) -> Result<CanonicalRequest, DecisionError> {
    let dialect = Dialect::OpenAi;
    crate::openai_validation::validate_lengths(&request)?;
    if request
        .safety_identifier
        .as_ref()
        .is_some_and(|s| s.chars().count() > 128)
    {
        return Err(DecisionError::validation(
            dialect,
            "safety_identifier exceeds its length limit",
        ));
    }
    validate_openai_input(&request.input)?;
    let input = serde_json::to_value(request.input)
        .map_err(|_| DecisionError::validation(dialect, "invalid input"))?;
    let questions = request
        .questions
        .into_iter()
        .enumerate()
        .map(|(ordinal, q)| {
            let (id, instructions, kind, candidates) = match q {
                openai::Question::Predicate { instructions, name } => (
                    name,
                    instructions,
                    QuestionKind::Predicate,
                    predicate_candidates(None, None),
                ),
                openai::Question::Choice {
                    instructions,
                    name,
                    choices,
                } => {
                    cardinality(choices.len(), 2, 255, dialect)?;
                    let mut seen = HashSet::new();
                    if choices.iter().any(|c| !seen.insert(&c.value)) {
                        return Err(DecisionError::validation(
                            dialect,
                            "choice values must be unique including their JSON type",
                        ));
                    }
                    (
                        name,
                        instructions,
                        QuestionKind::Choice,
                        choices
                            .into_iter()
                            .map(|c| candidate(c.value, c.description.map(Value::String)))
                            .collect(),
                    )
                }
                openai::Question::Score {
                    instructions,
                    name,
                    levels,
                } => {
                    cardinality(levels.len(), 2, 10, dialect)?;
                    let candidates = levels
                        .into_iter()
                        .enumerate()
                        .map(|(index, l)| Candidate {
                            value: ChoiceValue::String(index.to_string()),
                            description: l.description.map(Value::String),
                            label: Some(l.label),
                        })
                        .collect();
                    (name, instructions, QuestionKind::Score, candidates)
                }
            };
            Ok(CanonicalQuestion {
                ordinal,
                id,
                kind,
                instructions: Some(Value::String(instructions)),
                candidates,
            })
        })
        .collect::<Result<Vec<_>, DecisionError>>()?;
    Ok(CanonicalRequest {
        safety_identifier: request.safety_identifier,
        ..base(dialect, request.model, input, questions)
    })
}

fn validate_openai_input(input: &openai::Input) -> Result<(), DecisionError> {
    if let openai::Input::Messages(messages) = input {
        if messages.is_empty() {
            return Err(DecisionError::validation(
                Dialect::OpenAi,
                "input message list must not be empty",
            ));
        }
        for message in messages {
            if let openai::Content::Parts(parts) = &message.content {
                if parts
                    .iter()
                    .any(|p| matches!(p, openai::InputPart::InputImage { .. }))
                {
                    return Err(DecisionError::unsupported(
                        Dialect::OpenAi,
                        "this decision profile supports text input only",
                    ));
                }
                if parts.is_empty() {
                    return Err(DecisionError::validation(
                        Dialect::OpenAi,
                        "input content parts must not be empty",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn from_jev(request: systemone::Request) -> Result<CanonicalRequest, DecisionError> {
    let dialect = Dialect::Jev;
    text_value(&request.state, dialect)?;
    if !request.images.is_empty() {
        return Err(DecisionError::unsupported(
            dialect,
            "this decision profile supports text input only",
        ));
    }
    let questions = request
        .questions
        .into_iter()
        .enumerate()
        .map(|(ordinal, (id, q))| {
            let (instructions, kind, candidates) = match q {
                systemone::Question::Noul {
                    instructions,
                    criteria,
                } => {
                    let criteria = criteria.unwrap_or_default();
                    if [
                        instructions.as_ref(),
                        criteria.positive.as_ref(),
                        criteria.negative.as_ref(),
                    ]
                    .into_iter()
                    .all(|v| v.is_none_or(is_blank))
                    {
                        return Err(DecisionError::validation(
                            dialect,
                            "noul requires instructions or criteria",
                        ));
                    }
                    optional_text(criteria.positive.as_ref(), dialect)?;
                    optional_text(criteria.negative.as_ref(), dialect)?;
                    (
                        instructions,
                        QuestionKind::Predicate,
                        predicate_candidates(criteria.positive, criteria.negative),
                    )
                }
                systemone::Question::Choice {
                    instructions,
                    criteria,
                } => {
                    cardinality(criteria.len(), 1, 255, dialect)?;
                    let candidates = criteria
                        .into_iter()
                        .map(|(name, description)| {
                            optional_text(Some(&description), dialect)?;
                            Ok(candidate(ChoiceValue::String(name), Some(description)))
                        })
                        .collect::<Result<Vec<_>, DecisionError>>()?;
                    (instructions, QuestionKind::Choice, candidates)
                }
                systemone::Question::Score {
                    instructions,
                    criteria,
                } => {
                    cardinality(criteria.len(), 1, 10, dialect)?;
                    (
                        instructions,
                        QuestionKind::Score,
                        score_candidates(criteria, dialect)?,
                    )
                }
            };
            optional_text(instructions.as_ref(), dialect)?;
            Ok(CanonicalQuestion {
                ordinal,
                id: Some(id),
                kind,
                instructions,
                candidates,
            })
        })
        .collect::<Result<Vec<_>, DecisionError>>()?;
    Ok(CanonicalRequest {
        chat_template_kwargs: request.chat_template_kwargs,
        ..base(dialect, request.model, request.state, questions)
    })
}

fn score_candidates(levels: Vec<Value>, dialect: Dialect) -> Result<Vec<Candidate>, DecisionError> {
    levels
        .into_iter()
        .enumerate()
        .map(|(index, level)| {
            text_value(&level, dialect)?;
            Ok(candidate(
                ChoiceValue::String(index.to_string()),
                Some(level),
            ))
        })
        .collect()
}

fn text_value(value: &Value, dialect: Dialect) -> Result<(), DecisionError> {
    if !(value.is_string() || value.is_object() || value.is_array()) {
        return Err(DecisionError::validation(
            dialect,
            "text evidence must be a string, object, or array",
        ));
    }
    Ok(())
}
fn optional_text(value: Option<&Value>, dialect: Dialect) -> Result<(), DecisionError> {
    if let Some(value) = value.filter(|v| !v.is_null()) {
        text_value(value, dialect)?;
    }
    Ok(())
}

fn validate_controls(request: &CanonicalRequest) -> Result<(), DecisionError> {
    let dialect = request.dialect;
    if !request.temperature.is_finite() || request.temperature <= 0.0 {
        return Err(DecisionError::validation(
            dialect,
            "temperature must be finite and positive",
        ));
    }
    if request.prompt_format_version != 1 {
        return Err(DecisionError::unsupported(
            dialect,
            "unsupported decision prompt format version",
        ));
    }
    for (key, value) in &request.chat_template_kwargs {
        if !matches!(key.as_str(), "enable_thinking" | "thinking") || value != &Value::Bool(false) {
            return Err(DecisionError::unsupported(
                dialect,
                "only explicit disabled thinking controls are supported",
            ));
        }
    }
    Ok(())
}
