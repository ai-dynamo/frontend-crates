// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::distribution::modal_index;
use crate::*;
use protocols::{openai, systemone};
use serde::Serialize;
use serde_json::Value;

/// Project in canonical request order, never in executor completion order.
/// The caller must pair each outcome with its original question ordinal.
pub fn project_response(
    request: &CanonicalRequest,
    canonical_model: &str,
    outcomes: &[QuestionOutcome],
    usage: &Usage,
) -> Result<Value, DecisionError> {
    let dialect = request.dialect;
    if outcomes.len() != request.questions.len() {
        return Err(DecisionError::execution(
            dialect,
            "executor outcome count differs from the request",
        ));
    }
    let total = validate_usage(usage, dialect)?;
    for (question, outcome) in request.questions.iter().zip(outcomes) {
        match outcome {
            QuestionOutcome::Answer(distribution) => {
                distribution
                    .validate(question.candidates.len())
                    .map_err(|error| DecisionError { dialect, ..error })?;
                if question.kind == QuestionKind::Predicate && distribution.probabilities.len() != 2
                {
                    return Err(DecisionError::execution(
                        dialect,
                        "predicate outcomes require exactly two candidates",
                    ));
                }
            }
            QuestionOutcome::Refusal(_) if dialect != Dialect::OpenAi => {
                return Err(DecisionError::new(
                    dialect,
                    422,
                    "decision_refused",
                    "the model refused a question that this response contract cannot represent",
                ));
            }
            QuestionOutcome::Refusal(_) => {}
        }
    }
    match dialect {
        Dialect::OpenAi => {
            let answers = request
                .questions
                .iter()
                .zip(outcomes)
                .map(|(q, outcome)| openai_answer(q, outcome))
                .collect::<Result<Vec<_>, _>>()?;
            let cached_tokens=usage.cached_tokens.ok_or_else(||DecisionError::execution(dialect,"OpenAI usage requires measured cache reads or a verified zero-cache-read profile"))?;
            serialize(
                openai::Response {
                    model: canonical_model.into(),
                    answers,
                    usage: openai::Usage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                        total_tokens: total,
                        input_tokens_details: openai::InputTokenDetails {
                            cached_tokens,
                            cache_write_tokens: 0,
                        },
                        output_tokens_details: openai::OutputTokenDetails {
                            reasoning_tokens: usage.reasoning_tokens,
                        },
                    },
                },
                dialect,
            )
        }
        Dialect::Jev => {
            let answers = request
                .questions
                .iter()
                .zip(outcomes)
                .map(|(q, o)| {
                    Ok((
                        question_id(q, dialect)?,
                        jev_answer(q, answered(o, dialect)?)?,
                    ))
                })
                .collect::<Result<_, DecisionError>>()?;
            serialize(
                systemone::Response {
                    model: canonical_model.into(),
                    answers,
                    usage: systemone::Usage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                    },
                },
                dialect,
            )
        }
    }
}

fn serialize(value: impl Serialize, dialect: Dialect) -> Result<Value, DecisionError> {
    serde_json::to_value(value)
        .map_err(|_| DecisionError::execution(dialect, "response serialization failed"))
}
fn question_id(q: &CanonicalQuestion, dialect: Dialect) -> Result<String, DecisionError> {
    q.id.clone().ok_or_else(|| {
        DecisionError::execution(dialect, "canonical response correlation ID is missing")
    })
}
fn answered(outcome: &QuestionOutcome, dialect: Dialect) -> Result<&Distribution, DecisionError> {
    match outcome {
        QuestionOutcome::Answer(d) => Ok(d),
        QuestionOutcome::Refusal(_) => Err(DecisionError::execution(
            dialect,
            "unexpected refusal projection",
        )),
    }
}
fn validate_usage(usage: &Usage, dialect: Dialect) -> Result<u64, DecisionError> {
    let total = usage
        .input_tokens
        .checked_add(usage.output_tokens)
        .ok_or_else(|| DecisionError::execution(dialect, "token usage overflow"))?;
    if usage
        .cached_tokens
        .is_some_and(|cached| cached > usage.input_tokens)
        || usage.reasoning_tokens > usage.output_tokens
    {
        return Err(DecisionError::execution(
            dialect,
            "executor token usage details exceed their totals",
        ));
    }
    if dialect == Dialect::OpenAi && total > i32::MAX as u64 {
        return Err(DecisionError::execution(
            dialect,
            "token usage exceeds the OpenAI wire range",
        ));
    }
    Ok(total)
}

fn openai_answer(
    q: &CanonicalQuestion,
    outcome: &QuestionOutcome,
) -> Result<openai::Answer, DecisionError> {
    let QuestionOutcome::Answer(d) = outcome else {
        return Ok(openai::Answer::Refusal { name: q.id.clone() });
    };
    let name = q.id.clone();
    Ok(match q.kind {
        QuestionKind::Predicate => openai::Answer::Predicate {
            name,
            probability: d.probabilities[0],
        },
        QuestionKind::Choice => openai::Answer::Choice {
            name,
            choice: q.candidates[modal_index(&d.probabilities)].value.clone(),
            confidence: choice_confidence(&d.probabilities)?,
            probabilities: q
                .candidates
                .iter()
                .zip(&d.probabilities)
                .map(|(c, p)| openai::ChoiceProbability {
                    value: c.value.clone(),
                    probability: *p,
                })
                .collect(),
        },
        QuestionKind::Score => openai::Answer::Score {
            name,
            score: expected_score(d),
            confidence: score_confidence(&d.probabilities)?,
            probabilities: q
                .candidates
                .iter()
                .zip(&d.probabilities)
                .enumerate()
                .map(|(i, (c, p))| {
                    let label = c.label.clone().ok_or_else(|| {
                        DecisionError::execution(
                            Dialect::OpenAi,
                            "score response requires original labels",
                        )
                    })?;
                    Ok(openai::ScoreProbability {
                        value: i,
                        label,
                        probability: *p,
                    })
                })
                .collect::<Result<Vec<_>, DecisionError>>()?,
        },
    })
}

fn string_value(candidate: &Candidate, dialect: Dialect) -> Result<String, DecisionError> {
    match &candidate.value {
        ChoiceValue::String(s) => Ok(s.clone()),
        ChoiceValue::Bool(_) => Err(DecisionError::execution(
            dialect,
            "string choice contract received a boolean candidate",
        )),
    }
}
fn expected_score(d: &Distribution) -> f64 {
    d.probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| i as f64 * p)
        .sum()
}

fn jev_answer(q: &CanonicalQuestion, d: &Distribution) -> Result<systemone::Answer, DecisionError> {
    let dialect = Dialect::Jev;
    Ok(match q.kind {
        QuestionKind::Predicate => systemone::Answer::Noul {
            noul: d.probabilities[0],
            x_label_mass: d.label_mass,
        },
        QuestionKind::Choice => systemone::Answer::Choice {
            choice: string_value(&q.candidates[modal_index(&d.probabilities)], dialect)?,
            confidence: choice_confidence(&d.probabilities)?,
            x_label_mass: d.label_mass,
            probabilities: q
                .candidates
                .iter()
                .zip(&d.probabilities)
                .map(|(c, p)| Ok((string_value(c, dialect)?, *p)))
                .collect::<Result<_, DecisionError>>()?,
        },
        QuestionKind::Score => systemone::Answer::Score {
            score: expected_score(d),
            confidence: score_confidence(&d.probabilities)?,
            x_label_mass: d.label_mass,
            probabilities: d
                .probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| (i.to_string(), *p))
                .collect(),
            legend: q
                .candidates
                .iter()
                .enumerate()
                .map(|(i, c)| (i.to_string(), c.description.clone().unwrap_or(Value::Null)))
                .collect(),
        },
    })
}
