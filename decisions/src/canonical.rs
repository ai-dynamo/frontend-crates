// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ChoiceValue, DecisionError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    Decisions,
    SystemOne,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dialect {
    OpenAi,
    Jev,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuestionKind {
    Predicate,
    Choice,
    Score,
}

/// Ordered semantic request. External IDs are never model instructions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalRequest {
    pub dialect: Dialect,
    pub model: String,
    pub input: Value,
    pub questions: Vec<CanonicalQuestion>,
    pub temperature: f64,
    pub prompt_format_version: u32,
    pub chat_template_kwargs: Map<String, Value>,
    /// Opaque caller metadata, never an authenticated identity or cache partition.
    pub safety_identifier: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanonicalQuestion {
    pub ordinal: usize,
    pub id: Option<String>,
    pub kind: QuestionKind,
    pub instructions: Option<Value>,
    /// Predicates always list true first and false second.
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub value: ChoiceValue,
    pub description: Option<Value>,
    /// The original OpenAI score label, distinct from its ordinal value.
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub probabilities: Vec<f64>,
    /// Only genuine full-vocabulary-normalized evidence may populate this field.
    pub label_mass: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QuestionOutcome {
    Answer(Distribution),
    Refusal(String),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// None means unavailable, not zero cache reads.
    pub cached_tokens: Option<u64>,
    pub reasoning_tokens: u64,
}

/// Execution limits must originate in the selected, revision-bound deployment.
#[derive(Clone, Debug)]
pub struct Capabilities {
    pub max_questions: usize,
    pub max_candidates: usize,
    pub supports_predicate: bool,
    pub supports_choice: bool,
    pub supports_score: bool,
    pub measured_cache_reads: bool,
    pub prompt_format_version: u32,
}

impl CanonicalRequest {
    /// Call after model resolution, before rendering or dispatching GPU work.
    pub fn validate_capabilities(&self, capabilities: &Capabilities) -> Result<(), DecisionError> {
        let unsupported = self.questions.len() > capabilities.max_questions
            || self.prompt_format_version != capabilities.prompt_format_version
            || (self.dialect == Dialect::OpenAi && !capabilities.measured_cache_reads)
            || self.questions.iter().any(|question| {
                question.candidates.len() > capabilities.max_candidates
                    || !match question.kind {
                        QuestionKind::Predicate => capabilities.supports_predicate,
                        QuestionKind::Choice => capabilities.supports_choice,
                        QuestionKind::Score => capabilities.supports_score,
                    }
            });
        if unsupported {
            return Err(DecisionError::unsupported(
                self.dialect,
                "request exceeds the selected decision profile's capabilities",
            ));
        }
        Ok(())
    }
}
