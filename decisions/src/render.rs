// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{CanonicalQuestion, ChoiceValue, DecisionError, Dialect, QuestionKind};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct RenderedQuestion {
    pub content: String,
    pub labels: Vec<String>,
}

/// Render prompt-format 1 semantic text. The caller applies the revision-bound
/// model chat template and verifies each candidate at the exact answer position.
/// Message-list input stays structured, so message and part boundaries survive.
pub fn render_question_prompt(
    input: &Value,
    question: &CanonicalQuestion,
) -> Result<RenderedQuestion, DecisionError> {
    let count = question.candidates.len();
    if count == 0 || count > 255 || (question.kind == QuestionKind::Score && count > 10) {
        return Err(DecisionError::unsupported(
            Dialect::OpenAi,
            "question exceeds prompt label capacity",
        ));
    }
    let instruction = question
        .instructions
        .as_ref()
        .filter(|v| !crate::text::is_blank(v))
        .map(render_text);
    let (lines, labels) = match question.kind {
        QuestionKind::Choice => {
            let labels = choice_labels(count);
            let mut lines = instruction
                .map(|i| vec![format!("Question: {i}")])
                .unwrap_or_default();
            let typed = question
                .candidates
                .iter()
                .any(|c| matches!(c.value, ChoiceValue::Bool(_)));
            for (c, label) in question.candidates.iter().zip(&labels) {
                let name = match &c.value {
                    ChoiceValue::String(s) if !typed => s.clone(),
                    ChoiceValue::String(s) => Value::String(s.clone()).to_string(),
                    ChoiceValue::Bool(b) => b.to_string(),
                };
                let description = c.description.as_ref().map(render_text).unwrap_or_default();
                lines.push(if description.is_empty() {
                    format!("{label}: {name}")
                } else {
                    format!("{label}: {name} - {description}")
                });
            }
            lines.push("Answer with the letter of one option only.".into());
            (lines, labels)
        }
        QuestionKind::Score => {
            let labels: Vec<_> = (0..count).map(|i| i.to_string()).collect();
            let mut lines = instruction
                .map(|i| vec![format!("Question: {i}")])
                .unwrap_or_default();
            for (c, label) in question.candidates.iter().zip(&labels) {
                let description = c.description.as_ref().map(render_text).unwrap_or_default();
                let level = match &c.label {
                    Some(name) if description.is_empty() => name.clone(),
                    Some(name) => format!("{name} - {description}"),
                    None => description,
                };
                lines.push(format!("{label}: {level}"));
            }
            lines.push("Answer with the number of one level only.".into());
            (lines, labels)
        }
        QuestionKind::Predicate => {
            if count != 2 {
                return Err(DecisionError::execution(
                    Dialect::OpenAi,
                    "predicate requires two candidates",
                ));
            }
            let mut lines = vec![
                instruction
                    .map(|i| format!("Is the following true? {i}"))
                    .unwrap_or_else(|| "Is the following true?".into()),
            ];
            for (label, c) in ["yes", "no"].into_iter().zip(&question.candidates) {
                if let Some(description) =
                    c.description.as_ref().filter(|v| !crate::text::is_blank(v))
                {
                    lines.push(format!("{label}: {}", render_text(description)));
                }
            }
            lines.push("Answer with yes or no only.".into());
            (lines, vec!["yes".into(), "no".into()])
        }
    };
    Ok(RenderedQuestion {
        content: std::iter::once(render_text(input))
            .chain(std::iter::once(String::new()))
            .chain(lines)
            .collect::<Vec<_>>()
            .join("\n"),
        labels,
    })
}

fn choice_labels(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            if count <= 26 {
                char::from(b'A' + i as u8).to_string()
            } else {
                format!(
                    "{}{}",
                    char::from(b'A' + (i / 26) as u8),
                    char::from(b'A' + (i % 26) as u8)
                )
            }
        })
        .collect()
}
fn render_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}
