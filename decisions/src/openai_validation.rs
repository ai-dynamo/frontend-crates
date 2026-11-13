// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{DecisionError, Dialect, protocols::openai};

const FIELD_LIMIT: usize = 1_048_576;
const TEXT_LIMIT: usize = 10_485_760;

fn length(value: &str, limit: usize) -> Result<(), DecisionError> {
    if value.chars().take(limit + 1).count() > limit {
        return Err(DecisionError::validation(
            Dialect::OpenAi,
            "a request string exceeds its OpenAI schema length limit",
        ));
    }
    Ok(())
}
fn optional(value: Option<&str>) -> Result<(), DecisionError> {
    if let Some(value) = value {
        length(value, FIELD_LIMIT)?;
    }
    Ok(())
}

pub(crate) fn validate_lengths(request: &openai::Request) -> Result<(), DecisionError> {
    length(&request.model, FIELD_LIMIT)?;
    match &request.input {
        openai::Input::Text(text) => length(text, TEXT_LIMIT)?,
        openai::Input::Messages(messages) => {
            for message in messages {
                match &message.content {
                    openai::Content::Text(text) => length(text, TEXT_LIMIT)?,
                    openai::Content::Parts(parts) => {
                        for part in parts {
                            if let openai::InputPart::InputText { text } = part {
                                length(text, TEXT_LIMIT)?;
                            }
                        }
                    }
                }
            }
        }
    }
    for question in &request.questions {
        let (instructions, name) = match question {
            openai::Question::Predicate { instructions, name } => (instructions, name),
            openai::Question::Choice {
                instructions,
                name,
                choices,
            } => {
                for choice in choices {
                    optional(choice.description.as_deref())?;
                }
                (instructions, name)
            }
            openai::Question::Score {
                instructions,
                name,
                levels,
            } => {
                for level in levels {
                    length(&level.label, FIELD_LIMIT)?;
                    optional(level.description.as_deref())?;
                }
                (instructions, name)
            }
        };
        length(instructions, FIELD_LIMIT)?;
        optional(name.as_deref())?;
    }
    Ok(())
}
