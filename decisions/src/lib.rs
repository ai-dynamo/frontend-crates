// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-independent decision API contracts and numerical reduction.

mod canonical;
mod distribution;
mod error;
mod openai_validation;
mod parse;
mod projection;
mod render;
mod strict_json;
mod text;

pub use canonical::*;
pub use distribution::{choice_confidence, reduce_vocab_logprobs, score_confidence};
pub use dynamo_protocols::types::decisions::{self as protocols, ChoiceValue};
pub use error::DecisionError;
pub use parse::{ParseOptions, parse_request, parse_request_with_options};
pub use projection::project_response;
pub use render::{RenderedQuestion, render_question_prompt};
