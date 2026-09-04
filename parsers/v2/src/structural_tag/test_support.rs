// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use crate::Tool;

use super::{
    StructuralTagBuilder, StructuralTagContext, StructuralTagSchemaMode, StructuralTagToolChoice,
};

pub(super) fn build_context(
    builder: &StructuralTagBuilder,
    context: &StructuralTagContext<'_>,
) -> Value {
    builder
        .build(context)
        .expect("structural tag should build")
        .expect("request should need a structural tag")
}

pub(super) fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "add_numbers".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {"a": {"type": "number"}},
                "required": ["a"]
            }),
            strict: Some(false),
        },
        Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
            strict: Some(true),
        },
    ]
}

pub(super) fn build(
    builder: &StructuralTagBuilder,
    tool_choice: StructuralTagToolChoice<'_>,
    tools: &[Tool],
    parallel_tool_calls: Option<bool>,
    schema_mode: StructuralTagSchemaMode,
    starts_in_reasoning: bool,
) -> Value {
    build_context(
        builder,
        &StructuralTagContext {
            tool_choice,
            tools,
            parallel_tool_calls,
            schema_mode,
            structured_output_schema: None,
            starts_in_reasoning,
        },
    )
}
