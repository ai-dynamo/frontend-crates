// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::StructuralTagBuilder;
use super::templated_tool_calls::TemplatedToolCallFormat;
use crate::structural_tag::wire::JsonSchemaStyle;

pub(crate) static HARMONY: StructuralTagBuilder = StructuralTagBuilder::new(&FORMAT);

// The generation prompt already supplies `<|start|>assistant`. Subsequent
// directed messages explicitly reopen the assistant role after the handoff.
static FORMAT: TemplatedToolCallFormat = TemplatedToolCallFormat {
    tool_call_begin_prefix: "<|channel|>commentary to=functions.",
    tool_call_begin_suffix: "<|constrain|>json<|message|>",
    tool_call_end: "<|call|>",
    tool_call_trigger: "<|channel|>commentary to=functions.",
    first_tool_call_prefix: "",
    tool_call_separator: "<|start|>assistant",
    arguments_style: JsonSchemaStyle::Json,
    reasoning_begin: Some("<|channel|>analysis<|message|>"),
    reasoning_end: Some("<|end|>"),
    reasoning_suffix: "<|start|>assistant",
    free_text_excludes: &["to=functions."],
    tool_call_excludes: &["to=functions."],
    exclude_special_tokens: false,
};
