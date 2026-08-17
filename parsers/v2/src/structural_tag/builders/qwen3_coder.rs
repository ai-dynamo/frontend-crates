// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::StructuralTagBuilder;
use super::templated_tool_calls::TemplatedToolCallFormat;
use crate::structural_tag::wire::JsonSchemaStyle;

pub(crate) static QWEN3_CODER: StructuralTagBuilder =
    StructuralTagBuilder::new(&QWEN3_CODER_FORMAT);

static QWEN3_CODER_FORMAT: TemplatedToolCallFormat = TemplatedToolCallFormat {
    tool_call_begin_prefix: "<tool_call>\n<function=",
    tool_call_begin_suffix: ">\n",
    tool_call_end: "\n</function>\n</tool_call>",
    tool_call_trigger: "<tool_call>\n<function=",
    first_tool_call_prefix: "\n\n",
    tool_call_separator: "\n",
    arguments_style: JsonSchemaStyle::QwenXml,
    reasoning_begin: Some("<think>"),
    reasoning_end: Some("</think>"),
    tool_call_excludes: &["<tool_call>", "<function="],
    exclude_special_tokens: true,
};
