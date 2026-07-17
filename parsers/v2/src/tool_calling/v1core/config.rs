// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendored copy of the v1 batch-parser configs the v2 streaming parsers need
//! (XML / GLM-4.7 / MiniMax-M3 / Kimi-K2). Copied verbatim from
//! `dynamo_parsers::tool_calling::config` so `parsers/v2` owns its extraction
//! stack and never links `dynamo_parsers`. v1 is slated for deletion; this stays.
//! See the "v1-v2-independent-no-shared-code" project rule.

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct XmlParserConfig {
    /// Start token for individual tool calls (e.g., "<tool_call>")
    pub tool_call_start_token: String,
    /// End token for individual tool calls (e.g., `</tool_call>`)
    pub tool_call_end_token: String,
    /// Start token for function name (e.g., `<function=`)
    pub function_start_token: String,
    /// End token for function (e.g., `</function>`)
    pub function_end_token: String,
    /// Start token for parameter (e.g., `<parameter=`)
    pub parameter_start_token: String,
    /// End token for parameter (e.g., `</parameter>`)
    pub parameter_end_token: String,

    /// See v1's `JsonParserConfig::allow_eof_recovery`. Streaming jails MUST
    /// leave this `false`.
    #[serde(default)]
    pub allow_eof_recovery: bool,

    /// When true, the function- and parameter-regex omit the `|$` end-of-block
    /// fallback (so missing `</function>` / `</parameter>` causes the match to
    /// fail rather than being silently recovered). Finalize recovery may still
    /// accept a missing outer wrapper end marker if the inner blocks are
    /// complete. Used by families whose official reference parser is
    /// strict-match (e.g. MiniMax-M2 — see
    /// https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/docs/tool_calling_guide.md).
    #[serde(default)]
    pub strict_match: bool,

    /// When true, if `function_start_token` is absent anywhere in the input,
    /// short-circuit `try_tool_call_parse_xml` and return
    /// `(calls=[], normal_text=<input>)`. Matches the early-return passthrough
    /// in Qwen3-Coder's official reference parser
    /// (https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/qwen3coder_tool_parser.py).
    #[serde(default)]
    pub passthrough_when_no_function: bool,

    /// When true, if `tool_call_start_token` is absent but `function_start_
    /// token` is present, parse the entire input as a single tool-call block
    /// (back-off strategy). Used by XML families that can recover a complete
    /// function/invoke body even when the outer wrapper opener is missing.
    /// Independent of `passthrough_when_no_function` (different trigger: this
    /// one fires when function tags exist but the outer wrapper does not).
    #[serde(default)]
    pub backoff_when_no_wrapper: bool,
}

impl Default for XmlParserConfig {
    fn default() -> Self {
        Self {
            tool_call_start_token: "<tool_call>".to_string(),
            tool_call_end_token: "</tool_call>".to_string(),
            function_start_token: "<function=".to_string(),
            function_end_token: "</function>".to_string(),
            parameter_start_token: "<parameter=".to_string(),
            parameter_end_token: "</parameter>".to_string(),
            allow_eof_recovery: false,
            strict_match: false,
            passthrough_when_no_function: false,
            backoff_when_no_wrapper: false,
        }
    }
}

impl XmlParserConfig {
    /// Returns true when the chunk lacks the outer `<tool_call>` wrapper but
    /// contains `<function=...>`, and the family opts into back-off parsing
    /// (qwen3_coder, nemotron_nano). In this mode the function-level tokens
    /// act as the tool-call boundary for both start detection and end-position
    /// search, mirroring the wrapped path's behavior so streaming and batch
    /// agree on what counts as a tool call.
    pub fn is_bare_function_mode(&self, chunk: &str) -> bool {
        self.backoff_when_no_wrapper
            && !chunk.contains(self.tool_call_start_token.as_str())
            && chunk.contains(self.function_start_token.as_str())
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Glm47ParserConfig {
    /// Start token for tool call block (e.g., "<tool_call>")
    pub tool_call_start: String,
    /// End token for tool call block (e.g., "</tool_call>")
    pub tool_call_end: String,
    /// Start token for argument key (e.g., "<arg_key>")
    pub arg_key_start: String,
    /// End token for argument key (e.g., "</arg_key>")
    pub arg_key_end: String,
    /// Start token for argument value (e.g., "<arg_value>")
    pub arg_value_start: String,
    /// End token for argument value (e.g., "</arg_value>")
    pub arg_value_end: String,

    /// See v1's `JsonParserConfig::allow_eof_recovery`. Streaming jails MUST
    /// leave this `false`.
    #[serde(default)]
    pub allow_eof_recovery: bool,
}

impl Default for Glm47ParserConfig {
    fn default() -> Self {
        Self {
            tool_call_start: "<tool_call>".to_string(),
            tool_call_end: "</tool_call>".to_string(),
            arg_key_start: "<arg_key>".to_string(),
            arg_key_end: "</arg_key>".to_string(),
            arg_value_start: "<arg_value>".to_string(),
            arg_value_end: "</arg_value>".to_string(),
            allow_eof_recovery: false,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MiniMaxM3ParserConfig {
    /// Namespace token emitted before each XML-ish tag.
    pub namespace_token: String,
    /// Tool-call block tag name.
    pub tool_call_tag: String,

    /// See v1's `JsonParserConfig::allow_eof_recovery`. Streaming jails MUST
    /// leave this `false`.
    #[serde(default)]
    pub allow_eof_recovery: bool,
}

impl Default for MiniMaxM3ParserConfig {
    fn default() -> Self {
        Self {
            namespace_token: "]<]minimax[>[".to_string(),
            tool_call_tag: "tool_call".to_string(),
            allow_eof_recovery: false,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KimiK2ParserConfig {
    /// Primary start token for the tool calls section
    pub section_start: String,
    /// Primary end token for the tool calls section
    pub section_end: String,
    /// All recognized start tokens for the tool calls section (includes singular variants)
    pub section_start_variants: Vec<String>,
    /// All recognized end tokens for the tool calls section (includes singular variants)
    pub section_end_variants: Vec<String>,
    /// Start token for an individual tool call (e.g., "<|tool_call_begin|>")
    pub call_start: String,
    /// End token for an individual tool call (e.g., "<|tool_call_end|>")
    pub call_end: String,
    /// Token separating function ID from JSON arguments (e.g., "<|tool_call_argument_begin|>")
    pub argument_begin: String,
}

impl Default for KimiK2ParserConfig {
    fn default() -> Self {
        Self {
            section_start: "<|tool_calls_section_begin|>".to_string(),
            section_end: "<|tool_calls_section_end|>".to_string(),
            section_start_variants: vec![
                "<|tool_calls_section_begin|>".to_string(),
                "<|tool_call_section_begin|>".to_string(),
            ],
            section_end_variants: vec![
                "<|tool_calls_section_end|>".to_string(),
                "<|tool_call_section_end|>".to_string(),
            ],
            call_start: "<|tool_call_begin|>".to_string(),
            call_end: "<|tool_call_end|>".to_string(),
            argument_begin: "<|tool_call_argument_begin|>".to_string(),
        }
    }
}
