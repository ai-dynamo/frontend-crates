// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::collections::HashMap;
use std::sync::OnceLock;

mod base_parser;
mod gemma4_parser;
mod gpt_oss_parser;
mod granite_parser;
mod inkling_parser;
mod minimax_append_think_parser;

// Re-export main types and functions for convenience
pub(crate) use crate::tool_calling::config::MINIMAX_M3_TOOL_NAMESPACE;
pub use base_parser::BasicReasoningParser;
pub use gemma4_parser::Gemma4ReasoningParser;
pub use gpt_oss_parser::{GptOssReasoningParser, harmony_terminator_token_ids};
pub use granite_parser::GraniteReasoningParser;
pub use inkling_parser::InklingReasoningParser;
pub use minimax_append_think_parser::MiniMaxAppendThinkParser;

/// Kimi-K2/K2.5 tool-call section marker. Shared between the `kimi_k25` reasoning-parser
/// registration and its test fixtures so both stay in sync. Mirrors
/// `KimiK2ParserConfig::default().section_start` in `crate::tool_calling::config`.
pub(crate) const KIMI_K2_TOOL_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
pub(crate) const DEEPSEEK_TOOL_BLOCK_BEGIN: &str = "<｜DSML｜tool_calls>";
pub(crate) const DEEPSEEK_TOOL_INVOKE_BEGIN: &str = "<｜DSML｜invoke name=";

static REASONING_PARSER_MAP: OnceLock<HashMap<&'static str, ReasoningParserType>> = OnceLock::new();

/// Initialize the global reasoning parser map
fn get_reasoning_parser_map() -> &'static HashMap<&'static str, ReasoningParserType> {
    REASONING_PARSER_MAP.get_or_init(|| {
        let mut map = HashMap::new();
        map.insert("deepseek_r1", ReasoningParserType::DeepseekR1);
        // DeepSeek V3.x thinking mode uses the same output shape as R1:
        // reasoning text is already in progress and the completion emits
        // `</think>` before the final answer. The V3.x tool-call parsers are
        // distinct because their tool-call wire formats differ, but reasoning
        // shares this forced think-tag parser.
        map.insert("deepseek_v3", ReasoningParserType::DeepseekR1);
        map.insert("deepseek_v3_1", ReasoningParserType::DeepseekR1);
        map.insert("deepseek_v3_2", ReasoningParserType::DeepseekR1);
        map.insert("basic", ReasoningParserType::Basic);
        map.insert("gpt_oss", ReasoningParserType::GptOss);
        map.insert("qwen3", ReasoningParserType::Qwen);
        // DeepSeek-V4 uses the same `<think>` / `</think>` delimiters as Qwen
        // (confirmed against deepseek-ai/DeepSeek-V4-Pro's encoding_dsv4.py)
        // so it delegates to the same `BasicReasoningParser` config today. We
        // still route through a dedicated `DeepSeekV4` variant rather than
        // hard-aliasing to `Qwen` so future divergence (different special
        // tokens, max-thinking mode, etc.) has a place to land without rippling
        // through Qwen's own config.
        //
        // The three name aliases exist because callers set this via
        // `--dyn-reasoning-parser` / `--reasoning-parser` with whatever string
        // the HF model / vLLM recipe / chat-template author picked. We accept
        // all three separator conventions (snake / kebab / concat) rather than
        // force a single canonical form on users.
        map.insert("deepseek_v4", ReasoningParserType::DeepSeekV4);
        map.insert("deepseek-v4", ReasoningParserType::DeepSeekV4);
        map.insert("deepseekv4", ReasoningParserType::DeepSeekV4);
        map.insert("nemotron_deci", ReasoningParserType::NemotronDeci);
        map.insert("kimi", ReasoningParserType::Kimi);
        map.insert("kimi_k25", ReasoningParserType::KimiK25);
        // Kimi K3 uses XTML channel markers. The generation prompt normally
        // consumes the opening `think` channel, so callers should pair this
        // parser with `set_in_reasoning(true)` when thinking is enabled.
        map.insert("kimi_k3", ReasoningParserType::KimiK3);
        map.insert("kimi-k3", ReasoningParserType::KimiK3);
        map.insert("step3", ReasoningParserType::Step3);
        map.insert("mistral", ReasoningParserType::Mistral);
        map.insert("granite", ReasoningParserType::Granite);
        map.insert("nemotron_nano", ReasoningParserType::DeepseekR1); // nemotron nano is ...</think>
        map.insert("nemotron3", ReasoningParserType::DeepseekR1);
        map.insert("nemotron_v3", ReasoningParserType::DeepseekR1);
        map.insert("glm45", ReasoningParserType::NemotronDeci); // GLM-4.5/5 is <think>...</think>, no force_reasoning
        map.insert(
            "minimax_append_think",
            ReasoningParserType::MiniMaxAppendThink,
        );
        map.insert("minimax_m2", ReasoningParserType::MiniMaxM2);
        // MiniMax M3 thinking blocks use `<mm:think>...</mm:think>`. The chat
        // template pre-fills the opener, so the completion typically emits only
        // the closing marker; dangling-end recovery handles that.
        map.insert("minimax_m3", ReasoningParserType::MiniMaxM3);
        map.insert("minimax-m3", ReasoningParserType::MiniMaxM3);
        // Gemma 4 thinking models: reasoning is wrapped in `<|channel>...<channel|>`
        // with a `thought\n` role label that this parser strips. Pair with
        // `--dyn-tool-call-parser gemma4` for end-to-end Gemma 4 support.
        map.insert("gemma4", ReasoningParserType::Gemma4);
        map.insert("gemma-4", ReasoningParserType::Gemma4);
        // Block-structured, not a `<think>` prefix, so not a BasicReasoningParser.
        map.insert("inkling", ReasoningParserType::Inkling);
        map
    })
}

/// Get all available reasoning parser names
pub fn get_available_reasoning_parsers() -> Vec<&'static str> {
    get_reasoning_parser_map().keys().copied().collect()
}

#[derive(Debug, Clone, Default)]
pub struct ParserResult {
    /// The normal text outside of reasoning blocks.
    pub normal_text: String,

    /// The extracted reasoning text from within reasoning blocks.
    pub reasoning_text: String,
}

impl ParserResult {
    pub fn get_some_reasoning(&self) -> Option<String> {
        if self.reasoning_text.is_empty() {
            None
        } else {
            Some(self.reasoning_text.clone())
        }
    }

    pub fn get_some_normal_text(&self) -> Option<String> {
        if self.normal_text.is_empty() {
            None
        } else {
            Some(self.normal_text.clone())
        }
    }
}

pub trait ReasoningParser: Send + std::fmt::Debug {
    /// Parses a standalone, non-streaming input chunk. Implementations may reset or ignore
    /// internal streaming state and should return the split of normal vs reasoning text for
    /// this complete input. Marker tokens must not be included in either output.
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult;

    /// Parses a streaming chunk and updates internal state. The return value should be the
    /// delta: only the newly discovered normal and reasoning text attributable to this chunk
    /// (not the cumulative totals). Marker tokens must not be included in either output.
    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult;

    /// Finalizes a stream after the last chunk, before parser state is dropped.
    ///
    /// Incremental parsing may buffer a partial delimiter prefix instead of
    /// emitting it immediately because the next chunk could complete a marker
    /// like `<think>` or `</think>`. At EOF, no next chunk is coming, so the
    /// parser must flush the undecided bytes as normal or reasoning text based
    /// on its current state.
    fn finish_reasoning_stream(&mut self) -> ParserResult {
        ParserResult::default()
    }

    /// Override the parser's initial reasoning state. When called with `true`, the parser
    /// starts in reasoning mode without waiting for the start token in the completion stream.
    /// Use this when the chat template already injected the start token (e.g., `<think>`)
    /// into the prompt, so it won't appear in the model's output.
    fn set_in_reasoning(&mut self, _in_reasoning: bool) {
        // Default no-op for parsers that don't support per-request overrides.
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningParserType {
    DeepseekR1,
    Step3,
    Basic,
    GptOss,
    Qwen,
    /// DeepSeek-V4-Pro / V4-Flash. Currently uses the same `<think>` /
    /// `</think>` `BasicReasoningParser` config as Qwen (V4 never appends
    /// `<think>` in the completion — the chat template always pre-injects it,
    /// so the parser starts via `set_in_reasoning(true)` rather than
    /// `force_reasoning`). A dedicated variant keeps future V4-specific
    /// divergence (different delimiters, thinking-effort modes) from leaking
    /// into Qwen's behavior.
    DeepSeekV4,
    NemotronDeci,
    Kimi,
    KimiK25,
    /// Kimi K3 XTML reasoning channel:
    /// `<|open|>think<|sep|>...<|close|>think<|sep|>`.
    KimiK3,
    Mistral,
    Granite,
    MiniMaxAppendThink,
    /// MiniMax M2 emits reasoning from token one and may transition directly
    /// into its XML tool-call envelope without a closing reasoning marker.
    MiniMaxM2,
    /// MiniMax M3 thinking models. `<mm:think>...</mm:think>` delimiters with
    /// streaming dangling-end recovery (the chat template pre-fills the opener).
    MiniMaxM3,
    /// Google Gemma 4 thinking models. Custom `<|channel>...<channel|>`
    /// delimiters with a `thought\n` role-label prefix stripped by the parser.
    Gemma4,
    /// Inkling (thinkingmachines/Inkling-NVFP4): block-structured reasoning, tool-call
    /// blocks passed through verbatim. See [`crate::reasoning::inkling_parser`].
    Inkling,
}

#[derive(std::fmt::Debug)]
pub struct ReasoningParserWrapper {
    parser: Box<dyn ReasoningParser>,
}

impl ReasoningParser for ReasoningParserWrapper {
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        self.parser.detect_and_parse_reasoning(text, token_ids)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        self.parser
            .parse_reasoning_streaming_incremental(text, token_ids)
    }

    fn finish_reasoning_stream(&mut self) -> ParserResult {
        self.parser.finish_reasoning_stream()
    }

    fn set_in_reasoning(&mut self, in_reasoning: bool) {
        self.parser.set_in_reasoning(in_reasoning)
    }
}

impl ReasoningParserType {
    pub fn get_reasoning_parser(self) -> ReasoningParserWrapper {
        let basic_parser =
            BasicReasoningParser::new("<think>".into(), "</think>".into(), false, true);
        let force_reasoning_basic_parser =
            BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true);
        match self {
            ReasoningParserType::DeepseekR1 => ReasoningParserWrapper {
                parser: Box::new(force_reasoning_basic_parser),
            },
            ReasoningParserType::Step3 => ReasoningParserWrapper {
                parser: Box::new(force_reasoning_basic_parser),
            },
            ReasoningParserType::Basic => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::Qwen => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            // Same `<think>` / `</think>` config as Qwen today; kept as a
            // distinct variant so V4-specific divergence has somewhere to land.
            // See `ReasoningParserType::DeepSeekV4` docstring for rationale.
            ReasoningParserType::DeepSeekV4 => ReasoningParserWrapper {
                parser: Box::new(
                    BasicReasoningParser::new("<think>".into(), "</think>".into(), false, true)
                        .with_tool_start_token(DEEPSEEK_TOOL_BLOCK_BEGIN)
                        .with_tool_start_token(DEEPSEEK_TOOL_INVOKE_BEGIN),
                ),
            },
            ReasoningParserType::NemotronDeci => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::Kimi => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "◁think▷".into(),
                    "◁/think▷".into(),
                    false,
                    true,
                )),
            },
            ReasoningParserType::KimiK25 => ReasoningParserWrapper {
                parser: Box::new(
                    BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true)
                        .with_tool_start_token(KIMI_K2_TOOL_SECTION_BEGIN),
                ),
            },
            ReasoningParserType::KimiK3 => {
                let mut parser = BasicReasoningParser::new(
                    "<|open|>think<|sep|>".into(),
                    "<|close|>think<|sep|>".into(),
                    false,
                    true,
                );
                // All K3 structural channels begin with one of these reserved
                // control tokens. Three prefix probes cover canonical and
                // engine-spaced forms without scanning the reasoning text once
                // for every response/tools/call/argument marker.
                parser = parser
                    .with_tool_start_token("<|open|>")
                    .with_tool_start_token("<|close|>")
                    .with_tool_start_token("<|end_of_msg|>");
                ReasoningParserWrapper {
                    parser: Box::new(
                        parser
                            // K3 markers all begin with the unambiguous `<|`
                            // prefix, so retain a lone trailing `<` until the
                            // next backend chunk completes or disproves it.
                            .with_single_char_marker_buffering()
                            // The prompt normally consumes the think opener.
                            .with_dangling_end_recovery(),
                    ),
                }
            }
            ReasoningParserType::Mistral => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "[THINK]".into(),
                    "[/THINK]".into(),
                    true,
                    true,
                )),
            },
            ReasoningParserType::GptOss => match GptOssReasoningParser::new() {
                Ok(parser) => ReasoningParserWrapper {
                    parser: Box::new(parser),
                },
                Err(e) => {
                    tracing::warn!(
                        "GptOssReasoningParser could not be initialized, falling back to Basic Reasoning Parser: {e}"
                    );
                    ReasoningParserWrapper {
                        parser: Box::new(BasicReasoningParser::new(
                            "<think>".into(),
                            "</think>".into(),
                            false,
                            true,
                        )),
                    }
                }
            },
            ReasoningParserType::Granite => ReasoningParserWrapper {
                parser: Box::new(GraniteReasoningParser::new()),
            },
            ReasoningParserType::MiniMaxAppendThink => ReasoningParserWrapper {
                parser: Box::new(MiniMaxAppendThinkParser::new()),
            },
            ReasoningParserType::MiniMaxM2 => ReasoningParserWrapper {
                parser: Box::new(
                    BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true)
                        .with_tool_start_token("<minimax:tool_call>")
                        .with_tool_start_token("<invoke name="),
                ),
            },
            ReasoningParserType::MiniMaxM3 => ReasoningParserWrapper {
                parser: Box::new(
                    BasicReasoningParser::new(
                        "<mm:think>".into(),
                        "</mm:think>".into(),
                        false,
                        true,
                    )
                    .with_dangling_end_recovery()
                    .with_implicit_tool_start_recovery()
                    // M3 can begin a native tool call without `</mm:think>`.
                    .with_tool_start_token(MINIMAX_M3_TOOL_NAMESPACE),
                ),
            },
            ReasoningParserType::Gemma4 => ReasoningParserWrapper {
                parser: Box::new(Gemma4ReasoningParser::new()),
            },
            ReasoningParserType::Inkling => ReasoningParserWrapper {
                parser: Box::new(InklingReasoningParser::new()),
            },
        }
    }

    pub fn get_reasoning_parser_from_name(name: &str) -> ReasoningParserWrapper {
        tracing::debug!("Selected reasoning parser: {}", name);

        let parser_map = get_reasoning_parser_map();
        let normalized_name = name.to_lowercase();

        match parser_map.get(normalized_name.as_str()) {
            Some(parser_type) => parser_type.get_reasoning_parser(),
            None => {
                tracing::warn!(
                    parser_name = name,
                    "Unknown reasoning parser type, falling back to Basic Reasoning Parser",
                );
                Self::Basic.get_reasoning_parser()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] // registry helper
    fn test_get_available_reasoning_parsers() {
        let parsers = get_available_reasoning_parsers();
        assert!(!parsers.is_empty());
        // Update this list when adding a new parser
        let available_parsers = [
            "deepseek_r1",
            "deepseek_v3",
            "deepseek_v3_1",
            "deepseek_v3_2",
            "basic",
            "gpt_oss",
            "qwen3",
            "deepseek_v4",
            "deepseek-v4",
            "deepseekv4",
            "nemotron_deci",
            "kimi",
            "kimi_k25",
            "kimi_k3",
            "kimi-k3",
            "step3",
            "mistral",
            "granite",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
            "glm45",
            "minimax_append_think",
            "minimax_m2",
            "minimax_m3",
            "minimax-m3",
            "gemma4",
            "gemma-4",
            "inkling",
        ];
        for parser in available_parsers {
            assert!(parsers.contains(&parser));
        }
    }

    #[test]
    fn test_deepseek_v4_detect_and_parse_exits_reasoning_on_dsml_tool_start() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        let result = parser.detect_and_parse_reasoning(
            &format!(
                "<think>need a tool{DEEPSEEK_TOOL_BLOCK_BEGIN}\n<｜DSML｜invoke name=\"bash\">"
            ),
            &[],
        );

        assert_eq!(result.reasoning_text, "need a tool");
        assert_eq!(
            result.normal_text,
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">"
        );
    }

    #[test]
    fn test_deepseek_v4_detect_and_parse_exits_reasoning_without_visible_opener_on_dsml_tool_start()
    {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        parser.set_in_reasoning(true);

        let result = parser.detect_and_parse_reasoning(
            &format!(
                "What is the weather now?{DEEPSEEK_TOOL_BLOCK_BEGIN}\n<｜DSML｜invoke name=\"get_weather\">"
            ),
            &[],
        );

        assert_eq!(result.reasoning_text, "What is the weather now?");
        assert_eq!(
            result.normal_text,
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">"
        );
    }

    #[test]
    fn test_deepseek_v4_detect_and_parse_exits_reasoning_on_bare_dsml_invoke() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        let tool_call = format!(
            "{DEEPSEEK_TOOL_INVOKE_BEGIN}\"bash\">\n<｜DSML｜parameter name=\"cmd\" string=\"true\">pwd</｜DSML｜parameter>\n</｜DSML｜invoke>"
        );
        let result =
            parser.detect_and_parse_reasoning(&format!("<think>need a tool{tool_call}"), &[]);

        assert_eq!(result.reasoning_text, "need a tool");
        assert_eq!(result.normal_text, tool_call);
    }

    #[test]
    fn test_deepseek_v4_detect_and_parse_keeps_partial_dsml_tool_prefix_at_eof() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        let result = parser.detect_and_parse_reasoning("<think>need a tool<｜DSML｜tool_", &[]);

        assert_eq!(result.reasoning_text, "need a tool<｜DSML｜tool_");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_deepseek_v4_streaming_exits_reasoning_on_split_dsml_tool_start() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        parser.set_in_reasoning(true);

        let chunks = ["need a tool", "<", "｜DSML｜tool_calls>\nX"];
        let mut reasoning = String::new();
        let mut normal = String::new();

        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }

        assert_eq!(reasoning, "need a tool");
        assert_eq!(normal, "<｜DSML｜tool_calls>\nX");
    }

    #[test]
    fn test_deepseek_v4_streaming_exits_reasoning_on_split_dsml_invoke() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        parser.set_in_reasoning(true);

        let chunks = ["need a tool", "<｜DSML｜inv", "oke name=\"bash\">\nX"];
        let mut reasoning = String::new();
        let mut normal = String::new();

        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }

        assert_eq!(reasoning, "need a tool");
        assert_eq!(normal, "<｜DSML｜invoke name=\"bash\">\nX");
    }

    #[test]
    fn test_deepseek_v4_streaming_keeps_non_tool_dsml_prefix_in_reasoning() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        parser.set_in_reasoning(true);

        let chunks = ["literal", "<｜DSML｜", "not_a_tool"];
        let mut reasoning = String::new();
        let mut normal = String::new();

        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }

        assert_eq!(reasoning, "literal<｜DSML｜not_a_tool");
        assert_eq!(normal, "");
    }

    #[test]
    fn test_deepseek_v4_streaming_does_not_drop_non_dsml_lone_angle() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        parser.set_in_reasoning(true);

        let chunks = ["literal", "<", "not dsml"];
        let mut reasoning = String::new();
        let mut normal = String::new();

        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }

        assert_eq!(reasoning, "literal<not dsml");
        assert_eq!(normal, "");
    }

    #[test] // Inkling
    fn test_inkling_detect_and_parse() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("inkling");
        let result = parser.detect_and_parse_reasoning(
            "<|message_model|><|content_thinking|>thinking<|end_message|><|message_model|><|content_text|>answer<|end_message|><|content_model_end_sampling|>",
            &[],
        );
        assert_eq!(result.reasoning_text, "thinking");
        assert_eq!(result.normal_text, "answer");
    }

    #[test] // Inkling
    fn test_inkling_preserves_tool_block() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("inkling");
        let result = parser.detect_and_parse_reasoning(
            r#"<|message_model|><|content_thinking|>use weather</a><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"Paris"}}<|end_message|>"#,
            &[],
        );
        assert_eq!(result.reasoning_text, "use weather</a>");
        assert_eq!(
            result.normal_text,
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"Paris"}}<|end_message|>"#
        );
    }

    #[test] // MiniMax M2
    fn test_minimax_m2_force_reasoning_and_tool_transition() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m2");
        let result = parser.detect_and_parse_reasoning(
            "I should call weather.</think><minimax:tool_call><invoke name=\"get_weather\"></invoke></minimax:tool_call>",
            &[],
        );

        assert_eq!(result.reasoning_text, "I should call weather.");
        assert_eq!(
            result.normal_text,
            "<minimax:tool_call><invoke name=\"get_weather\"></invoke></minimax:tool_call>"
        );
    }

    #[test] // MiniMax M2
    fn test_minimax_m2_tool_start_exits_force_reasoning() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m2");
        let tool_call =
            "<minimax:tool_call><invoke name=\"get_weather\"></invoke></minimax:tool_call>";
        let result = parser.detect_and_parse_reasoning(tool_call, &[]);

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, tool_call);
    }

    #[test] // MiniMax M2
    fn test_minimax_m2_bare_invoke_exits_force_reasoning() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m2");
        let tool_call =
            "<invoke name=\"get_weather\"><parameter name=\"city\">NYC</parameter></invoke>";
        let result =
            parser.detect_and_parse_reasoning(&format!("I should call weather.{tool_call}"), &[]);

        assert_eq!(result.reasoning_text, "I should call weather.");
        assert_eq!(result.normal_text, tool_call);
    }

    #[test] // MiniMax M2
    fn test_minimax_m2_bare_invoke_streaming_exits_force_reasoning() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m2");

        let r1 = parser.parse_reasoning_streaming_incremental("I should call ", &[]);
        assert_eq!(r1.reasoning_text, "I should call ");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("weather.<invoke na", &[]);
        assert_eq!(r2.reasoning_text, "weather.");
        assert_eq!(r2.normal_text, "");

        let tail = "me=\"get_weather\"><parameter name=\"city\">NYC</parameter></invoke>";
        let r3 = parser.parse_reasoning_streaming_incremental(tail, &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(
            r3.normal_text,
            "<invoke name=\"get_weather\"><parameter name=\"city\">NYC</parameter></invoke>"
        );
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_detect_and_parse() {
        for parser_name in ["minimax_m3", "minimax-m3"] {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(parser_name);
            let result =
                parser.detect_and_parse_reasoning("<mm:think>thinking</mm:think>answer", &[]);
            assert_eq!(result.reasoning_text, "thinking");
            assert_eq!(result.normal_text, "answer");
        }
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_streaming() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");

        let r1 = parser.parse_reasoning_streaming_incremental("<mm:think>rea", &[]);
        let r2 = parser.parse_reasoning_streaming_incremental("son</mm:think>answer", &[]);

        assert_eq!(
            format!("{}{}", r1.reasoning_text, r2.reasoning_text),
            "reason"
        );
        assert_eq!(format!("{}{}", r1.normal_text, r2.normal_text), "answer");
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_streaming_with_prompt_prefilled_start_marker() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        parser.set_in_reasoning(true);

        let result = parser.parse_reasoning_streaming_incremental("reason</mm:think>answer", &[]);

        assert_eq!(result.reasoning_text, "reason");
        assert_eq!(result.normal_text, "answer");
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_native_tool_namespace_ends_prompt_prefilled_reasoning() {
        for parser_name in ["minimax_m3", "minimax-m3"] {
            for prompt_prefilled_state_set in [false, true] {
                let mut parser = ReasoningParserType::get_reasoning_parser_from_name(parser_name);
                if prompt_prefilled_state_set {
                    parser.set_in_reasoning(true);
                }

                let input = format!(
                    "reasoning{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>\n\
                     {MINIMAX_M3_TOOL_NAMESPACE}<invoke name=\"get_weather\">"
                );
                let result = parser.detect_and_parse_reasoning(&input, &[]);

                assert_eq!(
                    result.reasoning_text, "reasoning",
                    "parser {parser_name}, prompt-prefilled state set: \
                     {prompt_prefilled_state_set}"
                );
                assert_eq!(
                    result.normal_text,
                    format!(
                        "{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>\n\
                         {MINIMAX_M3_TOOL_NAMESPACE}<invoke name=\"get_weather\">"
                    ),
                    "parser {parser_name}, prompt-prefilled state set: \
                     {prompt_prefilled_state_set}"
                );
            }
        }
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_native_tool_namespace_is_lossless_across_stream_chunks() {
        let mut chunkings = vec![vec![
            "reasoning".to_string(),
            MINIMAX_M3_TOOL_NAMESPACE.to_string(),
            "<tool_call>".to_string(),
        ]];
        chunkings.extend((1..MINIMAX_M3_TOOL_NAMESPACE.len()).map(|split| {
            let (namespace_prefix, namespace_suffix) = MINIMAX_M3_TOOL_NAMESPACE.split_at(split);
            vec![
                format!("reasoning{namespace_prefix}"),
                format!("{namespace_suffix}<tool_call>"),
            ]
        }));

        for chunks in chunkings {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
            parser.set_in_reasoning(true);

            let results: Vec<_> = chunks
                .iter()
                .map(|chunk| parser.parse_reasoning_streaming_incremental(chunk, &[]))
                .collect();

            assert_eq!(
                results
                    .iter()
                    .map(|result| result.reasoning_text.as_str())
                    .collect::<String>(),
                "reasoning",
                "namespace chunking {chunks:?} leaked into reasoning"
            );
            assert_eq!(
                results
                    .iter()
                    .map(|result| result.normal_text.as_str())
                    .collect::<String>(),
                format!("{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>"),
                "namespace chunking {chunks:?} was not preserved for the tool parser"
            );
        }
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_native_tool_namespace_recovers_implicit_streaming_reasoning() {
        for parser_name in ["minimax_m3", "minimax-m3"] {
            let cases = [
                (
                    vec![format!("reasoning{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>")],
                    "reasoning",
                ),
                (
                    vec![
                        "reasoning]".to_string(),
                        "<]minimax[>[<tool_call>".to_string(),
                    ],
                    "reasoning",
                ),
                (
                    vec![
                        "reasoning".to_string(),
                        format!("{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>"),
                    ],
                    "reasoning",
                ),
                (
                    vec![
                        "let ".to_string(),
                        "me ".to_string(),
                        "think".to_string(),
                        MINIMAX_M3_TOOL_NAMESPACE.to_string(),
                        "<tool_call>".to_string(),
                    ],
                    "let me think",
                ),
            ];

            for (chunks, expected_reasoning) in cases {
                let mut parser = ReasoningParserType::get_reasoning_parser_from_name(parser_name);
                let results: Vec<_> = chunks
                    .iter()
                    .map(|chunk| parser.parse_reasoning_streaming_incremental(chunk, &[]))
                    .collect();

                assert_eq!(
                    results
                        .iter()
                        .map(|result| result.reasoning_text.as_str())
                        .collect::<String>(),
                    expected_reasoning,
                    "parser {parser_name}, chunks {chunks:?}"
                );
                assert_eq!(
                    results
                        .iter()
                        .map(|result| result.normal_text.as_str())
                        .collect::<String>(),
                    format!("{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>"),
                    "parser {parser_name}, chunks {chunks:?}"
                );
            }
        }
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_implicit_streaming_recovery_waits_for_a_decisive_boundary() {
        let chunks = ["see item [1]", " and [2]", " and [3]", " done."];
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");

        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            assert_eq!(result.reasoning_text, "");
            assert_eq!(result.normal_text, "");
        }

        let finished = parser.finish_reasoning_stream();
        assert_eq!(finished.reasoning_text, "");
        assert_eq!(finished.normal_text, "see item [1] and [2] and [3] done.");
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_partial_implicit_namespace_fakeout_is_normal() {
        for finish_after_prefix in [false, true] {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");

            let first = parser.parse_reasoning_streaming_incremental("plain]", &[]);
            let second = if finish_after_prefix {
                parser.finish_reasoning_stream()
            } else {
                parser.parse_reasoning_streaming_incremental("not-a-namespace", &[])
            };
            let finished = if finish_after_prefix {
                ParserResult::default()
            } else {
                parser.finish_reasoning_stream()
            };

            assert_eq!(
                format!(
                    "{}{}{}",
                    first.reasoning_text, second.reasoning_text, finished.reasoning_text
                ),
                ""
            );
            assert_eq!(
                format!(
                    "{}{}{}",
                    first.normal_text, second.normal_text, finished.normal_text
                ),
                if finish_after_prefix {
                    "plain]"
                } else {
                    "plain]not-a-namespace"
                }
            );
        }
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_explicit_normal_state_streams_each_chunk_immediately() {
        let chunks = ["see item [1]", " and [2]", " and [3]", " done."];
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        parser.set_in_reasoning(false);

        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            assert_eq!(result.reasoning_text, "");
            assert_eq!(result.normal_text, chunk);
        }

        let finished = parser.finish_reasoning_stream();
        assert_eq!(finished.reasoning_text, "");
        assert_eq!(finished.normal_text, "");
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_explicit_normal_state_disables_implicit_namespace_recovery() {
        let input = format!("normal{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>");

        let mut batch_parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        batch_parser.set_in_reasoning(false);
        let batch = batch_parser.detect_and_parse_reasoning(&input, &[]);

        let mut stream_parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        stream_parser.set_in_reasoning(false);
        let streamed = stream_parser.parse_reasoning_streaming_incremental(&input, &[]);

        for result in [batch, streamed] {
            assert_eq!(result.reasoning_text, "");
            assert_eq!(result.normal_text, input);
        }
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_native_tool_namespace_with_empty_reasoning() {
        let input = format!("{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>");

        let mut batch_parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        batch_parser.set_in_reasoning(true);
        let batch = batch_parser.detect_and_parse_reasoning(&input, &[]);

        let mut stream_parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        stream_parser.set_in_reasoning(true);
        let streamed = stream_parser.parse_reasoning_streaming_incremental(&input, &[]);

        for result in [batch, streamed] {
            assert_eq!(result.reasoning_text, "");
            assert_eq!(result.normal_text, input);
        }
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_namespace_after_explicit_end_keeps_normal_prefix() {
        let input = format!("reason</mm:think>answer{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>");
        let expected_normal = format!("answer{MINIMAX_M3_TOOL_NAMESPACE}<tool_call>");

        let mut batch_parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        let batch = batch_parser.detect_and_parse_reasoning(&input, &[]);

        let mut stream_parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        let streamed = stream_parser.parse_reasoning_streaming_incremental(&input, &[]);

        for result in [batch, streamed] {
            assert_eq!(result.reasoning_text, "reason");
            assert_eq!(result.normal_text, expected_normal);
        }
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_detect_and_parse_dangling_end_marker() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        let result = parser.detect_and_parse_reasoning("reason</mm:think>answer", &[]);

        assert_eq!(result.reasoning_text, "reason");
        assert_eq!(result.normal_text, "answer");
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_streaming_dangling_end_marker() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");

        let r1 = parser.parse_reasoning_streaming_incremental("reason</mm:", &[]);
        let r2 = parser.parse_reasoning_streaming_incremental("think>answer", &[]);

        assert_eq!(
            format!("{}{}", r1.reasoning_text, r2.reasoning_text),
            "reason"
        );
        assert_eq!(format!("{}{}", r1.normal_text, r2.normal_text), "answer");
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_streaming_close_marker_only_is_stripped() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");

        let result = parser.parse_reasoning_streaming_incremental(
            "</mm:think>I'll check the content of both files.",
            &[],
        );

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "I'll check the content of both files.");
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_detect_and_parse_multiple_spans() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        let result = parser.detect_and_parse_reasoning(
            "<mm:think>first</mm:think> middle <mm:think>second</mm:think> done",
            &[],
        );

        assert_eq!(result.reasoning_text, "firstsecond");
        assert_eq!(result.normal_text, "middle  done");
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_streaming_prompt_prefilled_close_after_complete_span() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");

        let r1 = parser.parse_reasoning_streaming_incremental("<mm:think>preamble</mm:think>", &[]);
        let r2 = parser.parse_reasoning_streaming_incremental("body</mm:think>", &[]);
        let r3 = parser.parse_reasoning_streaming_incremental("answer", &[]);

        assert_eq!(
            format!(
                "{}{}{}",
                r1.reasoning_text, r2.reasoning_text, r3.reasoning_text
            ),
            "preamblebody"
        );
        assert_eq!(
            format!("{}{}{}", r1.normal_text, r2.normal_text, r3.normal_text),
            "answer"
        );
    }

    #[test] // MiniMax M3
    fn test_minimax_m3_streaming_partial_start_prefix_becomes_normal_text_at_eof() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");

        let r1 = parser.parse_reasoning_streaming_incremental("plain <mm:th", &[]);
        let r2 = parser.parse_reasoning_streaming_incremental("esis answer", &[]);
        let finished = parser.finish_reasoning_stream();

        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r1.normal_text, "");
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");
        assert_eq!(finished.reasoning_text, "");
        assert_eq!(finished.normal_text, "plain <mm:thesis answer");
    }

    #[test] // REASONING.batch.2.c
    fn test_deepseek_v4_detect_and_parse() {
        for parser_name in ["deepseek_v4", "deepseek-v4", "deepseekv4"] {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(parser_name);
            let result = parser.detect_and_parse_reasoning("<think>thinking</think>answer", &[]);
            assert_eq!(result.reasoning_text, "thinking");
            assert_eq!(result.normal_text, "answer");
        }
    }

    #[test] // REASONING.batch.1.b
    fn test_deepseek_v4_no_forced_reasoning_without_tags() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        let result = parser.detect_and_parse_reasoning("answer only", &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "answer only");
    }

    #[test] // REASONING.stream.2.a, REASONING.batch.2.c
    fn test_deepseek_v4_streaming() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");

        let chunks = ["<think>rea", "son</think>answer"];
        let mut reasoning = String::new();
        let mut normal = String::new();

        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }

        assert_eq!(reasoning, "reason");
        assert_eq!(normal, "answer");
    }

    #[test] // REASONING.batch.2.a, REASONING.batch.2.c, REASONING.batch.2.e
    fn test_kimi_k25_detect_and_parse() {
        // (description, input, expected_reasoning, expected_normal)
        let cases = [
            (
                "force reasoning: no think tags",
                "no think tags here",
                "no think tags here",
                "",
            ),
            (
                "standard think tags",
                "<think>Let me reason about this.</think>Hello!",
                "Let me reason about this.",
                "Hello!",
            ),
            (
                "empty think block (instant mode)",
                "<think></think>Hello from instant mode!",
                "",
                "Hello from instant mode!",
            ),
            (
                "empty think block with newline",
                "<think>\n</think>Hello from instant mode!",
                "",
                "Hello from instant mode!",
            ),
        ];

        for (desc, input, expected_reasoning, expected_normal) in cases {
            let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
            let result = parser.detect_and_parse_reasoning(input, &[]);
            assert_eq!(
                result.reasoning_text, expected_reasoning,
                "FAILED reasoning: {desc}"
            );
            assert_eq!(result.normal_text, expected_normal, "FAILED normal: {desc}");
        }
    }

    #[test] // REASONING.stream.3.a, REASONING.stream.3.b, REASONING.batch.2.c
    fn test_kimi_k25_streaming_force_reasoning() {
        // Streaming: force_reasoning means tokens before <think> are treated as reasoning
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();

        // First chunk: partial think tag — buffered because it's a prefix of "<think>"
        let r1 = parser.parse_reasoning_streaming_incremental("<thi", &[]);
        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r1.normal_text, "");

        // Second chunk: completes the think tag + reasoning content
        let r2 = parser.parse_reasoning_streaming_incremental("nk>reasoning here", &[]);
        assert_eq!(r2.reasoning_text, "reasoning here");
        assert_eq!(r2.normal_text, "");

        // Third chunk: close tag + normal content
        let r3 = parser.parse_reasoning_streaming_incremental("</think>Hello!", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "Hello!");
    }

    #[test] // REASONING.stream.2.a, REASONING.batch.2.c, REASONING.batch.2.e
    fn test_kimi_k25_streaming() {
        // (description, tokens, expected_reasoning, expected_content)
        let cases: Vec<(&str, &[&str], &str, &str)> = vec![
            (
                "complete response",
                &[
                    "<think>",
                    "I need to",
                    " think about",
                    " this carefully.",
                    "</think>",
                    "Bonjour",
                    "!",
                ],
                "I need to think about this carefully.",
                "Bonjour!",
            ),
            (
                "empty think (instant mode)",
                &["<think>", "</think>", "Direct answer."],
                "",
                "Direct answer.",
            ),
        ];

        for (desc, tokens, expected_reasoning, expected_content) in cases {
            let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
            let mut all_reasoning = String::new();
            let mut all_content = String::new();
            for token in tokens {
                let r = parser.parse_reasoning_streaming_incremental(token, &[]);
                all_reasoning.push_str(&r.reasoning_text);
                all_content.push_str(&r.normal_text);
            }
            assert_eq!(
                all_reasoning, expected_reasoning,
                "FAILED reasoning: {desc}"
            );
            assert_eq!(all_content, expected_content, "FAILED content: {desc}");
        }
    }

    #[test] // registry lookup
    fn test_kimi_k25_parser_lookup_by_name() {
        // Verify the parser can be looked up by name
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi_k25");
        let result = parser.detect_and_parse_reasoning("<think>thinking</think>answer", &[]);
        assert_eq!(result.reasoning_text, "thinking");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn test_kimi_k3_explicit_reasoning_channel() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi_k3");
        let result = parser.detect_and_parse_reasoning(
            "<|open|>think<|sep|>check weather<|close|>think<|sep|><|open|>response<|sep|>It is raining.<|close|>response<|sep|>",
            &[],
        );

        assert_eq!(result.reasoning_text, "check weather");
        assert_eq!(
            result.normal_text,
            "<|open|>response<|sep|>It is raining.<|close|>response<|sep|>"
        );
    }

    #[test]
    fn test_kimi_k3_prompt_prefilled_reasoning_channel() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi-k3");
        parser.set_in_reasoning(true);

        let result = parser.parse_reasoning_streaming_incremental(
            "check weather<|close|>think<|sep|><|open|>response<|sep|>It is raining.",
            &[],
        );

        assert_eq!(result.reasoning_text, "check weather");
        assert_eq!(result.normal_text, "<|open|>response<|sep|>It is raining.");
    }

    #[test]
    fn test_kimi_k3_streaming_dangling_close_split_across_chunks() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi_k3");

        let first = parser.parse_reasoning_streaming_incremental("check weather<|close|>thi", &[]);
        let second = parser.parse_reasoning_streaming_incremental(
            "nk<|sep|><|open|>response<|sep|>It is raining.",
            &[],
        );

        assert_eq!(
            format!("{}{}", first.reasoning_text, second.reasoning_text),
            "check weather"
        );
        assert_eq!(
            format!("{}{}", first.normal_text, second.normal_text),
            "<|open|>response<|sep|>It is raining."
        );
    }

    #[test]
    fn test_kimi_k3_thinking_disabled_is_normal_text() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi_k3");
        let response = "answer<|close|>response<|sep|><|close|>message<|sep|>";
        let result = parser.detect_and_parse_reasoning(response, &[]);

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, response);
    }

    #[test]
    fn test_kimi_k3_response_marker_recovers_missing_think_close() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi_k3");
        parser.set_in_reasoning(true);
        let result = parser
            .detect_and_parse_reasoning("check weather<|open|>response<|sep|>It is raining.", &[]);

        assert_eq!(result.reasoning_text, "check weather");
        assert_eq!(result.normal_text, "<|open|>response<|sep|>It is raining.");
    }

    #[test]
    fn test_kimi_k3_reasoning_output_composes_with_tool_parser() {
        let output = concat!(
            "<|open|>think<|sep|>use the calculator<|close|>think<|sep|>",
            "<|open|>response<|sep|>I will calculate.<|close|>response<|sep|>",
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"calc\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"x\" type=\"number\"<|sep|>42",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>"
        );
        let mut reasoning = ReasoningParserType::KimiK3.get_reasoning_parser();
        let split = reasoning.detect_and_parse_reasoning(output, &[]);

        let (calls, content) = crate::tool_calling::try_tool_call_parse_kimi_k3(
            &split.normal_text,
            &crate::tool_calling::KimiK3ParserConfig::default(),
            None,
        )
        .unwrap();

        assert_eq!(split.reasoning_text, "use the calculator");
        assert_eq!(content.as_deref(), Some("I will calculate."));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "calc:0");
        assert_eq!(calls[0].function.name, "calc");
        assert_eq!(calls[0].function.arguments, r#"{"x":42}"#);
    }

    fn parse_streamed_kimi_k3(
        completion: &str,
        split: usize,
    ) -> (String, Vec<crate::tool_calling::ToolCallResponse>, String) {
        parse_streamed_kimi_k3_chunks(&[&completion[..split], &completion[split..]])
    }

    fn parse_streamed_kimi_k3_chunks(
        chunks: &[&str],
    ) -> (String, Vec<crate::tool_calling::ToolCallResponse>, String) {
        let mut parser = ReasoningParserType::KimiK3.get_reasoning_parser();
        parser.set_in_reasoning(true);

        let mut reasoning = String::new();
        let mut xtml = String::new();
        for chunk in chunks {
            let parsed = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&parsed.reasoning_text);
            xtml.push_str(&parsed.normal_text);
        }
        let finished = parser.finish_reasoning_stream();
        reasoning.push_str(&finished.reasoning_text);
        xtml.push_str(&finished.normal_text);

        let (calls, content) = crate::tool_calling::try_tool_call_parse_kimi_k3(
            &xtml,
            &crate::tool_calling::KimiK3ParserConfig::default(),
            None,
        )
        .unwrap();
        (reasoning, calls, content.unwrap_or_default())
    }

    fn assert_kimi_k3_all_splits<F>(completion: &str, mut assert_result: F)
    where
        F: FnMut(usize, &str, &[crate::tool_calling::ToolCallResponse], &str),
    {
        for split in completion
            .char_indices()
            .map(|(position, _)| position)
            .skip(1)
            .chain(std::iter::once(completion.len()))
        {
            let (reasoning, calls, content) = parse_streamed_kimi_k3(completion, split);
            assert_result(split, &reasoning, &calls, &content);
        }
    }

    #[test]
    fn test_kimi_k3_think_close_all_splits_preserve_response_content() {
        const THINK_CLOSE: &str = "<|close|>think<|sep|>";

        for answer in ["17", r#"{"answer":81}"#] {
            for split in THINK_CLOSE
                .char_indices()
                .map(|(position, _)| position)
                .chain(std::iter::once(THINK_CLOSE.len()))
            {
                let (reasoning, calls, content) = parse_streamed_kimi_k3_chunks(&[
                    "reasoning text",
                    &THINK_CLOSE[..split],
                    &THINK_CLOSE[split..],
                    answer,
                ]);

                assert_eq!(reasoning, "reasoning text", "split at byte {split}");
                assert!(calls.is_empty(), "split at byte {split}");
                assert_eq!(content, answer, "split at byte {split}");
            }
        }
    }

    #[test]
    fn test_kimi_k3_think_close_all_splits_preserve_native_tool_call() {
        const THINK_CLOSE: &str = "<|close|>think<|sep|>";
        let tool_call = concat!(
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"calc\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"x\" type=\"number\"<|sep|>5",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>"
        );

        for split in THINK_CLOSE
            .char_indices()
            .map(|(position, _)| position)
            .chain(std::iter::once(THINK_CLOSE.len()))
        {
            let (reasoning, calls, content) = parse_streamed_kimi_k3_chunks(&[
                "reasoning text",
                &THINK_CLOSE[..split],
                &THINK_CLOSE[split..],
                tool_call,
            ]);

            assert_eq!(reasoning, "reasoning text", "split at byte {split}");
            assert_eq!(content, "", "split at byte {split}");
            assert_eq!(calls.len(), 1, "split at byte {split}");
            assert_eq!(calls[0].function.name, "calc");
            assert_eq!(calls[0].function.arguments, r#"{"x":5}"#);
        }
    }

    #[test]
    fn test_kimi_k3_aggregated_think_close_does_not_leak() {
        let mut reasoning = ReasoningParserType::KimiK3.get_reasoning_parser();
        let split = reasoning.detect_and_parse_reasoning(
            concat!(
                "<|open|>think<|sep|>reasoning text",
                "<|close|>think<|sep|>answer"
            ),
            &[],
        );

        let (calls, content) = crate::tool_calling::try_tool_call_parse_kimi_k3(
            &split.normal_text,
            &crate::tool_calling::KimiK3ParserConfig::default(),
            None,
        )
        .unwrap();

        assert_eq!(split.reasoning_text, "reasoning text");
        assert!(calls.is_empty());
        assert_eq!(content.as_deref(), Some("answer"));
    }

    #[test]
    fn test_kimi_k3_live_response_handoff_is_chunk_boundary_independent() {
        let completion = concat!(
            "Final exactly '4'.",
            "<|open|>response<|sep|>4",
            "<|close|>response<|sep|>",
            "<|close|>message<|sep|>",
            "<|end_of_msg|>"
        );

        assert_kimi_k3_all_splits(completion, |split, reasoning, calls, content| {
            assert_eq!(
                reasoning, "Final exactly '4'.",
                "split at byte {split} leaked the response channel into reasoning"
            );
            assert!(calls.is_empty(), "split at byte {split}");
            assert_eq!(content, "4", "split at byte {split}");
        });
    }

    #[test]
    fn test_kimi_k3_live_multi_argument_call_is_chunk_boundary_independent() {
        let completion = concat!(
            "Now call add_numbers.",
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"add_numbers\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"a\" type=\"number\"<|sep|>17",
            "<|close|>argument<|sep|>",
            "<|open|>argument key=\"b\" type=\"number\"<|sep|>19",
            "<|close|>argument<|sep|>",
            "<|close|>call<|sep|>",
            "<|close|>tools<|sep|>",
            "<|close|>message<|sep|>",
            "<|end_of_msg|>"
        );

        assert_kimi_k3_all_splits(completion, |split, reasoning, calls, content| {
            assert_eq!(
                reasoning, "Now call add_numbers.",
                "split at byte {split} leaked argument framing into reasoning"
            );
            assert_eq!(content, "", "split at byte {split}");
            assert_eq!(calls.len(), 1, "split at byte {split}");
            assert_eq!(calls[0].function.name, "add_numbers");
            assert_eq!(calls[0].function.arguments, r#"{"a":17,"b":19}"#);
        });
    }

    #[test]
    fn test_kimi_k3_live_parallel_calls_are_chunk_boundary_independent() {
        let completion = concat!(
            "Fetch both URLs.",
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"fetch_url\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"url\" type=\"string\"<|sep|>https://a.example/x",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|open|>call tool=\"fetch_url\" index=\"2\"<|sep|>",
            "<|open|>argument key=\"url\" type=\"string\"<|sep|>https://b.example/y",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|>",
            "<|close|>message<|sep|><|end_of_msg|>"
        );

        assert_kimi_k3_all_splits(completion, |split, reasoning, calls, content| {
            assert_eq!(
                reasoning, "Fetch both URLs.",
                "split at byte {split} leaked the second call into reasoning"
            );
            assert_eq!(content, "", "split at byte {split}");
            assert_eq!(calls.len(), 2, "split at byte {split}");
            assert_eq!(
                calls[0].function.arguments,
                r#"{"url":"https://a.example/x"}"#
            );
            assert_eq!(
                calls[1].function.arguments,
                r#"{"url":"https://b.example/y"}"#
            );
        });
    }

    #[test]
    fn test_kimi_k3_live_bare_call_recovers_missing_outer_tools_wrapper() {
        let completion = concat!(
            "Call the calculator.",
            "<|open|>call tool=\"calc\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"x\" type=\"number\"<|sep|>5",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>message<|sep|><|end_of_msg|>"
        );

        assert_kimi_k3_all_splits(completion, |split, reasoning, calls, content| {
            assert_eq!(reasoning, "Call the calculator.", "split at byte {split}");
            assert_eq!(content, "", "split at byte {split}");
            assert_eq!(calls.len(), 1, "split at byte {split}");
            assert_eq!(calls[0].function.name, "calc");
            assert_eq!(calls[0].function.arguments, r#"{"x":5}"#);
        });
    }

    #[test]
    fn test_kimi_k3_live_orphan_argument_is_quarantined_not_leaked() {
        let completion = concat!(
            "Choose the unit.",
            "<|open|>argument key=\"unit\" type=\"string\"<|sep|>celsius",
            "<|close|>argument<|sep|>",
            "<|close|>call<|sep|><|close|>tools<|sep|>"
        );

        assert_kimi_k3_all_splits(completion, |split, reasoning, calls, content| {
            assert_eq!(reasoning, "Choose the unit.", "split at byte {split}");
            assert!(calls.is_empty(), "split at byte {split}");
            assert_eq!(content, "", "split at byte {split}");
        });
    }

    #[test]
    fn test_kimi_k3_added_token_spacing_is_chunk_boundary_independent() {
        let completion = concat!(
            "Use the calculator.",
            "<|open|> tools <|sep|>",
            "<|open|> call tool=\"calc\" index=\"1\" <|sep|>",
            "<|open|> argument key=\"x\" type=\"number\" <|sep|>5",
            "<|close|> argument <|sep|>",
            "<|close|> call <|sep|>",
            "<|close|> tools <|sep|>",
            "<|close|> message <|sep|>",
            "<|end_of_msg|>"
        );

        assert_kimi_k3_all_splits(completion, |split, reasoning, calls, content| {
            assert_eq!(reasoning, "Use the calculator.", "split at byte {split}");
            assert_eq!(content, "", "split at byte {split}");
            assert_eq!(calls.len(), 1, "split at byte {split}");
            assert_eq!(calls[0].function.name, "calc");
            assert_eq!(calls[0].function.arguments, r#"{"x":5}"#);
        });
    }

    #[test]
    fn test_kimi_k3_single_char_marker_buffering_preserves_literal_at_eof() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi_k3");
        parser.set_in_reasoning(true);

        let streamed = parser.parse_reasoning_streaming_incremental("literal<", &[]);
        let finished = parser.finish_reasoning_stream();

        assert_eq!(streamed.reasoning_text, "literal");
        assert_eq!(streamed.normal_text, "");
        assert_eq!(finished.reasoning_text, "<");
        assert_eq!(finished.normal_text, "");
    }

    #[test]
    fn test_non_k3_single_char_marker_behavior_is_unchanged() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("qwen3");
        parser.set_in_reasoning(true);

        let streamed = parser.parse_reasoning_streaming_incremental("literal<", &[]);
        let finished = parser.finish_reasoning_stream();

        assert_eq!(streamed.reasoning_text, "literal<");
        assert_eq!(streamed.normal_text, "");
        assert_eq!(finished.reasoning_text, "");
        assert_eq!(finished.normal_text, "");
    }

    #[test] // TOOLCALLING.fmt.3 — token-spelling differences across model variants
    fn test_kimi_vs_kimi_k25_different_tags() {
        // Kimi (original) uses ◁think▷/◁/think▷, KimiK25 uses <think>/</think>
        let mut kimi = ReasoningParserType::Kimi.get_reasoning_parser();
        let mut kimi_k25 = ReasoningParserType::KimiK25.get_reasoning_parser();

        // Kimi original does NOT parse <think> tags
        let r_kimi = kimi.detect_and_parse_reasoning("<think>reasoning</think>answer", &[]);
        assert_eq!(r_kimi.normal_text, "<think>reasoning</think>answer");
        assert_eq!(r_kimi.reasoning_text, "");

        // KimiK25 does parse <think> tags
        let r_k25 = kimi_k25.detect_and_parse_reasoning("<think>reasoning</think>answer", &[]);
        assert_eq!(r_k25.reasoning_text, "reasoning");
        assert_eq!(r_k25.normal_text, "answer");
    }

    // Scenario 1: Normal streaming flow with force_reasoning + set_in_reasoning.
    // Simulates the OpenAI path where the preprocessor detects prompt-injected
    // reasoning and calls set_in_reasoning(true). The parser should correctly
    // transition from reasoning to content when </think> arrives.
    #[test] // REASONING.stream.2.a, REASONING.batch.2.c — force-mode
    fn test_nemotron_streaming_with_set_in_reasoning() {
        let mut parser = ReasoningParserType::DeepseekR1.get_reasoning_parser();
        parser.set_in_reasoning(true); // OpenAI path calls this

        let tokens = &["Think", "ing about", " this", ".\n\n", "</think>", "Four"];

        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }
        assert_eq!(all_reasoning, "Thinking about this.\n\n");
        assert_eq!(all_content, "Four");
    }

    // Scenario 2: Streaming with force_reasoning but WITHOUT set_in_reasoning.
    // Simulates the Anthropic path bug where thinking_enabled=false and
    // set_in_reasoning is never called. The parser still starts in reasoning
    // mode (force_reasoning=true) but stripped_think_start=false. The </think>
    // boundary must still be detected correctly.
    #[test] // REASONING.stream.2.a, REASONING.batch.2.c — force-mode
    fn test_nemotron_streaming_force_reasoning_without_set_in_reasoning() {
        // DeepseekR1 has force_reasoning=true but we do NOT call set_in_reasoning
        let mut parser = ReasoningParserType::DeepseekR1.get_reasoning_parser();

        let tokens = &["Think", "ing about", " this", ".\n\n", "</think>", "Four"];

        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }
        assert_eq!(all_reasoning, "Thinking about this.\n\n");
        assert_eq!(all_content, "Four");
    }

    // Scenario 3: Token-by-token </think> split across chunks.
    // The '<' in '</think>' is a prefix of '<think>'. When stripped_think_start
    // is false, the parser's prefix-check could buffer '<' and interfere with
    // </think> detection. This test verifies the boundary is detected even when
    // </think> arrives as individual characters.
    #[test] // REASONING.stream.3.b, helper
    fn test_nemotron_streaming_split_end_think_tokens() {
        let mut parser = ReasoningParserType::DeepseekR1.get_reasoning_parser();
        parser.set_in_reasoning(true);

        // Simulate token-by-token arrival including </think> split across chunks
        let tokens = &[
            "reason", "ing", " done", ".", "</", "think", ">", "Hello", " world",
        ];

        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }
        assert_eq!(all_reasoning, "reasoning done.");
        assert_eq!(all_content, "Hello world");
    }

    // Scenario: vLLM's Nemotron v3 parser is force-reasoning. The model may
    // begin directly with reasoning text and only emit the closing </think>
    // boundary, or it may emit a full <think>...</think> block. Both forms
    // should split into reasoning_content before </think> and normal content
    // after it.
    #[test] // CASE.10 — vLLM nemotron_v3 parity
    fn test_nemotron_v3_detect_and_parse_vllm_cases() {
        let cases = [
            (
                "without start token",
                "This is a reasoning section</think>This is the rest",
                "This is a reasoning section",
                "This is the rest",
            ),
            (
                "with start token",
                "<think>This is a reasoning section</think>This is the rest",
                "This is a reasoning section",
                "This is the rest",
            ),
        ];

        for (desc, input, expected_reasoning, expected_content) in cases {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name("nemotron_v3");
            let result = parser.detect_and_parse_reasoning(input, &[]);
            assert_eq!(
                result.reasoning_text, expected_reasoning,
                "FAILED reasoning: {desc}"
            );
            assert_eq!(
                result.normal_text, expected_content,
                "FAILED content: {desc}"
            );
        }
    }

    // Scenario: same vLLM Nemotron v3 contract as the non-streaming test, but
    // exercised as streaming deltas. This verifies the parser keeps state
    // across chunks and handles both prompt-injected reasoning (no opening
    // <think> in output) and explicit <think> output.
    #[test] // CASE.8, CASE.10 — vLLM nemotron_v3 parity
    fn test_nemotron_v3_streaming_vllm_cases() {
        let cases: Vec<(&str, &[&str], &str, &str)> = vec![
            (
                "without start token",
                &[
                    "This is a reasoning section",
                    "</think>",
                    "This is the rest",
                ],
                "This is a reasoning section",
                "This is the rest",
            ),
            (
                "with start token",
                &[
                    "<think>",
                    "This is a reasoning section",
                    "</think>",
                    "This is the rest",
                ],
                "This is a reasoning section",
                "This is the rest",
            ),
        ];

        for (desc, tokens, expected_reasoning, expected_content) in cases {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name("nemotron_v3");
            let mut all_reasoning = String::new();
            let mut all_content = String::new();
            for token in tokens {
                let result = parser.parse_reasoning_streaming_incremental(token, &[]);
                all_reasoning.push_str(&result.reasoning_text);
                all_content.push_str(&result.normal_text);
            }
            assert_eq!(
                all_reasoning, expected_reasoning,
                "FAILED reasoning: {desc}"
            );
            assert_eq!(all_content, expected_content, "FAILED content: {desc}");
        }
    }

    // P2-1: V4 production regime where the prompt ends in <think>, so the stream
    // begins INSIDE a reasoning block (no opening <think> sentinel). The caller
    // initializes the parser via set_in_reasoning(true); bytes before </think>
    // must route to reasoning_content, bytes after to normal content.
    #[test]
    fn test_deepseek_v4_streaming_with_set_in_reasoning() {
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
        parser.set_in_reasoning(true);

        // Token-by-token stream, starting with raw reasoning (no <think> prefix),
        // </think> in the middle, then normal content.
        let tokens = &[
            "Wei", "gh", "ing ", "options", ".", "</think>", "Bei", "jing", " is", " sunny.",
        ];

        let mut all_reasoning = String::new();
        let mut all_content = String::new();
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }
        assert_eq!(all_reasoning, "Weighing options.");
        assert_eq!(all_content, "Beijing is sunny.");
    }
}
