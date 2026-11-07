// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod harmony_parser;

pub use super::config::JsonParserConfig;
pub use super::{config, response};
pub use harmony_parser::{detect_tool_call_start_harmony, parse_tool_calls_harmony_complete};

pub fn find_tool_call_end_position_harmony(chunk: &str, config: &JsonParserConfig) -> usize {
    config
        .tool_call_end_tokens
        .iter()
        .map(String::as_str)
        .chain(config.tool_call_end_tokens.is_empty().then_some("<|call|>"))
        .filter(|token| !token.is_empty())
        .filter_map(|token| {
            if matches!(token, "<|call|>" | "<|return|>") {
                harmony_parser::harmony_markers(chunk)
                    .filter(|(marker, in_json)| !in_json && marker.as_str() == token)
                    .map(|(marker, _)| marker.end())
                    .last()
            } else {
                chunk.rfind(token).map(|pos| pos + token.len())
            }
        })
        .max()
        .unwrap_or(chunk.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn harmony_config_declares_native_assistant_action_stops() {
        let config = config::ToolCallConfig::harmony().parser_config;
        let encoding = harmony_parser::get_harmony_encoding()
            .await
            .as_ref()
            .unwrap();
        let native = encoding.stop_tokens_for_assistant_actions().unwrap();
        let configured: std::collections::HashSet<_> = config
            .tool_call_end_tokens()
            .iter()
            .map(|token| {
                let ids = encoding.tokenizer().encode_with_special_tokens(token);
                assert_eq!(ids.len(), 1);
                ids[0]
            })
            .collect();
        assert_eq!(configured, native);
    }

    #[test]
    fn harmony_end_position_uses_all_configured_endings() {
        let config::ParserConfig::Harmony(config) = config::ToolCallConfig::harmony().parser_config
        else {
            unreachable!()
        };
        for ending in ["<|call|>", "<|return|>"] {
            let complete = format!("<|message|>{{}}{ending}");
            let text = format!("{complete} trailing text");
            assert_eq!(
                find_tool_call_end_position_harmony(&text, &config),
                complete.len()
            );
        }
        for literal in ["<|call|>", "<|return|>"] {
            for ending in ["<|call|>", "<|return|>"] {
                let complete = format!("<|message|>{{}}{ending}");
                let text = format!("{complete}<|message|>{{\"x\":\"literal {literal}\"}}");
                assert_eq!(
                    find_tool_call_end_position_harmony(&text, &config),
                    complete.len(),
                    "a quoted marker must not extend the completed span: {text}"
                );
            }
        }
        let mixed = "<|message|>{}<|call|><|message|>{}<|return|>";
        assert_eq!(
            find_tool_call_end_position_harmony(&format!("{mixed} trailing text"), &config),
            mixed.len()
        );
    }
}
