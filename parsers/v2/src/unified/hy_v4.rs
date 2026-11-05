// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified Tencent Hy4 (`hy_v4`) parser wiring.
//!
//! The default generation prompt ends with an open `<think:opensource>`, so the
//! family starts inside reasoning, as stock vLLM's `hy_v4` reasoning parser
//! does. A `no_think` prompt closes the block itself; callers pass
//! `UnifiedParserStartingState::Response` for it.

use crate::tool_calling::hy_v4::{THINK_END, THINK_START, hy_v4_scanner};
use crate::tool_calling::scan::ReasoningSpec;
use crate::tool_calling::traits::Tool;
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

pub(crate) fn hy_v4_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(ScannerUnified::new(
        hy_v4_scanner(tools).with_reasoning(ReasoningSpec {
            start: THINK_START,
            end: THINK_END,
            forced_start: true,
            preserve_special_tokens: false,
            ..Default::default()
        }),
    )))
}

#[cfg(test)]
mod tests {
    use crate::tool_calling::traits::Tool;
    use crate::unified::{
        UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, assemble,
        create_unified_parser_for_family,
    };
    use serde_json::json;

    fn tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "get_weather".into(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                }),
                strict: None,
            },
            Tool {
                name: "book".into(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "seats": {"type": "integer"},
                        "window": {"type": "boolean"},
                        "tags": {"type": "array"},
                        "meta": {"type": "object"},
                        "note": {"type": "string"}
                    }
                }),
                strict: None,
            },
        ]
    }

    fn parse(input: &str, state: UnifiedParserStartingState, split: usize) -> Vec<UnifiedEvent> {
        let mut parser = create_unified_parser_for_family("hy_v4", &tools()).expect("registry");
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: state,
                ..Default::default()
            })
            .expect("initialize");
        let mut deltas = parser.push(&input[..split]).expect("prefix");
        deltas.extend(parser.push(&input[split..]).expect("suffix"));
        deltas.extend(parser.finish().expect("finish").events);
        assemble(&deltas)
    }

    /// Parses `input` at every UTF-8 split and requires one result.
    fn parse_all_splits(input: &str, state: UnifiedParserStartingState) -> Vec<UnifiedEvent> {
        let whole = parse(input, state, input.len());
        for split in input.char_indices().map(|(at, _)| at) {
            assert_eq!(parse(input, state, split), whole, "split at {split}");
        }
        whole
    }

    fn reasoning(text: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning { text: text.into() }
    }

    fn text(text: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: text.into() }
    }

    fn weather(city: &str) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: json!({"city": city}),
        }
    }

    #[test]
    fn default_start_is_reasoning_then_text_then_call() {
        let input = concat!(
            "plan</think:opensource>Let me check.",
            "<tool_calls:opensource><tool_call:opensource>get_weather",
            "<arg_key:opensource>city</arg_key:opensource>",
            "<arg_value:opensource>Paris</arg_value:opensource>",
            "</tool_call:opensource></tool_calls:opensource>"
        );
        let want = vec![reasoning("plan"), text("Let me check."), weather("Paris")];
        assert_eq!(
            parse_all_splits(input, UnifiedParserStartingState::None),
            want
        );
        assert_eq!(
            parse_all_splits(input, UnifiedParserStartingState::Reasoning),
            want
        );
    }

    #[test]
    fn no_think_prompt_starts_in_the_response() {
        let input = "Hello <b>there</b>";
        assert_eq!(
            parse_all_splits(input, UnifiedParserStartingState::Response),
            vec![text(input)]
        );
    }

    #[test]
    fn unterminated_reasoning_stays_reasoning_at_eof() {
        assert_eq!(
            parse_all_splits("still thinking", UnifiedParserStartingState::Reasoning),
            vec![reasoning("still thinking")]
        );
    }

    #[test]
    fn parallel_calls_keep_schema_types_and_source_key_order() {
        let input = concat!(
            "<tool_calls:opensource>",
            "<tool_call:opensource>get_weather",
            "<arg_key:opensource>city</arg_key:opensource>",
            "<arg_value:opensource>北京</arg_value:opensource></tool_call:opensource>",
            "<tool_call:opensource>book",
            "<arg_key:opensource>window</arg_key:opensource>",
            "<arg_value:opensource>true</arg_value:opensource>",
            "<arg_key:opensource>seats</arg_key:opensource>",
            "<arg_value:opensource>2</arg_value:opensource>",
            "<arg_key:opensource>tags</arg_key:opensource>",
            "<arg_value:opensource>[\"a\", \"b\"]</arg_value:opensource>",
            "<arg_key:opensource>meta</arg_key:opensource>",
            "<arg_value:opensource>{\"k\": 1}</arg_value:opensource>",
            "<arg_key:opensource>note</arg_key:opensource>",
            "<arg_value:opensource>42</arg_value:opensource>",
            "</tool_call:opensource></tool_calls:opensource>"
        );
        let events = parse_all_splits(input, UnifiedParserStartingState::Response);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], weather("北京"));
        let UnifiedEvent::ToolCall { name, arguments } = &events[1] else {
            panic!("expected a tool call, got {:?}", events[1]);
        };
        assert_eq!(name, "book");
        assert_eq!(
            arguments,
            &json!({"window": true, "seats": 2, "tags": ["a", "b"], "meta": {"k": 1}, "note": "42"})
        );
        let keys: Vec<&str> = arguments
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["window", "seats", "tags", "meta", "note"]);
    }

    #[test]
    fn call_without_arguments_has_empty_object() {
        let input = "<tool_calls:opensource><tool_call:opensource>get_weather</tool_call:opensource></tool_calls:opensource>";
        assert_eq!(
            parse_all_splits(input, UnifiedParserStartingState::Response),
            vec![UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: json!({}),
            }]
        );
    }
}
