// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tencent Hy4 (`hy_v4`) tool-call structural tag.
//!
//! Mirrors stock vLLM's `get_hy_v4_structural_tag`: Hy4 writes argument values
//! verbatim, so only the skeleton (tool names, argument keys, tag order) is
//! constrained and every `<arg_value>` body stays free text. Required keys come
//! first in declaration order; optional keys may follow in any order.

use serde_json::Value;

use super::{StructuralTagBuilder, ToolCallGrammar, triggered_calls_format};
use crate::Tool;
use crate::structural_tag::policy::{ResolvedToolCallingPolicy, ToolCallingMode};
use crate::structural_tag::wire::{
    AnyTextFormat, ConstStringFormat, Format, NestedFormat, OrFormat, SequenceFormat, TagFormat,
    TagsWithSeparatorFormat,
};
use crate::tool_calling::hy_v4::{
    ARG_KEY_END, ARG_KEY_START, ARG_VALUE_END, ARG_VALUE_START, BLOCK_END, BLOCK_START, INVOKE_END,
    INVOKE_START, THINK_END, THINK_START,
};

pub(crate) static HY_V4: StructuralTagBuilder = StructuralTagBuilder::new(&HyV4Format);

const TOOL_CALL_MARKERS: &[&str] = &[
    BLOCK_START,
    BLOCK_END,
    INVOKE_START,
    INVOKE_END,
    ARG_KEY_START,
    ARG_KEY_END,
    ARG_VALUE_START,
    ARG_VALUE_END,
];

/// Reserved outside tool calls; the block opener is the trigger itself.
const FREE_TEXT_EXCLUDES: &[&str] = &[
    BLOCK_END,
    INVOKE_START,
    INVOKE_END,
    ARG_KEY_START,
    ARG_KEY_END,
    ARG_VALUE_START,
    ARG_VALUE_END,
];

struct HyV4Format;

fn any_text() -> Format {
    Format::AnyText(AnyTextFormat {
        excludes: Vec::new(),
    })
}

fn const_string(value: &str) -> Format {
    Format::ConstString(ConstStringFormat {
        value: value.to_string(),
    })
}

fn arg_pair(key: &str) -> Format {
    Format::Sequence(SequenceFormat {
        elements: vec![
            const_string(ARG_KEY_START),
            const_string(key),
            const_string(ARG_KEY_END),
            const_string(ARG_VALUE_START),
            any_text(),
            const_string(ARG_VALUE_END),
        ],
    })
}

/// Splits argument keys into `(required, optional)`, both in declaration order.
/// Required keys missing from `properties` are still required.
fn argument_keys(parameters: &Value) -> (Vec<&str>, Vec<&str>) {
    let properties: Vec<&str> = parameters
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|required| required.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut required_keys: Vec<&str> = properties
        .iter()
        .copied()
        .filter(|key| required.contains(key))
        .collect();
    required_keys.extend(required.iter().filter(|key| !properties.contains(key)));
    let optional_keys = properties
        .into_iter()
        .filter(|key| !required.contains(key))
        .collect();
    (required_keys, optional_keys)
}

fn tool_call_tag(tool: &Tool) -> TagFormat {
    let (required_keys, optional_keys) = argument_keys(&tool.parameters);
    let mut elements: Vec<Format> = required_keys.into_iter().map(arg_pair).collect();
    if !optional_keys.is_empty() {
        let mut optional_pairs: Vec<Format> = optional_keys.into_iter().map(arg_pair).collect();
        let pair = if optional_pairs.len() == 1 {
            optional_pairs.remove(0)
        } else {
            Format::Or(OrFormat {
                elements: optional_pairs,
            })
        };
        elements.push(Format::Star(NestedFormat {
            content: Box::new(pair),
        }));
    }
    let content = if elements.is_empty() {
        any_text()
    } else {
        Format::Sequence(SequenceFormat { elements })
    };
    TagFormat {
        begin: [INVOKE_START, &tool.name].concat(),
        content: Box::new(content),
        end: INVOKE_END.to_string(),
    }
}

fn tool_calls_block(policy: &ResolvedToolCallingPolicy<'_>) -> TagFormat {
    TagFormat {
        begin: BLOCK_START.to_string(),
        content: Box::new(Format::TagsWithSeparator(TagsWithSeparatorFormat {
            tags: policy
                .tools
                .iter()
                .map(|tool| tool_call_tag(tool))
                .collect(),
            separator: String::new(),
            at_least_one: true,
            stop_after_first: policy.stop_after_first(),
        })),
        end: BLOCK_END.to_string(),
    }
}

impl ToolCallGrammar for HyV4Format {
    fn reasoning_begin(&self) -> Option<&'static str> {
        Some(THINK_START)
    }

    fn reasoning_end(&self) -> Option<&'static str> {
        Some(THINK_END)
    }

    fn tool_call_excludes(&self) -> &'static [&'static str] {
        TOOL_CALL_MARKERS
    }

    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude_special_tokens: bool,
        _tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format> {
        match policy.mode {
            // Stock vLLM emits the block immediately for required choice.
            ToolCallingMode::Required => Ok(Format::Tag(tool_calls_block(policy))),
            ToolCallingMode::Auto => Ok(triggered_calls_format(
                BLOCK_START,
                vec![tool_calls_block(policy)],
                self.reasoning_begin(),
                self.reasoning_end(),
                FREE_TEXT_EXCLUDES,
                exclude_special_tokens,
                policy,
            )),
            _ => anyhow::bail!("auto or required policy expected"),
        }
    }

    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        _tool_arguments_any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(Format::Tag(tool_calls_block(policy)))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::HY_V4;
    use crate::Tool;
    use crate::structural_tag::test_support::{build, tools};
    use crate::structural_tag::{StructuralTagSchemaMode, StructuralTagToolChoice};

    fn pair(key: &str) -> serde_json::Value {
        json!({"type": "sequence", "elements": [
            {"type": "const_string", "value": "<arg_key:opensource>"},
            {"type": "const_string", "value": key},
            {"type": "const_string", "value": "</arg_key:opensource>"},
            {"type": "const_string", "value": "<arg_value:opensource>"},
            {"type": "any_text", "excludes": []},
            {"type": "const_string", "value": "</arg_value:opensource>"}
        ]})
    }

    #[test]
    fn auto_triggers_on_the_tool_calls_block() {
        let actual = build(
            &HY_V4,
            StructuralTagToolChoice::Auto,
            &tools(),
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );
        let format = &actual["format"];
        assert_eq!(format["type"], "triggered_tags");
        assert_eq!(format["triggers"], json!(["<tool_calls:opensource>"]));
        assert_eq!(format["at_least_one"], false);
        let block = &format["tags"][0];
        assert_eq!(block["begin"], "<tool_calls:opensource>");
        assert_eq!(block["end"], "</tool_calls:opensource>");
        assert_eq!(block["content"]["type"], "tags_with_separator");
        assert_eq!(block["content"]["separator"], "");
        let call = &block["content"]["tags"][1];
        assert_eq!(call["begin"], "<tool_call:opensource>get_weather");
        assert_eq!(call["end"], "</tool_call:opensource>");
        assert_eq!(
            call["content"],
            json!({"type": "sequence", "elements": [pair("city")]})
        );
        let excludes = format["excludes"].as_array().expect("excludes");
        assert!(excludes.contains(&json!("</think:opensource>")));
        assert!(!excludes.contains(&json!("<tool_calls:opensource>")));
    }

    #[test]
    fn required_emits_the_block_without_leading_text() {
        let actual = build(
            &HY_V4,
            StructuralTagToolChoice::Required,
            &tools(),
            Some(false),
            StructuralTagSchemaMode::Auto,
            false,
        );
        let format = &actual["format"];
        assert_eq!(format["type"], "tag");
        assert_eq!(format["begin"], "<tool_calls:opensource>");
        assert_eq!(format["content"]["at_least_one"], true);
        assert_eq!(format["content"]["stop_after_first"], true);
    }

    #[test]
    fn named_builds_one_call_for_the_selected_tool() {
        let actual = build(
            &HY_V4,
            StructuralTagToolChoice::Named("get_weather"),
            &tools(),
            Some(true),
            StructuralTagSchemaMode::Auto,
            false,
        );
        let format = &actual["format"];
        assert_eq!(format["begin"], "<tool_calls:opensource>");
        let tags = format["content"]["tags"].as_array().expect("tags");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0]["begin"], "<tool_call:opensource>get_weather");
        assert_eq!(format["content"]["stop_after_first"], true);
    }

    #[test]
    fn required_keys_lead_and_optional_keys_repeat_in_any_order() {
        let tools = [Tool {
            name: "search".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {"limit": {"type": "integer"}, "query": {"type": "string"}, "lang": {"type": "string"}},
                "required": ["query", "extra"]
            }),
            strict: None,
        }];
        let actual = build(
            &HY_V4,
            StructuralTagToolChoice::Named("search"),
            &tools,
            Some(true),
            StructuralTagSchemaMode::Auto,
            false,
        );
        assert_eq!(
            actual["format"]["content"]["tags"][0]["content"],
            json!({"type": "sequence", "elements": [
                pair("query"),
                pair("extra"),
                {"type": "star", "content": {"type": "or", "elements": [pair("limit"), pair("lang")]}}
            ]})
        );
    }

    #[test]
    fn tool_without_properties_takes_free_text() {
        let tools = [Tool {
            name: "now".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {}}),
            strict: None,
        }];
        let actual = build(
            &HY_V4,
            StructuralTagToolChoice::Named("now"),
            &tools,
            Some(true),
            StructuralTagSchemaMode::Auto,
            false,
        );
        assert_eq!(
            actual["format"]["content"]["tags"][0]["content"],
            json!({"type": "any_text", "excludes": []})
        );
    }

    #[test]
    fn none_excludes_every_tool_marker() {
        let actual = build(
            &HY_V4,
            StructuralTagToolChoice::None,
            &tools(),
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );
        assert_eq!(actual["format"]["content"]["type"], "any_text");
        assert_eq!(
            actual["format"]["content"]["excludes"]
                .as_array()
                .expect("excludes")
                .len(),
            8
        );
    }
}
