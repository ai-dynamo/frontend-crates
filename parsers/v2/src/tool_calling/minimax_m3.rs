// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy tool-only projection of the MiniMax M3 unified event stream.

use crate::tool_calling::traits::{Tool, ToolParseResult, ToolParser};
use crate::unified::{UnifiedParser, UnifiedParserExt, minimax_m3};

/// MiniMax M3 tool parser backed by the native unified parser.
pub struct MiniMaxM3ToolStreamParser {
    parser: Box<dyn UnifiedParser>,
}

impl MiniMaxM3ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            parser: minimax_m3::minimax_m3_unified(tools),
        }
    }
}

impl ToolParser for MiniMaxM3ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.parser.push(chunk)?))
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.parser.finish()?.events))
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.parser.tool_call_id(tool_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS: &str = "]<]minimax[>[";
    const BLOCK_START: &str = "]<]minimax[>[<tool_call>";
    const BLOCK_END: &str = "]<]minimax[>[</tool_call>";

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn parse_chunks(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = MiniMaxM3ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn null_constraints_preserve_argument_types() {
        let input = concat!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"get_weather\">",
            "]<]minimax[>[<location>null]<]minimax[>[</location>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        );
        for (schema, expected) in [
            (
                serde_json::json!({"anyOf": [{"type": "string"}, {"const": null}]}),
                "{\"location\":null}",
            ),
            (
                serde_json::json!({"type": "string", "anyOf": [{"type": "null"}, {"type": "string"}]}),
                "{\"location\":\"null\"}",
            ),
        ] {
            let mut tools = weather_tools();
            tools[0].parameters["properties"]["location"] = schema;
            let output = parse_chunks(&tools, &[input]).coalesce_calls();
            assert_eq!(output.calls.len(), 1);
            assert_eq!(output.calls[0].arguments, expected);
            assert!(output.calls[0].complete);
        }
    }

    #[test]
    fn array_items_preserve_schema_directed_null_at_every_split() {
        use serde_json::json;

        for (item_schema, permits_null) in [
            (json!({"type": ["number", "null"]}), true),
            (json!({"type": ["string", "null"]}), true),
            (json!({"type": ["integer", "null"]}), true),
            (json!({"type": ["boolean", "null"]}), true),
            (json!({"type": "number", "nullable": true}), true),
            (json!({"type": "string", "nullable": true}), true),
            (
                json!({"anyOf": [{"type": "number"}, {"const": null}]}),
                true,
            ),
            (
                json!({"oneOf": [{"type": "string"}, {"type": "null"}]}),
                true,
            ),
            (json!({"$ref": "#/$defs/Nullable"}), true),
            (json!({"type": "number"}), false),
            (json!({"type": "string"}), false),
            (
                json!({"allOf": [{"type": ["number", "null"]}, {"type": "number"}]}),
                false,
            ),
            (
                json!({"anyOf": [{"type": "number"}, {"$ref": "#/$defs/Missing"}]}),
                false,
            ),
        ] {
            for shape in ["direct", "reference", "nested"] {
                let array_schema = json!({"type": "array", "items": item_schema});
                let values_schema = match shape {
                    "reference" => json!({"$ref": "#/$defs/Values"}),
                    "nested" => json!({"type": "array", "items": array_schema}),
                    _ => array_schema.clone(),
                };
                let tools = vec![Tool {
                    name: "process_data".into(),
                    description: None,
                    parameters: json!({
                        "type": "object",
                        "properties": {"values": values_schema, "scalar": item_schema},
                        "$defs": {
                            "Nullable": {"type": ["number", "null"]},
                            "Values": array_schema
                        }
                    }),
                    strict: None,
                }];
                for raw in ["null", " NULL "] {
                    let item = if permits_null {
                        json!(null)
                    } else {
                        json!(raw)
                    };
                    let values = if shape == "nested" {
                        json!([[item, item]])
                    } else {
                        json!([item, item])
                    };
                    let expected = json!({"values": values, "scalar": item});
                    // The trailing text exercises array append_text as well as
                    // the completed <item> path; both must share null handling.
                    let items = format!("{NS}<item>{raw}{NS}</item>{raw}");
                    let items = if shape == "nested" {
                        format!("{NS}<item>{items}{NS}</item>")
                    } else {
                        items
                    };
                    let invoke = format!(
                        "{NS}<invoke name=\"process_data\">{NS}<values>{items}{NS}</values>\
                         {NS}<scalar>{raw}{NS}</scalar>{NS}</invoke>"
                    );
                    for wrapped in [false, true] {
                        let input = if wrapped {
                            format!("{BLOCK_START}{invoke}{BLOCK_END}")
                        } else {
                            invoke.clone()
                        };
                        for split in 0..=input.len() {
                            let output = parse_chunks(&tools, &[&input[..split], &input[split..]])
                                .coalesce_calls();
                            assert_eq!(output.normal_text, "");
                            assert_eq!(output.calls.len(), 1);
                            let call = &output.calls[0];
                            assert_eq!(call.name.as_deref(), Some("process_data"));
                            assert!(call.complete);
                            assert_eq!(
                                serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap(),
                                expected,
                                "schema={item_schema}, shape={shape}, raw={raw:?}, wrapped={wrapped}, split={split}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>",
                "NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_prefix_and_trailing_text() {
        // 8.c: prefix before the block AND narration after the close both flow
        // into normal_text verbatim; the block markup is suppressed.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check the weather. ]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
                " Let me know if you need more.",
            ],
        );
        assert_eq!(
            out.normal_text,
            "I will check the weather.  Let me know if you need more."
        );
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn suppresses_in_block_narration_between_invokes() {
        // 8.d: prose INSIDE the block between two invokes is part of the markup
        // block and is dropped, matching the v1 batch parser.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check both cities. ]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\nThen check LA.\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>LA]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check both cities. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn recovers_bare_invoke_without_wrapper() {
        // 5.g-shaped: prose + bare invoke (no `<tool_call>` opener) + orphan close.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. ]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check that. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn holds_back_split_orphan_close_after_bare_invoke() {
        // 5.g-shaped, but the orphan `BLOCK_END` close is split across two chunk
        // boundaries while `suppress_normal_text` is latched by the bare-invoke
        // recovery. The split close must be held back whole so the orphan-close
        // path can consume it and clear the latch; otherwise its remainder is
        // unrecognizable, suppression stays on, and the trailing prose is dropped.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. ]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax",
                "[>[</tool",
                "_call>Here is the weather.",
            ],
        );
        // The close marker never leaks, and the trailing prose survives.
        assert_eq!(out.normal_text, "I will check that. Here is the weather.");
        assert!(!out.normal_text.contains("tool_call"));
        assert!(!out.normal_text.contains("minimax"));
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn suppresses_incomplete_invoke_at_eof() {
        // 5.c: block start + invoke truncated mid-value never leaks.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        // Argument deltas can arrive before EOF, but truncation never commits
        // a completed call or leaks the native markup into visible content.
        assert!(out.calls.iter().all(|call| !call.complete));
        assert!(out.coalesce_calls().calls.is_empty());
    }

    #[test]
    fn holds_back_split_namespace_token() {
        // A chunk boundary inside the namespace token must not leak its prefix.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "text before ]<]mini",
                "max[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "text before ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_source_parameter_order_and_types() {
        // 7.a-shaped: multiple typed parameters keep model-emitted order.
        let tools = vec![Tool {
            name: "book_flight".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "destination": { "type": "string" },
                    "passengers": { "type": "integer" },
                    "first_class": { "type": "boolean" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"book_flight\">",
                "\n]<]minimax[>[<destination>Paris]<]minimax[>[</destination>",
                "\n]<]minimax[>[<passengers>2]<]minimax[>[</passengers>",
                "\n]<]minimax[>[<first_class>true]<]minimax[>[</first_class>",
                "\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"destination":"Paris","passengers":2,"first_class":true}"#
        );
    }

    #[test]
    fn nested_value_tags_do_not_shadow_top_level_order() {
        // 7.d-shaped: nested array/object tags inside a value must not be taken
        // for top-level parameter names when re-ordering.
        let tools = vec![Tool {
            name: "process_data".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "items": { "type": "array", "items": { "type": "integer" } },
                    "config": { "type": "object" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"process_data\">",
                "\n]<]minimax[>[<items>]<]minimax[>[<item>1]<]minimax[>[</item>]<]minimax[>[<item>2]<]minimax[>[</item>]<]minimax[>[</items>",
                "\n]<]minimax[>[<config>]<]minimax[>[<mode>fast]<]minimax[>[</mode>]<]minimax[>[</config>",
                "\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&merged.calls[0].arguments).unwrap();
        assert_eq!(args["items"], serde_json::json!([1, 2]));
        assert_eq!(args["config"]["mode"], "fast");
        // Top-level order: items before config.
        assert!(
            merged.calls[0].arguments.find("\"items\"").unwrap()
                < merged.calls[0].arguments.find("\"config\"").unwrap()
        );
    }

    #[test]
    fn nested_union_child_values_follow_payload_shape() {
        let tools = vec![Tool {
            name: "list_notes".into(),
            description: None,
            parameters: serde_json::json!({
                "properties": {"pagination": {"anyOf": [
                    {"type": "object", "properties": {
                        "page": {"type": "integer"},
                        "mode": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                        "after": {"type": ["object", "null"]},
                        "config": {"type": "object", "properties": {"enabled": {"type": "boolean"}}}
                    }},
                    {"type": "null"}
                ]}}
            }),
            strict: Some(true),
        }];
        let input = concat!(
            "]<]minimax[>[<tool_call>",
            "]<]minimax[>[<invoke name=\"list_notes\">",
            "]<]minimax[>[<pagination>",
            "]<]minimax[>[<page>2]<]minimax[>[</page>",
            "]<]minimax[>[<mode>one]<]minimax[>[</mode>",
            "]<]minimax[>[<after>null]<]minimax[>[</after>",
            "]<]minimax[>[<config>]<]minimax[>[<enabled>true]<]minimax[>[</enabled>]<]minimax[>[</config>",
            "]<]minimax[>[</pagination>",
            "]<]minimax[>[</invoke>",
            "]<]minimax[>[</tool_call>"
        );
        for width in [1, input.len()] {
            let chunks: Vec<_> = input
                .as_bytes()
                .chunks(width)
                .map(|chunk| std::str::from_utf8(chunk).unwrap())
                .collect();
            let merged = parse_chunks(&tools, &chunks).coalesce_calls();
            assert_eq!(merged.calls.len(), 1);
            let args: serde_json::Value = serde_json::from_str(&merged.calls[0].arguments).unwrap();
            assert_eq!(
                args,
                serde_json::json!({"pagination": {"page": 2, "mode": "one", "after": null, "config": {"enabled": true}}}),
                "width {width}"
            );
        }
    }

    #[test]
    fn nested_array_union_without_explicit_array_type_is_split_invariant() {
        let tools = vec![Tool {
            name: "process_data".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"anyOf": [
                            {"items": {"type": "number"}},
                            {"type": "object"}
                        ]}
                    }
                }
            }),
            strict: None,
        }];
        let input = concat!(
            "]<]minimax[>[<tool_call>",
            "]<]minimax[>[<invoke name=\"process_data\">",
            "]<]minimax[>[<values>",
            "]<]minimax[>[<item>",
            "]<]minimax[>[<item>9007199254740992.5",
            "]<]minimax[>[</item>",
            "]<]minimax[>[</item>",
            "]<]minimax[>[</values>",
            "]<]minimax[>[</invoke>",
            "]<]minimax[>[</tool_call>"
        );

        for split in 0..=input.len() {
            let merged = parse_chunks(&tools, &[&input[..split], &input[split..]]).coalesce_calls();
            assert_eq!(merged.calls.len(), 1, "split {split}");
            assert_eq!(
                merged.calls[0].arguments, r#"{"values":[[9007199254740992.5]]}"#,
                "split {split}"
            );
        }
    }

    #[test]
    fn literal_null_branch_does_not_erase_nested_integer_type() {
        for union in ["anyOf", "oneOf"] {
            for alternative in [
                serde_json::json!({"enum":[null]}),
                serde_json::json!({"const":null}),
            ] {
                let tools = vec![Tool {
                    name: "list_notes".into(),
                    description: None,
                    strict: None,
                    parameters: serde_json::json!({"properties":{"pagination":{union:[
                        {"type":"object","properties":{"page":{"type":"integer"}}},alternative
                    ]}}}),
                }];
                let input = concat!(
                    "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"list_notes\">",
                    "]<]minimax[>[<pagination>]<]minimax[>[<page>2]<]minimax[>[</page>",
                    "]<]minimax[>[</pagination>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"
                );
                for width in [1, input.len()] {
                    let chunks: Vec<_> = input
                        .as_bytes()
                        .chunks(width)
                        .map(|c| std::str::from_utf8(c).unwrap())
                        .collect();
                    let out = parse_chunks(&tools, &chunks).coalesce_calls();
                    assert_eq!(out.calls.len(), 1);
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(&out.calls[0].arguments).unwrap(),
                        serde_json::json!({"pagination":{"page":2}}),
                        "{union}, {alternative}, width {width}"
                    );
                }
            }
        }
    }

    #[test]
    fn nested_and_encoded_refs_preserve_types_across_chunk_boundaries() {
        let tools = vec![Tool {
            name: "create_order".into(),
            description: None,
            strict: None,
            parameters: serde_json::json!({
                "$defs": {
                    "postal code": {"type": "integer"},
                    "Address": {"type": "object", "properties": {
                        "zip": {"$ref": "#/$defs/postal%20code"},
                        "primary": {"$ref": "#/$defs/Flag"}
                    }},
                    "Flag": {"type": "boolean"}
                },
                "properties": {"shipping": {"$ref": "#/$defs/Address"}}
            }),
        }];
        let input = concat!(
            "Preparing. ]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"create_order\">",
            "]<]minimax[>[<shipping>]<]minimax[>[<zip>18956]<]minimax[>[</zip>",
            "]<]minimax[>[<primary>true]<]minimax[>[</primary>]<]minimax[>[</shipping>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call> Done."
        );
        for width in [1, 7, input.len()] {
            let chunks: Vec<_> = input
                .as_bytes()
                .chunks(width)
                .map(|chunk| std::str::from_utf8(chunk).unwrap())
                .collect();
            let out = parse_chunks(&tools, &chunks).coalesce_calls();
            assert_eq!(out.normal_text, "Preparing.  Done.");
            assert_eq!(out.calls.len(), 1);
            assert_eq!(out.calls[0].name.as_deref(), Some("create_order"));
            assert!(out.calls[0].complete);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&out.calls[0].arguments).unwrap(),
                serde_json::json!({"shipping": {"zip": 18956, "primary": true}}),
                "width {width}"
            );
        }
    }
}
