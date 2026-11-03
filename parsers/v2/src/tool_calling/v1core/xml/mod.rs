// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendored XML-ish batch extractors (generic XML, GLM-4.7, Kimi-K2, MiniMax-M3).

#[allow(dead_code)]
mod glm47_parser;
mod kimi_k2_parser;
pub(crate) mod minimax_m3_parser;
mod parser;

pub use super::response;
pub use glm47_parser::parse_glm47_invoke;
pub use kimi_k2_parser::try_tool_call_parse_kimi_k2;
pub use parser::{parse_qwen_invoke, try_tool_call_parse_xml};

#[cfg(test)]
mod differential_sweep {
    use std::io::Write;

    use super::try_tool_call_parse_xml;
    use crate::tool_calling::v1core::ToolDefinition;
    use crate::tool_calling::v1core::config::XmlParserConfig;

    /// Base-vs-head differential driver: `SWEEP_SCHEMAS` names a JSON list of
    /// `{"id", "parameters"}` tool schemas; every property of every schema is
    /// fed each value of a fixed matrix through one complete XML call, and one
    /// JSON line per (schema, property, value) with the serialized arguments is
    /// written to `SWEEP_OUT`. Run the same driver on two revisions and diff.
    #[test]
    #[ignore = "differential sweep driver; set SWEEP_SCHEMAS and SWEEP_OUT"]
    fn sweep_schema_corpus() {
        let Ok(schemas_path) = std::env::var("SWEEP_SCHEMAS") else {
            return;
        };
        let out_path = std::env::var("SWEEP_OUT").expect("SWEEP_OUT must name the output file");
        let schemas: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(schemas_path).unwrap()).unwrap();
        let values = [
            "42",
            "42.0",
            "1e3",
            "1.5e1",
            "-0.0",
            "2.5",
            "1e-400",
            "9007199254740992.5",
            "true",
            "null",
            "\"42\"",
            "{\"a\":1}",
            "[1]",
            "hello",
        ];
        let mut out = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
        for schema in &schemas {
            let id = schema["id"].as_str().unwrap();
            let parameters = schema["parameters"].clone();
            let Some(props) = parameters.get("properties").and_then(|p| p.as_object()) else {
                continue;
            };
            let tools = vec![ToolDefinition {
                name: "f".into(),
                parameters: Some(parameters.clone()),
            }];
            for prop in props.keys() {
                for value in values {
                    let xml = format!(
                        "<tool_call>\n<function=f>\n<parameter={prop}>\n{value}\n</parameter>\n</function>\n</tool_call>"
                    );
                    let output = match try_tool_call_parse_xml(
                        &xml,
                        &XmlParserConfig::default(),
                        Some(&tools),
                    ) {
                        Ok((calls, _)) if calls.len() == 1 => calls[0].function.arguments.clone(),
                        Ok((calls, _)) => format!("<{} calls>", calls.len()),
                        Err(error) => format!("<error {error}>"),
                    };
                    writeln!(
                        out,
                        "{}",
                        serde_json::json!({"id": id, "prop": prop, "value": value, "output": output})
                    )
                    .unwrap();
                }
            }
        }
    }
}
