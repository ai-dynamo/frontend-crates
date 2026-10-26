// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::process::{Command, Stdio};

use dynamo_parsers_v2::Tool;
use dynamo_parsers_v2::structural_tag::{
    StructuralTagContext, StructuralTagSchemaMode, StructuralTagToolChoice,
};
use dynamo_parsers_v2::structural_tag_builder_for_family;
use serde_json::json;

#[test]
#[ignore = "requires python3 with xgrammar >= 0.2.8 (includes legacy V4 validation)"]
fn xgrammar_enforces_v41_framing_schema_and_call_cardinality() {
    let tools = [Tool {
        name: "Bash".into(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"}
            },
            "required": ["command", "timeout"],
            "additionalProperties": false
        }),
        strict: Some(true),
    }];
    let mut probes = Vec::new();
    for family in ["deepseek_v4", "deepseek_v41"] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        for (name, choice) in [
            ("auto", StructuralTagToolChoice::Auto),
            ("required", StructuralTagToolChoice::Required),
            ("named", StructuralTagToolChoice::Named("Bash")),
            ("none", StructuralTagToolChoice::None),
        ] {
            for parallel in [true, false] {
                let tag = builder
                    .build(&StructuralTagContext {
                        tool_choice: choice,
                        tools: &tools,
                        parallel_tool_calls: Some(parallel),
                        schema_mode: StructuralTagSchemaMode::Strict,
                        structured_output_schema: None,
                        starts_in_reasoning: false,
                    })
                    .unwrap()
                    .unwrap();
                probes.push(
                    json!({"family": family, "choice": name, "parallel": parallel, "tag": tag}),
                );
            }
        }
    }
    let mut child = Command::new(std::env::var("DSML_GRAMMAR_PYTHON").unwrap_or("python3".into()))
        .args(["-c", include_str!("check_deepseek_v41_structural_tag.py")])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&probes).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    println!("{}", String::from_utf8_lossy(&output.stdout));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
