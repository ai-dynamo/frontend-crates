// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
pub struct Case {
    pub name: String,
    pub schema: Value,
    pub value: Value,
    pub raw: String,
    #[serde(default)]
    pub defs: Option<Value>,
    #[serde(default = "enabled")]
    pub grammar: bool,
    #[serde(default)]
    pub xgrammar_min: Option<String>,
    #[serde(default)]
    pub invalid_raw: Vec<String>,
}

fn enabled() -> bool {
    true
}

pub fn cases() -> Vec<Case> {
    serde_json::from_str(include_str!(
        "../../conformance/utils/src/glm47_schema_cases.json"
    ))
    .unwrap()
}

impl Case {
    pub fn parameters(&self) -> Value {
        let mut schema = json!({
            "type": "object", "properties": {"value": self.schema},
            "required": ["value"], "additionalProperties": false
        });
        if let Some(defs) = &self.defs {
            schema["$defs"] = defs.clone();
        }
        schema
    }

    pub fn wire(&self) -> String {
        wire(&self.raw)
    }

    pub fn expected(&self) -> Value {
        json!({"value": self.value})
    }

    pub fn probe(&self, tag: Value, arguments: Value) -> Value {
        json!({
            "name": self.name, "tag": tag, "wire": self.wire(),
            "schema": self.parameters(), "arguments": arguments,
            "xgrammar_min": self.xgrammar_min,
            "reject": self.invalid_raw.iter().map(|raw| wire(raw)).chain([
                "<tool_call>probe</tool_call>".into(),
                format!("<tool_call>probe<arg_key>value</arg_key><arg_value>{}</arg_value><arg_key>extra</arg_key><arg_value>1</arg_value></tool_call>", self.raw)
            ]).collect::<Vec<_>>()
        })
    }
}

pub fn wire(raw: &str) -> String {
    format!("<tool_call>probe<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>")
}

pub fn check_grammar(probes: &[Value]) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let script = include_str!("check_glm47_schema_roundtrip.py");
    let mut child = Command::new(std::env::var("GLM_SCHEMA_PYTHON").unwrap_or("python3".into()))
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(probes).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{stdout}");
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

pub fn write_report(label: &str, results: &[Value]) {
    if let Ok(directory) = std::env::var("GLM_SCHEMA_REPORT_DIR") {
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            std::path::Path::new(&directory).join(format!("{label}.json")),
            serde_json::to_vec_pretty(results).unwrap(),
        )
        .unwrap();
    }
}
