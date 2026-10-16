// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::reasoning::{
    get_available_reasoning_parsers, ReasoningParser, ReasoningParserType,
};
#[cfg(not(feature = "jail"))]
use dynamo_parsers::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;
use dynamo_parsers::tool_calling::parsers::get_available_tool_parsers;
use dynamo_parsers::tool_calling::ToolDefinition;
use serde_json::{json, Value};

#[cfg(feature = "jail")]
mod legacy_jail_probe;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected job path")?;
    let input: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut results = serde_json::Map::new();
    let reasoning_name = input["reasoning_parser"].as_str().unwrap_or("");
    let tool_name = input["tool_parser"].as_str().unwrap_or("");
    let registered = get_available_reasoning_parsers().contains(&reasoning_name)
        && get_available_tool_parsers().contains(&tool_name);
    for case in input["job"]["cases"].as_array().ok_or("expected cases")? {
        let request = &case["request"];
        #[cfg(feature = "jail")]
        let unsupported_schedule = legacy_jail_probe::text_chunks(request).is_err();
        #[cfg(not(feature = "jail"))]
        let unsupported_schedule = false;
        let mut result = if !registered {
            unavailable(
                "unsupported_family",
                "No registered reasoning + batch tool parser for this family",
            )
        } else if !["None", "Reasoning"].iter().any(|state| {
            request["init"]
                == json!({"starting_state":state, "tool_output_mode":"Native", "named_tool":null})
        }) || !matches!(request["finish_reason"].as_str(), Some("stop" | "length"))
            || unsupported_schedule
        {
            unavailable(
                "unsupported_request_mode",
                "Legacy pipeline cannot apply this authored initialization, finish reason, or chunk schedule",
            )
        } else {
            let tools: Vec<ToolDefinition> = request["tools"]
                .as_array()
                .ok_or("expected tools")?
                .iter()
                .map(|tool| {
                    let function = tool.get("function").unwrap_or(tool);
                    ToolDefinition {
                        name: function["name"].as_str().unwrap_or("").to_owned(),
                        parameters: function.get("parameters").cloned(),
                        strict: function["strict"].as_bool(),
                    }
                })
                .collect();
            #[cfg(not(feature = "jail"))]
            let raw_text = request["input"]
                .as_str()
                .ok_or("expected input")?
                .to_owned();
            let reasoning_name = reasoning_name.to_owned();
            let tool_name = tool_name.to_owned();
            let starts_in_reasoning = request["init"]["starting_state"] == "Reasoning";
            #[cfg(feature = "jail")]
            let jail_request = request.clone();
            // A panic belongs to this case, never to a silently missing result.
            match tokio::spawn(async move {
                let mut parser = ReasoningParserType::get_reasoning_parser_from_name(&reasoning_name);
                if starts_in_reasoning {
                    parser.set_in_reasoning(true);
                }
                #[cfg(feature = "jail")]
                return legacy_jail_probe::capture(parser, &tool_name, tools, jail_request).await;
                #[cfg(not(feature = "jail"))]
                {
                let reasoning = parser.detect_and_parse_reasoning(&raw_text, &[]);
                let parsed = detect_and_parse_tool_call_with_recovery(&reasoning.normal_text, Some(&tool_name), Some(&tools)).await;
                parsed.map(|(calls, content)| json!({
                    "reasoning_text": reasoning.reasoning_text,
                    "normal_text": content.unwrap_or_default(),
                    "calls": calls.into_iter().map(|call| json!({"name":call.function.name, "arguments":serde_json::from_str::<Value>(&call.function.arguments).unwrap_or(Value::String(call.function.arguments))})).collect::<Vec<_>>()
                }))
                }
            }).await {
                Ok(Ok(final_fields)) => json!({"batch_final": final_fields}),
                Ok(Err(error)) => exception(error.to_string()),
                Err(error) => exception(format!("parser panic: {error}")),
            }
        };
        result["capture_input"] = request.clone();
        results.insert(
            case["id"].as_str().ok_or("expected case id")?.to_owned(),
            result,
        );
    }
    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}

fn unavailable(code: &str, detail: &str) -> Value {
    json!({"unavailable":detail, "capture_observation":{"unavailable":{"code":code,"detail":detail}}})
}

fn exception(detail: String) -> Value {
    json!({"error":detail, "capture_observation":{"error":{"code":"parser_exception","detail":detail}}})
}
