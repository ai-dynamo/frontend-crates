// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::reasoning::{
    ReasoningParser, ReasoningParserType, get_available_reasoning_parsers,
};
#[cfg(not(conformance_no_tool_definitions))]
use dynamo_parsers::tool_calling::ToolDefinition;
#[cfg(all(not(feature = "jail"), not(conformance_old_tool_api)))]
use dynamo_parsers::tool_calling::config::ToolCallConfig;
#[cfg(all(not(feature = "jail"), conformance_old_tool_api))]
use dynamo_parsers::tool_calling::parsers::detect_and_parse_tool_call;
#[cfg(all(not(feature = "jail"), not(conformance_old_tool_api)))]
use dynamo_parsers::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;
use dynamo_parsers::tool_calling::parsers::get_available_tool_parsers;
use serde_json::{Value, json};

#[cfg(all(feature = "jail", not(conformance_no_jail)))]
mod legacy_jail_probe;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected job path")?;
    let input: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut results = serde_json::Map::new();
    let reasoning_name = input["reasoning_parser"].as_str().unwrap_or("");
    let tool_name = input["tool_parser"].as_str().unwrap_or("");
    let tool_registered = get_available_tool_parsers().contains(&tool_name);
    let reasoning_registered = get_available_reasoning_parsers().contains(&reasoning_name);
    for case in input["job"]["cases"].as_array().ok_or("expected cases")? {
        let request = &case["request"];
        let response = request["init"]["starting_state"] == "Response";
        let registered = tool_registered && (response || reasoning_registered);
        let custom_jail_unsupported =
            cfg!(feature = "jail") && input["job"]["family"] == "deepseek_v41";
        #[cfg(all(feature = "jail", not(conformance_no_jail)))]
        let unsupported_schedule = legacy_jail_probe::text_chunks(request).is_err();
        #[cfg(any(not(feature = "jail"), conformance_no_jail))]
        let unsupported_schedule = false;
        let mut result = if cfg!(all(feature = "jail", conformance_no_jail)) {
            unavailable(
                "unsupported_capture_mode",
                "This published crate does not expose a JailedStream parser",
            )
        } else if !registered {
            unavailable(
                "unsupported_family",
                "No registered reasoning + batch tool parser for this family",
            )
        } else if custom_jail_unsupported {
            unavailable(
                "unsupported_family",
                "The public jail only accepts registered parser names; its DSv4 factory cannot receive the spaced DSv4.1 config.",
            )
        } else if response && matches!(tool_name, "harmony" | "kimi_k3" | "kimi-k3") {
            unavailable(
                "unsupported_request_mode",
                "This public tool parser consumes reasoning channel markers and has no persistent Response initialization.",
            )
        } else if !["None", "Reasoning", "Response"].iter().any(|state| {
            request["init"]
                == json!({"starting_state":state, "tool_output_mode":"Native", "named_tool":null})
        }) || !matches!(request["finish_reason"].as_str(), Some("stop" | "length"))
            || unsupported_schedule
            || cfg!(conformance_old_tool_api) && input.get("tool_config").is_some()
            || cfg!(conformance_no_reasoning_init)
                && request["init"]["starting_state"] == "Reasoning"
        {
            unavailable(
                "unsupported_request_mode",
                "Legacy pipeline cannot apply this authored initialization, finish reason, or chunk schedule",
            )
        } else {
            #[cfg(not(conformance_no_tool_definitions))]
            let tools: Vec<ToolDefinition> = request["tools"]
                .as_array()
                .ok_or("expected tools")?
                .iter()
                .map(|tool| {
                    let function = tool.get("function").unwrap_or(tool);
                    ToolDefinition {
                        name: function["name"].as_str().unwrap_or("").to_owned(),
                        parameters: function.get("parameters").cloned(),
                        #[cfg(not(conformance_no_tool_strict))]
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
            #[cfg(not(feature = "jail"))]
            let tool_config = input.get("tool_config").cloned();
            #[cfg(all(feature = "jail", not(conformance_no_jail)))]
            let jail_request = request.clone();
            // A panic belongs to this case, never to a silently missing result.
            match tokio::spawn(async move {
                // Historical Harmony constructors load vocabularies with a blocking HTTP client.
                let mut parser = if response { None } else { Some(tokio::task::spawn_blocking(move || {
                    ReasoningParserType::get_reasoning_parser_from_name(&reasoning_name)
                }).await.map_err(|error| format!("reasoning parser initialization failed: {error}"))?) };
                if starts_in_reasoning {
                    #[cfg(not(conformance_no_reasoning_init))]
                    parser.as_mut().expect("reasoning state requires a parser").set_in_reasoning(true);
                }
                #[cfg(all(feature = "jail", not(conformance_no_jail)))]
                return legacy_jail_probe::capture(parser, &tool_name, tools, jail_request).await;
                #[cfg(all(feature = "jail", conformance_no_jail))]
                return Err::<Value, String>("JailedStream is unavailable".to_owned());
                #[cfg(not(feature = "jail"))]
                {
                let (reasoning_text, normal_text) = if let Some(parser) = parser.as_mut() {
                    let result = parser.detect_and_parse_reasoning(&raw_text, &[]);
                    (result.reasoning_text, result.normal_text)
                } else { (String::new(), raw_text) };
                #[cfg(not(conformance_old_tool_api))]
                let parsed = if let Some(config) = tool_config {
                    let config: ToolCallConfig = serde_json::from_value(config).map_err(|error| error.to_string())?;
                    dynamo_parsers::tool_calling::parsers::try_tool_call_parse(&normal_text, &config, Some(&tools)).await
                } else { detect_and_parse_tool_call_with_recovery(&normal_text, Some(&tool_name), Some(&tools)).await };
                #[cfg(all(conformance_old_tool_api, not(conformance_async_tool_api)))]
                let parsed = detect_and_parse_tool_call(&normal_text, Some(&tool_name));
                #[cfg(all(conformance_old_tool_api, conformance_async_tool_api, not(conformance_tool_api_schemas)))]
                let parsed = detect_and_parse_tool_call(&normal_text, Some(&tool_name)).await;
                #[cfg(all(conformance_old_tool_api, conformance_async_tool_api, conformance_tool_api_schemas))]
                let parsed = detect_and_parse_tool_call(&normal_text, Some(&tool_name), Some(&tools)).await;
                parsed.map(|(calls, content)| json!({
                    "reasoning_text": reasoning_text,
                    "normal_text": content.unwrap_or_default(),
                    "calls": calls.into_iter().map(|call| json!({"name":call.function.name, "arguments":serde_json::from_str::<Value>(&call.function.arguments).unwrap_or(Value::String(call.function.arguments))})).collect::<Vec<_>>()
                })).map_err(|error| error.to_string())
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
