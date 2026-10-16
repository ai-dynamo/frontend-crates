// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Export the production GPT-OSS constraint for replay against a native serving engine.
use dynamo_parsers_v2::structural_tag::{
    ReasoningBoundary, StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
    StructuralTagToolChoice,
};
use dynamo_parsers_v2::{Tool, structural_tag_builder_for_family};
use std::io::Read;

fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let request: serde_json::Value = serde_json::from_str(&input)?;
    let tools: Vec<Tool> = request["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|tool| serde_json::from_value(tool["function"].clone()))
        .collect::<Result<_, _>>()?;
    let choice = match request["tool_choice"].as_str() {
        Some("none") => StructuralTagToolChoice::None,
        Some("required") => StructuralTagToolChoice::Required,
        Some("auto") | None
            if request["tool_choice"].is_string() || request["tool_choice"].is_null() =>
        {
            StructuralTagToolChoice::Auto
        }
        _ => StructuralTagToolChoice::Named(
            request["tool_choice"]["function"]["name"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing named tool"))?,
        ),
    };
    let tag = structural_tag_builder_for_family("harmony")
        .unwrap()
        .build_with_options(
            &StructuralTagContext {
                tool_choice: choice,
                tools: &tools,
                parallel_tool_calls: request["parallel_tool_calls"].as_bool(),
                schema_mode: StructuralTagSchemaMode::Strict,
                structured_output_schema: None,
                starts_in_reasoning: true,
            },
            &StructuralTagOptions {
                // Native SGLang already delays grammar enforcement until reasoning ends.
                reasoning_boundary: ReasoningBoundary::External,
                ..Default::default()
            },
        )?;
    println!("{}", serde_json::to_string(&tag)?);
    Ok(())
}
