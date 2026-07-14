// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod debug;
pub mod dsml;
pub mod harmony;
mod harmony_grammar;
mod harmony_recovery;
pub mod qwen3_coder;
pub mod traits;

use traits::{Tool, ToolParser};

use self::debug::DebugToolParser;
use self::dsml::DeepSeekV4ToolStreamParser;
use self::harmony::HarmonyToolStreamParser;
use self::qwen3_coder::Qwen3CoderToolStreamParser;

/// Create the Dynamo v2 tool parser for a conformance family.
pub fn create_tool_parser_for_family(
    family: &str,
    tools: &[Tool],
) -> anyhow::Result<Box<dyn ToolParser>> {
    let parser = match family {
        "harmony" | "harmony_text" => HarmonyToolStreamParser::create(tools),
        "deepseek_v4" => DeepSeekV4ToolStreamParser::create(tools),
        "qwen3_coder" => Qwen3CoderToolStreamParser::create(tools),
        other => anyhow::bail!("no Dynamo parser v2 for family '{other}'"),
    }?;

    // Optional stderr instrumentation so a host (e.g. vLLM's experimental Rust
    // frontend) can confirm the Dynamo parser was selected and is parsing.
    if debug::debug_enabled() {
        return Ok(DebugToolParser::wrap(family, parser));
    }
    Ok(parser)
}
