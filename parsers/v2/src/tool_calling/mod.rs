// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod debug;
pub mod dsml;
pub mod gemma4;
pub mod glm47;
pub mod harmony;
mod harmony_grammar;
mod harmony_recovery;
pub mod kimi_k2;
pub mod minimax_m2;
pub mod minimax_m3;
pub mod muse_glimmer;
pub mod qwen3_coder;
/// Shared marker-scan core. Crate-visible because `crate::unified` builds on the
/// same scanner rather than reimplementing marker handling.
pub(crate) mod scan;
pub mod traits;
/// Vendored batch extraction copied from v1 so v2 is standalone (see module docs).
mod v1core;

// Vendored types that surface in the public streaming API.
pub use v1core::{CalledFunctionStream, ToolCallResponseChunk, ToolCallType};

use traits::{Tool, ToolParser};

use self::debug::DebugToolParser;
use self::dsml::DeepSeekV4ToolStreamParser;
use self::gemma4::Gemma4ToolStreamParser;
use self::glm47::Glm47ToolStreamParser;
use self::harmony::HarmonyToolStreamParser;
use self::kimi_k2::KimiK2ToolStreamParser;
use self::minimax_m2::MiniMaxM2ToolStreamParser;
use self::minimax_m3::MiniMaxM3ToolStreamParser;
use self::muse_glimmer::MuseGlimmerToolStreamParser;
use self::qwen3_coder::Qwen3CoderToolStreamParser;

/// Every family name `create_tool_parser_for_family` accepts, exactly one entry
/// per match arm. Test harnesses iterate this to enforce coverage: a family
/// registered below without sweep/parity coverage must fail the suite, not
/// silently skip (`registered_families_all_create` guards const/match drift).
pub const REGISTERED_FAMILIES: &[&str] = &[
    "harmony",
    "harmony_text",
    "deepseek_v4",
    "qwen3_coder",
    "muse_glimmer",
    "minimax_m2",
    "minimax_m3",
    "gemma4",
    "glm47",
    "kimi_k2",
];

/// Create the Dynamo v2 tool parser for a conformance family.
pub fn create_tool_parser_for_family(
    family: &str,
    tools: &[Tool],
) -> anyhow::Result<Box<dyn ToolParser>> {
    let parser = match family {
        "harmony" | "harmony_text" => HarmonyToolStreamParser::create(tools),
        "deepseek_v4" => DeepSeekV4ToolStreamParser::create(tools),
        "qwen3_coder" => Qwen3CoderToolStreamParser::create(tools),
        "muse_glimmer" => MuseGlimmerToolStreamParser::create(tools),
        "minimax_m2" => MiniMaxM2ToolStreamParser::create(tools),
        "minimax_m3" => MiniMaxM3ToolStreamParser::create(tools),
        "gemma4" => Gemma4ToolStreamParser::create(tools),
        "glm47" => Glm47ToolStreamParser::create(tools),
        "kimi_k2" => KimiK2ToolStreamParser::create(tools),
        other => anyhow::bail!("no Dynamo parser v2 for family '{other}'"),
    }?;

    // Optional stderr instrumentation so a host (e.g. vLLM's experimental Rust
    // frontend) can confirm the Dynamo parser was selected and is parsing.
    if debug::debug_enabled() {
        return Ok(DebugToolParser::wrap(family, parser));
    }
    Ok(parser)
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    /// Guard `REGISTERED_FAMILIES` against drifting from the match in
    /// `create_tool_parser_for_family`: every listed family must construct.
    /// (The reverse direction — an arm missing from the const — is caught by
    /// the conformance sweep, which fails a family with zero swept cases.)
    #[test]
    fn registered_families_all_create() {
        for family in REGISTERED_FAMILIES {
            create_tool_parser_for_family(family, &[]).unwrap_or_else(|e| {
                panic!("REGISTERED_FAMILIES entry '{family}' does not create: {e}")
            });
        }
    }
}
