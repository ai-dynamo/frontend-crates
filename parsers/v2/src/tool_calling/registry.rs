// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family capabilities shared by parser construction and guided decoding.

use crate::structural_tag::StructuralTagBuilder;
use crate::unified::resolve_unified_parser;

use super::debug::{self, DebugToolParser};
use super::dsml::DeepSeekV4ToolStreamParser;
use super::gemma4::Gemma4ToolStreamParser;
use super::glm47::Glm47ToolStreamParser;
use super::harmony::HarmonyToolStreamParser;
use super::kimi_k2::KimiK2ToolStreamParser;
use super::kimi_k3::KimiK3ToolStreamParser;
use super::minimax_m2::MiniMaxM2ToolStreamParser;
use super::minimax_m3::MiniMaxM3ToolStreamParser;
use super::muse_glimmer::MuseGlimmerToolStreamParser;
use super::muse_spark::MuseSparkToolStreamParser;
use super::qwen3_coder::Qwen3CoderToolStreamParser;
use super::traits::{Tool, ToolParser};

type ParserFactory = fn(&[Tool]) -> anyhow::Result<Box<dyn ToolParser>>;

struct FamilySpec {
    name: &'static str,
    parser_factory: ParserFactory,
    standalone_batch: bool,
}

macro_rules! family_registry {
    ($($name:literal => ($parser:path, $standalone_batch:literal)),+ $(,)?) => {
        const FAMILY_SPECS: &[FamilySpec] = &[
            $(FamilySpec {
                name: $name,
                parser_factory: $parser,
                standalone_batch: $standalone_batch,
            }),+
        ];

        /// Every model family registered with a v2 tool parser.
        pub const REGISTERED_FAMILIES: &[&str] = &[$($name),+];
    };
}

family_registry! {
    "harmony" => (HarmonyToolStreamParser::create, false),
    "harmony_text" => (HarmonyToolStreamParser::create, false),
    "deepseek_v4" => (DeepSeekV4ToolStreamParser::create, true),
    "qwen3_coder" => (Qwen3CoderToolStreamParser::create, true),
    "muse_glimmer" => (MuseGlimmerToolStreamParser::create, false),
    "muse_spark" => (MuseSparkToolStreamParser::create, false),
    "minimax_m2" => (MiniMaxM2ToolStreamParser::create, false),
    "minimax_m3" => (MiniMaxM3ToolStreamParser::create, false),
    "gemma4" => (Gemma4ToolStreamParser::create, false),
    "glm47" => (Glm47ToolStreamParser::create, false),
    "kimi_k2" => (KimiK2ToolStreamParser::create, false),
    "kimi_k3" => (KimiK3ToolStreamParser::create, false),
}

/// Resolve a selector qualified for standalone tool-only batch parsing.
pub fn standalone_batch_tool_family(selector: &str) -> Option<&'static str> {
    let spec = family_spec(selector)?;
    spec.standalone_batch.then_some(spec.name)
}

fn family_spec(family: &str) -> Option<&'static FamilySpec> {
    // Public and compatibility names must select the same parser and grammar.
    let family = if family == "muse_spark_open" {
        "muse_spark"
    } else {
        family
    };
    FAMILY_SPECS
        .iter()
        .find(|spec| spec.name == family)
        .or_else(|| {
            let entry = crate::unified::builtin_unified_registrations()
                .iter()
                .find(|entry| entry.descriptor.accepts(family))?;
            FAMILY_SPECS
                .iter()
                .find(|spec| spec.standalone_batch && spec.name == entry.descriptor.family)
        })
}

/// Create the Dynamo v2 tool parser for a registered model family.
pub fn create_tool_parser_for_family(
    family: &str,
    tools: &[Tool],
) -> anyhow::Result<Box<dyn ToolParser>> {
    let spec = family_spec(family)
        .ok_or_else(|| anyhow::anyhow!("no Dynamo parser v2 for family '{family}'"))?;
    let parser = (spec.parser_factory)(tools)?;

    // Optional stderr instrumentation so a host can confirm which parser ran.
    if debug::debug_enabled() {
        return Ok(DebugToolParser::wrap(family, parser));
    }
    Ok(parser)
}

/// Return the structural-tag builder for a model family, accepting Unified aliases: from
/// the tool-calling registry, or for a family served only by a unified parser (no
/// tool-call-only parser to carry it), from the structural-tag builders.
pub fn structural_tag_builder_for_family(family: &str) -> Option<&'static StructuralTagBuilder> {
    let selector = if family == "harmony_text" {
        "gpt_oss"
    } else {
        family
    };
    resolve_unified_parser(selector)?
        .descriptor
        .structural_tag_builder
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::structural_tag::{
        StructuralTagContext, StructuralTagSchemaMode, StructuralTagToolChoice,
    };

    #[test]
    fn standalone_batch_uses_tool_only_names_and_qualified_aliases() {
        // Pin the built-in selector spellings and standalone batch eligibility.
        for selector in ["deepseek_v4", "deepseek-v4", "deepseekv4"] {
            assert_eq!(standalone_batch_tool_family(selector), Some("deepseek_v4"));
            assert!(create_tool_parser_for_family(selector, &[]).is_ok());
        }
        assert_eq!(
            standalone_batch_tool_family("qwen3_coder"),
            Some("qwen3_coder")
        );
        for selector in ["qwen3", "glm47", "muse_spark_open", "unknown"] {
            assert_eq!(standalone_batch_tool_family(selector), None);
        }
    }

    #[test]
    fn registered_families_all_create() {
        for family in REGISTERED_FAMILIES {
            create_tool_parser_for_family(family, &[]).unwrap_or_else(|e| {
                panic!("REGISTERED_FAMILIES entry '{family}' does not create: {e}")
            });
        }
    }

    #[test]
    fn structural_tag_capability_is_declared_in_the_family_registry() {
        assert!(structural_tag_builder_for_family("qwen3_coder").is_some());
        assert!(structural_tag_builder_for_family("deepseek_v4").is_some());
        assert!(structural_tag_builder_for_family("glm47").is_some());
        assert!(structural_tag_builder_for_family("kimi_k2").is_some());
        assert!(structural_tag_builder_for_family("kimi_k3").is_some());
        // Unified-only family: no tool-calling entry, grammar still resolves.
        assert!(structural_tag_builder_for_family("beam").is_some());
        assert!(create_tool_parser_for_family("beam", &[]).is_err());
        assert!(structural_tag_builder_for_family("gemma4").is_none());
        assert!(structural_tag_builder_for_family("unknown").is_none());
    }

    #[test]
    fn structural_tag_registry_routes_families_to_expected_builders() {
        let tools = [Tool {
            name: "search".to_string(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: Some(true),
        }];

        for (family, trigger) in [
            ("qwen3", "<tool_call>"),
            ("qwen3_coder", "<tool_call>"),
            ("deepseek_v4", "<｜DSML｜tool_calls>"),
            ("deepseek_v41", "<｜DSML｜ calls>"),
            ("glm47", "<tool_call>"),
        ] {
            let builder = structural_tag_builder_for_family(family)
                .unwrap_or_else(|| panic!("{family} should have a structural tag builder"));
            let tag = builder
                .build(&StructuralTagContext {
                    tool_choice: StructuralTagToolChoice::Auto,
                    tools: &tools,
                    parallel_tool_calls: None,
                    schema_mode: StructuralTagSchemaMode::Auto,
                    structured_output_schema: None,
                    starts_in_reasoning: false,
                })
                .unwrap_or_else(|error| panic!("{family} should build: {error}"))
                .unwrap_or_else(|| panic!("{family} should need a structural tag"));

            assert_eq!(tag["format"]["triggers"], json!([trigger]), "{family}");
        }
    }
}
