// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family capabilities shared by parser construction and guided decoding.

use crate::structural_tag::StructuralTagBuilder;

use super::debug::{self, DebugToolParser};
use super::dsml::DeepSeekV4ToolStreamParser;
use super::gemma4::Gemma4ToolStreamParser;
use super::glm47::Glm47ToolStreamParser;
use super::harmony::HarmonyToolStreamParser;
use super::kimi_k2::KimiK2ToolStreamParser;
use super::minimax_m2::MiniMaxM2ToolStreamParser;
use super::minimax_m3::MiniMaxM3ToolStreamParser;
use super::qwen3_coder::Qwen3CoderToolStreamParser;
use super::traits::{Tool, ToolParser};

type ParserFactory = fn(&[Tool]) -> anyhow::Result<Box<dyn ToolParser>>;

struct FamilySpec {
    name: &'static str,
    parser_factory: ParserFactory,
    structural_tag_builder: Option<&'static StructuralTagBuilder>,
}

macro_rules! family_registry {
    ($($name:literal => ($parser:path, $structural_tag:expr)),+ $(,)?) => {
        const FAMILY_SPECS: &[FamilySpec] = &[
            $(FamilySpec {
                name: $name,
                parser_factory: $parser,
                structural_tag_builder: $structural_tag,
            }),+
        ];

        /// Every model family registered with a v2 tool parser.
        pub const REGISTERED_FAMILIES: &[&str] = &[$($name),+];
    };
}

family_registry! {
    "harmony"      => (HarmonyToolStreamParser::create, None),
    "harmony_text" => (HarmonyToolStreamParser::create, None),
    "deepseek_v4"  => (
        DeepSeekV4ToolStreamParser::create,
        Some(&crate::structural_tag::builders::DEEPSEEK_DSML)
    ),
    "qwen3_coder"  => (
        Qwen3CoderToolStreamParser::create,
        Some(&crate::structural_tag::builders::QWEN3_CODER)
    ),
    "minimax_m2"   => (MiniMaxM2ToolStreamParser::create, None),
    "minimax_m3"   => (MiniMaxM3ToolStreamParser::create, None),
    "gemma4"       => (Gemma4ToolStreamParser::create, None),
    "glm47"        => (
        Glm47ToolStreamParser::create,
        Some(&crate::structural_tag::builders::GLM47)
    ),
    "kimi_k2"      => (KimiK2ToolStreamParser::create, None),
}

fn family_spec(family: &str) -> Option<&'static FamilySpec> {
    FAMILY_SPECS.iter().find(|spec| spec.name == family)
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

/// Return the structural-tag builder registered for a model family, if any.
pub fn structural_tag_builder_for_family(family: &str) -> Option<&'static StructuralTagBuilder> {
    family_spec(family).and_then(|spec| spec.structural_tag_builder)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::structural_tag::{
        StructuralTagContext, StructuralTagSchemaMode, StructuralTagToolChoice,
    };

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
            ("qwen3_coder", "<tool_call>\n<function="),
            ("deepseek_v4", "<｜DSML｜tool_calls>"),
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
