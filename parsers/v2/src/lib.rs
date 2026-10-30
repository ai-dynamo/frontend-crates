// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo parser v2 implementations.

pub mod structural_tag;
pub mod tool_calling;
pub mod unified;

pub use tool_calling::debug::{DEBUG_ENV, debug_enabled};
pub use tool_calling::dsml::DeepSeekV4ToolStreamParser;
pub use tool_calling::gemma4::Gemma4ToolStreamParser;
pub use tool_calling::glm47::Glm47ToolStreamParser;
pub use tool_calling::harmony::{
    HarmonyToolStreamParser, ToolStreamResult, assemble_tool_calls, decode_harmony, encode_harmony,
};
pub use tool_calling::kimi_k2::KimiK2ToolStreamParser;
pub use tool_calling::kimi_k3::KimiK3ToolStreamParser;
pub use tool_calling::minimax_m2::MiniMaxM2ToolStreamParser;
pub use tool_calling::minimax_m3::MiniMaxM3ToolStreamParser;
pub use tool_calling::muse_glimmer::MuseGlimmerToolStreamParser;
// Expose Spark's parser without changing the existing Glimmer entry point.
pub use tool_calling::muse_spark::MuseSparkToolStreamParser;
pub use tool_calling::qwen3_coder::Qwen3CoderToolStreamParser;
pub use tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser, ToolParserInput};
pub use tool_calling::{
    REGISTERED_FAMILIES, create_tool_parser_for_family, structural_tag_builder_for_family,
};
// Vendored batch-extraction types that surface in the public streaming API
// (e.g. `ToolStreamResult::tool_call_chunks`). v2 owns these now — see
// `tool_calling::v1core`.
pub use tool_calling::{CalledFunctionStream, ToolCallResponseChunk, ToolCallType};
// One state machine per stream owning reasoning + content + tool calls, emitting
// ONE ordered event stream (see `unified`).
pub use unified::{ArgumentFraming, MalformedToolCall, MalformedToolCallPolicy, ToolCallPolicy};
pub use unified::{
    CommittedCall, GuidedJsonCursor, InvalidGuidedPayload, InvalidGuidedPayloadKind,
    InvalidGuidedPayloadPolicy, REGISTERED_UNIFIED_FAMILIES, UnifiedEvent, UnifiedParser,
    UnifiedParserContinuation, UnifiedParserEvent, UnifiedParserExt, UnifiedParserFactory,
    UnifiedParserInit, UnifiedParserInputCapabilities, UnifiedParserOutput,
    UnifiedParserStartingState, UnifiedToolOutputMode, assemble, assemble_with_tool_indices,
    builtin_unified_families, canonical_unified_family, create_unified_parser_for_family,
    register_unified_parser, tool_arguments_raw, unregister_unified_parser,
    vendor_unified_families,
};

#[cfg(test)]
mod feature_tests {
    #[cfg(feature = "arbitrary-precision")]
    #[test]
    fn default_schema_values_preserve_large_fractions_and_exponents() {
        let text = r#"{"const":9007199254740992.5,"enum":[1e9999]}"#;
        let schema: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(schema["const"].to_string(), "9007199254740992.5");
        assert_eq!(schema["enum"][0].to_string(), "1e+9999");
    }

    #[cfg(not(feature = "arbitrary-precision"))]
    #[test]
    fn compatibility_mode_keeps_generic_serde_numeric_visitors() {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Number {
            Integer(i64),
            Float(f64),
        }
        assert!(matches!(
            serde_json::from_str("42").unwrap(),
            Number::Integer(42)
        ));
        match serde_json::from_str("0.7").unwrap() {
            Number::Float(value) => assert_eq!(value, 0.7),
            _ => panic!("fraction must reach a numeric visitor"),
        }
    }
}
