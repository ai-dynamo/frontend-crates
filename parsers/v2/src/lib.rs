// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo parser v2 implementations.

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
pub use tool_calling::minimax_m2::MiniMaxM2ToolStreamParser;
pub use tool_calling::minimax_m3::MiniMaxM3ToolStreamParser;
pub use tool_calling::muse_glimmer::MuseGlimmerToolStreamParser;
pub use tool_calling::qwen3_coder::Qwen3CoderToolStreamParser;
pub use tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser, ToolParserInput};
pub use tool_calling::{REGISTERED_FAMILIES, create_tool_parser_for_family};
// Vendored batch-extraction types that surface in the public streaming API
// (e.g. `ToolStreamResult::tool_call_chunks`). v2 owns these now — see
// `tool_calling::v1core`.
pub use tool_calling::{CalledFunctionStream, ToolCallResponseChunk, ToolCallType};
// One state machine per stream owning reasoning + content + tool calls, emitting
// ONE ordered event stream (see `unified`).
pub use unified::{
    CommittedCall, GuidedJsonCursor, InvalidGuidedPayload, InvalidGuidedPayloadKind,
    InvalidGuidedPayloadPolicy, REGISTERED_UNIFIED_FAMILIES, UnifiedEvent, UnifiedParser,
    UnifiedParserEvent, UnifiedParserExt, UnifiedParserFactory, UnifiedParserInit,
    UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
    builtin_unified_families, canonical_unified_family, create_unified_parser_for_family,
    register_unified_parser, tool_arguments_raw, unregister_unified_parser,
    vendor_unified_families,
};
