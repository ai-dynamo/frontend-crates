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
pub mod qwen3_coder;
mod registry;
/// Shared marker-scan core. Crate-visible because `crate::unified` builds on the
/// same scanner rather than reimplementing marker handling.
pub(crate) mod scan;
pub mod traits;
/// Vendored batch extraction copied from v1 so v2 is standalone (see module docs).
mod v1core;

// Vendored types that surface in the public streaming API.
pub use v1core::{CalledFunctionStream, ToolCallResponseChunk, ToolCallType};

pub use registry::{
    REGISTERED_FAMILIES, create_tool_parser_for_family, structural_tag_builder_for_family,
};
