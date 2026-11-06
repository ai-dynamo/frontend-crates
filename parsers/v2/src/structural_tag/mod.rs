// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4.1 argument guards require `JSONSchemaFormat.excludes` support
//! with preserved string-length constraints. XGrammar's initial implementation
//! (mlc-ai/xgrammar#912) drops these constraints; it is not a suitable backend
//! for enabling this guard with constrained strings. See frontend-crates#530.
//!
//! Model-family structural tags for tool calling, optionally combined with
//! structured output.
//!
//! The public API describes request semantics. The xgrammar JSON wire format is
//! intentionally private so support can grow without exposing backend-specific
//! format nodes as a stable parser API.
//!
//! Kimi K3 structural tags require XGrammar 0.2 or newer for the `optional`
//! format node. Check the version bundled with the inference backend.
//! XGrammar 0.1.32 also ignores JSON Schema `any_order`, so it does not honor
//! reordered JSON keys requested by [`StructuralTagOptions::tool_arguments_any_order`].

pub(crate) mod builders;
mod context;
mod policy;
mod wire;

pub use builders::StructuralTagBuilder;
pub use context::{
    PromptChannel, ReasoningBoundary, StructuralTagContext, StructuralTagOptions,
    StructuralTagSchemaMode, StructuralTagToolChoice,
};

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
