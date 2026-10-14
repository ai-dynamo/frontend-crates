// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family structural tags for tool calling, optionally combined with
//! structured output.
//!
//! The public API describes request semantics. The xgrammar JSON wire format is
//! intentionally private so support can grow without exposing backend-specific
//! format nodes as a stable parser API.

pub(crate) mod builders;
mod context;
mod policy;
mod wire;

pub use builders::StructuralTagBuilder;
pub use context::{
    ReasoningBoundary, StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
    StructuralTagToolChoice,
};

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
