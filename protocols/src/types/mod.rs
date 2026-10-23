// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Types used in inference API requests and responses.
//
// Base OpenAI types are re-exported from upstream async-openai.
// Inference-serving extensions and Anthropic types are locally defined.

// --- Locally defined modules ---
pub mod anthropic;
mod chat;
mod completion;
pub mod decisions;
mod embeddings;
pub mod realtime;
pub mod responses;
pub mod systemone;

// --- Local type re-exports ---
pub use chat::*;
pub use completion::*;

// Embeddings: `Embedding` and `CreateEmbeddingResponse` are owned
// locally (in the `embeddings` submodule) so a per-input embedding
// can be either a float vector or a base64 string per the OpenAI
// `encoding_format` spec. Every other embedding type is still
// re-exported from `async_openai::types::embeddings::*`.
pub use embeddings::*;

// Batches: re-exported wholesale from async-openai (no Dynamo overrides).
pub use async_openai::types::batches::*;

// Files: re-exported wholesale from async-openai (no Dynamo overrides).
pub use async_openai::types::files::*;

// Images: re-exported wholesale from async-openai (no Dynamo overrides).
pub use async_openai::types::images::*;

/// Build the schema for a nullable map of token IDs to biases in `[-100, 100]`.
///
/// Both request types use this with `schema_with` because Dynamo accepts fractional
/// biases, while upstream declares integers. The returned schema includes the
/// property's nullability and description, which `schema_with` replaces as well.
/// It does not parse token IDs, validate requests, or change stored values.
#[cfg(feature = "protocol-schema")]
fn logit_bias_schema() -> utoipa::openapi::schema::Object {
    use utoipa::openapi::schema::{ObjectBuilder, SchemaType, Type};

    ObjectBuilder::new()
        .schema_type(SchemaType::from_iter([Type::Object, Type::Null]))
        .description(Some(
            "Map token IDs to numeric logit biases from -100 to 100. Dynamo accepts \
             fractional biases as well as integers. Token ID parsing and validation \
             are performed by the serving layer.",
        ))
        .additional_properties(Some(
            ObjectBuilder::new()
                .schema_type(Type::Number)
                .minimum(Some(-100))
                .maximum(Some(100)),
        ))
        .build()
}

// --- Convenience impls for locally-defined types ---
mod impls;

use crate::error::OpenAIError;
use derive_builder::UninitializedFieldError;

impl From<UninitializedFieldError> for OpenAIError {
    fn from(value: UninitializedFieldError) -> Self {
        OpenAIError::InvalidArgument(value.to_string())
    }
}
