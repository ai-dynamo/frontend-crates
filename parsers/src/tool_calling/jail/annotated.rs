// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal `Annotated<R>` stream wrapper.
//!
//! The streaming tool-call jail was ported from dynamo `lib/llm`, where the
//! stream item is `dynamo_runtime::protocols::annotated::Annotated<R>`.
//! frontend-crates cannot depend on `dynamo-runtime` (dynamo depends on these
//! crates, not the other way around), so the jail carries its own minimal copy
//! of the wrapper. It preserves the fields the jail reads and forwards — the
//! stream payload plus the `id`/`event`/`comment` metadata — so dynamo can map
//! its own `Annotated` to and from this type at its boundary after the move.
//!
//! The dynamo-runtime `error` field is `Option<DynamoError>`; the jail never
//! constructs an error (every literal is `error: None`), so this copy uses
//! `Option<String>` and avoids pulling in the dynamo error type.

use serde::{Deserialize, Serialize};

/// A stream item plus optional annotation metadata.
///
/// Mirrors `dynamo_runtime::protocols::annotated::Annotated` for the fields the
/// jail depends on.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Annotated<R> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<R>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
