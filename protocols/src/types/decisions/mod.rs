// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decision wire contracts. Semantic validation belongs to `dynamo-decisions`.
pub mod openai;
pub mod sglang;
pub mod systemone;

use serde::{Deserialize, Serialize};

/// Choice values retain their JSON type; `true` differs from `"true"`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "utoipa", schema(as = DecisionSharedChoiceValue))]
pub enum ChoiceValue {
    String(String),
    Bool(bool),
}
