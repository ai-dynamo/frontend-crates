// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Server-authenticated scope used for every checkpoint read and write.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AuthorizationScope {
    pub tenant_id: String,
    pub principal_id: String,
}

/// Per-turn runtime resource limits selected by frontend policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLimits {
    pub max_tool_rounds: u32,
    pub max_parallel_tools: u32,
    pub max_tool_output_bytes: u64,
    pub max_external_work_millis: u64,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_tool_rounds: 8,
            max_parallel_tools: 4,
            max_tool_output_bytes: 1024 * 1024,
            max_external_work_millis: 60_000,
        }
    }
}

/// Trusted authorization issued by the frontend after authenticating a caller.
///
/// This is deliberately separate from inference metadata and raw headers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAuthorization {
    pub scope: AuthorizationScope,
    pub permitted_connectors: BTreeSet<String>,
    pub limits: RuntimeLimits,
}

impl RuntimeAuthorization {
    pub fn permits_connector(&self, connector: &str) -> bool {
        self.permitted_connectors.contains(connector)
    }
}
