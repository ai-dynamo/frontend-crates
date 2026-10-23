// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native model grammars, independent of parser generation and request policy.
//! Callers select tools and resolve schema enforcement before building formats.
pub mod format;
pub mod kimi_k2;
pub mod kimi_k3;

/// A selected tool with its resolved argument constraint (`true` means relaxed).
pub struct Tool<'a> {
    pub name: &'a str,
    pub parameters: &'a serde_json::Value,
}
