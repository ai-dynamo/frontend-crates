// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendored Gemma-4 grammar helpers.

mod parser;

pub(crate) use parser::{has_recoverable_tool_call_boundaries_gemma4, is_call_name_char};
pub use parser::{is_call_prefix_boundary, parse_one_tool_call_gemma4};
