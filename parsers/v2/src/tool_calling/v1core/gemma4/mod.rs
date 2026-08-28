// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendored Gemma-4 grammar helpers.

mod parser;

pub use parser::{
    find_complete_wrapped_call_after_gemma4, find_leading_tool_call_end_gemma4,
    has_bare_call_body_start_gemma4, is_call_prefix_boundary, parse_one_tool_call_gemma4,
};
