// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendored Gemma-4 batch extractor.

mod parser;

pub use parser::{
    detect_tool_call_start_gemma4, find_tool_call_end_position_gemma4, try_tool_call_parse_gemma4,
};
