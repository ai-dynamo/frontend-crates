// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsers for ATEM channel-routed tool-call grammars (Muse Glimmer).

mod muse_glimmer_parser;

pub(crate) use muse_glimmer_parser::{
    EOM, EOT, FUNCTION_CALLS_OPEN, INVOKE_OPEN_PREFIX, MESSAGE, REASONING_RECIPIENT, START,
    USER_RECIPIENT, bare_header_pos, normalized_header, push_stripped, resolve_header,
};
pub use muse_glimmer_parser::{
    detect_tool_call_start_muse_glimmer, find_tool_call_end_position_muse_glimmer,
    try_tool_call_parse_muse_glimmer,
};
