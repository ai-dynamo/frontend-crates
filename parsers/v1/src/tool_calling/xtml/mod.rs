// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsers for XTML channel grammars.

mod kimi_k3_parser;

pub(crate) use kimi_k3_parser::{
    END_OF_MSG, JAIL_BOUNDARIES, MESSAGE_CLOSE, RESPONSE_CLOSE, RESPONSE_OPEN,
    SPACED_JAIL_BOUNDARIES, SPACED_MESSAGE_CLOSE, SPACED_RESPONSE_CLOSE, SPACED_RESPONSE_OPEN,
    SPACED_TOOLS_CLOSE, TOOLS_CLOSE, split_reasoning_handoff,
};
pub use kimi_k3_parser::{
    detect_tool_call_start_kimi_k3, find_tool_call_end_position_kimi_k3,
    try_tool_call_parse_kimi_k3,
};
