// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod parser;

pub(crate) use parser::{END_MESSAGE, INVOKE, MESSAGE_MODEL};
pub use parser::{
    detect_tool_call_start_inkling, find_tool_call_end_position_inkling,
    try_tool_call_parse_inkling,
};
