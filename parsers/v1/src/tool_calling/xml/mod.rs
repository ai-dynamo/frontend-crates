// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod glm47_parser;
mod kimi_k2_parser;
mod minimax_m3_parser;
mod parsed_value;
mod parser;

use serde::ser::{Serialize, SerializeMap, Serializer};

use parsed_value::ParsedValue;

struct OrderedArguments<'a>(&'a [(String, ParsedValue)]);

impl Serialize for OrderedArguments<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

pub use super::response;
pub use glm47_parser::{
    detect_tool_call_start_glm47, find_tool_call_end_position_glm47, try_tool_call_parse_glm47,
};
pub use kimi_k2_parser::{
    detect_tool_call_start_kimi_k2, find_tool_call_end_position_kimi_k2,
    try_tool_call_parse_kimi_k2,
};
pub use minimax_m3_parser::{
    detect_tool_call_start_minimax_m3, find_tool_call_end_position_minimax_m3,
    try_tool_call_parse_minimax_m3,
};
pub use parser::{
    detect_tool_call_start_xml, find_tool_call_end_position_xml, try_tool_call_parse_xml,
};
