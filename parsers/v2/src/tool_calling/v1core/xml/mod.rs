// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendored XML-ish batch extractors (generic XML, GLM-4.7, Kimi-K2, MiniMax-M3).

mod glm47_parser;
mod kimi_k2_parser;
mod minimax_m3_parser;
mod parsed_value;
mod parser;

pub use super::response;
pub use glm47_parser::try_tool_call_parse_glm47;
pub use kimi_k2_parser::try_tool_call_parse_kimi_k2;
pub use minimax_m3_parser::try_tool_call_parse_minimax_m3;
pub use parser::try_tool_call_parse_xml;
