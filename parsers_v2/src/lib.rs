// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! frontend-crate parser implementations.

pub mod tool_calling;

pub use tool_calling::harmony::{
    HarmonyToolStreamParser, ToolStreamResult, assemble_tool_calls, decode_harmony, encode_harmony,
};
