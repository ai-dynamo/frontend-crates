// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tencent Hy4 (`hy_v4`) tool-call grammar.
//!
//! Hy4 wraps back-to-back GLM-style invokes in a plural block, with every
//! marker carrying the checkpoint's `:opensource` suffix:
//!
//! ```text
//! <tool_calls:opensource>
//!   <tool_call:opensource>NAME<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>...</tool_call:opensource>
//!   ...
//! </tool_calls:opensource>
//! ```
//!
//! `WrappedBlockScanner` owns buffering, recovery and chunk-boundary handling.
//! Each delimited invoke is typed by the shared GLM emitter configured with the
//! Hy4 markers.

use crate::tool_calling::glm47::Glm47Emitter;
use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::traits::Tool;
use crate::tool_calling::v1core::Glm47ParserConfig;

pub(crate) const BLOCK_START: &str = "<tool_calls:opensource>";
pub(crate) const BLOCK_END: &str = "</tool_calls:opensource>";
pub(crate) const INVOKE_START: &str = "<tool_call:opensource>";
pub(crate) const INVOKE_END: &str = "</tool_call:opensource>";
pub(crate) const ARG_KEY_START: &str = "<arg_key:opensource>";
pub(crate) const ARG_KEY_END: &str = "</arg_key:opensource>";
pub(crate) const ARG_VALUE_START: &str = "<arg_value:opensource>";
pub(crate) const ARG_VALUE_END: &str = "</arg_value:opensource>";
pub(crate) const THINK_START: &str = "<think:opensource>";
pub(crate) const THINK_END: &str = "</think:opensource>";

fn config() -> Glm47ParserConfig {
    Glm47ParserConfig {
        tool_call_start: INVOKE_START.to_string(),
        tool_call_end: INVOKE_END.to_string(),
        arg_key_start: ARG_KEY_START.to_string(),
        arg_key_end: ARG_KEY_END.to_string(),
        arg_value_start: ARG_VALUE_START.to_string(),
        arg_value_end: ARG_VALUE_END.to_string(),
        allow_eof_recovery: false,
    }
}

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "hy_v4",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: INVOKE_START.to_string(),
        invoke_end: INVOKE_END.to_string(),
        orphan_markers: vec![BLOCK_END.to_string()],
        // BLOCK_END is held back so a split orphan close is stripped whole.
        holdback_markers: vec![
            BLOCK_START.to_string(),
            INVOKE_START.to_string(),
            BLOCK_END.to_string(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Clear,
        invoke_latch: InvokeLatch::IfEmitted,
        drop_invoke_crossing_block_end: false,
        // Hy4's tool and think markers are ordinary added tokens (ids
        // 120029-120043 have `special: false`), so default decoding keeps them.
        preserve_special_tokens: false,
        ..Default::default()
    }
}

/// The one Hy4 scanner construction site.
pub(crate) fn hy_v4_scanner(tools: &[Tool]) -> WrappedBlockScanner<Glm47Emitter> {
    WrappedBlockScanner::new(spec(), Glm47Emitter::new(config(), tools))
}
