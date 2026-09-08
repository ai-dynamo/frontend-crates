// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capability parity in a process that cannot observe vendor-registry mutation.
//!
//! `vendor_registry.rs` intentionally mutates process-global parser factories.
//! Keeping read-only built-in assertions in that same test binary let them
//! observe a temporary qwen3 override when Rust ran the tests in parallel.

use dynamo_parsers_v2::tool_calling::create_tool_parser_for_family;
use dynamo_parsers_v2::unified::create_unified_parser_for_family;

fn tool_only_value() -> bool {
    create_tool_parser_for_family("qwen3_coder", &[])
        .expect("qwen3_coder tool parser")
        .preserve_special_tokens()
}

#[test]
fn canonical_family_matches_the_tool_only_adapter() {
    let unified = create_unified_parser_for_family("qwen3", &[]).expect("qwen3 unified");
    assert_eq!(
        unified.preserve_special_tokens(),
        tool_only_value(),
        "unified and tool-only must not report different decoding requirements"
    );
}

#[test]
fn alias_matches_the_tool_only_adapter() {
    let unified =
        create_unified_parser_for_family("qwen3_coder", &[]).expect("qwen3_coder unified");
    assert_eq!(
        unified.preserve_special_tokens(),
        tool_only_value(),
        "the alias must agree with the canonical family and the tool-only adapter"
    );
}
