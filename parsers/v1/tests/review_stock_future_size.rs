// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::jail::apply_tool_calling_jail;
use futures::stream;

#[test]
fn review_f2_the_stock_stream_keeps_its_size_before_held_reports() {
    let output = apply_tool_calling_jail(
        Some("hermes".to_string()),
        None,
        None,
        false,
        stream::empty(),
    );
    // Boxing this stream allocates its full size once per request.
    let size = std::mem::size_of_val(&output);
    let before_held_reports = 8728;
    assert!(
        size <= before_held_reports,
        "the stock stream is {size} bytes, above {before_held_reports} at 20522ba"
    );
}
