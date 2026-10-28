// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The jail's per-chunk completion future stays at its size before source
//! recording.
//!
//! `check_jail_completion` runs on every jailed chunk, and its future holds a
//! parse result across an await. When that result carried a `Spans` beside its
//! normal text, even with recording off, the future grew from 784 to 832
//! bytes, and the stock `apply` ran 3.7% slower on brev with the same
//! allocation count. Unrecorded text is now a bare `String`, so the result is
//! the size it was. The sizes are the same in debug and release builds.

use super::*;

/// The completion future's size before source recording.
const COMPLETION_FUTURE_LIMIT: usize = 784;

#[test]
fn unrecorded_normal_text_is_no_larger_than_a_string() {
    assert_eq!(
        std::mem::size_of::<Option<SourcedText>>(),
        std::mem::size_of::<Option<String>>()
    );
}

#[test]
fn the_completion_future_is_no_larger_than_before_recording() {
    let jail = JailedStream::builder().tool_call_parser("hermes").build();
    let mut progress = JailCompletionProgress::default();
    let size = std::mem::size_of_val(&jail.check_jail_completion("x", &mut progress));
    assert!(
        size <= COMPLETION_FUTURE_LIMIT,
        "check_jail_completion's future is {size} bytes, above {COMPLETION_FUTURE_LIMIT}"
    );
}
