// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reproduction: UTF-8 char-boundary panic in `DecodeStream::step`.
//
// `step` slices the freshly decoded text at `new_text[prefix_text.len()..]`
// without checking that `prefix_text.len()` is a char boundary. When a decode
// window begins on orphaned multi-byte continuation bytes, `prefix_text` starts
// with a U+FFFD replacement char and the byte offset lands *inside* a multi-byte
// char in `new_text`, panicking with "byte index N is not a char boundary".
//
// The sibling `Sequence::append_token_id` already guards this (walks the split
// point back to a char boundary, then strips U+FFFD); `DecodeStream::step` does
// not. This test drives the bundled TinyLlama tokenizer with a minimal 3-token
// sequence that triggers it.

use dynamo_tokenizers::Tokenizer;

const TINYLLAMA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llm/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
);

#[test]
fn decode_stream_handles_multibyte_boundary() {
    let tokenizer = Tokenizer::from_file(TINYLLAMA).expect("load TinyLlama tokenizer");
    let mut stream = tokenizer.decode_stream(&[], false);

    // Minimal reproducing sequence (found by fuzzing the real tokenizer).
    let mut out = String::new();
    for id in [9u32, 248, 14259] {
        // Pre-fix: this panics on the third token with
        //   "byte index 1 is not a char boundary; it is inside '\u{FFFD}'".
        if let Some(chunk) = stream.step(id).expect("step should not error") {
            out.push_str(&chunk);
        }
    }

    // Only reached once the boundary is handled safely.
    assert!(out.ends_with("wir"), "unexpected decode output: {out:?}");
}
