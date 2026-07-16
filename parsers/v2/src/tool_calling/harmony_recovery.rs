// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Harmony recovery policy: EOF recovery and normal-text suppression (audit B10).
//!
//! Decides what survives as user-visible `normal_text` after the grammar has run:
//! strips Harmony protocol envelopes (commentary/analysis/final/message-call) out of
//! the residual, and drops bare text that never carried a Harmony final/commentary
//! message. Depends only on `harmony_grammar` (the cleanup regexes + special-token
//! recording); the parser state machine calls this after `parse_harmony_snapshot`.

use regex::Captures;

use super::harmony_grammar::{
    analysis_block_cleanup_regex, commentary_block_cleanup_regex, commentary_header_cleanup_regex,
    final_block_cleanup_regex, message_call_cleanup_regex, push_unique, record_special_tokens,
    special_token_regex,
};

pub(super) fn strip_harmony_protocol_from_normal_text(text: &str, reason: &'static str) -> String {
    let mut stripped = Vec::new();

    let cleaned = commentary_block_cleanup_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            let item = match caps.name("name").map(|m| m.as_str()) {
                Some(name) => format!("commentary_tool_call:functions.{name}"),
                None => "commentary_tool_call:missing_recipient".to_string(),
            };
            push_unique(&mut stripped, item);
            ""
        })
        .into_owned();

    let cleaned = commentary_header_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            let item = match caps.name("name").map(|m| m.as_str()) {
                Some(name) => format!("commentary_tool_call_without_message:functions.{name}"),
                None => "commentary_tool_call_without_message:missing_recipient".to_string(),
            };
            push_unique(&mut stripped, item);
            ""
        })
        .into_owned();

    let cleaned = analysis_block_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            push_unique(&mut stripped, "analysis_envelope".to_string());
            ""
        })
        .into_owned();

    let cleaned = final_block_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            push_unique(&mut stripped, "final_envelope".to_string());
            caps.name("body")
                .map(|m| m.as_str())
                .unwrap_or_default()
                .to_string()
        })
        .into_owned();

    let cleaned = message_call_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            push_unique(&mut stripped, "message_call_payload".to_string());
            ""
        })
        .into_owned();

    let cleaned = special_token_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            push_unique(&mut stripped, format!("special_token:{}", &caps[0]));
            ""
        })
        .into_owned();

    if stripped.is_empty() {
        return text.to_string();
    }

    // Only the protocol envelopes are removed — the surrounding plain text is
    // kept VERBATIM, including the boundary space touching a stripped envelope
    // (e.g. `"I will check the weather. "` before `<|channel|>` keeps its
    // trailing space). The v1 jail passes that text through untouched; trimming
    // here made the stream output lose model-emitted whitespace.
    tracing::warn!(
        family = "harmony",
        reason,
        stripped = ?stripped,
        original_len = text.len(),
        cleaned_len = cleaned.len(),
        "stripped harmony protocol content from normal_text"
    );
    cleaned
}

pub(super) fn normal_text_after_parse_failure(text: &str, reason: &'static str) -> String {
    // No calls were parsed: strip any protocol residue and pass the remaining
    // plain text through VERBATIM. Marker-free text (a model answering in bare
    // prose without Harmony framing, or a whitespace-only response) is the
    // user's content and cannot leak markup by definition — dropping it here
    // used to swallow whole answers (the DIS-2322 class). The v1 jail passes
    // such text through untouched; the strict v1 batch parser still drops it —
    // that divergence is documented in the batch-via-stream allowlist.
    let cleaned = strip_harmony_protocol_from_normal_text(text, reason);
    if cleaned == text && !text.trim().is_empty() {
        tracing::warn!(
            family = "harmony",
            reason,
            original_len = text.len(),
            "passing through bare text without a Harmony final/commentary message"
        );
    }
    cleaned
}
