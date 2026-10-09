// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::tool_calling::json_prefix::JsonPrefixState;
use crate::tool_calling::traits::ToolCallDelta;
use crate::unified::GuidedJsonCursor;

/// Native call identity and append-only wire progress, independent of value closure.
pub(crate) struct KimiProgress {
    pub(crate) tool_index: usize,
    name: String,
    published: bool,
    pub(crate) released: usize,
    released_arguments: String,
    json: GuidedJsonCursor,
    syntax: JsonPrefixState,
    lexed: usize,
    stable_end: usize,
    pending_marker: Option<(usize, usize)>,
    started: bool,
}

impl KimiProgress {
    pub(crate) fn new(tool_index: usize, name: String) -> Self {
        Self {
            tool_index,
            json: GuidedJsonCursor::named(name.clone()),
            name,
            published: false,
            released: 0,
            released_arguments: String::new(),
            syntax: JsonPrefixState::default(),
            lexed: 0,
            stable_end: 0,
            pending_marker: None,
            started: false,
        }
    }

    pub(crate) fn published(&self) -> bool {
        self.published
    }

    pub(crate) fn delta(&mut self, arguments: String, complete: bool) -> ToolCallDelta {
        let name = (!self.published).then(|| self.name.clone());
        self.published = true;
        self.released += arguments.len();
        self.released_arguments.push_str(&arguments);
        ToolCallDelta {
            tool_index: self.tool_index,
            name,
            arguments,
            complete,
        }
    }

    pub(crate) fn advance_json(&mut self, payload: &str) -> Option<ToolCallDelta> {
        self.advance_json_with_markers(payload, &[], false)
    }

    pub(crate) fn advance_json_with_markers(
        &mut self,
        payload: &str,
        markers: &[&str],
        trim_unfinished_string_tail: bool,
    ) -> Option<ToolCallDelta> {
        let mut updates = Vec::new();
        let is_candidate = |at: usize| {
            let suffix = &payload[at..];
            markers.iter().any(|marker| {
                #[cfg(test)]
                count_work(2, 2 * suffix.len().min(marker.len()));
                suffix.starts_with(marker) || marker.starts_with(suffix)
            })
        };
        if self.pending_marker.is_some_and(|(at, _)| !is_candidate(at)) {
            self.pending_marker = None;
        }
        if !self.syntax.invalid && !self.syntax.complete {
            let start = self.lexed;
            for (at, ch) in payload[start..].char_indices() {
                #[cfg(test)]
                count_work(2, ch.len_utf8());
                if self.syntax.in_string()
                    && self.pending_marker.is_none()
                    && ch == '<'
                    && is_candidate(start + at)
                {
                    // EOF recovery may reinterpret native closers inside an open
                    // string; keep them until JSON proves they are literal data.
                    self.pending_marker = Some((start + at, self.stable_end));
                }
                if !self.started {
                    if ch != '{' {
                        self.syntax.invalid = true;
                        break;
                    }
                    self.syntax = JsonPrefixState::new(ch);
                    self.started = true;
                } else {
                    self.syntax.consume(ch);
                }
                if self.syntax.invalid {
                    break;
                }
                if self.syntax.complete {
                    self.pending_marker = None;
                }
                self.lexed = start + at + ch.len_utf8();
                // K2 recovery trims an unfinished string too. A later non-space
                // character or closing quote settles that pending whitespace as data.
                if (self.syntax.in_string() && !trim_unfinished_string_tail) || !ch.is_whitespace()
                {
                    self.stable_end = self.lexed;
                }
                if self.syntax.complete {
                    break;
                }
            }
        }
        let end = self.pending_marker.map_or(self.stable_end, |(_, end)| end);
        self.json.advance(&payload[..end], &mut updates);
        if updates.is_empty() {
            return None;
        }
        // JSON object closure is argument evidence only; native call syntax owns completion.
        let arguments = updates.into_iter().map(|delta| delta.arguments).collect();
        Some(self.delta(arguments, false))
    }

    pub(crate) fn finish_json(&mut self, payload: &str) -> Option<ToolCallDelta> {
        // Recovery may reinterpret native closers that were streamed inside an
        // unterminated string. Previously published bytes cannot be retracted.
        if !payload.starts_with(&self.released_arguments) {
            tracing::warn!(
                why = "kimi_recovery_conflicts_with_streamed_arguments",
                released_bytes = self.released,
                recovered_bytes = payload.len(),
                "abandoning Kimi call whose recovered arguments rewrite streamed bytes"
            );
            return None;
        }
        Some(self.delta(payload[self.released..].to_string(), true))
    }
}

#[cfg(test)]
std::thread_local! {
    static WORK: std::cell::Cell<[usize; 3]> = const { std::cell::Cell::new([0; 3]) };
}

#[cfg(test)]
pub(crate) fn count_work(category: usize, bytes: usize) {
    WORK.with(|work| {
        let mut measured = work.get();
        measured[category] += bytes;
        work.set(measured);
    });
}

#[cfg(test)]
pub(crate) fn reset_work() {
    WORK.with(|work| work.set([0; 3]));
}

#[cfg(test)]
pub(crate) fn work() -> [usize; 3] {
    WORK.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_calling::kimi_k2::KimiK2ToolStreamParser;
    use crate::tool_calling::kimi_k3::KimiK3ToolStreamParser;
    use crate::tool_calling::traits::ToolParser;

    #[test]
    // Per-chunk capture rows cannot record scan-work counts. Exercise the full
    // scanner here so an incremental argument cursor cannot hide wrapper rescans.
    fn wrapper_boundary_and_argument_work_scale_linearly() {
        for form in 0..10 {
            let measure = |size| {
                let content = "q".repeat(size);
                let input = match form {
                    0 => format!(
                        "<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{{\"count\":7,\"content\":\"{content}\"}}<|tool_call_end|><|tool_calls_section_end|>"
                    ),
                    1 | 3 => format!(
                        "<|open|>tools<|sep|><|open|>call tool=\"write_file\" index=\"1\"<|sep|><|open|>argument key=\"count\" type=\"number\"<|sep|>7<|close|>argument<|sep|><|open|>argument key=\"content\" type=\"string\"<|sep|>{content}<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
                    ),
                    4 => format!(
                        "<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{{not-json{content}<|tool_call_end|><|tool_calls_section_end|>"
                    ),
                    5 => format!(
                        "<|tool_calls_section_begin|><|tool_call_begin|>functions.{content}:0<|tool_call_argument_begin|>{{\"count\":7,\"content\":\"q\"}}<|tool_call_end|><|tool_calls_section_end|>"
                    ),
                    6 => format!(
                        "<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{{not-json\"{}\"<|tool_call_end|><|tool_calls_section_end|>",
                        "<|tool_call_end|>".repeat(size / 16)
                    ),
                    7 => format!(
                        "<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{{\"count\":7,\"content\":\"q\"}}{}",
                        "<|tool_call_begin|>".repeat(size / 16)
                    ),
                    8 => format!(
                        "<|tool_calls_section_begin|><|tool_call_begin|>functions.{content}:abc<|tool_call_argument_begin|>{{\"count\":7,\"content\":\"{content}\"}}<|tool_call_end|><|tool_calls_section_end|>"
                    ),
                    9 => format!(
                        "<|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{{\"count\":7,\"content\":\"q\"}}{}",
                        "<|tool_call_begin|>".repeat(size / 16)
                    ),
                    _ => format!(
                        "<|open|>tools<|sep|><|open|>call tool=\"write_file\" index=\"1\"<|sep|><|open|>json type=\"object\"<|sep|>{{\"count\":7,\"content\":\"{content}\"}}<|close|>json<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
                    ),
                };
                let input = if form == 3 {
                    input.replace(
                        "<|close|>argument<|sep|><|open|>argument",
                        &format!(
                            "<|close|>argument<|sep|>{}<|open|>argument",
                            " ".repeat(size)
                        ),
                    )
                } else {
                    input
                };
                let mut parser: Box<dyn ToolParser> = if form == 0 || form >= 4 {
                    Box::new(KimiK2ToolStreamParser::new(&[]))
                } else {
                    Box::new(KimiK3ToolStreamParser::new(&[]))
                };
                reset_work();
                let mut updates = crate::tool_calling::traits::ToolParseResult::default();
                for ch in input.chars() {
                    updates.append(parser.push(&ch.to_string()).unwrap());
                }
                updates.append(parser.finish().unwrap());
                let calls = updates.coalesce_calls().calls;
                if form >= 8 {
                    let measured = work();
                    let mut whole = KimiK2ToolStreamParser::new(&[]);
                    let mut expected = whole.push(&input).unwrap();
                    expected.append(whole.finish().unwrap());
                    assert_eq!(calls, expected.coalesce_calls().calls);
                    return measured;
                }
                assert_eq!(calls.len(), 1);
                if form == 4 {
                    assert_eq!(calls[0].arguments, format!("{{not-json{content}"));
                } else if form == 6 {
                    assert_eq!(
                        calls[0].arguments,
                        format!("{{not-json\"{}\"", "<|tool_call_end|>".repeat(size / 16))
                    );
                } else {
                    let expected_content = if form == 5 || form == 7 {
                        "q"
                    } else {
                        &content
                    };
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(&calls[0].arguments).unwrap(),
                        serde_json::json!({"count":7,"content":expected_content})
                    );
                }
                work()
            };
            let small = measure(4096);
            let large = measure(8192);
            eprintln!("form={form} wrapper/boundary/arguments: {small:?} -> {large:?}");
            for category in 0..3 {
                assert!(
                    large[category] <= small[category] * 2 + 256,
                    "form={form} category={category}: {small:?} -> {large:?}"
                );
            }
        }
    }
}
