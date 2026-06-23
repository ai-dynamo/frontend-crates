// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::ParserResult;
use crate::ReasoningParser;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraniteReasoningParser {
    think_start_tokens: Vec<String>,
    think_end_tokens: Vec<String>,
    buffer: String,
    stripped_think_start: bool,
    in_reasoning: bool,
}

impl GraniteReasoningParser {
    pub fn new() -> Self {
        Self {
            think_start_tokens: ["Here's my thought process:", "Here is my thought process:"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            think_end_tokens: ["Here's my response:", "Here is my response:"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            buffer: String::new(),
            stripped_think_start: false,
            in_reasoning: false,
        }
    }
}

impl Default for GraniteReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

impl GraniteReasoningParser {
    /// Earliest occurrence of any of `tokens` in `text`, as `(byte_index, token_len)`.
    fn find_first(text: &str, tokens: &[String]) -> Option<(usize, usize)> {
        tokens
            .iter()
            .filter_map(|t| text.find(t.as_str()).map(|i| (i, t.len())))
            .min_by_key(|&(i, _)| i)
    }

    /// Length of the longest suffix of `text` that is a *proper* prefix of an
    /// end marker. Used in streaming to hold a partial `Here is my response:`
    /// split across chunks, so it is matched once complete instead of leaking
    /// into `reasoning_text`.
    fn partial_end_suffix_len(&self, text: &str) -> usize {
        let max = self
            .think_end_tokens
            .iter()
            .map(|t| t.len())
            .max()
            .unwrap_or(0)
            .min(text.len());
        for n in (1..=max).rev() {
            let idx = text.len() - n;
            if !text.is_char_boundary(idx) {
                continue;
            }
            let suffix = &text[idx..];
            if self
                .think_end_tokens
                .iter()
                .any(|t| t.len() > suffix.len() && t.starts_with(suffix))
            {
                return n;
            }
        }
        0
    }

    /// Re-parse text that still carries reasoning markers after a single-span
    /// pass, consuming *every* thought-process / response marker so none leak
    /// into the output. Text between a thought marker and the next response
    /// marker is reasoning; everything else (including the neighbours of a
    /// dangling response marker) is normal text. This diverges from vLLM,
    /// which stops after the first span and leaks the remainder.
    fn parse_all_spans(&self, text: &str) -> ParserResult {
        let mut reasoning: Vec<&str> = Vec::new();
        let mut normal: Vec<&str> = Vec::new();
        let mut rest = text;
        loop {
            let start = Self::find_first(rest, &self.think_start_tokens);
            let end = Self::find_first(rest, &self.think_end_tokens);
            match (start, end) {
                // A thought-process span opens before any response marker.
                (Some((si, sl)), end_opt) if end_opt.is_none_or(|(ei, _)| si < ei) => {
                    normal.push(rest[..si].trim());
                    rest = &rest[si + sl..];
                    match Self::find_first(rest, &self.think_end_tokens) {
                        Some((ei, el)) => {
                            reasoning.push(rest[..ei].trim());
                            rest = &rest[ei + el..];
                        }
                        None => {
                            // Open span with no close: remainder is reasoning.
                            reasoning.push(rest.trim());
                            break;
                        }
                    }
                }
                // A dangling response marker with no preceding thought marker:
                // consume the marker, keep both sides as normal text.
                (_, Some((ei, el))) => {
                    normal.push(rest[..ei].trim());
                    rest = &rest[ei + el..];
                }
                // No span opens in the remainder: it is all normal text.
                _ => {
                    normal.push(rest.trim());
                    break;
                }
            }
        }
        let join = |parts: Vec<&str>| {
            parts
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        };
        ParserResult {
            reasoning_text: join(reasoning),
            normal_text: join(normal),
        }
    }

    /// Single-span batch extraction: the original Granite behavior. Strips the
    /// first thought-process marker and splits once on the first response
    /// marker. Kept verbatim so single-span inputs stay byte-identical.
    fn parse_single_span(&self, text: &str) -> ParserResult {
        let think_start_token = self
            .think_start_tokens
            .iter()
            .find(|&token| text.contains(token))
            .unwrap_or_else(|| self.think_start_tokens.first().unwrap());

        let think_end_token = self
            .think_end_tokens
            .iter()
            .find(|&token| text.contains(token))
            .unwrap_or_else(|| self.think_end_tokens.first().unwrap());
        // Implement parsing logic specific to Granite format
        let in_reasoning = self.in_reasoning
            || self
                .think_start_tokens
                .iter()
                .any(|token| text.contains(token));
        if !in_reasoning {
            return ParserResult {
                normal_text: text.to_string(),
                reasoning_text: String::new(),
            };
        }

        // The text is considered to be in a reasoning block.
        let processed_text = text.replacen(think_start_token, "", 1).trim().to_string();

        if !processed_text.contains(think_end_token) {
            // Assume reasoning was truncated before `think_end_token`
            return ParserResult {
                normal_text: String::new(),
                reasoning_text: processed_text,
            };
        }

        // Extract reasoning content
        let splits: Vec<&str> = processed_text.splitn(2, think_end_token).collect();
        let reasoning_text = splits.first().unwrap_or(&"").to_string();
        let normal_text = splits
            .get(1)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        ParserResult {
            normal_text,
            reasoning_text,
        }
    }
}

impl ReasoningParser for GraniteReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _: &[u32]) -> ParserResult {
        let result = self.parse_single_span(text);
        // If a marker survived in normal_text, the input has multiple spans or
        // a dangling response marker; re-parse so no markup leaks downstream.
        if Self::find_first(&result.normal_text, &self.think_start_tokens).is_some()
            || Self::find_first(&result.normal_text, &self.think_end_tokens).is_some()
        {
            return self.parse_all_spans(text);
        }
        result
    }

    fn parse_reasoning_streaming_incremental(&mut self, text: &str, _: &[u32]) -> ParserResult {
        // Implement streaming parsing logic specific to Granite format

        // Incrementally parse the streaming text
        self.buffer.push_str(text);
        let mut current_text = self.buffer.to_string();
        // If the current text is a prefix of the think token, keep buffering

        for think_start_token in &self.think_start_tokens {
            if think_start_token.starts_with(&current_text)
                && think_start_token.as_str() != current_text.as_str()
            {
                return ParserResult {
                    normal_text: String::new(),
                    reasoning_text: String::new(),
                };
            }
        }
        for think_end_token in &self.think_end_tokens {
            if think_end_token.starts_with(&current_text)
                && think_end_token.as_str() != current_text.as_str()
            {
                return ParserResult {
                    normal_text: String::new(),
                    reasoning_text: String::new(),
                };
            }
        }

        let think_start_token = self
            .think_start_tokens
            .iter()
            .find(|&token| current_text.contains(token))
            .unwrap_or_else(|| self.think_start_tokens.first().unwrap());

        let think_end_token = self
            .think_end_tokens
            .iter()
            .find(|&token| current_text.contains(token))
            .unwrap_or_else(|| self.think_end_tokens.first().unwrap());

        if !self.stripped_think_start && current_text.contains(think_start_token) {
            current_text = current_text.replacen(think_start_token, "", 1);
            self.buffer = current_text.to_string();
            self.stripped_think_start = true;
            self.in_reasoning = true;
        }
        // Handle end of reasoning block
        let mut think_end_idx = current_text.len();
        if self.in_reasoning {
            think_end_idx = current_text
                .find(think_end_token)
                .unwrap_or(current_text.len());
        }
        if self.in_reasoning && think_end_idx < current_text.len() {
            let reasoning_text = &current_text[..think_end_idx];
            self.buffer.clear();
            self.in_reasoning = false;
            // Allow a later thought-process marker to open a fresh span, so a
            // multi-span stream is parsed in full instead of leaking the
            // second span's markers into normal_text.
            self.stripped_think_start = false;
            let start_idx = think_end_idx + think_end_token.len();
            let normal_text = if start_idx < current_text.len() {
                &current_text[start_idx..]
            } else {
                ""
            };
            return ParserResult {
                normal_text: normal_text.to_string(),
                reasoning_text: reasoning_text.to_string(),
            };
        }
        // Continue with reasoning content
        if self.in_reasoning {
            // Hold back a trailing partial end marker (e.g. an end token split
            // across chunks) so it is matched once complete rather than leaked.
            let hold = self.partial_end_suffix_len(&current_text);
            let split = current_text.len() - hold;
            let reasoning_text = current_text[..split].to_string();
            self.buffer = current_text[split..].to_string();
            ParserResult {
                normal_text: String::new(),
                reasoning_text,
            }
        } else {
            // If we're not in a reasoning block return as normal text
            let normal_text = current_text;
            self.buffer.clear();
            ParserResult {
                normal_text,
                reasoning_text: String::new(),
            }
        }
    }

    fn finish_reasoning_stream(&mut self) -> ParserResult {
        if self.buffer.is_empty() {
            return ParserResult::default();
        }

        let buffered = std::mem::take(&mut self.buffer);
        if self.in_reasoning {
            ParserResult {
                normal_text: String::new(),
                reasoning_text: buffered,
            }
        } else {
            ParserResult {
                normal_text: buffered,
                reasoning_text: String::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] // helper
    fn test_basic_reasoning_detection() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: I need to think about this. Here's my response: The answer is 42.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.reasoning_text, " I need to think about this. ");
        assert_eq!(result.normal_text, " The answer is 42.");
    }

    #[test] // helper, TOOLCALLING.fmt.3
    fn test_alternative_start_token() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here is my thought process: Different thinking here. Here is my response: Final answer.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.reasoning_text, " Different thinking here. ");
        assert_eq!(result.normal_text, " Final answer.");
    }

    #[test] // REASONING.stream.3.a, helper
    fn test_streaming_partial_tokens() {
        let mut parser = GraniteReasoningParser::new();

        // Test partial start token
        let result1 = parser.parse_reasoning_streaming_incremental("Here's", &[]);
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "");

        // Complete the start token and add reasoning
        let result2 = parser
            .parse_reasoning_streaming_incremental(" my thought process: This is reasoning", &[]);
        assert_eq!(result2.reasoning_text, " This is reasoning");
        assert_eq!(result2.normal_text, "");
    }

    #[test] // REASONING.stream.3.b, helper
    fn test_streaming_partial_end_tokens() {
        let mut parser = GraniteReasoningParser::new();

        // Start reasoning
        parser
            .parse_reasoning_streaming_incremental("Here's my thought process: Thinking... ", &[]);

        parser.parse_reasoning_streaming_incremental("Here", &[]);

        // Partial end token should buffer
        let result = parser.parse_reasoning_streaming_incremental("'s my", &[]);
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "");

        // Complete end token
        let result2 = parser.parse_reasoning_streaming_incremental(" response: Done!", &[]);
        assert_eq!(result2.reasoning_text, "");
        assert_eq!(result2.normal_text, " Done!");
    }

    #[test] // REASONING.batch.1.b, helper
    fn test_no_reasoning_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "This is just normal text without any special tokens.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.normal_text, text);
        assert_eq!(result.reasoning_text, "");
    }

    #[test] // REASONING.batch.5, helper
    fn test_only_start_token_no_end() {
        let mut parser = GraniteReasoningParser::new();

        let result1 = parser.parse_reasoning_streaming_incremental(
            "Here's my thought process: This is reasoning content",
            &[],
        );
        assert_eq!(result1.reasoning_text, " This is reasoning content");
        assert_eq!(result1.normal_text, "");

        // More reasoning content without end token
        let result2 = parser.parse_reasoning_streaming_incremental(" and more thinking", &[]);
        assert_eq!(result2.reasoning_text, " and more thinking");
        assert_eq!(result2.normal_text, "");
    }

    #[test] // REASONING.batch.2.e
    fn test_empty_reasoning_block() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process:Here's my response: Direct answer.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, " Direct answer.");
    }

    #[test] // REASONING.batch.2.f, TOOLCALLING.fmt.2
    fn test_reasoning_with_whitespace() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process:   \n  Indented reasoning  \n  Here's my response:   Final result  ";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(result.reasoning_text, "   \n  Indented reasoning  \n  ");
        assert_eq!(result.normal_text, "   Final result  ");
    }

    #[test] // TOOLCALLING.fmt.1 — token case sensitivity
    fn test_case_sensitive_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "here's my thought process: lowercase. here's my response: answer.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        // Should not detect lowercase tokens
        assert_eq!(result.normal_text, text);
        assert_eq!(result.reasoning_text, "");
    }

    #[test] // REASONING.batch.2.f
    fn test_nested_or_repeated_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: I think Here's my thought process: is confusing. Here's my response: Done.";
        let result = parser.parse_reasoning_streaming_incremental(text, &[]);

        assert_eq!(
            result.reasoning_text,
            " I think Here's my thought process: is confusing. "
        );
        assert_eq!(result.normal_text, " Done.");
    }

    #[test] // REASONING.batch.2.c
    fn test_detect_and_parse_reasoning_basic() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: I need to analyze this problem. Here's my response: The solution is clear.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "I need to analyze this problem. ");
        assert_eq!(result.normal_text, "The solution is clear.");
    }

    #[test] // REASONING.batch.2.c, TOOLCALLING.fmt.3
    fn test_detect_and_parse_reasoning_alternative_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here is my thought process: Different reasoning approach. Here is my response: Final conclusion.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "Different reasoning approach. ");
        assert_eq!(result.normal_text, "Final conclusion.");
    }

    #[test] // REASONING.batch.1.b
    fn test_detect_and_parse_reasoning_no_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "This is just normal text without special markers.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.normal_text, text);
        assert_eq!(result.reasoning_text, "");
    }

    #[test] // REASONING.batch.5
    fn test_detect_and_parse_reasoning_only_start_token() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: This reasoning has no end marker.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "This reasoning has no end marker.");
        assert_eq!(result.normal_text, "");
    }

    #[test] // REASONING.batch.2.e
    fn test_detect_and_parse_reasoning_empty_sections() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process:Here's my response:";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "");
    }

    #[test] // REASONING.batch.2.f, TOOLCALLING.fmt.2
    fn test_detect_and_parse_reasoning_whitespace_handling() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process:   \n\tSpaced reasoning\n   Here's my response:  \n  Spaced response\n";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "Spaced reasoning\n   ");
        assert_eq!(result.normal_text, "Spaced response");
    }

    #[test] // REASONING.batch.2.f, TOOLCALLING.fmt.3
    fn test_detect_and_parse_reasoning_multiple_end_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: Thinking about Here's my response: in the middle. Here's my response: Real end.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        // The trailing dangling response marker is consumed rather than left
        // in normal_text (no markup leak).
        assert_eq!(result.reasoning_text, "Thinking about");
        assert_eq!(result.normal_text, "in the middle. Real end.");
    }

    #[test] // REASONING.batch.4 — dangling response marker without a thought marker
    fn test_detect_and_parse_reasoning_dangling_response_marker() {
        let mut parser = GraniteReasoningParser::new();
        let text = "normal Here is my response: answer";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "normal answer");
    }

    #[test] // REASONING.batch.6.a — multiple thought-process / response spans
    fn test_detect_and_parse_reasoning_multiple_spans() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here is my thought process: first Here is my response: middle Here is my thought process: second Here is my response: done";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "first second");
        assert_eq!(result.normal_text, "middle done");
    }

    // Drive a chunk sequence through the streaming parser and concatenate the
    // emitted reasoning / normal text (including the final flush).
    fn run_stream(chunks: &[&str]) -> (String, String) {
        let mut parser = GraniteReasoningParser::new();
        let (mut reasoning, mut normal) = (String::new(), String::new());
        for chunk in chunks {
            let r = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        let r = parser.finish_reasoning_stream();
        reasoning.push_str(&r.reasoning_text);
        normal.push_str(&r.normal_text);
        (reasoning, normal)
    }

    #[test] // REASONING.stream.2.b — multiple spans across chunks
    fn test_streaming_multiple_spans() {
        let (reasoning, normal) = run_stream(&[
            "Here is my thought process: first Here is my response:",
            " middle ",
            "Here is my thought process: second Here is my response: done",
        ]);

        assert_eq!(reasoning, " first  second ");
        assert_eq!(normal, " middle  done");
    }

    #[test] // REASONING.stream.3.b — end marker split across chunks
    fn test_streaming_split_end_marker() {
        let (reasoning, normal) = run_stream(&[
            "Here is my thought process: thinking Here is my res",
            "ponse: answer",
        ]);

        // The split `Here is my response:` is buffered until complete, so it
        // never leaks into reasoning_text.
        assert_eq!(reasoning, " thinking ");
        assert_eq!(normal, " answer");
    }

    #[test] // TOOLCALLING.fmt.1
    fn test_detect_and_parse_reasoning_case_sensitivity() {
        let mut parser = GraniteReasoningParser::new();
        let text =
            "here's my thought process: lowercase tokens. here's my response: should not work.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.normal_text, text);
        assert_eq!(result.reasoning_text, "");
    }

    #[test] // REASONING.batch.2.c, TOOLCALLING.fmt.3
    fn test_detect_and_parse_reasoning_mixed_tokens() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: First reasoning. Here is my response: Mixed token response.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(result.reasoning_text, "First reasoning. ");
        assert_eq!(result.normal_text, "Mixed token response.");
    }

    #[test] // REASONING.batch.2.f
    fn test_detect_and_parse_reasoning_long_content() {
        let mut parser = GraniteReasoningParser::new();
        let text = "Here's my thought process: This is a very long reasoning section that spans multiple sentences. I need to consider various factors. The analysis requires careful thought. Here's my response: After all that thinking, here is the comprehensive answer with multiple parts and detailed explanation.";
        let result = parser.detect_and_parse_reasoning(text, &[]);

        assert_eq!(
            result.reasoning_text,
            "This is a very long reasoning section that spans multiple sentences. I need to consider various factors. The analysis requires careful thought. "
        );
        assert_eq!(
            result.normal_text,
            "After all that thinking, here is the comprehensive answer with multiple parts and detailed explanation."
        );
    }
}
