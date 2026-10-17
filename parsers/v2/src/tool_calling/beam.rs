// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy tool-only projection of the Beam unified event stream (NOT GPT-OSS Harmony).
//!
//! This is a thin view over the SAME [`UnifiedParser`] the registry builds through
//! [`crate::unified::beam::beam_unified`]: [`ToolParseResult::from_deltas`] folds
//! reasoning and text deltas together into `normal_text` and keeps the calls. It owns
//! no buffer, grammar, policy or index of its own, so the two API surfaces cannot drift
//! on holdback, recovery or call numbering.

use crate::tool_calling::traits::{Tool, ToolParseResult, ToolParser};
use crate::unified::{UnifiedParser, UnifiedParserExt, beam};

/// Stream parser for Beam tool calls.
pub struct BeamToolStreamParser {
    parser: Box<dyn UnifiedParser>,
}

impl BeamToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            parser: beam::beam_unified(tools),
        }
    }
}

impl ToolParser for BeamToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.parser.push(chunk)?))
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.parser.finish()?.events))
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.parser.tool_call_id(tool_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_calling::traits::ToolCallDelta;
    use crate::unified::UnifiedParserEvent;
    use serde_json::json;

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "weather".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {"location": {"type": "string"}}}),
            strict: None,
        }]
    }

    const INPUT: &str = concat!(
        "<|start|>assistant<|channel|>analysis<|message|>Look it up.<|end|>",
        "<|start|>assistant to=weather<|channel|>analysis <|constrain|>xml<|message|>",
        "<rfl:parameter name=\"location\">\nLondon\n</rfl:parameter><|end|>",
        "<|start|>assistant<|channel|>final<|message|>Then 日本 \"<|call|>\"<|return|>",
        "<|start|>assistant to=weather<|channel|>analysis <|constrain|>xml<|message|>",
        "<rfl:parameter name=\"location\">Tōkyō</rfl:parameter><|call|>",
    );

    /// The folded view of one stream: `normal_text`, and per call `(tool_index, name,
    /// argument bytes)` concatenated delta by delta exactly as a streaming client
    /// would, with indices preserved and argument bytes never re-serialized.
    #[derive(Debug, PartialEq, Eq)]
    struct Projected {
        text: String,
        calls: Vec<(usize, String, String)>,
    }

    impl Projected {
        fn fold(&mut self, result: ToolParseResult) {
            self.text.push_str(&result.normal_text);
            for call in result.calls {
                self.fold_call(call);
            }
        }

        fn fold_call(&mut self, call: ToolCallDelta) {
            match self
                .calls
                .iter_mut()
                .find(|(index, _, _)| *index == call.tool_index)
            {
                Some((_, name, arguments)) => {
                    if let Some(n) = call.name {
                        name.push_str(&n);
                    }
                    arguments.push_str(&call.arguments);
                }
                None => self.calls.push((
                    call.tool_index,
                    call.name.unwrap_or_default(),
                    call.arguments,
                )),
            }
        }
    }

    /// Through the legacy surface, chunk by chunk.
    fn legacy(chunks: &[&str]) -> Projected {
        let mut parser = BeamToolStreamParser::new(&tools());
        let mut out = Projected {
            text: String::new(),
            calls: Vec::new(),
        };
        for chunk in chunks {
            out.fold(parser.push(chunk).expect("push"));
        }
        out.fold(parser.finish().expect("finish"));
        out
    }

    /// Through the public unified surface, projecting each push's raw deltas the
    /// same way `from_deltas` does, so the comparison is delta-for-delta.
    fn unified(chunks: &[&str]) -> Projected {
        let mut parser = beam::beam_unified(&tools());
        let mut out = Projected {
            text: String::new(),
            calls: Vec::new(),
        };
        let mut project = |deltas: Vec<UnifiedParserEvent>| {
            for delta in deltas {
                match delta {
                    UnifiedParserEvent::Reasoning(text) | UnifiedParserEvent::Text(text) => {
                        out.text.push_str(&text)
                    }
                    UnifiedParserEvent::ToolCall(call) => out.fold_call(call),
                }
            }
        };
        for chunk in chunks {
            project(parser.push(chunk).expect("push"));
        }
        project(parser.finish().expect("finish").events);
        out
    }

    fn check(got: &Projected) {
        // The lossy legacy view folds reasoning and text together; markup never leaks,
        // and a quoted call marker inside the `final` body stays literal.
        assert_eq!(got.text, "Look it up.Then 日本 \"<|call|>\"");
        let parsed: Vec<(usize, &str, serde_json::Value)> = got
            .calls
            .iter()
            .map(|(index, name, arguments)| {
                (
                    *index,
                    name.as_str(),
                    serde_json::from_str(arguments).expect("argument bytes are JSON"),
                )
            })
            .collect();
        assert_eq!(
            parsed,
            vec![
                (0, "weather", json!({"location": "London"})),
                (1, "weather", json!({"location": "Tōkyō"})),
            ]
        );
    }

    #[test]
    fn legacy_projection_agrees_with_unified_on_every_split() {
        let want = unified(&[INPUT]);
        check(&want);
        assert_eq!(legacy(&[INPUT]), want);
        for at in (0..=INPUT.len()).filter(|&at| INPUT.is_char_boundary(at)) {
            let chunks = [&INPUT[..at], &INPUT[at..]];
            let split = unified(&chunks);
            check(&split);
            assert_eq!(legacy(&chunks), split, "split at {at}");
        }
        let chars: Vec<String> = INPUT.chars().map(String::from).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        let per_char = unified(&refs);
        check(&per_char);
        assert_eq!(legacy(&refs), per_char);
    }

    #[test]
    fn legacy_projection_has_no_model_call_ids_and_preserves_special_tokens() {
        let mut parser = BeamToolStreamParser::new(&tools());
        parser.push(INPUT).unwrap();
        parser.finish().unwrap();
        assert!(parser.preserve_special_tokens());
        assert_eq!(parser.tool_call_id(0), None);
    }
}
