// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-4.7/5.x XML tool calls and their legacy ToolParser projection.
//!
//! GLM's outer `<tool_call>` block is itself the invoke: the function name is
//! followed directly by `<arg_key>`/`<arg_value>` pairs. `WrappedBlockScanner`
//! owns all buffering, recovery, and chunk-boundary handling; this module only
//! supplies the grammar and value emitter.

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
    reorder_arguments,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};
use crate::tool_calling::v1core::{Glm47ParserConfig, ToolDefinition, try_tool_call_parse_glm47};

pub(crate) const BLOCK_START: &str = "<tool_call>";
pub(crate) const BLOCK_END: &str = "</tool_call>";
const ARG_KEY_START: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE_START: &str = "<arg_value>";

const ORPHAN_ANCHORS: [&str; 4] = [BLOCK_END, ARG_KEY_START, ARG_KEY_END, ARG_VALUE_START];

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "glm47",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        // The block opener is also the invoke opener. The scanner consumes the
        // opener as block markup and passes the body plus closer to the emitter.
        invoke_start: BLOCK_START.to_string(),
        invoke_end: BLOCK_END.to_string(),
        orphan_markers: ORPHAN_ANCHORS
            .iter()
            .map(|marker| (*marker).to_string())
            .collect(),
        holdback_markers: [
            BLOCK_START,
            BLOCK_END,
            ARG_KEY_START,
            ARG_KEY_END,
            ARG_VALUE_START,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        bare_recovery_latch: BareRecoveryLatch::Clear,
        invoke_latch: InvokeLatch::IfEmitted,
        bare_invoke_start: Some(find_bare_invoke_start),
        bare_invoke_holdback: Some(trailing_holdback_len),
        preserve_special_tokens: true,
        ..Default::default()
    }
}

pub(crate) struct Glm47Emitter {
    config: Glm47ParserConfig,
    tools: Vec<ToolDefinition>,
}

impl InvokeEmitter for Glm47Emitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        // The scanner has already found the invoke boundary. Re-wrap only for
        // the v1 value typer; it must not rediscover a boundary in the payload.
        let wrapped = format!("{BLOCK_START}{invoke}");
        let (calls, _content) =
            try_tool_call_parse_glm47(&wrapped, &self.config, Some(&self.tools))?;
        let Some(call) = calls.into_iter().next() else {
            return Ok(None);
        };
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(call.function.name),
            arguments: reorder_arguments(&call.function.arguments, &source_arg_key_order(invoke)),
            complete: true,
        }))
    }
}

/// The one GLM scanner construction site shared by native UnifiedParser and
/// the legacy ToolParser compatibility surface.
pub(crate) fn glm47_scanner(tools: &[Tool]) -> WrappedBlockScanner<Glm47Emitter> {
    WrappedBlockScanner::new(
        spec(),
        Glm47Emitter {
            config: Glm47ParserConfig::default(),
            tools: tools.iter().map(ToolDefinition::from).collect(),
        },
    )
}

/// Compatibility projection for callers that still use the tool-only trait.
pub struct Glm47ToolStreamParser {
    scanner: WrappedBlockScanner<Glm47Emitter>,
}

impl Glm47ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: glm47_scanner(tools),
        }
    }
}

impl ToolParser for Glm47ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.scanner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

fn find_bare_invoke_start(text: &str) -> Option<usize> {
    let marker_idx = ORPHAN_ANCHORS
        .iter()
        .filter(|marker| **marker != BLOCK_END)
        .filter_map(|marker| text.find(marker))
        .min()?;
    if text
        .find(BLOCK_START)
        .is_some_and(|wrapped| wrapped < marker_idx)
    {
        return None;
    }
    let before = text[..marker_idx].trim_end();
    let name_start = before
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);
    let candidate = before[name_start..].trim();
    (!candidate.is_empty()
        && candidate
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')))
    .then_some(name_start)
}

fn trailing_holdback_len(text: &str) -> usize {
    let mut marker_keep = 0;
    let mut orphan_partial = false;
    for marker in [
        BLOCK_START,
        BLOCK_END,
        ARG_KEY_START,
        ARG_KEY_END,
        ARG_VALUE_START,
    ] {
        let is_orphan = marker != BLOCK_START;
        for length in 1..marker.len() {
            if text.ends_with(&marker[..length]) {
                if length > marker_keep {
                    marker_keep = length;
                    orphan_partial = is_orphan;
                } else if length == marker_keep && is_orphan {
                    orphan_partial = true;
                }
            }
        }
    }
    if !orphan_partial && marker_keep != 0 {
        return marker_keep;
    }
    let identifier_end = text.len() - marker_keep;
    let name_start = text[..identifier_end]
        .char_indices()
        .rev()
        .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
        .last()
        .map(|(idx, _)| idx)
        .unwrap_or(identifier_end);
    text.len() - name_start
}

fn source_arg_key_order(block: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = block[cursor..].find(ARG_KEY_START) {
        let start = cursor + relative + ARG_KEY_START.len();
        let Some(end) = block[start..].find(ARG_KEY_END) else {
            break;
        };
        let name = block[start..start + end].trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
        cursor = start + end + ARG_KEY_END.len();
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::UnifiedParserExt;

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn legacy(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = Glm47ToolStreamParser::new(tools);
        let mut output = ToolParseResult::default();
        for chunk in chunks {
            output.append(parser.push(chunk).expect("push"));
        }
        output.append(parser.finish().expect("finish"));
        output
    }

    #[test]
    fn native_legacy_projection_parses_glm_xml() {
        let output = legacy(
            &tools(),
            &[
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            ],
        );
        let calls = output.coalesce_calls();
        assert_eq!(calls.normal_text, "");
        assert_eq!(calls.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls.calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn legacy_projection_is_split_invariant() {
        let input = "before <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> after";
        let whole = legacy(&tools(), &[input]).coalesce_calls();
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            let split_output =
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls();
            assert_eq!(split_output, whole, "split at {split}");
        }
    }

    #[test]
    fn malformed_and_eof_tool_markup_is_dropped() {
        let output = legacy(&tools(), &["visible <tool_call>get_weather<arg_key>city"]);
        assert_eq!(output.normal_text, "visible ");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn bare_name_is_held_until_its_argument_marker_arrives() {
        let mut parser = Glm47ToolStreamParser::new(&tools());
        assert!(
            parser
                .push("get_weather")
                .expect("push")
                .normal_text
                .is_empty()
        );
        let mut output = parser
            .push("<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>")
            .expect("push");
        output.append(parser.finish().expect("finish"));
        let calls = output.coalesce_calls();
        assert_eq!(calls.normal_text, "");
        assert_eq!(calls.calls.len(), 1);
    }

    #[test]
    fn legacy_and_unified_share_the_same_events() {
        let input = "<think>look</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call><think>answer</think>Done";
        let legacy_output = legacy(&tools(), &[input]).coalesce_calls();
        let mut parser =
            crate::unified::create_unified_parser_for_family("glm47", &tools()).unwrap();
        let unified = parser.parse_complete(input).unwrap();
        assert_eq!(legacy_output.calls.len(), 1);
        assert_eq!(unified.len(), 4);
    }
}
