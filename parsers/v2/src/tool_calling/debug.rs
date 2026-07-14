// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ENV-gated stderr debug tracing for the v2 stream parsers.
//!
//! Set `DYNAMO_PARSERS_DEBUG=1` to make the crate print a marker to stderr when
//! a parser is created via `create_tool_parser_for_family` and whenever a
//! parser emits tool-call updates. This lets a host process (for example
//! vLLM's experimental Rust frontend) confirm that a Dynamo parser was
//! selected and is actually parsing, without installing a `tracing` subscriber.
//!
//! The toggle is read once and defaults to off, so the normal path pays no cost.

use std::io::Write;
use std::sync::OnceLock;

use super::traits::{Result, Tool, ToolParseResult, ToolParser};

/// Env var that turns on `dynamo-parsers-v2` stderr debug output.
pub const DEBUG_ENV: &str = "DYNAMO_PARSERS_DEBUG";

/// Write one debug line to stderr, discarding any I/O error. Uses fallible
/// `writeln!` rather than `eprintln!` so debug output can never panic the host
/// (e.g. when stderr is closed and a write returns `EPIPE`).
fn emit(args: std::fmt::Arguments<'_>) {
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "[dynamo-parsers-v2] {args}");
}

/// Returns true for the same truthy strings Dynamo uses: "1", "true", "on", "yes"
/// (case-insensitive). Everything else, including an unset var, is false.
pub fn is_truthy(val: &str) -> bool {
    matches!(val.to_lowercase().as_str(), "1" | "true" | "on" | "yes")
}

/// Returns true if the named environment variable is set to a truthy value.
pub fn env_is_truthy(env: &str) -> bool {
    match std::env::var(env) {
        Ok(val) => is_truthy(val.as_str()),
        Err(_) => false,
    }
}

/// Whether debug output is enabled. Read once from the environment.
pub fn debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_is_truthy(DEBUG_ENV))
}

/// Wraps a family stream parser and logs creation plus emitted tool calls to
/// stderr. Constructed only when [`debug_enabled`] is true, so it adds no
/// overhead on the default path. One wrapper at the dispatch choke point
/// instruments every family, so there is no per-parser duplication.
pub(super) struct DebugToolParser {
    family: String,
    inner: Box<dyn ToolParser>,
}

impl DebugToolParser {
    pub(super) fn wrap(family: &str, inner: Box<dyn ToolParser>) -> Box<dyn ToolParser> {
        emit(format_args!("family={family} created"));
        Box::new(Self {
            family: family.to_string(),
            inner,
        })
    }

    fn log(&self, method: &str, result: &ToolParseResult) {
        if result.calls.is_empty() {
            return;
        }
        let names: Vec<&str> = result
            .calls
            .iter()
            .filter_map(|c| c.name.as_deref())
            .collect();
        emit(format_args!(
            "family={} {} emitted {} call update(s) names={:?}",
            self.family,
            method,
            result.calls.len(),
            names
        ));
    }
}

impl ToolParser for DebugToolParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        anyhow::bail!("DebugToolParser wraps an existing parser; use create_tool_parser_for_family")
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn prefers_tokens(&self) -> bool {
        self.inner.prefers_tokens()
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        let result = self.inner.push(chunk)?;
        self.log("push", &result);
        Ok(result)
    }

    fn push_tokens(&mut self, ids: &[u32]) -> Result<ToolParseResult> {
        let result = self.inner.push_tokens(ids)?;
        self.log("push_tokens", &result);
        Ok(result)
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        let result = self.inner.finish()?;
        self.log("finish", &result);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::super::traits::{Tool, ToolParser};
    use super::DebugToolParser;
    use crate::tool_calling::qwen3_coder::Qwen3CoderToolStreamParser;

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    // The wrapper must be a transparent pass-through: same parsed result as the
    // wrapped parser, only with an added stderr side effect.
    #[test]
    fn wrapper_is_transparent_passthrough() {
        let tools = weather_tools();
        let inner = Qwen3CoderToolStreamParser::create(&tools).unwrap();
        let mut wrapped = DebugToolParser::wrap("qwen3_coder", inner);

        let result = wrapped
            .parse_complete(
                "<tool_call> <function=get_weather> \
<parameter=location> NYC </parameter> </function> </tool_call>",
            )
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, r#"{"location":"NYC"}"#);
    }
}
