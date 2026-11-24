// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::OAIChatLikeRequest;
use anyhow::Result;
use minijinja::{Value, value::merge_maps};

#[derive(Debug, Clone, Copy)]
pub(super) enum EffortTemplate {
    GptOss,
    Glm,
    Beam,
    MuseGlimmer,
}

impl EffortTemplate {
    pub(super) fn detect(source: &str) -> Option<Self> {
        if source.contains("emit_default_thinking_effort")
            && super::formatters::template_references_variable(
                source,
                "emit_default_thinking_effort",
            )
        {
            return None;
        }
        if source.contains("{%- macro build_system_message() -%}")
            && source.contains(r#"{%- set reasoning_effort = "medium" %}"#)
            && source.contains(GPT_OSS_LINE)
        {
            Some(Self::GptOss)
        } else if source.starts_with("[gMASK]<sop>")
            && source.contains("<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}")
            && [GLM52_DEFAULT, GLM53_DEFAULT].iter().any(|tag| source.contains(tag))
        {
            Some(Self::Glm)
        } else if source.contains("{%- macro build_system_message(builtin_tool_definitions) -%}")
            && source.contains("{%- set reasoning_effort = 512 %}")
            && source.contains(BEAM_LINE)
        {
            Some(Self::Beam)
        } else if source.contains("{%- macro render_reasoning() -%}")
            && source.contains("set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high'")
            && source.contains(MUSE_CALL)
        {
            Some(Self::MuseGlimmer)
        } else {
            None
        }
    }

    // Called at executable Jinja tag boundaries, so quoted examples and raw blocks stay literal.
    pub(super) fn guarded_tag(self, source: &str) -> Option<(usize, String)> {
        let (block, end) = match self {
            Self::GptOss if source.starts_with(GPT_OSS_LINE) => (GPT_OSS_LINE, "{% endif %}"),
            Self::Beam if source.starts_with(BEAM_LINE) => (BEAM_LINE, "{% endif %}"),
            Self::Glm => {
                let tag = [GLM52_DEFAULT, GLM53_DEFAULT]
                    .into_iter()
                    .find(|tag| source.starts_with(tag))?;
                let replacement = tag.replace(
                    "else 'max'",
                    "else ('max' if emit_default_thinking_effort is not false else none)",
                );
                return Some((tag.len(), replacement));
            }
            Self::MuseGlimmer => {
                let rest = source.strip_prefix(MUSE_SEPARATOR)?.trim_start();
                let after = rest.strip_prefix(MUSE_CALL)?;
                (&source[..source.len() - after.len()], "{% endif -%}")
            }
            _ => return None,
        };
        Some((
            block.len(),
            format!("{{%- if emit_default_thinking_effort is not false %}}{block}{end}"),
        ))
    }

    pub(super) fn context(self, req: &dyn OAIChatLikeRequest, ctx: Value) -> Result<Value> {
        let args = req.chat_template_args();
        if crate::emit_default_thinking_effort(args)? {
            return Ok(ctx);
        }
        let key = match self {
            Self::MuseGlimmer => "reasoning_strength",
            _ => "reasoning_effort",
        };
        let native_effort = ctx.get_attr(key)?;
        let has_explicit_effort = match self {
            Self::MuseGlimmer => native_effort.is_true(),
            _ => !native_effort.is_undefined() && !native_effort.is_none(),
        };
        let explicit = has_explicit_effort
            .then_some(native_effort)
            .or_else(|| req.reasoning_effort())
            .or_else(|| args?.get("reasoning_effort").map(Value::from_serialize));
        if let Some(effort) = explicit.filter(|v| !v.is_none() && !v.is_undefined()) {
            let overrides = Value::from_iter([
                ("emit_default_thinking_effort", Value::from(true)),
                (key, effort),
            ]);
            return Ok(merge_maps([ctx, overrides]));
        }
        Ok(ctx)
    }
}

const GPT_OSS_LINE: &str = r#"{{- "Reasoning: " + reasoning_effort + "\n\n" }}"#;
const BEAM_LINE: &str = r#"{{- "\n\nReasoning effort: " + (reasoning_effort | string) }}"#;
const GLM52_DEFAULT: &str = "{%- set effective_reasoning_effort = 'high' if reasoning_effort is defined and reasoning_effort == 'high' else 'max' -%}";
const GLM53_DEFAULT: &str = "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}";
const MUSE_SEPARATOR: &str = r"{{- '\n\n' -}}";
const MUSE_CALL: &str = "{{- render_reasoning() -}}";
