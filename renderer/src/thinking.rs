// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request policy shared by template preparation and parser gating.
use std::collections::HashMap;

type TemplateArgs = HashMap<String, serde_json::Value>;

#[derive(Clone, Copy)]
enum ReasoningMode {
    Fixed(bool),
    BooleanDefault(bool),
    TemplateModeDefault(bool),
    Effort,
    Adaptive,
}

impl ReasoningMode {
    fn enabled(self, args: Option<&TemplateArgs>) -> bool {
        let value = |key| {
            args.and_then(|args| args.get(key))
                .and_then(|value| value.as_str())
        };
        match self {
            Self::Fixed(enabled) => enabled,
            Self::BooleanDefault(default) => {
                crate::thinking_bool_from_args(args).unwrap_or(default)
            }
            Self::TemplateModeDefault(default) => template_mode_reasoning_enabled(args, default),
            Self::Effort => value("reasoning_effort").is_some_and(|effort| effort != "none"),
            Self::Adaptive => value("thinking_mode") != Some("disabled"),
        }
    }
}

/// Evaluate thinking aliases and the template's chat/thinking mode in priority order.
pub fn template_mode_reasoning_enabled(args: Option<&TemplateArgs>, default: bool) -> bool {
    crate::thinking_bool_from_args(args).unwrap_or_else(|| {
        match args
            .and_then(|args| args.get("thinking_mode"))
            .and_then(serde_json::Value::as_str)
        {
            Some("chat") => false,
            Some("thinking") => true,
            _ => default,
        }
    })
}

fn reasoning_modes(selector: Option<&str>) -> (ReasoningMode, ReasoningMode) {
    use ReasoningMode::{Adaptive, BooleanDefault, Effort, Fixed, TemplateModeDefault};
    let common = match selector {
        Some(
            "kimi_k25" | "kimi_k3" | "kimi-k3" | "nemotron_nano" | "nemotron3" | "nemotron_v3",
        ) => BooleanDefault(true),
        Some("deepseek_v3" | "deepseek_v3_1") => TemplateModeDefault(false),
        Some(
            "deepseek_v3_2" | "deepseek_v4" | "deepseek-v4" | "deepseekv4" | "deepseek_v41"
            | "minimax_m2",
        ) => TemplateModeDefault(true),
        Some("gemma4" | "gemma-4") => BooleanDefault(false),
        Some("mistral") => Effort,
        Some("minimax_m3" | "minimax-m3") => Adaptive,
        Some("qwen3" | "glm45") => return (Fixed(true), BooleanDefault(true)),
        Some("deepseek_r1") => return (TemplateModeDefault(true), Fixed(true)),
        Some("muse_spark_open" | "muse_spark" | "step3" | "gpt_oss" | "kimi") => Fixed(true),
        _ => return (Fixed(true), Fixed(false)),
    };
    (common, common)
}

/// Whether the request disables the legacy reasoning parser.
pub fn legacy_reasoning_disabled(selector: Option<&str>, args: Option<&TemplateArgs>) -> bool {
    !reasoning_modes(selector).0.enabled(args)
}

/// Whether the SGLang renderer contract permits reasoning before computing native guidance.
pub fn sglang_reasoning_enabled(selector: Option<&str>, args: Option<&TemplateArgs>) -> bool {
    reasoning_modes(selector).1.enabled(args)
}

/// Model rules applied before rendering; independent of parser generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThinkingPolicy {
    pub adaptive: bool,
    pub named_tool_disables: bool,
    pub default_enabled: Option<bool>,
}

impl ThinkingPolicy {
    /// Resolve template behavior from the configured tool and reasoning selectors.
    pub fn for_selectors(tool: Option<&str>, reasoning: Option<&str>) -> Self {
        Self {
            adaptive: matches!(reasoning, Some("minimax_m3" | "minimax-m3"))
                || matches!(
                    tool,
                    Some("minimax_m3" | "minimax-m3" | "minimax_m3_nom" | "minimax-m3-nom")
                ),
            named_tool_disables: matches!(tool, Some("kimi_k3" | "kimi-k3")),
            default_enabled: matches!(reasoning, Some("kimi_k25" | "kimi_k3" | "kimi-k3"))
                .then_some(true),
        }
    }
}

/// Normalize adaptive template arguments after the host resolves public boolean aliases.
pub fn normalize_adaptive_thinking(
    chat_template_args: &mut Option<HashMap<String, serde_json::Value>>,
    normalized: Option<bool>,
    thinking_control_from_client: bool,
    constrained_generation: bool,
) {
    // MiniMax M3 defaults `thinking_mode` to "adaptive", which breaks
    // constrained JSON/schema and forced-tool generation. If the client
    // did not set thinking controls, force "disabled" for those requests.
    let explicit_thinking_mode_is_adaptive = chat_template_args
        .as_ref()
        .and_then(|args| args.get("thinking_mode"))
        .and_then(|v| v.as_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("adaptive"));
    if !thinking_control_from_client && constrained_generation {
        let args = chat_template_args.get_or_insert_default();
        args.insert(
            "thinking_mode".to_string(),
            serde_json::Value::String("disabled".to_string()),
        );
        args.insert("thinking".to_string(), serde_json::Value::Bool(false));
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        return;
    }

    apply_normalized_thinking_aliases(chat_template_args, normalized);

    let Some(normalized) = normalized else {
        return;
    };

    // Canonicalize for M3 because its template only treats exact
    // "disabled" as off. Preserve explicit "adaptive" as client intent.
    if !explicit_thinking_mode_is_adaptive {
        let args = chat_template_args.get_or_insert_default();
        args.insert(
            "thinking_mode".to_string(),
            serde_json::Value::String(if normalized { "enabled" } else { "disabled" }.to_string()),
        );
    }
}

/// Apply one resolved boolean to both public template aliases.
pub fn apply_normalized_thinking_aliases(
    chat_template_args: &mut Option<HashMap<String, serde_json::Value>>,
    normalized: Option<bool>,
) {
    let Some(normalized) = normalized else {
        return;
    };
    let args = chat_template_args.get_or_insert_default();
    args.insert("thinking".to_string(), serde_json::Value::Bool(normalized));
    args.insert(
        "enable_thinking".to_string(),
        serde_json::Value::Bool(normalized),
    );
}
