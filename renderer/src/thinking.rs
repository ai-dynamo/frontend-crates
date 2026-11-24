// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request policy shared by template preparation and parser gating.
use std::collections::HashMap;

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
