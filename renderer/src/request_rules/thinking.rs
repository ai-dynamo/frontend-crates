// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thinking-mode rules for the chat-template arguments.
//!
//! Clients turn thinking on and off with several spellings (`thinking`,
//! `enable_thinking`, `thinking_mode`), and templates read different ones.
//! These rules rewrite `chat_template_args` so every spelling a template might
//! read agrees, apply the deployment's default mode, and apply the MiniMax M3
//! rules. The Dynamo frontend runs them before it renders the prompt, and
//! again on a copy when it decides whether to run the reasoning parser, so
//! both decisions see the same mode.

use std::collections::HashMap;

use super::truthy::{is_truthy, parse_bool_opt};
use crate::{OAIChatLikeRequest, PromptRenderError, thinking_bool_from_args};

/// The chat-template argument map that the thinking rules read and rewrite.
pub type ChatTemplateArgs = HashMap<String, serde_json::Value>;

/// The parser names and deployment default that select which thinking rules
/// apply to a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThinkingRules<'a> {
    /// The deployment's reasoning parser name, such as `"qwen3"`.
    pub reasoning_parser: Option<&'a str>,
    /// The deployment's tool-call parser name, such as `"minimax_m3"`.
    pub tool_call_parser: Option<&'a str>,
    /// The deployment's default thinking mode: `"enabled"` or `"disabled"`.
    /// Any other value is ignored with a warning.
    pub default_thinking_mode: Option<&'a str>,
}

impl ThinkingRules<'_> {
    /// Apply the default thinking mode, then normalize the thinking aliases.
    ///
    /// `client_thinking_field` is true when the request carries a thinking
    /// control outside `chat_template_args`, such as a top-level `thinking`
    /// object. It counts as client intent together with the thinking keys in
    /// `chat_template_args`, and client intent is read before the default is
    /// applied, so the default never counts as client intent.
    ///
    /// `constrained_output` is true when the request constrains output from the
    /// first token. Compute it with [`has_minimax_m3_constrained_generation`].
    /// Only MiniMax M3 deployments read it.
    pub fn apply(
        &self,
        chat_template_args: &mut Option<ChatTemplateArgs>,
        client_thinking_field: bool,
        constrained_output: bool,
    ) {
        let thinking_control_from_client =
            client_thinking_field || has_request_thinking_control(chat_template_args.as_ref());
        apply_default_thinking_mode(
            self.default_thinking_mode,
            chat_template_args,
            thinking_control_from_client,
        );
        let constrained_generation = !thinking_control_from_client
            && is_minimax_m3_family(self.reasoning_parser, self.tool_call_parser)
            && constrained_output;
        normalize_thinking_args(
            chat_template_args,
            self.reasoning_parser,
            self.tool_call_parser,
            thinking_control_from_client,
            constrained_generation,
        );
    }
}

/// True when `chat_template_args` carries a thinking control from the client:
/// `thinking`, `enable_thinking`, `thinking_mode`, or `reasoning_effort`.
pub fn has_request_thinking_control(chat_template_args: Option<&ChatTemplateArgs>) -> bool {
    chat_template_args.is_some_and(|args| {
        [
            "thinking",
            "enable_thinking",
            "thinking_mode",
            "reasoning_effort",
        ]
        .iter()
        .any(|key| args.contains_key(*key))
    })
}

/// Write the deployment's default thinking mode into all three aliases, unless
/// the client set a thinking control.
pub fn apply_default_thinking_mode(
    default_mode: Option<&str>,
    chat_template_args: &mut Option<ChatTemplateArgs>,
    thinking_control_from_client: bool,
) {
    if thinking_control_from_client {
        return;
    }
    let Some(default_mode) = default_mode else {
        return;
    };
    let enabled = match default_mode {
        "enabled" => true,
        "disabled" => false,
        other => {
            tracing::warn!(
                default_thinking_mode = other,
                "Ignoring invalid runtime_config default_thinking_mode; expected 'enabled' or 'disabled'"
            );
            return;
        }
    };

    let args = chat_template_args.get_or_insert_with(HashMap::new);
    args.insert("thinking".to_string(), serde_json::Value::Bool(enabled));
    args.insert(
        "enable_thinking".to_string(),
        serde_json::Value::Bool(enabled),
    );
    args.insert(
        "thinking_mode".to_string(),
        serde_json::Value::String(if enabled { "enabled" } else { "disabled" }.to_string()),
    );
}

/// Normalize the thinking aliases for the deployment's parsers.
///
/// The first alias the client set wins, in the order `thinking`,
/// `enable_thinking`, `thinking_mode`. Its value is written to `thinking` and
/// `enable_thinking`. MiniMax M3 deployments also get a canonical
/// `thinking_mode`, and a constrained request with no client control renders
/// with thinking disabled. Kimi K2.5 and K3 parsers default to thinking on.
pub fn normalize_thinking_args(
    chat_template_args: &mut Option<ChatTemplateArgs>,
    reasoning_parser: Option<&str>,
    tool_call_parser: Option<&str>,
    thinking_control_from_client: bool,
    constrained_generation: bool,
) {
    let normalized = normalize_thinking_aliases(chat_template_args.as_ref(), reasoning_parser);

    if is_minimax_m3_family(reasoning_parser, tool_call_parser) {
        normalize_minimax_m3_thinking_mode(
            chat_template_args,
            normalized,
            thinking_control_from_client,
            constrained_generation,
        );
        return;
    }

    apply_normalized_thinking_aliases(chat_template_args, normalized);
}

fn normalize_thinking_aliases(
    chat_template_args: Option<&ChatTemplateArgs>,
    reasoning_parser: Option<&str>,
) -> Option<bool> {
    chat_template_args
        .and_then(|args| {
            // Normalize public aliases in array order: `thinking`,
            // `enable_thinking`, then `thinking_mode`.
            // Preserve "adaptive" as tri-state by returning None.
            for key in ["thinking", "enable_thinking", "thinking_mode"] {
                match args.get(key) {
                    Some(serde_json::Value::Bool(b)) => {
                        // `thinking_bool_from_args` ignores `thinking_mode`,
                        // so read that bool directly.
                        if key == "thinking_mode" {
                            return Some(*b);
                        }
                        return thinking_bool_from_args(Some(args));
                    }
                    Some(serde_json::Value::String(value)) => {
                        // Parse `thinking_mode` explicitly so
                        // "enabled"/"disabled" and boolean spellings
                        // behave predictably. Preserve unrecognized strings
                        // such as "adaptive" or DeepSeek's "chat"/"thinking".
                        if key == "thinking_mode" {
                            if value.eq_ignore_ascii_case("enabled") {
                                return Some(true);
                            }
                            if value.eq_ignore_ascii_case("disabled") {
                                return Some(false);
                            }
                            if let Some(b) = parse_bool_opt(value) {
                                return Some(b);
                            }
                            return None;
                        }
                        return Some(is_truthy(value));
                    }
                    Some(serde_json::Value::Number(value)) => {
                        if let Some(value) = value.as_f64() {
                            return Some(value != 0.0);
                        }
                    }
                    _ => {}
                }
            }
            None
        })
        .or_else(|| {
            matches!(reasoning_parser, Some("kimi_k25" | "kimi_k3" | "kimi-k3")).then_some(true)
        })
}

fn normalize_minimax_m3_thinking_mode(
    chat_template_args: &mut Option<ChatTemplateArgs>,
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

fn apply_normalized_thinking_aliases(
    chat_template_args: &mut Option<ChatTemplateArgs>,
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

/// True when the deployment renders MiniMax M3's `thinking_mode` template.
/// Either parser is sufficient, and tool-call aliases include `_nom`.
pub fn is_minimax_m3_family(
    reasoning_parser: Option<&str>,
    tool_call_parser: Option<&str>,
) -> bool {
    matches!(reasoning_parser, Some("minimax_m3") | Some("minimax-m3"))
        || matches!(
            tool_call_parser,
            Some("minimax_m3")
                | Some("minimax-m3")
                | Some("minimax_m3_nom")
                | Some("minimax-m3-nom")
        )
}

/// A structured `response_format`, top-level `guided_*`, and a forced tool
/// choice each constrain output from the first token.
///
/// `explicit_guided_decoding` is true when the request's own guided-decoding
/// fields (Dynamo's `guided_json`, `guided_regex`, a non-empty
/// `guided_choice`, or `guided_grammar`) install a grammar. They are not part
/// of [`OAIChatLikeRequest`], so the caller checks them.
///
/// With no client thinking control, a request this returns true for
/// renders thinking disabled (see [`normalize_thinking_args`]),
/// even when the grammar itself contains `<mm:think>...</mm:think>`. The
/// reasoning parser is then skipped, and a generated think block stays
/// verbatim in `content`. A client that wants that block split into
/// `reasoning_content` must send a thinking control that does not disable
/// thinking, preferably an adaptive one (`thinking_mode: "adaptive"` or
/// `thinking: {type: "adaptive"}`): its prompt prefills no marker for the
/// grammar to repeat.
pub fn has_minimax_m3_constrained_generation<R: OAIChatLikeRequest + ?Sized>(
    request: &R,
    explicit_guided_decoding: bool,
) -> bool {
    has_structured_response_format(request)
        || has_guided_tool_choice(request)
        || explicit_guided_decoding
}

/// True when `response_format` asks for anything other than `text`.
pub fn has_structured_response_format<R: OAIChatLikeRequest + ?Sized>(request: &R) -> bool {
    request.response_format().is_some_and(|format| {
        format
            .get_attr("type")
            .ok()
            .is_some_and(|kind| kind.as_str().is_some_and(|kind| kind != "text"))
    })
}

/// True when `tool_choice` forces a tool call.
/// Shared by guided-output reasoning checks and the M3 default.
pub fn has_guided_tool_choice<R: OAIChatLikeRequest + ?Sized>(request: &R) -> bool {
    request.tool_choice().is_some_and(|tool_choice| {
        match tool_choice.as_str() {
            Some("required") => true,
            Some(_) => false,
            // The only supported non-string tool choice is a named function.
            None => true,
        }
    })
}

/// Refuse `reasoning_effort: "none"` for GPT-OSS, which has no off mode.
///
/// `parser_family` is the parser family the deployment selected, such as
/// `"gpt_oss"`. The error is a client error with the message the Dynamo
/// frontend returns.
pub fn reject_unsupported_reasoning_effort(
    parser_family: Option<&str>,
    chat_template_args: Option<&ChatTemplateArgs>,
) -> Result<(), PromptRenderError> {
    if parser_family == Some("gpt_oss")
        && chat_template_args
            .and_then(|args| args.get("reasoning_effort"))
            .and_then(serde_json::Value::as_str)
            == Some("none")
    {
        return Err(PromptRenderError::invalid_request(
            "GPT-OSS does not support reasoning_effort=none; use low, medium, or high",
        ));
    }
    Ok(())
}
