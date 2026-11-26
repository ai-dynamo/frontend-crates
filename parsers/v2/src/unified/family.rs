// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a host needs to know about a family beyond "text in, events out".
//!
//! A parser turns model text into events. The host also has to know which names
//! select the family, which markers open reasoning and tool calls (so it can hold
//! back streamed text and read the prompt's channel state), and whether reasoning
//! can be turned off. That knowledge lives here, next to the parser, so a new
//! family needs no host-side code: a crate bump or a vendor registration brings it.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use super::{Result, Tool, UnifiedParser, UnifiedParserFactory, UnifiedParserStartingState};

/// Reasoning open and close markers, as the model emits them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningMarkers {
    pub open: &'static str,
    pub close: &'static str,
}

/// Whether a request may turn reasoning off.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ThinkingPolicy {
    /// The family does not reason.
    #[default]
    None,
    /// `enable_thinking=false` (and its aliases) turns reasoning off.
    Optional,
    /// The model always reasons; a request to turn it off is invalid.
    Mandatory,
}

/// Host-facing facts about one family.
///
/// `&'static` so the built-in table is a `const` and lookups never allocate. A
/// vendor building one at runtime (for example, from a loaded `.so`) leaks its
/// strings once at registration, which is startup-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnifiedFamilyInfo {
    /// Canonical name.
    pub family: &'static str,
    /// Other names that select this family.
    pub aliases: &'static [&'static str],
    pub reasoning: Option<ReasoningMarkers>,
    /// The first marker of a native tool call. The host holds back streamed text
    /// that could be the start of it.
    pub tool_call_opener: Option<&'static str>,
    /// Other control markers that mean "this text is framed output, not prose".
    pub control_markers: &'static [&'static str],
    pub thinking: ThinkingPolicy,
    /// Where bare guided JSON output starts.
    pub bare_json_starting_state: UnifiedParserStartingState,
}

impl UnifiedFamilyInfo {
    pub const fn new(family: &'static str) -> Self {
        Self {
            family,
            aliases: &[],
            reasoning: None,
            tool_call_opener: None,
            control_markers: &[],
            thinking: ThinkingPolicy::None,
            bare_json_starting_state: UnifiedParserStartingState::None,
        }
    }

    pub const fn aliases(mut self, aliases: &'static [&'static str]) -> Self {
        self.aliases = aliases;
        self
    }

    pub const fn reasoning(mut self, open: &'static str, close: &'static str) -> Self {
        self.reasoning = Some(ReasoningMarkers { open, close });
        if matches!(self.thinking, ThinkingPolicy::None) {
            self.thinking = ThinkingPolicy::Optional;
        }
        self
    }

    pub const fn tool_call_opener(mut self, opener: &'static str) -> Self {
        self.tool_call_opener = Some(opener);
        self
    }

    pub const fn control_markers(mut self, markers: &'static [&'static str]) -> Self {
        self.control_markers = markers;
        self
    }

    pub const fn thinking(mut self, thinking: ThinkingPolicy) -> Self {
        self.thinking = thinking;
        self
    }

    pub const fn bare_json_starts_in(mut self, state: UnifiedParserStartingState) -> Self {
        self.bare_json_starting_state = state;
        self
    }

    fn answers_to(&self, name: &str) -> bool {
        self.family == name || self.aliases.contains(&name)
    }
}

const THINK: (&str, &str) = ("<think>", "</think>");
const MUSE_FRAMING: &[&str] = &["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"];

/// Built-in families. A family missing here still parses; the host just gets
/// no markers for it.
const BUILTIN: &[UnifiedFamilyInfo] = &[
    UnifiedFamilyInfo::new("deepseek_v4")
        .aliases(&["deepseek-v4", "deepseekv4"])
        .reasoning(THINK.0, THINK.1)
        .tool_call_opener("<｜DSML｜tool_calls>"),
    UnifiedFamilyInfo::new("deepseek_v41")
        .reasoning(THINK.0, THINK.1)
        .tool_call_opener("<｜DSML｜ calls>"),
    UnifiedFamilyInfo::new("gemma4")
        .aliases(&["gemma-4"])
        .reasoning("<|channel>", "<channel|>")
        .tool_call_opener("<|tool_call>"),
    UnifiedFamilyInfo::new("glm47")
        .aliases(&["glm45"])
        .reasoning(THINK.0, THINK.1)
        .tool_call_opener("<tool_call>"),
    UnifiedFamilyInfo::new("hermes").reasoning(THINK.0, THINK.1),
    UnifiedFamilyInfo::new("qwen3")
        .aliases(&["qwen3_coder"])
        .reasoning(THINK.0, THINK.1)
        .tool_call_opener("<tool_call>"),
    UnifiedFamilyInfo::new("nemotron_v3")
        .aliases(&["nemotron3"])
        .reasoning(THINK.0, THINK.1)
        .tool_call_opener("<tool_call>")
        .bare_json_starts_in(UnifiedParserStartingState::Response),
    UnifiedFamilyInfo::new("minimax_m3")
        .aliases(&["minimax-m3"])
        .reasoning("<mm:think>", "</mm:think>")
        .tool_call_opener("]<]minimax[>[<tool_call>"),
    UnifiedFamilyInfo::new("muse_glimmer")
        .aliases(&["muse"])
        .tool_call_opener("<atem:invoke")
        .control_markers(MUSE_FRAMING),
    UnifiedFamilyInfo::new("muse_spark")
        .aliases(&["muse_spark_open"])
        .tool_call_opener("<atem:invoke")
        .control_markers(MUSE_FRAMING),
    UnifiedFamilyInfo::new("beam")
        .tool_call_opener("<|start|>assistant to=")
        .control_markers(&[
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
        ])
        .thinking(ThinkingPolicy::Mandatory),
    UnifiedFamilyInfo::new("kimi_k2")
        .aliases(&["kimi_k25"])
        .reasoning(THINK.0, THINK.1)
        .tool_call_opener("<|tool_call_begin|>"),
    UnifiedFamilyInfo::new("kimi_k3")
        .aliases(&["kimi-k3"])
        .tool_call_opener("<|open|>call")
        .control_markers(&[
            "<|open|>think",
            "<|close|>think",
            "<|open|>response",
            "<|close|>response",
            "<|open|>tools",
            "<|close|>tools",
            "<|open|>call",
            "<|close|>call",
            "<|sep|>",
        ])
        .thinking(ThinkingPolicy::Optional),
    UnifiedFamilyInfo::new("mistral_4")
        .aliases(&["mistral"])
        .thinking(ThinkingPolicy::Optional),
    UnifiedFamilyInfo::new("gpt_oss").aliases(&["harmony"]),
];

static VENDOR_INFO: LazyLock<RwLock<HashMap<&'static str, UnifiedFamilyInfo>>> =
    LazyLock::new(Default::default);

/// Facts about a family, by canonical name or alias. A vendor registration wins
/// over the built-in of the same name, matching parser selection.
pub fn unified_family_info(name: &str) -> Option<UnifiedFamilyInfo> {
    let vendor = VENDOR_INFO.read().unwrap_or_else(|e| e.into_inner());
    vendor
        .values()
        .find(|info| info.answers_to(name))
        .or_else(|| BUILTIN.iter().find(|info| info.answers_to(name)))
        .copied()
}

/// Every name that selects a family, built-in or vendor, sorted.
pub fn unified_family_names() -> Vec<&'static str> {
    let vendor = VENDOR_INFO.read().unwrap_or_else(|e| e.into_inner());
    let mut names: Vec<&'static str> = super::REGISTERED_UNIFIED_FAMILIES
        .iter()
        .copied()
        .chain(
            BUILTIN
                .iter()
                .chain(vendor.values())
                .flat_map(|info| std::iter::once(info.family).chain(info.aliases.iter().copied())),
        )
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// A parser family: its facts and how to build its parser.
///
/// ```ignore
/// struct MyModel;
/// impl UnifiedFamily for MyModel {
///     const INFO: UnifiedFamilyInfo = UnifiedFamilyInfo::new("my_model")
///         .reasoning("<think>", "</think>")
///         .tool_call_opener("<tool_call>");
///     fn create(tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> { ... }
/// }
/// MyModel::register(); // once, at startup
/// ```
pub trait UnifiedFamily {
    const INFO: UnifiedFamilyInfo;

    fn create(tools: &[Tool]) -> Result<Box<dyn UnifiedParser>>;

    /// Returns the factory this replaced, if any.
    fn register() -> Option<UnifiedParserFactory> {
        register_unified_family(Self::INFO, Self::create)
    }
}

/// Register a parser together with its facts, for hosts that build `info` at
/// runtime (a loaded `.so`). Startup-only, like
/// [`register_unified_parser`](super::register_unified_parser).
pub fn register_unified_family(
    info: UnifiedFamilyInfo,
    factory: UnifiedParserFactory,
) -> Option<UnifiedParserFactory> {
    let previous = super::register_unified_parser(info.family, factory);
    VENDOR_INFO
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(info.family, info);
    previous
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_family_has_info() {
        for family in super::super::REGISTERED_UNIFIED_FAMILIES {
            assert!(unified_family_info(family).is_some(), "{family}");
        }
    }

    #[test]
    fn aliases_resolve_to_one_canonical_family() {
        for info in BUILTIN {
            for name in std::iter::once(&info.family).chain(info.aliases) {
                assert_eq!(unified_family_info(name).unwrap().family, info.family);
            }
        }
    }
}
