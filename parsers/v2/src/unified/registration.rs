// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;

use super::{Tool, UnifiedParser, UnifiedParserFactory, UnifiedParserStartingState};
use crate::structural_tag::StructuralTagBuilder;

/// Selection restrictions applied only when the host automatically chooses a generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoSelection {
    #[default]
    Unified,
    /// A reasoning-only selector does not identify the model's tool grammar.
    RequiresToolSelector,
    /// Rendering and decoding require both configured selectors.
    RequiresBothSelectors,
}

/// Vocabulary needed to distinguish a supported native wire format.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct RequiredVocabularyToken {
    pub token: String,
    pub tokenizer_description: String,
    pub legacy_parser: Option<String>,
}

/// How request thinking controls affect parsing of emitted reasoning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningControl {
    #[default]
    Request,
    AlwaysParse,
    OptInUnlessToolSelected,
    DeepSeek,
}

/// Native grammar inspection used by hosts that reconcile backend completion evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeGrammar {
    #[default]
    Opaque,
    QwenXml,
    GlmXml,
}

/// Recognize call boundaries when parsed events interrupt the raw model stream.
pub trait NativeToolBoundary: Send {
    fn observe(&mut self, text: &str);
    fn outside_call(&self) -> bool;
    fn can_restart_call(&self) -> bool;
    fn resolve_opener(&self, held: &str, text: &str) -> Option<bool>;
    fn partial_opener_suffix_start(&self, text: &str) -> Option<usize>;
    fn first_call_marker(&self, text: &str) -> Option<usize>;
}

/// A grammar that consumes a tokenized assistant prefix before generated deltas.
#[derive(Clone, Copy)]
pub struct TokenPromptPrefix {
    pub encode: fn(&str) -> anyhow::Result<Vec<u32>>,
    pub empty_assistant_prefix: &'static str,
    pub batch_prefix: fn(&str) -> Option<&'static str>,
}

pub type NativeResponsePrefix = fn(&str) -> (&str, &'static str);
pub type UnfinishedNativeCallStart = fn(&str) -> anyhow::Result<Option<usize>>;

/// Immutable integration policy supplied with a parser factory.
#[derive(Clone, Serialize)]
pub struct UnifiedParserDescriptor {
    pub family: String,
    pub aliases: Vec<String>,
    /// Additional tool selectors compatible with this family's reasoning grammar.
    pub compatible_tool_selectors: Vec<String>,
    pub auto_selection: AutoSelection,
    /// Selectors whose reasoning-only Auto route is known to identify this tool grammar.
    pub auto_without_tool_selectors: Vec<String>,
    pub required_tokens: Vec<RequiredVocabularyToken>,
    pub text_input: bool,
    pub token_id_input: bool,
    pub continuation: bool,
    pub notify_model_eos: bool,
    pub preserve_special_tokens: bool,
    /// A completed native call may carry argument text that is not strict JSON.
    pub native_completion_authoritative: bool,
    pub errors_are_fatal: bool,
    pub native_grammar: NativeGrammar,
    #[serde(skip)]
    pub native_tool_boundary: Option<fn() -> Box<dyn NativeToolBoundary>>,
    pub preserve_message_phases: bool,
    #[serde(skip)]
    pub native_response_prefix: Option<NativeResponsePrefix>,
    pub legacy_backend_completion: bool,
    pub reasoning_control: ReasoningControl,
    pub reasoning_mandatory: bool,
    pub disabled_reasoning_starts_response: bool,
    pub deprecated_aliases: Vec<String>,
    pub deprecation_note: Option<String>,
    pub reasoning_markers: Option<(&'static str, &'static str)>,
    pub native_tool_call_marker: Option<&'static str>,
    #[serde(skip)]
    pub unfinished_native_call_start: Option<UnfinishedNativeCallStart>,
    pub batch_prefill: BatchPrefill,
    pub batch_markers: Option<(&'static str, &'static str)>,
    pub marker_free_batch_response: bool,
    pub native_call_owns_following_markers: bool,
    pub implicit_stream_reasoning: bool,
    pub prompt_style: super::prefill::PromptStyle,
    pub prompt_closure: super::prefill::PromptClosure,
    #[serde(skip)]
    pub token_prompt_prefix: Option<TokenPromptPrefix>,
    #[serde(skip)]
    pub structural_tag_builder: Option<&'static StructuralTagBuilder>,
    pub guidance: Guidance,
}

/// How raw output reveals the channel left open by an unavailable prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchPrefill {
    #[default]
    ModelOpensChannels,
    MarkerPair,
    NamedChannels,
    HarmonyChannels,
}

/// Constraint formats a parser can consume without changing its native wire format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Guidance {
    pub structural_tag: bool,
    pub whole_turn: bool,
    pub mistral_lark: bool,
    pub argument_framing: Option<super::ArgumentFraming>,
    pub reject_malformed_calls: bool,
    pub eos_completes_calls: bool,
    pub forced_tool_choice: ForcedToolGuidance,
    pub from_reasoning_selector: bool,
    pub owns_reasoning_boundary: bool,
    pub frames_structured_output: bool,
}

/// Constraint needed for a forced tool choice on this parser's wire format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForcedToolGuidance {
    #[default]
    Generic,
    Native,
    PreferTagForNamed,
    PreferTagForForced,
    RequireTagForNamed,
    RequireTagForForced,
}

impl ForcedToolGuidance {
    pub fn prefers_tag(self, named: bool) -> bool {
        self.requires_tag(named)
            || self == Self::PreferTagForForced
            || (named && self == Self::PreferTagForNamed)
    }

    pub fn requires_tag(self, named: bool) -> bool {
        self == Self::RequireTagForForced || (named && self == Self::RequireTagForNamed)
    }
}

impl UnifiedParserDescriptor {
    /// Default policy for a text parser with no additional integration requirements.
    pub fn new(family: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            aliases: Vec::new(),
            compatible_tool_selectors: Vec::new(),
            auto_selection: AutoSelection::Unified,
            auto_without_tool_selectors: Vec::new(),
            required_tokens: Vec::new(),
            text_input: true,
            token_id_input: false,
            continuation: false,
            notify_model_eos: false,
            preserve_special_tokens: false,
            native_completion_authoritative: false,
            errors_are_fatal: false,
            native_grammar: NativeGrammar::Opaque,
            native_tool_boundary: None,
            preserve_message_phases: false,
            native_response_prefix: None,
            legacy_backend_completion: false,
            reasoning_control: ReasoningControl::Request,
            reasoning_mandatory: false,
            disabled_reasoning_starts_response: false,
            deprecated_aliases: Vec::new(),
            deprecation_note: None,
            reasoning_markers: None,
            native_tool_call_marker: None,
            unfinished_native_call_start: None,
            batch_prefill: BatchPrefill::ModelOpensChannels,
            batch_markers: None,
            marker_free_batch_response: false,
            native_call_owns_following_markers: false,
            implicit_stream_reasoning: false,
            prompt_style: super::prefill::PromptStyle::NoPrefill,
            prompt_closure: super::prefill::PromptClosure::Unspecified,
            token_prompt_prefix: None,
            structural_tag_builder: None,
            guidance: Guidance::default(),
        }
    }

    pub(crate) fn builtin(family: &str, aliases: &[&str], configure: fn(&mut Self)) -> Self {
        let mut descriptor = Self::new(family);
        descriptor.aliases = aliases.iter().map(|s| (*s).to_owned()).collect();
        descriptor.preserve_special_tokens = true;
        descriptor.prompt_style = super::prefill::PromptStyle::Think;
        descriptor.reasoning_markers = Some(("<think>", "</think>"));
        descriptor.batch_prefill = BatchPrefill::MarkerPair;
        descriptor.marker_free_batch_response = true;
        configure(&mut descriptor);
        if let Some(builder) = descriptor.structural_tag_builder {
            descriptor.guidance.structural_tag = true;
            descriptor.guidance.whole_turn = builder.constrains_every_turn();
        }
        descriptor
    }

    /// Whether Auto must retain a split parser until a tool selector disambiguates the grammar.
    pub fn requires_tool_selector_in_auto(&self, reasoning: Option<&str>) -> bool {
        self.auto_selection == AutoSelection::RequiresToolSelector
            && !reasoning.is_some_and(|name| {
                self.auto_without_tool_selectors
                    .iter()
                    .any(|selector| selector == name)
            })
    }

    /// Whether a selector names this registration, including deprecated aliases.
    pub fn accepts(&self, selector: &str) -> bool {
        self.family == selector || self.aliases.iter().any(|alias| alias == selector)
    }

    /// Resolve a stream's initial channel from rendered prompt evidence and family defaults.
    pub fn stream_starting_state(
        &self,
        prompt_opened_reasoning: bool,
    ) -> UnifiedParserStartingState {
        if prompt_opened_reasoning || self.implicit_stream_reasoning {
            UnifiedParserStartingState::Reasoning
        } else {
            UnifiedParserStartingState::None
        }
    }

    /// Infer parser initialization from raw batch output when no rendered prompt is available.
    pub fn infer_batch_starting_state(&self, content: &str) -> UnifiedParserStartingState {
        use super::prefill::{first_unquoted_marker_position as position, infer_marker_prefill};
        match self.batch_prefill {
            BatchPrefill::ModelOpensChannels => UnifiedParserStartingState::None,
            BatchPrefill::NamedChannels => super::prefill::named_channel_prefill(content),
            BatchPrefill::HarmonyChannels => super::prefill::beam_prefill(content),
            BatchPrefill::MarkerPair => {
                let (opener, closer) = self
                    .batch_markers
                    .or(self.reasoning_markers)
                    .expect("marker-pair descriptor");
                let open = position(content, opener);
                let close = position(content, closer);
                if self.native_call_owns_following_markers
                    && self
                        .native_tool_call_marker
                        .and_then(|marker| position(content, marker))
                        .is_some_and(|tool| open.into_iter().chain(close).all(|at| tool < at))
                {
                    return UnifiedParserStartingState::None;
                }
                if open.is_none() && close.is_none() && self.marker_free_batch_response {
                    UnifiedParserStartingState::Response
                } else {
                    infer_marker_prefill(content, opener, closer)
                }
            }
        }
    }
}

type OwnedFactory = dyn Fn(&[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>> + Send + Sync;

#[derive(Clone)]
pub(crate) enum Factory {
    Linked(UnifiedParserFactory),
    Loaded(std::sync::Arc<OwnedFactory>),
}

impl Factory {
    pub(crate) fn linked(&self) -> Option<UnifiedParserFactory> {
        match self {
            Self::Linked(factory) => Some(*factory),
            Self::Loaded(_) => None,
        }
    }
}

/// Host adapter for a native parser plugin's constraint builder.
pub type LoadedGuidance = dyn Fn(super::plugin::GuidanceRequest) -> anyhow::Result<Option<serde_json::Value>>
    + Send
    + Sync;

/// Host adapter for a native plugin's rendered-prompt inspector.
pub type LoadedPrompt = dyn Fn(super::plugin::PromptRequest) -> anyhow::Result<super::prefill::ParserPromptState>
    + Send
    + Sync;

/// A resolved descriptor and factory, retained together even after registry changes.
#[derive(Clone)]
pub struct UnifiedParserRegistration {
    pub descriptor: UnifiedParserDescriptor,
    pub(crate) factory: Factory,
    pub(crate) vendor: bool,
    pub(crate) plugin_identity: Option<String>,
    pub(crate) plugin_guidance: Option<std::sync::Arc<LoadedGuidance>>,
    pub(crate) plugin_prompt: Option<std::sync::Arc<LoadedPrompt>>,
}

impl UnifiedParserRegistration {
    /// Retain the loaded library through the factory and every instance it creates.
    pub fn loaded(
        descriptor: UnifiedParserDescriptor,
        identity: String,
        guidance: Option<std::sync::Arc<LoadedGuidance>>,
        prompt: Option<std::sync::Arc<LoadedPrompt>>,
        factory: impl Fn(&[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            descriptor,
            factory: Factory::Loaded(std::sync::Arc::new(factory)),
            vendor: true,
            plugin_identity: Some(identity),
            plugin_guidance: guidance,
            plugin_prompt: prompt,
        }
    }

    /// Identity advertised by workers that use an external plugin.
    pub fn plugin_identity(&self) -> Option<&str> {
        self.plugin_identity.as_deref()
    }

    /// Return a retained plugin builder usable by the host's guidance resolver.
    pub fn plugin_guidance(&self) -> Option<std::sync::Arc<LoadedGuidance>> {
        self.plugin_guidance.clone()
    }

    /// Inspect the rendered prompt through the registered implementation.
    pub fn inspect_prompt(
        &self,
        prompt: Option<&str>,
        header_search_end: Option<usize>,
    ) -> anyhow::Result<super::prefill::ParserPromptState> {
        match (&self.plugin_prompt, prompt) {
            (Some(inspect), Some(prompt)) => inspect(super::plugin::PromptRequest {
                prompt: prompt.to_owned(),
                header_search_end,
            }),
            _ => Ok(self.descriptor.resolve_prompt(prompt, header_search_end)),
        }
    }

    /// Whether the plugin supplies its own rendered-prompt inspector.
    pub fn has_plugin_prompt(&self) -> bool {
        self.plugin_prompt.is_some()
    }

    /// Whether this registration came from the startup override API.
    pub fn is_vendor(&self) -> bool {
        self.vendor
    }

    /// Create fresh parser state for one choice using this registration's factory.
    pub fn create(&self, tools: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>> {
        let family = &self.descriptor.family;
        let parser = match &self.factory {
            Factory::Linked(factory) => factory(tools),
            Factory::Loaded(factory) => factory(tools),
        }?;
        let parser = super::blank_tool_turn::wrap(family, parser);
        if crate::tool_calling::debug::debug_enabled() {
            return Ok(super::DebugUnifiedParser::wrap(family, parser));
        }
        Ok(parser)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn descriptors_match_the_effective_parsers_and_aliases() {
        for registration in super::super::unified_parser_registrations() {
            let descriptor = &registration.descriptor;
            let parser = registration.create(&[]).unwrap();
            assert_eq!(
                descriptor.preserve_special_tokens,
                parser.preserve_special_tokens(),
                "{}",
                descriptor.family
            );
            let inputs = parser.input_capabilities();
            assert_eq!(descriptor.text_input, inputs.text, "{}", descriptor.family);
            assert_eq!(
                descriptor.token_id_input, inputs.token_ids,
                "{}",
                descriptor.family
            );
            for selector in std::iter::once(&descriptor.family).chain(&descriptor.aliases) {
                let resolved = super::super::resolve_unified_parser(selector).unwrap();
                assert_eq!(resolved.descriptor.family, descriptor.family);
                assert!(
                    super::super::resolve_unified_parser_pair(Some(selector), Some(selector))
                        .is_some()
                );
            }
        }
    }

    #[test]
    fn reasoning_only_auto_exceptions_are_selector_specific() {
        let qwen = super::super::resolve_unified_parser("qwen3")
            .unwrap()
            .descriptor;
        assert!(qwen.requires_tool_selector_in_auto(Some("qwen3")));
        assert!(!qwen.requires_tool_selector_in_auto(Some("qwen3_coder")));
        for selector in ["nemotron_v3", "nemotron3"] {
            let descriptor = super::super::resolve_unified_parser(selector)
                .unwrap()
                .descriptor;
            assert!(descriptor.requires_tool_selector_in_auto(Some(selector)));
        }
    }

    #[test]
    fn selector_pair_policy_preserves_borrowed_grammars_and_rejects_mismatches() {
        let resolve = super::super::resolve_unified_parser_pair;
        assert_eq!(
            resolve(Some("qwen3_coder"), Some("nemotron3"))
                .unwrap()
                .descriptor
                .family,
            "nemotron_v3"
        );
        assert!(resolve(Some("qwen3_coder"), Some("mistral")).is_none());
        assert!(resolve(Some("unknown"), Some("unknown")).is_none());
        assert!(resolve(None, None).is_none());
        assert_eq!(
            resolve(Some("mistral"), None)
                .unwrap()
                .descriptor
                .required_tokens[0]
                .token,
            "[ARGS]"
        );
    }
}
