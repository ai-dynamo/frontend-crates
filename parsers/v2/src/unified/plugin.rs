// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version 1 of the native parser plugin protocol. Only C ABI values and UTF-8 JSON
//! cross the library boundary. The SDK continues to use the Rust parser trait locally.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::{Deserialize, Serialize};

use super::{
    RequiredVocabularyToken, Tool, ToolCallPolicy, UnifiedParser, UnifiedParserDescriptor,
    UnifiedParserEvent, UnifiedParserFactory, UnifiedParserInit, UnifiedParserOutput,
};

/// Version of the entry point and its message schema.
pub const ABI_VERSION: u32 = 1;

/// Host callback that copies one reply before the plugin returns.
pub type ReplyCallback = unsafe extern "C" fn(*mut c_void, *const u8, usize);

/// Exported function signature. Returns a handle only for a successful Create.
pub type Entry = unsafe extern "C" fn(
    u32,
    *mut c_void,
    *const u8,
    usize,
    ReplyCallback,
    *mut c_void,
) -> *mut c_void;

/// Capabilities available to a v1 plugin. All instances support decoded text.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub family: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub preserve_special_tokens: bool,
    #[serde(default)]
    pub native_completion_authoritative: bool,
    #[serde(default)]
    pub required_tokens: Vec<RequiredVocabularyToken>,
    #[serde(default)]
    pub structural_tag: bool,
    #[serde(default)]
    pub inspect_prompt: bool,
}

impl Manifest {
    /// Project the wire metadata into the shared registration policy.
    pub fn descriptor(self) -> UnifiedParserDescriptor {
        let mut descriptor = UnifiedParserDescriptor::new(self.family);
        descriptor.aliases = self.aliases;
        descriptor.preserve_special_tokens = self.preserve_special_tokens;
        descriptor.native_completion_authoritative = self.native_completion_authoritative;
        descriptor.required_tokens = self.required_tokens;
        descriptor.guidance.structural_tag = self.structural_tag;
        descriptor.errors_are_fatal = true;
        descriptor
    }
}

/// Normalized request facts for building a triggered structural tag.
#[derive(Serialize, Deserialize)]
pub struct GuidanceRequest {
    pub tools: Vec<Tool>,
    pub choice: GuidanceChoice,
    pub parallel_tool_calls: Option<bool>,
    pub schema_mode: crate::structural_tag::StructuralTagSchemaMode,
    pub structured_output_schema: Option<serde_json::Value>,
    pub starts_in_reasoning: bool,
    pub options: crate::structural_tag::StructuralTagOptions,
}

/// The normalized tool choice sent across the plugin boundary.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidanceChoice {
    None,
    Auto,
    Required,
    Named(String),
}

impl GuidanceRequest {
    /// Build with a static frontend-crates grammar using the same request policy as built-ins.
    pub fn build(
        &self,
        builder: &crate::structural_tag::StructuralTagBuilder,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        use crate::structural_tag::{StructuralTagContext, StructuralTagToolChoice};
        builder.build_with_options(
            &StructuralTagContext {
                tools: &self.tools,
                tool_choice: match &self.choice {
                    GuidanceChoice::None => StructuralTagToolChoice::None,
                    GuidanceChoice::Auto => StructuralTagToolChoice::Auto,
                    GuidanceChoice::Required => StructuralTagToolChoice::Required,
                    GuidanceChoice::Named(name) => StructuralTagToolChoice::Named(name),
                },
                parallel_tool_calls: self.parallel_tool_calls,
                schema_mode: self.schema_mode,
                structured_output_schema: self.structured_output_schema.as_ref(),
                starts_in_reasoning: self.starts_in_reasoning,
            },
            &self.options,
        )
    }
}

/// Optional plugin callback for parser-specific constraints.
pub type Guide = fn(GuidanceRequest) -> anyhow::Result<Option<serde_json::Value>>;

/// Rendered prompt and optional boundary before a continued assistant message.
#[derive(Serialize, Deserialize)]
pub struct PromptRequest {
    pub prompt: String,
    pub header_search_end: Option<usize>,
}

/// Optional startup capabilities that dispatch without a live parser instance.
#[derive(Default)]
pub struct Hooks {
    pub guide: Option<Guide>,
    pub inspect_prompt:
        Option<fn(PromptRequest) -> anyhow::Result<super::prefill::ParserPromptState>>,
}

/// Calls to a plugin entry point. Describe and Create take a null handle.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Describe,
    Guide(Box<GuidanceRequest>),
    InspectPrompt(PromptRequest),
    Create(Vec<Tool>),
    Initialize(UnifiedParserInit),
    Policy(ToolCallPolicy),
    PreserveMessagePhases(bool),
    Push(String),
    Finish,
    FinishAtEos,
    Reset,
    Destroy,
}

/// A plugin may commit events before reporting a parse error.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Reply {
    pub manifest: Option<Manifest>,
    #[serde(default)]
    pub structural_tag: Option<serde_json::Value>,
    pub prompt: Option<super::prefill::ParserPromptState>,
    pub events: Vec<UnifiedParserEvent>,
    pub tool_ids: Vec<(usize, String)>,
    pub recovered: String,
    pub error: Option<String>,
}

type Parser = Box<dyn UnifiedParser>;

/// SDK implementation used by the exported entry point.
///
/// # Safety
///
/// The request must point to a readable slice for this call. Non-null handles must
/// come from this entry point, have exclusive access, and be destroyed exactly once.
/// The callback must copy the reply during the call and must never unwind.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch(
    manifest: fn() -> Manifest,
    factory: UnifiedParserFactory,
    hooks: Hooks,
    abi: u32,
    handle: *mut c_void,
    request: *const u8,
    request_len: usize,
    callback: ReplyCallback,
    context: *mut c_void,
) -> *mut c_void {
    let mut reply = Reply::default();
    let mut created = None;
    let result = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<()> {
        anyhow::ensure!(abi == ABI_VERSION, "unsupported parser plugin ABI {abi}");
        anyhow::ensure!(!request.is_null(), "missing plugin request");
        // SAFETY: required by dispatch's caller contract.
        let bytes = unsafe { std::slice::from_raw_parts(request, request_len) };
        match serde_json::from_slice::<Command>(bytes)? {
            Command::Describe => {
                let manifest = manifest();
                anyhow::ensure!(
                    manifest.structural_tag == hooks.guide.is_some(),
                    "structural-tag capability must match the guidance callback"
                );
                anyhow::ensure!(
                    manifest.inspect_prompt == hooks.inspect_prompt.is_some(),
                    "prompt inspection capability must match its callback"
                );
                reply.manifest = Some(manifest);
            }
            Command::Guide(request) => {
                reply.structural_tag = hooks
                    .guide
                    .ok_or_else(|| anyhow::anyhow!("plugin has no structural-tag builder"))?(
                    *request,
                )?;
            }
            Command::InspectPrompt(request) => {
                reply.prompt =
                    Some(hooks.inspect_prompt.ok_or_else(|| {
                        anyhow::anyhow!("plugin has no prompt inspector")
                    })?(request)?);
            }
            Command::Create(tools) => created = Some(Box::new(factory(&tools)?)),
            Command::Destroy => {
                anyhow::ensure!(!handle.is_null(), "missing parser handle");
                // SAFETY: this handle was created below and is consumed once.
                drop(unsafe { Box::from_raw(handle.cast::<Parser>()) });
            }
            command => {
                // SAFETY: the caller provides exclusive access to a live handle.
                let parser = unsafe { handle.cast::<Parser>().as_mut() }
                    .ok_or_else(|| anyhow::anyhow!("missing parser handle"))?;
                apply(parser.as_mut(), command, &mut reply)?;
            }
        }
        Ok(())
    }));
    reply.error = match result {
        Ok(result) => result.err().map(|error| format!("{error:#}")),
        Err(_) => Some("parser plugin panicked".into()),
    };
    let bytes = serde_json::to_vec(&reply).unwrap_or_else(|_| {
        br#"{"manifest":null,"events":[],"tool_ids":[],"recovered":"","error":"cannot encode plugin reply"}"#.to_vec()
    });
    // SAFETY: the callback's context and lifetime are controlled by the host.
    unsafe { callback(context, bytes.as_ptr(), bytes.len()) };
    created.map_or(std::ptr::null_mut(), |parser| Box::into_raw(parser).cast())
}

fn apply(
    parser: &mut dyn UnifiedParser,
    command: Command,
    reply: &mut Reply,
) -> anyhow::Result<()> {
    let mut output = UnifiedParserOutput::default();
    let result = match command {
        Command::Initialize(init) => parser.initialize_request(init),
        Command::Policy(policy) => parser.set_tool_call_policy(policy),
        Command::PreserveMessagePhases(enabled) => {
            parser.preserve_message_phases(enabled);
            Ok(())
        }
        Command::Push(text) => parser.parse_into(&text, &mut output),
        Command::Finish => parser.finish().map(|value| output = value),
        Command::FinishAtEos => parser.finish_at_eos().map(|value| output = value),
        Command::Reset => {
            reply.recovered = parser.reset();
            Ok(())
        }
        Command::Describe
        | Command::Guide(_)
        | Command::InspectPrompt(_)
        | Command::Create(_)
        | Command::Destroy => {
            unreachable!()
        }
    };
    for event in &output.events {
        if let UnifiedParserEvent::ToolCall(call) = event
            && let Some(id) = parser.tool_call_id(call.tool_index)
        {
            reply.tool_ids.push((call.tool_index, id.to_owned()));
        }
    }
    reply.events = output.events;
    result
}

/// Export a v1 parser plugin from a Rust `cdylib`.
#[macro_export]
macro_rules! export_unified_parser_plugin {
    ($manifest:path, $factory:path) => {
        $crate::export_unified_parser_plugin!(@entry $manifest, $factory, $crate::unified::plugin::Hooks::default());
    };
    ($manifest:path, $factory:path, $guide:path) => {
        $crate::export_unified_parser_plugin!(@entry $manifest, $factory, $crate::unified::plugin::Hooks { guide: Some($guide), inspect_prompt: None });
    };
    ($manifest:path, $factory:path, hooks = $hooks:expr) => {
        $crate::export_unified_parser_plugin!(@entry $manifest, $factory, $hooks);
    };
    (@entry $manifest:path, $factory:path, $hooks:expr) => {
        /// Native parser plugin entry point.
        ///
        /// # Safety
        /// The caller must follow the SDK's buffer and handle contracts.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn dynamo_parser_plugin_v1(
            abi: u32,
            handle: *mut std::ffi::c_void,
            request: *const u8,
            length: usize,
            callback: $crate::unified::plugin::ReplyCallback,
            context: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void {
            // SAFETY: the caller must follow the exported protocol.
            unsafe {
                $crate::unified::plugin::dispatch(
                    $manifest, $factory, $hooks, abi, handle, request, length, callback, context,
                )
            }
        }
    };
}
