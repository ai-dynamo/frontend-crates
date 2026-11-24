// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native parser plugin protocols. Only C ABI values and UTF-8 JSON
//! cross the library boundary. The SDK continues to use the Rust parser trait locally.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::{Deserialize, Serialize};

use super::{
    ForcedToolGuidance, RequiredVocabularyToken, Tool, ToolCallPolicy, UnifiedParser,
    UnifiedParserContinuation, UnifiedParserDescriptor, UnifiedParserEvent, UnifiedParserFactory,
    UnifiedParserInit, UnifiedParserInputCapabilities, UnifiedParserOutput,
};

/// Version of the entry point and its message schema.
pub const ABI_VERSION: u32 = 1;
pub const ABI_VERSION_V2: u32 = 2;

/// Host callback that copies one reply before the plugin returns.
pub type ReplyCallback = unsafe extern "C" fn(*mut c_void, *const u8, usize);

/// Exported function signature. A successful handle-producing command transfers
/// ownership of the returned handle to the host.
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
        descriptor.guidance.from_reasoning_selector = self.structural_tag;
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

/// V2 keeps the text protocol payloads and adds token and checkpoint capabilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestV2 {
    pub base: Manifest,
    pub text_input: bool,
    pub token_id_input: bool,
    pub continuation: bool,
    #[serde(default)]
    pub discard_native_flush_calls_on_truncation: bool,
    #[serde(default)]
    pub forced_tool_guidance: ForcedToolGuidance,
}

impl ManifestV2 {
    pub fn input_capabilities(&self) -> UnifiedParserInputCapabilities {
        UnifiedParserInputCapabilities {
            text: self.text_input,
            token_ids: self.token_id_input,
        }
    }

    pub fn descriptor(self) -> UnifiedParserDescriptor {
        let mut descriptor = self.base.descriptor();
        descriptor.text_input = self.text_input;
        descriptor.token_id_input = self.token_id_input;
        descriptor.continuation = self.continuation;
        descriptor.discard_native_flush_calls_on_truncation =
            self.discard_native_flush_calls_on_truncation;
        descriptor.notify_model_eos = true;
        descriptor.guidance.forced_tool_choice = self.forced_tool_guidance;
        descriptor
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.text_input || self.token_id_input,
            "plugin must accept decoded text or token IDs"
        );
        anyhow::ensure!(
            !self.forced_tool_guidance.prefers_tag(true) || self.base.structural_tag,
            "forced-tool tag policy requires a structural-tag builder"
        );
        Ok(())
    }
}

/// V2 wire requests extend the V1 commands through a separate entry point.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandV2 {
    Describe,
    V1(Command),
    PushTokenIds(Vec<u32>),
    Checkpoint,
    Fork,
    DropCheckpoint,
}

#[derive(Default, Serialize, Deserialize)]
pub struct ReplyV2 {
    pub manifest_v2: Option<ManifestV2>,
    pub base: Reply,
}

type Parser = Box<dyn UnifiedParser>;
type Checkpoint = Box<dyn UnifiedParserContinuation>;

enum Handle {
    Parser(Parser),
    Checkpoint(Checkpoint),
}

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
                validate_hooks(&manifest, &hooks)?;
                reply.manifest = Some(manifest);
            }
            Command::Guide(request) => {
                reply.structural_tag = guide(&hooks, *request)?;
            }
            Command::InspectPrompt(request) => {
                reply.prompt = Some(inspect_prompt(&hooks, request)?);
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

/// Dispatch the version 2 protocol.
///
/// # Safety
///
/// Buffers and callbacks follow [`dispatch`]. Handles are owned exactly once
/// and may only be supplied to the entry point that created them.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dispatch_v2(
    manifest: fn() -> ManifestV2,
    factory: UnifiedParserFactory,
    hooks: Hooks,
    abi: u32,
    handle: *mut c_void,
    request: *const u8,
    request_len: usize,
    callback: ReplyCallback,
    context: *mut c_void,
) -> *mut c_void {
    let mut reply = ReplyV2::default();
    let result = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<Option<Handle>> {
        anyhow::ensure!(abi == ABI_VERSION_V2, "unsupported parser plugin ABI {abi}");
        anyhow::ensure!(!request.is_null(), "missing plugin request");
        // SAFETY: the request buffer is valid for this call.
        let bytes = unsafe { std::slice::from_raw_parts(request, request_len) };
        let command = serde_json::from_slice::<CommandV2>(bytes)?;
        match command {
            CommandV2::Describe => {
                anyhow::ensure!(handle.is_null(), "describe takes no handle");
                let manifest = manifest();
                validate_hooks(&manifest.base, &hooks)?;
                manifest.validate()?;
                reply.manifest_v2 = Some(manifest);
            }
            CommandV2::V1(Command::Create(tools)) => {
                anyhow::ensure!(handle.is_null(), "create takes no handle");
                let parser = factory(&tools)?;
                anyhow::ensure!(
                    parser.input_capabilities() == manifest().input_capabilities(),
                    "parser input capabilities differ from the manifest"
                );
                return Ok(Some(Handle::Parser(parser)));
            }
            CommandV2::V1(Command::Guide(request)) => {
                anyhow::ensure!(handle.is_null(), "guide takes no handle");
                reply.base.structural_tag = guide(&hooks, *request)?;
            }
            CommandV2::V1(Command::InspectPrompt(request)) => {
                anyhow::ensure!(handle.is_null(), "prompt inspection takes no handle");
                reply.base.prompt = Some(inspect_prompt(&hooks, request)?);
            }
            CommandV2::V1(Command::Describe) => {
                anyhow::bail!("use the v2 describe command");
            }
            CommandV2::Fork => {
                // SAFETY: the caller supplies a live handle and serializes access.
                let Some(Handle::Checkpoint(checkpoint)) =
                    (unsafe { handle.cast::<Handle>().as_ref() })
                else {
                    anyhow::bail!("fork requires a checkpoint handle");
                };
                return Ok(Some(Handle::Parser(checkpoint.fork()?)));
            }
            CommandV2::DropCheckpoint | CommandV2::V1(Command::Destroy) => {
                // SAFETY: the caller supplies a live handle and serializes access.
                let live = unsafe { handle.cast::<Handle>().as_ref() };
                anyhow::ensure!(
                    matches!(
                        (command, live),
                        (CommandV2::DropCheckpoint, Some(Handle::Checkpoint(_)))
                            | (CommandV2::V1(Command::Destroy), Some(Handle::Parser(_)))
                    ),
                    "destroy command does not match handle type"
                );
                // SAFETY: the checked handle is owned by this entry point.
                drop(unsafe { Box::from_raw(handle.cast::<Handle>()) });
            }
            command => {
                // SAFETY: the caller supplies exclusive access to a live handle.
                let Some(Handle::Parser(parser)) = (unsafe { handle.cast::<Handle>().as_mut() })
                else {
                    anyhow::bail!("command requires a parser handle");
                };
                match command {
                    CommandV2::Checkpoint => {
                        anyhow::ensure!(
                            manifest().continuation,
                            "checkpoint capability is disabled"
                        );
                        return Ok(parser.continuation()?.map(Handle::Checkpoint));
                    }
                    CommandV2::PushTokenIds(ids) => {
                        let mut output = UnifiedParserOutput::default();
                        let result = parser.parse_token_ids_into(&ids, &mut output);
                        record_output(parser.as_ref(), output, &mut reply.base);
                        result?;
                    }
                    CommandV2::V1(command) => apply(parser.as_mut(), command, &mut reply.base)?,
                    _ => unreachable!(),
                }
            }
        }
        Ok(None)
    }));
    let created = match result {
        Ok(Ok(created)) => created,
        Ok(Err(error)) => {
            reply.base.error = Some(format!("{error:#}"));
            None
        }
        Err(_) => {
            reply.base.error = Some("parser plugin panicked".into());
            None
        }
    };
    match serde_json::to_vec(&reply) {
        Ok(bytes) => {
            // SAFETY: the callback copies the reply before returning.
            unsafe { callback(context, bytes.as_ptr(), bytes.len()) };
            created.map_or(std::ptr::null_mut(), |handle| {
                Box::into_raw(Box::new(handle)).cast()
            })
        }
        Err(_) => {
            let bytes = br#"{"manifest_v2":null,"base":{"error":"cannot encode plugin reply"}}"#;
            // SAFETY: the callback copies the reply before returning.
            unsafe { callback(context, bytes.as_ptr(), bytes.len()) };
            std::ptr::null_mut()
        }
    }
}

fn validate_hooks(manifest: &Manifest, hooks: &Hooks) -> anyhow::Result<()> {
    anyhow::ensure!(
        manifest.structural_tag == hooks.guide.is_some(),
        "structural-tag capability must match the guidance callback"
    );
    anyhow::ensure!(
        manifest.inspect_prompt == hooks.inspect_prompt.is_some(),
        "prompt inspection capability must match its callback"
    );
    Ok(())
}

fn guide(hooks: &Hooks, request: GuidanceRequest) -> anyhow::Result<Option<serde_json::Value>> {
    hooks
        .guide
        .ok_or_else(|| anyhow::anyhow!("plugin has no structural-tag builder"))?(request)
}

fn inspect_prompt(
    hooks: &Hooks,
    request: PromptRequest,
) -> anyhow::Result<super::prefill::ParserPromptState> {
    hooks
        .inspect_prompt
        .ok_or_else(|| anyhow::anyhow!("plugin has no prompt inspector"))?(request)
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
    record_output(parser, output, reply);
    result
}

fn record_output(parser: &dyn UnifiedParser, output: UnifiedParserOutput, reply: &mut Reply) {
    for event in &output.events {
        if let UnifiedParserEvent::ToolCall(call) = event
            && let Some(id) = parser.tool_call_id(call.tool_index)
        {
            reply.tool_ids.push((call.tool_index, id.to_owned()));
        }
    }
    reply.events = output.events;
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

/// Export a v2 parser plugin from a Rust `cdylib`.
#[macro_export]
macro_rules! export_unified_parser_plugin_v2 {
    ($manifest:path, $factory:path) => {
        $crate::export_unified_parser_plugin_v2!(
            $manifest,
            $factory,
            hooks = $crate::unified::plugin::Hooks::default()
        );
    };
    ($manifest:path, $factory:path, hooks = $hooks:expr) => {
        /// Native parser plugin entry point.
        ///
        /// # Safety
        /// The caller must follow the SDK's buffer and handle contracts.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn dynamo_parser_plugin_v2(
            abi: u32,
            handle: *mut std::ffi::c_void,
            request: *const u8,
            length: usize,
            callback: $crate::unified::plugin::ReplyCallback,
            context: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void {
            // SAFETY: the caller must follow the exported protocol.
            unsafe {
                $crate::unified::plugin::dispatch_v2(
                    $manifest, $factory, $hooks, abi, handle, request, length, callback, context,
                )
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct TokenParser(Vec<u32>);

    impl UnifiedParser for TokenParser {
        fn input_capabilities(&self) -> UnifiedParserInputCapabilities {
            manifest().input_capabilities()
        }

        fn continuation(&mut self) -> anyhow::Result<Option<Box<dyn UnifiedParserContinuation>>> {
            Ok(Some(Box::new(self.clone())))
        }

        fn parse_into(&mut self, _: &str, _: &mut UnifiedParserOutput) -> anyhow::Result<()> {
            anyhow::bail!("token input required");
        }

        fn parse_token_ids_into(
            &mut self,
            ids: &[u32],
            _: &mut UnifiedParserOutput,
        ) -> anyhow::Result<()> {
            self.0.extend_from_slice(ids);
            Ok(())
        }

        fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            Ok(UnifiedParserOutput::default())
        }

        fn finish_at_eos(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            Ok(UnifiedParserOutput {
                events: vec![UnifiedParserEvent::Text(format!("{:?}", self.0))],
            })
        }
    }

    impl UnifiedParserContinuation for TokenParser {
        fn fork(&self) -> anyhow::Result<Box<dyn UnifiedParser>> {
            Ok(Box::new(self.clone()))
        }
    }

    fn factory(_: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>> {
        Ok(Box::new(TokenParser::default()))
    }

    fn manifest() -> ManifestV2 {
        ManifestV2 {
            base: Manifest {
                family: "test_token_parser".into(),
                aliases: vec!["test-token-parser".into()],
                preserve_special_tokens: false,
                native_completion_authoritative: false,
                required_tokens: Vec::new(),
                structural_tag: false,
                inspect_prompt: false,
            },
            text_input: false,
            token_id_input: true,
            continuation: true,
            discard_native_flush_calls_on_truncation: false,
            forced_tool_guidance: ForcedToolGuidance::Generic,
        }
    }

    fn exchange(handle: *mut c_void, command: CommandV2) -> (ReplyV2, *mut c_void) {
        unsafe extern "C" fn reply(context: *mut c_void, data: *const u8, len: usize) {
            // SAFETY: exchange supplies the output slot and the dispatcher owns the reply.
            unsafe {
                let out = &mut *context.cast::<Vec<u8>>();
                out.extend_from_slice(std::slice::from_raw_parts(data, len));
            }
        }
        let request = serde_json::to_vec(&command).unwrap();
        let mut output = Vec::<u8>::new();
        // SAFETY: buffers live for the call and the test owns all handles.
        let returned = unsafe {
            dispatch_v2(
                manifest,
                factory,
                Hooks::default(),
                ABI_VERSION_V2,
                handle,
                request.as_ptr(),
                request.len(),
                reply,
                (&mut output as *mut Vec<u8>).cast(),
            )
        };
        (serde_json::from_slice(&output).unwrap(), returned)
    }

    fn call(handle: *mut c_void, command: CommandV2) -> (ReplyV2, *mut c_void) {
        let (reply, returned) = exchange(handle, command);
        assert!(reply.base.error.is_none(), "{:?}", reply.base.error);
        (reply, returned)
    }

    #[test]
    fn token_checkpoints_survive_parent_and_forks_finish_independently() {
        let null = std::ptr::null_mut();
        let (description, _) = call(null, CommandV2::Describe);
        let descriptor = description.manifest_v2.unwrap().descriptor();
        assert!(descriptor.token_id_input && descriptor.continuation);
        assert!(!descriptor.text_input);
        assert_eq!(descriptor.aliases, ["test-token-parser"]);
        let (_, parent) = call(null, CommandV2::V1(Command::Create(Vec::new())));
        call(parent, CommandV2::PushTokenIds(vec![11, 12]));
        let (_, checkpoint) = call(parent, CommandV2::Checkpoint);
        let (bad_drop, _) = exchange(checkpoint, CommandV2::V1(Command::Destroy));
        assert!(bad_drop.base.error.unwrap().contains("handle type"));
        call(parent, CommandV2::V1(Command::Destroy));
        let (_, first) = call(checkpoint, CommandV2::Fork);
        let (_, second) = call(checkpoint, CommandV2::Fork);
        call(checkpoint, CommandV2::DropCheckpoint);

        call(first, CommandV2::PushTokenIds(vec![13]));
        call(second, CommandV2::PushTokenIds(vec![21]));
        let (completed, _) = call(first, CommandV2::V1(Command::FinishAtEos));
        assert!(matches!(
            completed.base.events.as_slice(),
            [UnifiedParserEvent::Text(text)] if text == "[11, 12, 13]"
        ));
        let (truncated, _) = call(second, CommandV2::V1(Command::Finish));
        assert!(truncated.base.events.is_empty());
        call(first, CommandV2::V1(Command::Destroy));
        call(second, CommandV2::V1(Command::Destroy));
    }

    #[test]
    fn mandatory_tag_requires_a_declared_builder() {
        let mut manifest = manifest();
        manifest.forced_tool_guidance = ForcedToolGuidance::RequireTagForForced;
        assert!(manifest.validate().is_err());
        manifest.base.structural_tag = true;
        assert!(manifest.validate().is_ok());
    }
}
