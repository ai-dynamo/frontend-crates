// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The rules an OpenAI chat frontend uses to pick and drive Dynamo's parsers.
//!
//! `dynamo-parsers` (v1) and `dynamo-parsers-v2` parse model output, but neither
//! chooses which parser serves a request, nor changes the request so the parser can
//! work. The Dynamo frontend does both, keyed on the deployment's tool-call and
//! reasoning parser names. This crate holds those rules, so another frontend gives a
//! client the same reply as Dynamo:
//!
//! - [`version`]: the parser generation (`auto`, `1`, `2`). Callers pass it in; the
//!   crate never reads the environment.
//! - [`families`]: the unified v2 family a parser pair selects, with Dynamo's aliases.
//! - [`route`]: the path a request takes (unified v2, the v1 jail, or no tool parsing),
//!   and the start-up version check.
//! - [`prefill`]: whether the rendered prompt already opened a reasoning block.
//! - [`special_tokens`]: the `skip_special_tokens` default per parser.
//! - [`stop_tokens`]: a tool-call end token that is a hidden EOS becomes a visible stop.
//! - [`tools`]: the effective tools and the tool-parsing gate.
//! - [`tool_choice`]: the grammar a forced `tool_choice` installs, and the guided-tool
//!   constraint a request implies.
//! - [`unified_stream`]: the v2 stream adapter. One unified parser per choice turns
//!   raw model text into OpenAI reasoning, content, and tool-call chunks.
//! - [`stream_item`]: the stream item the adapter reads, so Dynamo keeps its own
//!   chunk fields.
//! - [`guided`]: the tool-output grammar a request installed.

pub mod families;
pub mod guided;
pub mod prefill;
pub mod route;
mod spans;
pub mod special_tokens;
pub mod stop_tokens;
pub mod stream_item;
pub mod tool_choice;
pub mod tools;
pub mod unified_stream;
pub mod version;

pub use route::{InvalidRequest, ToolProcessingRoute, tool_processing_route};
pub use version::ParserVersion;
