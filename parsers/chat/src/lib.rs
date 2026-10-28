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
//! - [`unified_stream`]: the v2 stream adapter. One unified parser per choice turns
//!   raw model text into OpenAI reasoning, content, and tool-call chunks. Also the
//!   batch parse of a whole reply.
//! - [`aggregator`]: the non-streaming finalize. The chunks of a `stream: false`
//!   reply become one response.
//! - [`stream_item`]: the stream item the adapter reads, so Dynamo keeps its own
//!   chunk fields.
//! - [`guided`]: the tool-output grammar a request installed.
//! - [`chat_stream`]: the parse stage of a chat stream. It picks the stages for a
//!   request (the v2 adapter, or the v1 reasoning stream and the caller's jail)
//!   and chains the stream policies after them.
//! - [`reasoning_off`]: whether a request turned reasoning off, and the fold of
//!   reasoning into content when it did.
//! - [`nonempty_content`]: the `force_nonempty_content` contract on a stream, and
//!   whether a stream can hold back all output (the keep-alive predicate).
//! - [`tool_call_policy`]: the request's tool-call policy on parsed output, and the
//!   assistant role on each choice's first delta.
//! - [`reasoning_usage`]: the `reasoning_tokens` estimate in the usage chunk.

pub mod aggregator;
pub mod chat_stream;
pub mod families;
pub mod guided;
pub mod nonempty_content;
pub mod prefill;
pub mod reasoning_off;
pub mod reasoning_usage;
pub mod route;
pub mod special_tokens;
pub mod stop_tokens;
pub mod stream_item;
pub mod tool_call_policy;
pub mod tools;
pub mod unified_stream;
pub mod version;

pub use chat_stream::{
    ChatStreamParsers, ChatStreamPlan, JailRequest, PolicyStreamItem, parse_chat_stream,
};
pub use reasoning_off::ChatStreamRequest;
pub use route::{InvalidRequest, ToolProcessingRoute, tool_processing_route};
pub use version::ParserVersion;
