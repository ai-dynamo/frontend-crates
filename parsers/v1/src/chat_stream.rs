// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The stream item that the chat stream stages read and rewrite.
//!
//! The reasoning stream stages and the jail's terminal-error contract were
//! moved from dynamo `lib/llm/src/preprocessor.rs`. There they run on
//! dynamo-runtime's `Annotated` over dynamo-llm's chunk type, which carries
//! dynamo-only fields (metrics, `nvext`, prompt logprobs) beside the OpenAI
//! chunk. This crate cannot name those types. The stages are generic over
//! [`ChatStreamItem`] instead, so dynamo keeps its own item and its fields pass
//! through each stage untouched. A concrete item type here would make dynamo
//! strip and re-attach those fields around every stage, and the end-of-stream
//! flush chunk has no input chunk to take them from.
//!
//! This crate implements the trait for its own [`Annotated`] over
//! [`CreateChatCompletionStreamResponse`], the item the jail already uses.

use dynamo_protocols::types::CreateChatCompletionStreamResponse;

use crate::tool_calling::jail::Annotated;

/// A stream item that may carry one OpenAI chat completion chunk.
pub trait ChatStreamItem: Clone + Send + 'static {
    /// Whether this item is a terminal error from the backend.
    ///
    /// dynamo-runtime decides this from the annotation event (`event ==
    /// "error"`), not from the payload.
    fn is_error(&self) -> bool;

    /// The chat chunk, if this item carries one.
    fn chunk(&self) -> Option<&CreateChatCompletionStreamResponse>;

    /// The chat chunk, if this item carries one, for in-place rewrite.
    fn chunk_mut(&mut self) -> Option<&mut CreateChatCompletionStreamResponse>;

    /// Clear the transport and per-chunk fields of a copy that becomes a
    /// synthetic chunk.
    ///
    /// A stage that flushes held text at the end of the stream clones an
    /// earlier item as the envelope. That chunk consumed no backend data, so
    /// it must not repeat the earlier item's usage or metadata.
    fn scrub_synthetic(&mut self);
}

impl ChatStreamItem for Annotated<CreateChatCompletionStreamResponse> {
    fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }

    fn chunk(&self) -> Option<&CreateChatCompletionStreamResponse> {
        self.data.as_ref()
    }

    fn chunk_mut(&mut self) -> Option<&mut CreateChatCompletionStreamResponse> {
        self.data.as_mut()
    }

    fn scrub_synthetic(&mut self) {
        self.event = None;
        self.comment = None;
        self.error = None;
        if let Some(data) = self.data.as_mut() {
            data.usage = None;
        }
    }
}
