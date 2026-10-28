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
//! [`CreateChatCompletionStreamResponse`], the item the jail already uses, and
//! for [`SourcedChunk`], the same item with a source slot.
//!
//! An item may carry a [`SourceSlot`]. When the input items have one, each
//! stage records where its text came from and writes, for each choice of each
//! output chunk, the input byte ranges that tile its `content` and its
//! `reasoning_content`. Items without a slot record nothing.

use dynamo_protocols::types::CreateChatCompletionStreamResponse;

use crate::sources::{ChoiceSources, SourceSlot};
use crate::tool_calling::jail::Annotated;

/// A stream item that may carry one OpenAI chat completion chunk.
pub trait ChatStreamItem: SourceSlot + Clone + Send + 'static {
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

/// [`Annotated`] carries no source slot.
impl SourceSlot for Annotated<CreateChatCompletionStreamResponse> {}

/// A chat chunk with a source slot: the item to feed the v1 stream stages and
/// the jail when the caller needs to know which input bytes each output byte
/// came from.
///
/// `sources` is `Some` on every item of a recording stream, with one entry
/// per choice of `chunk` in the same order. On input, an entry gives the
/// caller's byte ranges of that choice's `content` (and `reasoning_content`, if
/// any). On output, the stages write the ranges of the text they emit. A
/// choice whose stage cannot record gets no entry, so the caller can tell that
/// its text has no source.
#[derive(Clone, Debug)]
pub struct SourcedChunk {
    pub chunk: Annotated<CreateChatCompletionStreamResponse>,
    pub sources: Option<Vec<ChoiceSources>>,
}

impl SourceSlot for SourcedChunk {
    fn sources(&self) -> Option<&[ChoiceSources]> {
        self.sources.as_deref()
    }

    fn sources_mut(&mut self) -> Option<&mut Vec<ChoiceSources>> {
        self.sources.as_mut()
    }
}

impl ChatStreamItem for SourcedChunk {
    fn is_error(&self) -> bool {
        self.chunk.is_error()
    }

    fn chunk(&self) -> Option<&CreateChatCompletionStreamResponse> {
        self.chunk.chunk()
    }

    fn chunk_mut(&mut self) -> Option<&mut CreateChatCompletionStreamResponse> {
        self.chunk.chunk_mut()
    }

    fn scrub_synthetic(&mut self) {
        self.chunk.scrub_synthetic();
    }
}
