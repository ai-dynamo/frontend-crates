// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared multimodal request driver.
//!
//! Owns the request control flow — parallel fan-out, layout application,
//! failure semantics — while every model decision lives behind
//! [`MmFamilyProcessor`] (see `pipeline.rs`). Families produce data; they
//! cannot alter orchestration.

#[cfg(feature = "fetch")]
use crate::fetch;
use crate::pipeline::{MmFamilyProcessor, NamedTensors, PositionOutput, Tensor};

/// Per-request bounds: together with the per-source fetch cap
/// (`fetch::MAX_FETCH_BYTES`) they cap what one request can make the
/// pipeline buffer.
pub const MAX_ITEMS_PER_REQUEST: usize = 64;
pub const MAX_REQUEST_BYTES: u64 = 256 << 20;

/// One raw image source from the request.
#[derive(Debug)]
pub enum ImageSource {
    /// `data:`/base64/file/http — resolved by `fetch::fetch_bytes` (requires
    /// the `fetch` feature; rejected without it).
    String(String),
    /// Already-raw encoded image bytes.
    Bytes(Vec<u8>),
}

/// Typed multimodal request input. The serving engine's message layer owns
/// the wire format and parses its payload into this before calling
/// [`process`].
pub struct MmInput {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub images: Vec<ImageSource>,
}

/// One processed media item at the request boundary.
pub struct OutputItem {
    pub feature: Tensor,
    pub aux: NamedTensors,
    /// [`content_hash_u64`](crate::content_hash_u64) of the raw encoded
    /// source bytes.
    pub hash: u64,
}

/// The per-request result handed back to the serving engine.
pub struct Output {
    pub input_ids: Vec<i32>,
    /// In prompt order; `offsets[i]` is `items[i]`'s inclusive token range.
    pub items: Vec<OutputItem>,
    pub offsets: Vec<(u32, u32)>,
    pub positions: PositionOutput,
}

/// Options for [`process_with`]; [`Default`] matches [`process`].
#[derive(Default)]
#[non_exhaustive]
pub struct ProcessOptions {
    /// Knobs of the inline fetch of string sources.
    #[cfg(feature = "fetch")]
    pub fetch: fetch::FetchOptions,
}

#[cfg(feature = "fetch")]
impl From<fetch::FetchOptions> for ProcessOptions {
    fn from(fetch: fetch::FetchOptions) -> Self {
        Self { fetch }
    }
}

/// Run one request through the pipeline. Any `Err` rejects the request back
/// to the client — including inputs merely outside the pipeline's scope
/// (video/audio, precomputed features, undecodable images), since there is
/// no fallback path.
pub fn process(
    family: &dyn MmFamilyProcessor,
    input: MmInput,
    tokenize: impl Fn(&str) -> Result<Vec<i32>, String>,
) -> Result<Output, String> {
    process_with(family, input, tokenize, &ProcessOptions::default())
}

/// [`process`] with explicit [`ProcessOptions`].
pub fn process_with(
    family: &dyn MmFamilyProcessor,
    input: MmInput,
    tokenize: impl Fn(&str) -> Result<Vec<i32>, String>,
    opts: &ProcessOptions,
) -> Result<Output, String> {
    let _ = (family, input, tokenize, opts);
    // PR3, staged as:
    // 1. reject empty-image and over-cap requests (items, total bytes);
    // 2. resolve sources — inline and sequential, never on the CPU pool
    //    (callers on a fixed worker pool prefetch I/O sources themselves);
    // 3. per item, in parallel via `par::try_map`: content hash → decode →
    //    `family.process_item`;
    // 4. input ids from the request, else `tokenize(text)`;
    // 5. `family.layout` → `token_layout::apply_layout` → expanded ids +
    //    per-item offsets;
    // 6. `family.positions`.
    todo!("PR3: the request control flow")
}
