// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The model-family seam: the interface a family implements and the data
//! carriers it exchanges with the engine's driver.
//!
//! Design rule: **families produce data, the driver owns control flow.** A
//! family never sees the request loop, thread pool, or failure protocol — it
//! turns decoded media into named tensors and describes its prompt geometry
//! as a [`TokenLayout`], which the driver applies mechanically
//! ([`crate::token_layout::apply_layout`]).
//!
//! The carriers are `#[non_exhaustive]` so new families, modalities, and
//! position schemes are semver-minor additions; README §5 has the growth
//! plan.

/// Typed tensor payload. Grows a variant per dtype actually produced by a
/// family — not speculatively.
#[non_exhaustive]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

/// Named auxiliary tensors that reach the model runner as kwargs — e.g.
/// qwen's `image_grid_thw`.
pub type NamedTensors = Vec<(String, Tensor)>;

/// One decoded media item handed to [`MmFamilyProcessor::process_item`].
/// Grows a variant per modality as families that need it are ported.
#[non_exhaustive]
pub enum DecodedMedia {
    /// HWC u8 RGB.
    Image {
        rgb: Vec<u8>,
        height: usize,
        width: usize,
    },
}

/// Family-internal geometry of one processed item, consumed by
/// [`MmFamilyProcessor::layout`] / [`MmFamilyProcessor::positions`]. Grows a
/// variant per family style; the driver never interprets it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Geometry {
    /// `[t, h, w]` patch grid (`t` = 1 for still images).
    Grid([u32; 3]),
}

/// One processed media item: the primary feature tensor, named auxiliary
/// tensors, and the geometry the family's own `layout`/`positions` hooks need.
pub struct ProcessedItem {
    /// The model's feature tensor for this item (qwen: `pixel_values`).
    pub feature: Tensor,
    pub aux: NamedTensors,
    pub geometry: Geometry,
}

/// The tokens one media item occupies in the expanded prompt.
#[non_exhaustive]
pub enum TokenPattern {
    /// N copies of one placeholder id (qwen-style).
    Repeat { id: i32, n: usize },
    /// An explicit id sequence — tile markers, row separators, wrapper
    /// tokens (minicpm/internvl-style structured expansions).
    Explicit(Vec<i32>),
}

/// One span of the expanded prompt.
pub enum Segment {
    /// Copy `src` (a range into the original ids) verbatim.
    Text(std::ops::Range<usize>),
    /// Media item `item`'s token span.
    Media { item: usize, pattern: TokenPattern },
}

/// Prompt geometry as data: the family describes the expansion,
/// [`crate::token_layout::apply_layout`] derives final input ids and
/// per-item offsets from it.
pub struct TokenLayout {
    pub segments: Vec<Segment>,
}

/// Modalities a family accepts; a serving engine rejects anything a family
/// does not declare.
#[derive(Clone, Copy, Debug, Default)]
pub struct Capabilities {
    pub video: bool,
    pub audio: bool,
}

/// Position scheme of the expanded prompt.
#[non_exhaustive]
pub enum PositionOutput {
    /// Plain sequential positions — the consumer needs nothing extra.
    Rope1D,
    /// M-RoPE: flattened row-major `[3, input_len]` positions + the position
    /// delta (`max + 1 - input_len`).
    MRope { positions: Vec<i64>, delta: i64 },
}

/// The per-model-family hooks: implement in `src/models/<model>.rs` and add a
/// `family` arm to [`crate::registry::ProcessorSpec`]. All parameters come
/// from the runtime spec; nothing is hardcoded per model.
pub trait MmFamilyProcessor: Send + Sync {
    /// Modalities beyond images this family accepts. Default: images only.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Tokens one image will occupy in the expanded prompt, from its source
    /// dimensions alone — no pixel work (HF's `_get_num_multimodal_tokens`).
    /// Lets routers account prompt length before fetching past the header.
    fn num_media_tokens(&self, width: usize, height: usize) -> Result<usize, String>;

    /// Preprocess one decoded media item — the model's HF processor
    /// equivalent (resize/tile/normalize/patchify) — into tensors plus the
    /// geometry `layout`/`positions` will need.
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String>;

    /// Describe how the prompt expands around the processed items (in prompt
    /// order). Sees the full prompt and all items, so structured schemes
    /// (tile markers, separators) are expressible.
    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String>;

    /// Positions for the expanded prompt. Families without a custom scheme
    /// keep the default.
    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let _ = (input_len, offsets, items);
        Ok(PositionOutput::Rope1D)
    }
}
