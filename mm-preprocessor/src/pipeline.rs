// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The model-family seam of the preprocessing pipeline.
//!
//! Design rule: **families produce data, the driver owns control flow.** A
//! family never sees the request loop, the thread pool, or the failure
//! protocol — it implements [`MmFamilyProcessor`], turning decoded media into
//! named tensors and describing its prompt geometry as a [`TokenLayout`]
//! value. `driver::process` applies the layout mechanically, so expansion,
//! per-item offsets, and position inputs all derive from one declarative
//! structure and every family gets identical failure semantics for free.
//!
//! The carriers below are `#[non_exhaustive]` so a new family can grow them
//! without a breaking change. Known future needs, validated against the
//! GLM-4V and Kimi K2.5/K3 Python processors:
//! * GLM-4V's `<|begin_of_image|>` … `<|end_of_image|>` framing is a
//!   [`TokenPattern::Explicit`] span; its M-RoPE variant is a new
//!   [`PositionOutput`] variant if it diverges from the Qwen shape.
//! * Kimi K3 interleaves tokenized text (`image {w}x{h}`) inside the media
//!   span, so `layout` will need encode access — planned as a defaulted
//!   `layout_with(&LayoutContext)` method carrying the driver's tokenize
//!   closure, added without breaking existing families.
//! * Video/audio grow [`DecodedMedia`] variants and [`Capabilities`] flags.

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

/// Prompt geometry as data: the family *describes* the expansion, the driver
/// *applies* it ([`crate::token_layout::apply_layout`]) — deriving final input
/// ids and per-item offsets, and validating that every item is placed exactly
/// once.
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

/// The per-model-family hooks of the pipeline. Adding a family =
/// implementing this in `src/models/<model>.rs` and adding its `family` arm
/// to [`crate::registry::PipelineSpec`]. All parameters come from the runtime
/// spec (resolved from the HF config by the consumer); nothing is hardcoded
/// per model.
pub trait MmFamilyProcessor: Send + Sync {
    /// Modalities beyond images this family accepts. Default: images only.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Preprocess one decoded media item: the model's HF processor
    /// equivalent (resize/tile/normalize/patchify → named tensors) plus the
    /// geometry `layout`/`positions` will need.
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String>;

    /// Describe how the prompt expands around the processed items (in
    /// prompt order). Sees the full original prompt and all items, so
    /// structured schemes (tile markers, separators) are expressible.
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
