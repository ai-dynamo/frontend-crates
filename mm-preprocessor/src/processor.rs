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

/// A media modality understood by processor capability and metadata APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Modality {
    Image,
    Video,
    Audio,
}

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

impl DecodedMedia {
    pub fn modality(&self) -> Modality {
        match self {
            Self::Image { .. } => Modality::Image,
        }
    }
}

/// Lightweight media metadata for pixel-free token accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaMetadata {
    Image {
        width: u32,
        height: u32,
    },
    Video {
        width: u32,
        height: u32,
        num_frames: u32,
    },
    Audio {
        num_samples: u64,
        sample_rate: u32,
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

/// One processed media item and the metadata needed for prompt preparation.
pub struct ProcessedItem {
    pub modality: Modality,
    /// Number of model feature tokens this item contributes.
    pub feature_token_count: usize,

    pub feature: Tensor,
    pub aux: NamedTensors,

    pub geometry: Option<Geometry>,
}

/// One part of a media item's expansion. The two variants tell the engine
/// which positions receive feature embeddings, so it never has to recognize
/// special ids itself.
#[non_exhaustive]
pub enum ExpansionPart {
    /// `n` copies of the placeholder `id`. These are the positions the
    /// engine fills with the item's feature embeddings (for a Qwen image:
    /// `<|image_pad|>` repeated once per feature token).
    Feature { id: i32, n: usize },
    /// Fixed ids inserted as-is; the model reads them as normal text.
    /// Examples: `<|vision_start|>` / `<|vision_end|>` markers, video
    /// timestamps, tile separators.
    Literal(Vec<i32>),
}

/// One span of the expanded prompt.
pub enum Segment {
    /// Copy this range of the original ids unchanged.
    Text(std::ops::Range<usize>),
    /// Replace the original tokens at `src` (the item's placeholder) with
    /// `expansion`. An image is a single `Feature` part; a video can mix
    /// both kinds, e.g. `[<vision_start>, timestamp, feature × N,
    /// <vision_end>]`.
    Media {
        item: usize,
        src: std::ops::Range<usize>,
        expansion: Vec<ExpansionPart>,
    },
}

/// Prompt geometry as data: the family describes the expansion,
/// [`crate::token_layout::apply_layout`] derives final input ids and
/// per-item offsets from it.
pub struct TokenLayout {
    pub segments: Vec<Segment>,
}

/// Modalities a family accepts.
#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    image: bool,
    video: bool,
    audio: bool,
}

impl Capabilities {
    pub const fn new(image: bool, video: bool, audio: bool) -> Self {
        Self {
            image,
            video,
            audio,
        }
    }

    pub const fn supports(&self, modality: Modality) -> bool {
        match modality {
            Modality::Image => self.image,
            Modality::Video => self.video,
            Modality::Audio => self.audio,
        }
    }
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::new(true, false, false)
    }
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
    /// Modalities this family accepts. Default: images only.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Tokens one media item will occupy in the expanded prompt, from
    /// lightweight metadata alone — no pixel or feature-extraction work (HF's
    /// `_get_num_multimodal_tokens`). Lets routers account prompt length after
    /// probing only the modality-specific header.
    fn num_media_tokens(&self, media: &MediaMetadata) -> crate::Result<usize>;

    /// Preprocess one decoded media item — the model's HF processor
    /// equivalent (resize/tile/normalize/patchify) — into tensors plus the
    /// geometry `layout`/`positions` will need.
    fn process_item(&self, media: &DecodedMedia) -> crate::Result<ProcessedItem>;

    /// Describe how the prompt expands around the processed items (in prompt
    /// order). Sees the full prompt and all items, so structured schemes
    /// (tile markers, separators) are expressible.
    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> crate::Result<TokenLayout>;

    /// Positions for the expanded prompt. Families without a custom scheme
    /// keep the default.
    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[ProcessedItem],
    ) -> crate::Result<PositionOutput> {
        let _ = (input_len, offsets, items);
        Ok(PositionOutput::Rope1D)
    }
}
