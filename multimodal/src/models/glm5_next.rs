// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Geometry math adapted from transformers' image_processing_glm5_next.py (Apache-2.0).
// Pinned source: huggingface/transformers@c4c4605f961877852b89b97478536bb3a26b6fd1 (v5.19.0).

//! Pixel-free GLM-5.3-Flash (`glm5_next`) image geometry, shared by routers
//! and workers.
//!
//! The registered processor supports metadata-only token counting and prompt
//! layout. The chat template renders each image as
//! `<|begin_of_image|><|image|><|end_of_image|>`; the HF processor replaces
//! the single `<|image|>` with one copy per merged patch of the canvas that
//! [`smart_resize`] selects, and the delimiters stay ordinary text. vLLM,
//! SGLang, and TensorRT-LLM all take `image_grid_thw` from that HF image
//! processor, so this planner is the single geometry contract for all three.
//!
//! It does not decode, resize pixels, or produce vision embeddings, and it
//! does not cover video: GLM expands video frames with the *image* token, so
//! [`MmFamilyProcessor::layout_from_metadata`] refuses prompts that carry the
//! video placeholder rather than mis-attributing those positions.

use crate::processor::{
    DecodedMedia, MediaMetadata, MmFamilyProcessor, ProcessedItem, TokenLayout,
};
use crate::{MmError, Result};

/// Largest integer operand for which Python's `int / int` true division,
/// followed by `math.floor` / `math.ceil`, equals integer floor / ceiling
/// division. The planner stays inside this domain or fails closed.
const EXACT_DIVISION_LIMIT: u128 = 1 << 53;

/// Image-processor knobs that decide the canvas, with HF's field names
/// (`Glm5NextImageProcessorKwargs`). The bounds are vision-token counts, not
/// pixels; [`smart_resize`] converts them.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct Glm5NextGeometrySpec {
    pub patch_size: u32,
    pub merge_size: u32,
    pub temporal_patch_size: u32,
    pub patch_expand_factor: u32,
    pub min_image_tokens: u32,
    pub max_image_tokens: u32,
}

/// Planning parameters resolved from the checkpoint, without serving defaults.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Glm5NextSpec {
    /// `<|image|>`: the chat-template placeholder and the expanded feature id.
    pub image_token_id: i32,
    /// `<|video|>`, when the checkpoint declares one. Prompts containing it
    /// are refused: their worker-side expansion reuses `image_token_id`.
    #[serde(default)]
    pub video_token_id: Option<i32>,
    #[serde(flatten)]
    pub geometry: Glm5NextGeometrySpec,
}

/// GLM-5.3-Flash metadata-only processor. Pixel preprocessing is explicitly
/// unsupported.
///
/// ```
/// use dynamo_multimodal::{Result, registry, token_layout};
/// use dynamo_multimodal::processor::MediaMetadata;
/// # fn main() -> Result<()> {
/// // zai-org/GLM-5.3-Flash: config.json ids and processor_config.json knobs.
/// let family = registry::processor_from_spec(r#"{
///     "family": "glm5_next", "image_token_id": 154854, "video_token_id": 154855,
///     "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
///     "patch_expand_factor": 1, "min_image_tokens": 16, "max_image_tokens": 8000
/// }"#)?;
/// let media = [MediaMetadata::Image { width: 1024, height: 768 }];
/// // <|begin_of_image|> <|image|> <|end_of_image|>, as the chat template renders it.
/// let ids = [154830, 154854, 154831];
/// let counts = media.iter().map(|item| family.num_media_tokens(item))
///     .collect::<Result<Vec<_>>>()?;
/// assert_eq!(counts, [1036]); // 784x1036 canvas -> 56x74 patches / 2x2 merge
/// let layout = family.layout_from_metadata(&ids, &media)?;
/// let expanded = token_layout::apply_layout(&ids, &layout, &counts, 2048)?;
/// assert_eq!(expanded.input_ids.len(), 1038);
/// assert_eq!(expanded.offsets, [(1, 1036)]); // the delimiters stay text
/// assert_eq!(expanded.feature_ranges[0][0], 1..1037);
/// # Ok(())
/// # }
/// ```
pub struct Glm5NextProcessor {
    spec: Glm5NextSpec,
}

impl Glm5NextProcessor {
    pub fn new(spec: Glm5NextSpec) -> Result<Self> {
        spec.geometry.validate()?;
        if spec.image_token_id < 0 {
            return Err(MmError::invalid_input(
                "glm5_next: image_token_id must be nonnegative",
            ));
        }
        if spec.video_token_id.is_some_and(|id| id < 0) {
            return Err(MmError::invalid_input(
                "glm5_next: video_token_id must be nonnegative",
            ));
        }
        Ok(Self { spec })
    }

    pub fn plan_image(&self, width: u32, height: u32) -> Result<Glm5NextImagePlan> {
        plan_image_grid(width, height, &self.spec.geometry)
    }
}

impl MmFamilyProcessor for Glm5NextProcessor {
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        match media {
            MediaMetadata::Image { width, height } => {
                usize::try_from(self.plan_image(*width, *height)?.num_image_tokens)
                    .map_err(|_| MmError::limit_exceeded("glm5_next: token count exceeds usize"))
            }
            _ => Err(MmError::unsupported("glm5_next supports only images")),
        }
    }

    fn layout_from_metadata(
        &self,
        input_ids: &[i32],
        media: &[MediaMetadata],
    ) -> Result<TokenLayout> {
        let counts = media
            .iter()
            .map(|item| self.num_media_tokens(item))
            .collect::<Result<Vec<_>>>()?;
        if let Some(video_token_id) = self.spec.video_token_id
            && input_ids.contains(&video_token_id)
        {
            return Err(MmError::unsupported(
                "glm5_next: prompt carries the video placeholder; GLM expands video frames \
                 with the image token, which this planner does not reproduce",
            ));
        }
        // The chat template emits the delimiters itself; only `<|image|>` expands.
        crate::token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }

    fn process_item(&self, _media: &DecodedMedia) -> Result<ProcessedItem> {
        Err(MmError::unsupported(
            "glm5_next pixel preprocessing is not implemented; use metadata-only planning",
        ))
    }

    fn layout(&self, _input_ids: &[i32], _items: &[ProcessedItem]) -> Result<TokenLayout> {
        Err(MmError::unsupported(
            "glm5_next processed-item layout is not implemented; use layout_from_metadata",
        ))
    }
}

/// Canvas, patch grid, and merged token count for one image; dimensions are
/// height/width. `image_grid_thw` is `[1, grid_h, grid_w]`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct Glm5NextImagePlan {
    pub resized_height: u32,
    pub resized_width: u32,
    pub grid_h: u32,
    pub grid_w: u32,
    /// `grid_h * grid_w / merge_size²`: the `<|image|>` copies in the prompt.
    pub num_image_tokens: u32,
}

impl Glm5NextGeometrySpec {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.patch_size == 0
            || self.merge_size == 0
            || self.temporal_patch_size == 0
            || self.patch_expand_factor == 0
        {
            return Err(MmError::invalid_input(
                "glm5_next geometry: patch, merge, temporal, and expand factors must be positive",
            ));
        }
        if self.max_image_tokens == 0 {
            return Err(MmError::invalid_input(
                "glm5_next geometry: max_image_tokens must be positive",
            ));
        }
        Ok(())
    }

    /// `patch_size * merge_size * patch_expand_factor`: the spatial multiple
    /// the canvas aligns to (`_preprocess` passes it as `factor`).
    fn alignment_factor(&self) -> Result<u32> {
        self.patch_size
            .checked_mul(self.merge_size)
            .and_then(|factor| factor.checked_mul(self.patch_expand_factor))
            .ok_or_else(|| {
                MmError::invalid_input("glm5_next geometry: alignment factor overflows u32")
            })
    }
}

/// `math.ceil(value / factor) * factor` for `value < 2**53`.
fn align(value: u128, factor: u128) -> u128 {
    value.div_ceil(factor) * factor
}

/// Mirror of the pinned `smart_resize`: the aligned canvas within the
/// spatiotemporal budget. `min_image_tokens` / `max_image_tokens` are the
/// token-denominated `min_pixels` / `max_pixels` parameters.
///
/// Python's float steps are reproduced exactly: `round` is ties-to-even, and
/// every `int / int` division followed by `floor` / `ceil` is replaced by its
/// integer equivalent, which is identical while the numerator stays below
/// `2**53` ([`MmError::LimitExceeded`] otherwise).
pub fn smart_resize(
    num_frames: u32,
    height: u32,
    width: u32,
    temporal_factor: u32,
    factor: u32,
    min_image_tokens: u32,
    max_image_tokens: u32,
) -> Result<(u32, u32)> {
    if temporal_factor == 0 || factor == 0 {
        return Err(MmError::invalid_input(
            "glm5_next geometry: temporal_factor and factor must be positive",
        ));
    }
    if num_frames == 0 || height == 0 || width == 0 {
        // Python divides by `num_frames * height * width` on the upscale path.
        return Err(MmError::invalid_input(
            "glm5_next geometry: frame count and image dimensions must be positive",
        ));
    }
    let (num_frames, height, width) = (
        u128::from(num_frames),
        u128::from(height),
        u128::from(width),
    );
    let (temporal_factor, factor) = (u128::from(temporal_factor), u128::from(factor));
    let pixels_per_token = temporal_factor * factor * factor;
    let min_pixels = u128::from(min_image_tokens) * pixels_per_token;
    let max_pixels = u128::from(max_image_tokens) * pixels_per_token;

    // `round(num_frames / temporal_factor)`: both operands are small ints, so
    // the quotient is the exactly rounded f64 Python sees.
    let rounded_frames = (num_frames as f64 / temporal_factor as f64).round_ties_even() as u128;
    let aligned_frames = temporal_factor.max(rounded_frames * temporal_factor);
    let mut aligned_height = align(height, factor);
    let mut aligned_width = align(width, factor);
    let mut aligned_pixel_budget = aligned_frames * aligned_height * aligned_width;

    if aligned_pixel_budget < min_pixels {
        let content_pixels = num_frames * height * width;
        if content_pixels >= EXACT_DIVISION_LIMIT {
            return Err(MmError::limit_exceeded(
                "glm5_next geometry: image is too large for exact upscale planning",
            ));
        }
        let scale = (min_pixels as f64 / content_pixels as f64).sqrt();
        let upscale = |side: u128| -> u128 {
            // `math.ceil(side * scale)` with `side` exactly representable.
            let scaled = (side as f64 * scale).ceil();
            align((scaled as u128).max(1), factor)
        };
        aligned_height = upscale(height);
        aligned_width = upscale(width);
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width;
    }

    if aligned_pixel_budget > max_pixels {
        let minimum_pixels = aligned_frames * factor * factor;
        if max_pixels < minimum_pixels {
            return Err(MmError::invalid_input(format!(
                "glm5_next geometry: max_pixels={max_pixels} is too small; at least \
                 {minimum_pixels} pixels are required for one aligned patch"
            )));
        }
        // Binary search over the content height for the largest aligned
        // canvas that fits the budget, exactly as `fit_within_budget` does.
        let (mut low, mut high) = (1u128, height);
        let (mut best_height, mut best_width) = (factor, factor);
        while low <= high {
            let content_height = (low + high) / 2;
            let numerator = width * content_height;
            if numerator >= EXACT_DIVISION_LIMIT {
                return Err(MmError::limit_exceeded(
                    "glm5_next geometry: image is too large for exact budget planning",
                ));
            }
            let content_width = (numerator / height).max(1);
            let candidate_height = align(content_height, factor);
            let candidate_width = align(content_width, factor);
            if aligned_frames * candidate_height * candidate_width <= max_pixels {
                (best_height, best_width) = (candidate_height, candidate_width);
                low = content_height + 1;
            } else {
                high = content_height - 1;
            }
        }
        (aligned_height, aligned_width) = (best_height, best_width);
    }

    let dimension = |value: u128| {
        u32::try_from(value).map_err(|_| {
            MmError::limit_exceeded("glm5_next geometry: planned dimension exceeds u32 range")
        })
    };
    Ok((dimension(aligned_height)?, dimension(aligned_width)?))
}

/// Mirror `Glm5NextImageProcessor._preprocess` geometry for a still image
/// from its original dimensions, without reading pixels: the canvas from
/// [`smart_resize`] (one temporal group of `temporal_patch_size` duplicated
/// frames), the `patch_size` grid, and the merged token count.
pub fn plan_image_grid(
    width: u32,
    height: u32,
    spec: &Glm5NextGeometrySpec,
) -> Result<Glm5NextImagePlan> {
    spec.validate()?;
    let (resized_height, resized_width) = smart_resize(
        spec.temporal_patch_size,
        height,
        width,
        spec.temporal_patch_size,
        spec.alignment_factor()?,
        spec.min_image_tokens,
        spec.max_image_tokens,
    )?;
    let grid_h = resized_height / spec.patch_size;
    let grid_w = resized_width / spec.patch_size;
    let merge_length = u64::from(spec.merge_size) * u64::from(spec.merge_size);
    let num_image_tokens = u32::try_from(u64::from(grid_h) * u64::from(grid_w) / merge_length)
        .map_err(|_| MmError::limit_exceeded("glm5_next geometry: token count exceeds u32"))?;
    Ok(Glm5NextImagePlan {
        resized_height,
        resized_width,
        grid_h,
        grid_w,
        num_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glm53_flash() -> Glm5NextGeometrySpec {
        Glm5NextGeometrySpec {
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            patch_expand_factor: 1,
            min_image_tokens: 16,
            max_image_tokens: 8000,
        }
    }

    /// Hand-derived from the pinned Python: 1024x768 aligns to 1036x784
    /// (within budget), a 1x1 image upscales to the 16-token minimum, and a
    /// 4000x4000 image is cut to the 8000-token budget by the binary search.
    #[test]
    fn plans_the_three_smart_resize_branches() {
        let plan = plan_image_grid(1024, 768, &glm53_flash()).unwrap();
        assert_eq!((plan.resized_height, plan.resized_width), (784, 1036));
        assert_eq!(
            (plan.grid_h, plan.grid_w, plan.num_image_tokens),
            (56, 74, 1036)
        );

        let plan = plan_image_grid(1, 1, &glm53_flash()).unwrap();
        assert_eq!((plan.resized_height, plan.resized_width), (112, 112));
        assert_eq!(plan.num_image_tokens, 16);

        let plan = plan_image_grid(4000, 4000, &glm53_flash()).unwrap();
        assert!(plan.num_image_tokens <= 8000);
        assert_eq!(plan.resized_height, plan.resized_width);
        // The next aligned step up would overflow the budget.
        let step = plan.resized_height as u64 + 28;
        assert!(2 * step * step > 8000 * 2 * 28 * 28);
    }

    /// min above max: the upscale runs first and overshoots the budget, then
    /// `fit_within_budget` searches content heights of the *original* 1-pixel
    /// image, so the canvas collapses to one aligned patch group.
    #[test]
    fn upscale_then_budget_cut_follows_python_order() {
        let spec = Glm5NextGeometrySpec {
            min_image_tokens: 9000,
            ..glm53_flash()
        };
        let plan = plan_image_grid(1, 1, &spec).unwrap();
        assert_eq!((plan.resized_height, plan.resized_width), (28, 28));
        assert_eq!(plan.num_image_tokens, 1);
    }

    #[test]
    fn rejects_invalid_inputs_and_configuration() {
        for (w, h) in [(0, 1), (1, 0), (0, 0)] {
            assert!(matches!(
                plan_image_grid(w, h, &glm53_flash()),
                Err(MmError::InvalidInput { .. })
            ));
        }
        let reject = |spec: Glm5NextGeometrySpec| {
            assert!(matches!(
                plan_image_grid(28, 28, &spec),
                Err(MmError::InvalidInput { .. })
            ));
        };
        for field in 0..5 {
            let mut spec = glm53_flash();
            match field {
                0 => spec.patch_size = 0,
                1 => spec.merge_size = 0,
                2 => spec.temporal_patch_size = 0,
                3 => spec.patch_expand_factor = 0,
                _ => spec.max_image_tokens = 0,
            }
            reject(spec);
        }
        reject(Glm5NextGeometrySpec {
            patch_size: u32::MAX,
            merge_size: 2,
            ..glm53_flash()
        });
    }

    /// Beyond `2**53` pixels Python's float division could diverge from
    /// integer arithmetic, so the planner fails closed instead of guessing.
    #[test]
    fn huge_images_fail_closed_rather_than_approximating() {
        assert!(matches!(
            plan_image_grid(u32::MAX, u32::MAX, &glm53_flash()),
            Err(MmError::LimitExceeded { .. })
        ));
        // One side at u32::MAX stays exact: the product is below 2**53.
        assert!(plan_image_grid(1, u32::MAX, &glm53_flash()).is_ok());
    }

    #[test]
    fn processor_counts_and_refuses_video() {
        let spec = Glm5NextSpec {
            image_token_id: 154854,
            video_token_id: Some(154855),
            geometry: glm53_flash(),
        };
        let processor = Glm5NextProcessor::new(spec).unwrap();
        let image = MediaMetadata::Image {
            width: 448,
            height: 448,
        };
        assert_eq!(processor.num_media_tokens(&image).unwrap(), 256);
        let video = MediaMetadata::Video {
            width: 448,
            height: 448,
            num_frames: 8,
        };
        assert!(matches!(
            processor.num_media_tokens(&video),
            Err(MmError::Unsupported { .. })
        ));
        assert!(matches!(
            processor.layout_from_metadata(&[154832, 154855, 154833, 154854], &[image]),
            Err(MmError::Unsupported { .. })
        ));
        assert!(
            processor
                .layout_from_metadata(&[154830, 154854, 154831], &[image])
                .is_ok()
        );
    }
}
