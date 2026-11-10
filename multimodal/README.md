<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-multimodal

Shared multimodal input processing and accounting for LLM inference routers
and engines. The first model-family implementation reproduces the image
pipelines behind HF `AutoProcessor`: decode → resize → normalize → patchify,
prompt placeholder expansion, and position math (M-RoPE), all **bit-exact**
against the mirrored HF processor.

| feature    | default | adds |
| ---------- | ------- | ---- |
| `parallel` | **on**  | links rayon; kernels still run inline until `execution::init_pool` arms the crate-owned pool |
| `fetch`    | off     | `fetch::MediaFetcher`, a protected http(s)/data: fetcher for untrusted URLs (SSRF policy, redirect revalidation, byte caps); links reqwest and tokio. The trusted-source `fetch_bytes*` helpers in the same module are still stubs |
| `video-decode` | off | `video_decode`: MP4 / WebM / Matroska demuxing and decoding to RGB frames, through NVDEC or (VP8 / VP9) libvpx, both loaded at runtime; links `matroska-demuxer`. 64-bit Linux only |

**The boundary.** The crate carries what HF ships, plus what a router and an
engine must compute *identically* or routing keys and token accounting
diverge: spec resolution from HF config files, pixel-free token accounting,
and content-hash identity. The optional `fetch` module is
a convenience helper, not a request security boundary.

## 1. Where the crate sits

An **inference engine** preprocesses each request's multimodal inputs using this crate:

```
engine                              │   dynamo-multimodal (this crate)
────────────────────────────────────│──────────────────────────────────────
boot: locate model configs ────────►│  registry::spec_from_model_dir (or a
                                    │  pre-resolved spec) ─► build_processor
                                    │           ─► Box<dyn MmFamilyProcessor>
per request:                        │
image sources, caps ───────────────►│  fetch::fetch_bytes (stub)  ─► raw bytes
      ├─ supplied mm_hashes ───────►│  use unchanged
      ├─ bytes, no supplied hash ──►│  content_hash_bytes
      └─ bytes ────────────────────►│  family.decode_image
tokenize (if text) ─ ids            │
        └─ ids, decoded media ─────►│  family.process_items     (per request; its default maps
                                    │                            family.process_item per item)
                                    │     │ ProcessedItems
        └─ ids, items ─────────────►│  family.layout ─► token_layout::apply_layout
                                    │     │ expanded input_ids + per-item offsets
                                    │     │ + feature ranges (scatter targets)
        └─ offsets, items ─────────►│  family.positions          (optional, e.g. M-RoPE)
                                    │
```

A **router** uses this crate for multimodal requests' load accounting and cache-aware routing:

```
router                              │   dynamo-multimodal (this crate)
────────────────────────────────────│──────────────────────────────────────
boot: locate model configs ────────►│  registry::spec_from_model_dir
                                    │    ─► build_processor ─► Box<dyn MmFamilyProcessor>
per request                         │
   image_url ──────────────────────►│  fetch::fetch_bytes (stub)  ─► raw bytes
      ├─ decoded image ────────────►│  content_hash_canonical_image
                                    │                           ─► cache-affinity key
      └─ bytes ────────────────────►│  image::decode::dimensions  ─► (h, w), header-only
              └─ (w, h) ───────────►│  family.num_media_tokens    ─► token cost per image
                                    │
route: pick the engine by prefix-   │
cache affinity (hashes) + expanded  │
prompt length (token costs)         │
```

| module | responsibility |
| --- | --- |
| `processor` | the model-family seam: `MmFamilyProcessor` trait + data carriers |
| `registry` | family selection from a typed/JSON spec, or resolved straight from the HF config files (the `AutoProcessor` entry) |
| `models/` | one module per family — `models::qwen_vl` first |
| `image/` | decode (8-bit only, PIL-matching), header-only dimension probe, bit-exact resize kernels, transforms |
| `fetch` *(feature)* | `MediaFetcher`: protected fetcher for untrusted http(s)/data: URLs. Also a planned trusted-source helper (`fetch_bytes*`: data:/base64/file/http, `requests`-parity proxy semantics) that is still a stub returning `MmError::Unsupported` and is not for untrusted request URLs |
| `video_decode` *(feature)* | `Video` (open once: `info`, `decode`, `decode_sampled`) and the one-shot `probe`, `decode_frames`, `decode_sampled`: MP4 / WebM / Matroska to the RGB frames a caller names, through NVDEC or libvpx, each frame resized by a `Fit` as it is decoded (§2.4) |
| `video` | per-request video inputs: `VideoOptions` (`fps`, `max_frames`, `num_frames`) → `resolve_num_frames`; `VideoPixelBudget` |
| `frames` | `VideoFrames`, what a decoder hands over (RGB + timestamps), and `encode_jpeg_frames` for a backend that takes images but not video |
| `token_layout` | validating placeholder expansion of the *already tokenized* prompt |
| `execution` | the crate's only parallelism seam: inline by default, a runtime-armed crate-owned rayon pool otherwise |

Host applications normally own scheduling and invoke processors synchronously;
they should leave the crate pool unarmed to avoid nested parallelism.
`execution::init_pool` is an explicit opt-in for callers with little
host-side concurrency (for example, a Python binding). Repeating the resolved
thread count is a no-op; requesting a different count returns
`MmError::InvalidInput`.

## 2. The API, by consumer

One trait carries the family contract; consumers differ in which methods
they call and how they obtain the spec:

```rust
pub trait MmFamilyProcessor: Send + Sync {
    fn capabilities(&self) -> Capabilities; // query with supports(Modality)
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize>;
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem>;
    fn process_items(&self, input_ids: &[i32], media: &[DecodedMedia])
        -> Result<Vec<ProcessedItem>>;                            // maps process_item by default
    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout>;
    fn positions(&self, input_len: usize, offsets: &[(u32, u32)], items: &[ProcessedItem])
        -> Result<PositionOutput>;                                // Rope1D default
}

pub struct ProcessedItem {
    pub modality: Modality,
    pub feature_token_count: usize,
    pub feature: Tensor,
    pub aux: NamedTensors,
    pub geometry: Option<Geometry>,
}
pub enum PositionOutput { Rope1D, MRope { positions: Vec<i64>, delta: i64 } }
```

Specs arrive pre-resolved (e.g. by SGLang's Python gate) or from the HF config files. 
Resolution is conservative either way: an unknown
`model_type` or a knob the pipeline cannot honor bit-exactly is an `Err`,
never a silent approximation.

```rust
#[serde(tag = "family", rename_all = "snake_case")]
pub enum ProcessorSpec { QwenVl(QwenVlSpec) }          // one variant per family

pub fn build_processor(spec: ProcessorSpec) -> Result<Box<dyn MmFamilyProcessor>>;
pub fn processor_from_spec(json: &str)     -> Result<Box<dyn MmFamilyProcessor>>;

// AutoProcessor.from_pretrained parity for Python-free consumers:
pub fn spec_from_hf_configs(config_json: &str, preprocessor_config_json: &str)
    -> Result<ProcessorSpec>;
pub fn spec_from_model_dir(dir: &Path) -> Result<ProcessorSpec>;
```

`spec_from_model_dir` reads `preprocessor_config.json`, or the nested
transformers v5 `processor_config.json` when a checkpoint ships only that
file (GLM-5.3-Flash). A Mistral Large 4 directory resolves from
`params.json` and `processor_config.json` to the HF `PixtralProcessor`
pipeline; a mistral-common engine such as vLLM counts differently (§8).

### 2.1 A router (e.g. dynamo) — accounting and routing, no pixel work

| when | API | in → out |
| --- | --- | --- |
| boot, per model | `registry::spec_from_model_dir` | config dir (hub download is the router's concern) → `ProcessorSpec`; `Err` = model unsupported |
| boot, per model | `registry::build_processor` | `ProcessorSpec` → `Box<dyn MmFamilyProcessor>` |
| per media part | `fetch::MediaFetcher::fetch` *(feature `fetch`)*, or the consumer's own protected fetcher | untrusted media source → raw bytes, under the SSRF policy and a byte cap |
| per trusted media part | `fetch::fetch_bytes` *(feature `fetch`, stub)* | trusted media source → raw bytes once implemented; currently returns `MmError::Unsupported` |
| per decoded image | `content_hash_canonical_image` | shape + dtype + RGB bytes → Dynamo-compatible identity |
| per image part | `image::decode::dimensions` | bytes → `MediaMetadata::Image` dimensions — header probe, no pixel decode |
| per media part | `MmFamilyProcessor::num_media_tokens` | typed lightweight metadata → the item's expanded token count: every id the item puts in the prompt, literal ids included (Kimi-K3 prints a size header around its pad run) |

Token counts give the expanded prompt length for routing (a family that
sizes media by the whole request needs `layout_from_metadata` for the exact
length, §9); hashes drive
prefix-cache-affinity routing and travel downstream as `mm_hashes`. The engine
uses supplied hashes unchanged, whether Dynamo derived them from canonical
decoded media or a passthrough URL.

### 2.2 An inference engine (e.g. SGLang) — the full pixel path

| when | API | in → out |
| --- | --- | --- |
| boot, per worker pool | `registry::build_processor` | spec pre-resolved by the engine's gate or config dir → `Box<dyn MmFamilyProcessor>` |
| per image without a supplied hash | `content_hash_bytes` | encoded bytes → SGLang's fallback `u64` identity |
| per image | `MmFamilyProcessor::decode_image` | bytes + `DecodeLimits` → `DecodedMedia::Image`, or `DecodedMedia::ImageRgba` for a family that composites alpha after its resize (8-bit only, header-checked caps, family-specific EXIF and alpha policy) |
| per image | `MmFamilyProcessor::process_item` | `DecodedMedia` → `ProcessedItem` (preserves modality, feature-token count, tensors, and optional geometry) |
| per request, in place of the per-image call | `MmFamilyProcessor::process_items` | unexpanded input_ids + every `DecodedMedia` of the request → `ProcessedItem`s. The same result for most families; a family that sizes media by the whole request needs it and refuses `process_item` (§7) |
| per request | `MmFamilyProcessor::layout` | input_ids + processed items → `TokenLayout` (a description, not yet applied) |
| per request | `token_layout::apply_layout` | ids + `TokenLayout` + per-item feature-token counts → expanded ids, per-item offsets, and feature ranges (where feature embeddings go); validates the layout (source covered exactly once, each item placed once and in prompt order, feature ranges match the produced embeddings) |
| per request, if needed | `MmFamilyProcessor::positions` | expanded length + offsets + processed items → `PositionOutput` (e.g. M-RoPE) |

Example:

```rust
let family = registry::build_processor(ProcessorSpec::QwenVl(QwenVlSpec {
    image_token_id, patch_size: 14, merge_size: 2, temporal_patch_size: 2,
    min_pixels, max_pixels, image_mean, image_std, resample: Resampler::AtenU8,
}))?;
```

**Per request.** The engine's driver (SGLang: its `sglang-mm` adapter crate)
owns fetching, hashing, caps, and the failure contract, and composes the
crate:

```rust
// sketch
let input_ids: Vec<i32> = match request_ids { Some(ids) => ids, None => tokenizer.encode(&text)? };
let mut media: Vec<DecodedMedia> = Vec::with_capacity(images.len());
let mut hashes: Vec<u64> = Vec::with_capacity(images.len());
for (bytes, supplied_hash) in images {                  // fetched + capped by the engine
    let hash = match supplied_hash {
        Some(hash) => hash,
        None => content_hash_bytes(&bytes),
    };
    hashes.push(hash);
    media.push(family.decode_image(&bytes, &limits)?);
}
// The default maps `process_item`; a family that sizes media by the whole request overrides it.
let items: Vec<ProcessedItem> = family.process_items(&input_ids, &media)?;
let layout: TokenLayout = family.layout(&input_ids, &items)?;
let feature_token_counts: Vec<usize> =
    items.iter().map(|item| item.feature_token_count).collect();
let expanded: ExpandedPrompt =
    token_layout::apply_layout(&input_ids, &layout, &feature_token_counts, max_tokens)?;
let positions: PositionOutput =
    family.positions(expanded.input_ids.len(), &expanded.offsets, &items)?;
```


### 2.3 A Detailed example of process_item() — one 100×76 image, Qwen2.5-VL

(`patch_size 14, merge_size 2, temporal_patch_size 2`; prompt already
tokenized with one placeholder)

```
index:  0        1                2               3              4
ids:  [ …,  <|vision_start|>, <|image_pad|>, <|vision_end|>,     … ]
```

1. `process_item` — `smart_resize` rounds 100×76 up to 112×84 (multiples of
   `patch·merge = 28`), the bit-exact kernel resizes, then normalize +
   patchify: a 6×8 grid of 14×14 patches, each flattened to
   `3·2·14·14 = 1176` floats (temporal 2 duplicates a still's frame).
   Returns `feature = [48, 1176] f32` (HF's `pixel_values`),
   `modality = Image`, `feature_token_count = 12`,
   `aux = [("image_grid_thw", [1, 6, 8])]`, and
   `geometry = Some(Grid([1, 6, 8]))`.
2. `layout` — the image costs `1·6·8 / 2² = 12` tokens (the ViT merges 2×2
   patches per token): keep text `0..2`, replace the placeholder (span `2..3`)
   with a `Feature` part of 12 `<|image_pad|>` tokens, keep text `3..5`.
   `apply_layout` expands and validates: 16 ids, `offsets = [(2, 13)]`,
   `feature_ranges = [[2..14]]`. (A video expansion like
   `[<vision_start>, timestamp, feature × N, <vision_end>]` would also
   contain `Literal` parts, which the feature ranges skip.)
3. `positions(16, …)` — M-RoPE. Text advances the three (t, h, w) rows
   together; the 12 image tokens span a 3×4 grid (`6/2 × 8/2`) at base 2;
   text after resumes at `2 + max(1, 3, 4) = 6`. Returns flat `[3, 16]`
   positions and `delta = 7 + 1 − 16 = −8` (added to the sequence length at
   decode — the image packed 12 tokens into 4 position steps).

The engine scatters the ViT's 12 output embeddings into positions `2..14`
(from `feature_ranges`) and feeds `positions` to the model's rotary path.

### 2.4 Video (feature `video-decode`)

| when | API | in → out |
| --- | --- | --- |
| boot, per model | `serde_json::from_str::<Qwen3VlVideo>` | `video_preprocessor_config.json` → the Qwen3-VL / Cosmos3 sizing rule |
| boot | `VideoDecodeLimits::default()` | operator caps: 8192² frames, 512 MiB output, 512 MiB WebM / Matroska file |
| per video | `video_decode::Video::open` | `&bytes` or any `ByteSource` (a file, a ranged download) + limits → the demuxed video; `info()` gives `VideoInfo` (codec, size, duration, fps, frame count) without decoding |
| per video | `video::resolve_num_frames` | request `VideoOptions` (`fps` / `max_frames` / `num_frames`) + duration + frame count → frames to keep |
| per video | `Video::decode` | indices (display order, non-decreasing) + `Option<&dyn Fit>` + limits → `VideoFrames` |
| per video, no rule | `Video::decode_sampled` | `num_frames` + limits → evenly spaced frames at source size (`sample_indices` gives the same indices for `decode`) |
| one-shot | `video_decode::probe`, `decode_frames`, `decode_sampled` (+ `_from`) | the same on bytes or a `ByteSource`, each reading the container itself |
| image-only backends | `frames::encode_jpeg_frames` | `VideoFrames` + JPEG quality → one JPEG per frame with its timestamp |

`fit` is optional: `None` keeps the selected frames at the source size;
`FrameResize` is a fixed size, `models::qwen_vl::Qwen3VlVideoFit` the
Qwen3-VL / Cosmos3 rule under a request `VideoPixelBudget`, and a consumer
implements `Fit` for any other model.

Decoding runs on CUDA device 0: the first device `CUDA_VISIBLE_DEVICES`
leaves, in CUDA's order (fastest first unless `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
so not always `nvidia-smi`'s GPU 0). `cuvidGetDecoderCaps` on that GPU decides
which codecs and frame sizes NVDEC takes. VP8 always decodes on the CPU
through libvpx, and VP9 does when NVDEC reports it unsupported (no driver
library, no GPU, or a codec or size the GPU lacks); other NVDEC errors fail
the decode. NVDEC needs the driver's `libnvcuvid.so.1`: in a container,
`NVIDIA_DRIVER_CAPABILITIES` must include `video` (the NVIDIA Container
Toolkit's default, for an image that does not set it, is `compute,utility`),
e.g. `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.

```rust
use dynamo_multimodal::models::qwen_vl::{Qwen3VlVideo, Qwen3VlVideoFit};
use dynamo_multimodal::video::resolve_num_frames;
use dynamo_multimodal::video_decode::{self, Video, VideoDecodeLimits};

let model: Qwen3VlVideo = serde_json::from_str(&video_preprocessor_config_json)?; // boot
let limits = VideoDecodeLimits::default();

let video = Video::open(&bytes, &limits)?;                                        // per request
let info = video.info();
let n = resolve_num_frames(&options, info.duration, info.frame_count)?;
let indices = video_decode::sample_indices(info.frame_count, n)?; // evenly spaced, or the model's own rule
let fit = Qwen3VlVideoFit { video: &model, budget };        // `budget`: the request's VideoPixelBudget
let frames = video.decode(&indices, Some(&fit), &limits)?;
// frames.frame(i) is Some(RGB) at frames.width() x frames.height(); frames.timestamps()[i] its time
```

## 3. Python-parity map

Each item reproduces a specific Python behavior, most of them **bit-exactly**:

| this crate | on-par Python API | parity |
| --- | --- | --- |
| `registry::spec_from_hf_configs` / `spec_from_model_dir` | `AutoProcessor.from_pretrained` (config parsing + processor selection) | selection semantics; unknown knobs → `Err`, never approximation |
| `registry::processor_from_spec` | building the processor from already-resolved kwargs | selection semantics |
| `MmFamilyProcessor::num_media_tokens` | `_get_num_multimodal_tokens(…)` | exact token counts from typed image/video/audio metadata, no pixel or feature-extraction work; for a family that sizes media by the whole request, the span of the item alone (§7) |
| `image::resize::resize_rgb(Pil(_))` | `PIL.Image.resize` (LANCZOS/BICUBIC, u8) | **bitwise** (PIL's i32 fixed-point kernels, and Pillow 12.3's pass order: images taller than 100× their width shrink vertically first) |
| `image::resize::resize_rgb(AtenU8)` | `torchvision resize(antialias=True)` on uint8 | **bitwise** (ATen's per-axis i16 weight precision) |
| normalize LUT (family-internal) | slow path `rescale→normalize` vs fast path `_fuse_mean_std_and_rescale_factor` | **bitwise** — the roundings differ on 128 of 256 u8 inputs; the spec selects which to mirror |
| `image::decode::decode_rgb` | `PIL.Image.open(...).convert("RGB")` | same accepted formats; >8-bit samples rejected rather than silently diverging (PIL clips, Rust would rescale) |
| `image::decode::dimensions` | lazy `PIL.Image.open(...).size` | header-only probe |
| `image::decode::decode_rgb_exif_transposed` | `PIL.ImageOps.exif_transpose(Image.open(...)).convert("RGB")` | opt-in EXIF orientation; caps checked against transposed dimensions before pixel decoding |
| `image::decode::dimensions_exif_transposed` | EXIF-oriented `PIL.ImageOps.exif_transpose(Image.open(...)).size` | metadata-only oriented geometry for JPEG, PNG and WebP, without pixel decoding |
| `fetch::fetch_bytes` *(signatures only so far)* | `transformers.image_utils.load_image` (`requests` proxy + `NO_PROXY` semantics, source precedence) | optional trusted-source compatibility behavior, plus streaming byte caps; the protected untrusted-URL path is `fetch::MediaFetcher` |
| `content_hash_bytes` | SGLang Rust tokenizer fallback | BLAKE3 over encoded bytes |
| `content_hash_canonical_image` | Dynamo decoded-image identity | XXH3-64 over rank, dimensions, dtype, and contiguous RGB bytes |
| `token_layout::apply_layout` + `layout_by_placeholder` | HF `Qwen2VLProcessor`'s own `<|image_pad|>` expansion / SGLang `_expand_input_ids` + `get_mm_items_offset` | exact ids/offsets, plus full-coverage validation |
| `models::qwen_vl::QwenVlProcessor::process_item` | HF `Qwen2VLImageProcessor(Fast)` / `Qwen2VLImageProcessorPil` `__call__` → `pixel_values`, `image_grid_thw` | **bitwise** |
| `models::qwen_vl::smart_resize` | HF/SGLang `smart_resize` (incl. Python banker's rounding) | exact, including thin-image clamping |
| `models::glm5_next::plan_image_grid` | `Glm5NextImageProcessor` `smart_resize` + `_preprocess` grid (transformers v5.19.0) | exact canvas, patch grid, and merged token count from dimensions alone |
| `models::nemotron_vl::NemotronVlProcessor` with `sizing = sglang` | SGLang `NanoNemotronVLImageProcessor.process_mm_data_async` → `[1, 3, H, W]` bfloat16 features, expanded ids, item offsets (`feature_ranges`) | **bitwise** for a prompt given as token ids, including the request budget |
| `models::nemotron_vl::NemotronVlProcessor` with `sizing = hf` | the checkpoint's `image_processing_nemotron_h_omni.py` `_preprocess` sizes and `num_tokens` | exact tile sizes and token counts; no pixel path |
| `models::qwen_vl::mrope_image_only` | `get_rope_index` (in transformers' Qwen model code; image-only branch, identical across Qwen generations) | exact, optional model-input preparation beyond strict `AutoProcessor` parity |
| `image::resize::resize_rgba` | `PIL.Image.resize` on an RGBA or LA image (BICUBIC, u8) | **bitwise** against Pillow 12.3 (Pillow resizes premultiplied `RGBa` and divides back with truncation) |
| `models::kimi_k3::KimiK3Processor::decode_image` | `PIL.Image.open(...)` and `load()`, then `convert("RGBA")` when the mode has alpha or a `transparency` entry and is not RGB, else `convert("RGB")` | no EXIF transpose. JPEG: same pixels (libjpeg-turbo on both sides). PNG, BMP, WebP, GIF: no known difference on the forms the gate accepts; the 8 forms known to differ are an error, and an unknown one would decode with other pixels (§7) |
| `models::kimi_k3::KimiK3Processor::process_item` | the Kimi-K3 checkpoint's `KimiK3VisionProcessor.preprocess` (SGLang's CPU path) → `pixel_values`, `grid_thws` | **bitwise** against Pillow 12.3 from the decoded pixels on (resize, composite, pad, normalize, patchify); the decode has the scope of the row above |
| `models::pixtral::PixtralProcessor::process_item` | HF `PixtralProcessor` with `PixtralImageProcessor` / `PixtralImageProcessorPil` (transformers v5.19.0), on the CPU → `pixel_values`, `image_sizes` | **bitwise**, one item per image at its own size |
| `models::pixtral::PixtralProcessor::grid` / `layout` | HF `get_resize_output_image_size` and `replace_image_token`; SGLang's one offset run per grid row | exact sizes, ids, and per-row feature ranges |

For workers that transpose EXIF, pair `dimensions_exif_transposed` for token
accounting with `decode_rgb_exif_transposed` for pixels. Raw APIs remain
unchanged; forwarding and raw-content hashing still use the original bytes.
PNG EXIF includes `eXIf` chunks and legacy `Raw profile type exif` tEXt/zTXt
profiles, including trailing text metadata, with a 1 MiB profile-text cap.
The cap applies to both encodings; for plain tEXt it is stricter than Pillow.
Competing tEXt/zTXt profiles are rejected. XMP-only orientation is outside
these EXIF APIs; Pillow also supports that separate fallback.


## 4. Testing

Three layers, all pinned to bitwise-equality:

1. **Crate-local unit tests** — smart_resize against Python-derived reference
   values (including rounding ties), patchify layout, normalize-LUT
   divergence, layout coverage validation, config-resolution gating, plus a
   thread-count guard proving the crate owns no threads while the pool is
   unarmed (the default).
2. **Crate-local golden replay** — this repo's CI has no Python/HF, so
   committed tables drive the §2.2 composition (`build_processor` →
   `decode_image` → `process_items` → `layout` → `apply_layout` → `positions`)
   and compare **every output field bitwise**: `tests/mm_parity.rs` replays
   one table per model from `tests/fixtures/mm_parity/`. For Qwen-VL, the
   table covers both resamplers, both smart_resize branches, multi-image, and
   a tall image that crosses Pillow's pass-order rule, and
   `tests/qwen_vl_golden.rs` checks the resampler stage on its own so a
   failure names the stage. `tests/fixtures/mm_parity/generate.py`
   regenerates each table: the checkpoint's own HF processor writes
   `expected`, SGLang, vLLM and llm-multimodal each get a cell against it,
   and the crate must equal the table's target column.
   `tests/fixtures/mm_parity/PARITY.md` shows every column side by side. To
   add a family, follow `.agents/skills/mm-preprocessor-generator/SKILL.md`.
   The decode stage has its own table: `tests/fixtures/mm_parity/decode.json`
   gives Pillow's output for each image of the shared corpus
   (`tests/fixtures/mm_parity/images/`) through each decode entry point, and
   marks each form where the crate refuses or decodes other pixels. See
   `tests/fixtures/mm_parity/README.md`.
3. **Consumer parity (SGLang CI)** — per-step and end-to-end pytest suites
   compare the Rust path against the live HF/Python processors field-by-field
   with `.tobytes()` equality, plus a GPU e2e test and an MMMU accuracy gate
   (a systematic skew reads as fluent text; only the benchmark catches it
   end-to-end).

## 5. DSV4.1 geometry planning

`models::deepseek_v41::plan_image_grid(width, height, &spec)` returns resized
dimensions, ViT/LLM grids, and the full `llm_height * (llm_width + 1) + 2`
placeholder span, including row separators and boundary positions. Supply
the same resolved `DeepseekV41GeometrySpec` on the frontend and encoder.

The pixel-free planner mirrors [pinned SGLang](https://github.com/sgl-project/sglang/blob/ffac53d779c08dcdab2d07e5e2a41dba83f0e65c/python/sglang/srt/multimodal/deepseek_v41_image_processing.py).
Invalid inputs/settings and unrepresentable `u32` output dimensions return
errors.

The registry also exposes **metadata-only** DSV4.1 planning:
`ProcessorSpec::DeepseekV41` / `{"family": "deepseek_v41", ...}` accept
`image_token_id` and the geometry fields above. `spec_from_hf_configs` selects
this family for `deepseek_v4` and `deepseek_v41`; it accepts flat `vision_*`
fields or nested `vision_config` aliases (`patch_size`, `downsample_ratio`,
`min_pixels`, `max_image_tokens`, `max_wh_ratio`). Required geometry fields
and the image id must be present; there are no serving defaults. The
preprocessor config must be an object with no non-null processing overrides
(the inert `processor_class` field is allowed). Preserve the checkpoint's
original `model_type`: a config serialized as `deepseek_v3` is unsupported.

Use `num_media_tokens` and `layout_from_metadata`, then pass the counts and
layout to `token_layout::apply_layout` with the context limit.
See the compiled example on [`DeepseekV41Processor`](src/models/deepseek_v41.rs).


Every expanded position uses the same vocabulary `image_token_id`, including
boundaries and row separators. The model supplies embeddings for the entire
span, so `feature_ranges` includes those positions too.
`DeepseekV41ImagePlan::token_types()` separately iterates the pinned processor's
START/IMAGE/ROW_SEPARATOR/END types (0/1/2/3); these are not vocabulary ids.
Adjacent images retain distinct item offsets even when their repeated ids
form one contiguous run. Consumers must pass those offsets to workers rather
than rediscovering items by scanning token ids.

`process_item` and processed-item `layout` explicitly return `Unsupported`.
Full pixel processing and frontend/worker integration remain separate work;
this registration does not make a DSV4.1 image encoding pipeline available.

Regenerate/check the Python-derived fixtures with:

```sh
python3 multimodal/tests/fixtures/deepseek_v41/generate.py
python3 multimodal/tests/fixtures/deepseek_v41/generate.py --check
```

The stdlib-only generator executes the pinned source's pure geometry
functions. Fixture settings are mathematical test inputs, not model defaults.

## 6. GLM-5.3-Flash images


`Glm5NextImageProcessor` adds the full image pipeline on top of the shared
planner. Select `ProcessorSpec::Glm5NextImage`, or pass a `glm5_next` JSON
spec with `image_mean`, `image_std`, and `resample` (`aten_u8` or `pil`) in
addition to the planning fields below. Supplying pixel fields requires both
normalization arrays; an incomplete pixel spec is an error.
PIL preprocessing requires `patch_expand_factor = 1`; other values return
`Unsupported` because the pinned HF PIL processor ignores this knob during
resize and would disagree with the shared metadata planner.

Call `family.decode_image(bytes, &limits)` before `family.process_item`.
The default trait method preserves Qwen's plain RGB conversion. GLM uses
`image::decode::decode_rgb_smart`: EXIF transpose followed by alpha compositing
onto an edge-contrasting background, matching SGLang's `smart_to_rgb` for PNG,
WebP, GIF, and alpha-bitmask BMP. GLM then resizes the content without
unnecessary upscaling, zero-pads to the planned canvas, normalizes and
patchifies, expands placeholders, and emits image-only M-RoPE positions.
Video and samples deeper than 8 bits remain unsupported.

The metadata-only `Glm5NextSpec` API and `spec_from_hf_configs` keep their
planning contract. Python-free consumers can combine that planning spec
with explicitly resolved normalization and resampling in `Glm5NextImageSpec`.

The `mm_parity` test replays the 26 Python CPU cases captured in
[SGLang PR #46563](https://github.com/sgl-project/sglang/pull/46563), including
both resamplers, multiple images, token overrides, EXIF, and alpha formats.
See `tests/fixtures/mm_parity/README.md` for provenance and regeneration.


`models::glm5_next::plan_image_grid(width, height, &spec)` returns the padded
canvas, the `patch_size` grid (`image_grid_thw = [1, grid_h, grid_w]`), and the
merged token count `grid_h * grid_w / merge_size²`, which is how many
`<|image|>` copies the HF processor writes into the prompt. The canvas comes
from the model's own `smart_resize`: bounds are vision-token counts
(`min_image_tokens` / `max_image_tokens`), converted at
`temporal_patch_size * factor²` pixels per token with
`factor = patch_size * merge_size * patch_expand_factor`; sides align up,
small images upscale to the minimum, and oversize images are cut by the
processor's binary search over the content height. vLLM, SGLang, and
TensorRT-LLM all take `image_grid_thw` from that HF image processor, so
routers and workers share one geometry contract.

The pixel-free planner mirrors the pinned
[transformers v5.19.0 source](https://github.com/huggingface/transformers/blob/c4c4605f961877852b89b97478536bb3a26b6fd1/src/transformers/models/glm5_next/image_processing_glm5_next.py)
(unchanged since v5.16.1). Python's `int / int` divisions are reproduced with
integer arithmetic, which is exact while the numerator stays below 2⁵³;
larger products return `LimitExceeded` rather than risk a one-patch drift.

The registry exposes **metadata-only** planning: `ProcessorSpec::Glm5Next` /
`{"family": "glm5_next", ...}` take `image_token_id`, an optional
`video_token_id`, and the six geometry knobs above; there are no serving
defaults. `spec_from_hf_configs` selects this family for `glm5_next` and
accepts either the flat image-processor object or the nested
`processor_config.json` (`image_processor` / `video_processor` /
`processor_class`). Knobs that only shape pixel values are ignored; `do_resize`
must stay on, crop/pad must stay off, unknown knobs are refused, and processor
`patch_size` / `merge_size` / `temporal_patch_size` must agree with
`vision_config`.

The chat template renders each image as
`<|begin_of_image|><|image|><|end_of_image|>`; `layout_from_metadata` expands
only `<|image|>` (`layout_by_placeholder`), so the delimiters stay text and
`feature_ranges` covers exactly the repeated image ids. GLM expands **video**
frames with the same image token, which this planner does not reproduce:
video metadata is `Unsupported`, and a prompt containing the declared
`video_token_id` is refused rather than mis-attributed. `process_item` and
processed-item `layout` also return `Unsupported`.

Regenerate/check the Python-derived fixtures with:

```sh
python3 multimodal/tests/fixtures/glm5_next/generate.py
python3 multimodal/tests/fixtures/glm5_next/generate.py --check
python3 multimodal/tests/fixtures/glm5_next/generate.py --verify-hf  # needs transformers + torch
```

The stdlib-only generator executes the pinned source's pure `smart_resize`
and repeats the geometry lines of `resize` / `_preprocess`; `--verify-hf`
cross-checks the planned grid against the installed HF processor's
`image_grid_thw`. Fixture settings beyond the first group are mathematical
test inputs, not model defaults.

## 7. Kimi-K3 images

`KimiK3Processor` (`ProcessorSpec::KimiK3`, `{"family": "kimi_k3", ...}`) is
the image pipeline of Kimi-K3 and Kimi-K3.1. It mirrors the checkpoint's own
processor (`kimi_k3_vision_processing.py`, `media_utils.py` of
`moonshotai/Kimi-K3@f831ab66` and `moonshotai/Kimi-K3.1-Pro@d9e99b07`) on
Pillow 12.3, which SGLang runs on its CPU path. From the decoded pixels on,
the output is bit-exact. The decode is bit-exact for JPEG, and has no known
difference for PNG, BMP, WebP and GIF on the forms it accepts; see "What
each claim rests on" below. Pillow before 12.2 gives other pixels for an
image taller than 100 times its width. `spec_from_hf_configs` selects the
family for `model_type` `kimi_k3` and `kimi_k31`, from the shipped
`preprocessor_config.json` or from the files that `save_pretrained` writes
(checked on transformers 5.19.0).

Call `family.decode_image(bytes, &limits)` before `family.process_item`: it
keeps alpha (`DecodedMedia::ImageRgba`), applies no EXIF transpose, and holds
the input checks below. The output is `[grid_h * grid_w, 3, 14, 14]` patches
with `image_grid_thw`; positions are 1-D.

The expansion prints the source size as text:
`<|media_begin|>image {W}x{H}<|media_content|>`, the pad run, `<|media_end|>`.
The crate has no tokenizer, so the spec carries the ids of that text in
`size_tokens`. `spec_from_hf_configs` leaves it empty: pixels and
`KimiK3Processor::plan` work, and `layout`, `layout_from_metadata` and
`num_media_tokens` return `Unsupported`. The consumer builds the table with
the model's tokenizer, passes it as JSON, and must hold the table's rule
itself (see the `SizeTokens` doc); the crate checks only the table's shape.

Three counts for one image (300x200: 96, 88, 88):

- `num_media_tokens`: the whole expansion, header and end token included.
- The `Feature` parts of `layout_from_metadata`: the pad run, which takes the
  embeddings. Pass these counts to `apply_layout`. With decoded pixels this
  is `ProcessedItem::feature_token_count`.
- `feature_ranges`: the position of the pad run, which SGLang calls the item
  offset. `offsets` also cover the header and the end token.

These return an error, never an approximation:

- Spec: a non-null `fixed_output_tokens`, a `transparent_bg_fill_stage` that
  is absent or not `after_resize`, a background pattern other than
  `chessboard`, a chessboard field left to the checkpoint's default. From the
  HF files the error is `Unsupported`.
- Input, `Unsupported` from the decode gate. The gate reads the format and a
  few header fields. It refuses a palette, 1-bit, L with tRNS or CMYK source,
  or a GIF, that needs a resize (Pillow resizes these in the native mode),
  and any format other than JPEG, PNG, GIF, WebP and BMP. It also refuses
  the 8 forms whose decode is known to differ from Pillow:
  1. a BMP that is not uncompressed 24-bit or 32-bit, or has a pixel offset
     of 0;
  2. animated WebP;
  3. a lossy WebP where a segment's filter level is outside 0..63 and a
     loop-filter delta then moves it (libwebp clamps after the deltas);
  4. tRNS on 2-bit or 4-bit grayscale, and a 1-bit tRNS sample above 1;
  5. a palette PNG whose tRNS bytes are `FF* 00 FF* 0A` (Pillow reads them as
     one transparent entry);
  6. a PNG with a text chunk named `interlace` before IDAT (Pillow reads the
     keyword back as its deinterlace switch);
  7. a GIF whose first frame has a gray-ramp local color table beside a global
     table that is not a gray ramp (Pillow paints with the global table);
  8. a GIF with more than one Graphic Control Extension before its first
     image (Pillow keeps a transparent index that a later extension clears).
- Input, `InvalidInput` from a decoder. Every other file goes to the crate's
  shared decoders, which refuse what they cannot read: samples deeper than 8
  bits, an OS/2 v2 BMP, a 32-bit BITFIELDS BMP with all-zero masks, a
  lossless grayscale JPEG, and a JPEG that makes libjpeg warn. Pillow decodes
  each of these.
- Size: more than 2^53 patch cells, where f64 no longer plans exactly, and a
  feature buffer that does not fit `usize`.

What each claim rests on:

- After the decode: the resize, composite, pad, normalize, patchify and
  layout are bit-exact against the checkpoint's processor. The fixtures pin
  this.
- JPEG decode: bit-exact. The crate and Pillow both decode with
  libjpeg-turbo.
- PNG, BMP, WebP and GIF decode: no known difference on the forms the gate
  accepts. The crate decodes these with Rust crates (`png`, `image`,
  `image-webp`, `gif`) that are second implementations of the libraries
  Pillow uses. The gate refuses each of the 8 forms known to differ, and a
  fixture pins each refusal. Review sweeps, which are not in this repo,
  found no other: 2,050 generated files in the last round, and 194 + 2,461
  generated files and 3,137 files from a development host before it. A form
  that is not known would decode with other pixels and no error.
- In each difference found so far, the decoded size was right. So the grid,
  the token counts and the prompt span were right, and only pixel values
  differed.

Not covered: a file that breaks its format's rules, or that carries a field
an encoder should not write. It can decode with other pixels than Pillow's
and no error. Known forms: a WebP whose VP8X alpha flag does not match its
data; a WebP whose VP8L `alpha_is_used` bit, which the specification calls a
hint, does not match its pixels; a PNG whose tRNS chunk is after IDAT,
before PLTE, longer than PLTE, repeated, or has a non-zero high byte. The
other direction is open too: the crate decodes some files that Pillow or
the checkpoint rejects, such as a 32-bit BMP with masks outside Pillow's
list or a PNG text chunk with the keyword `transparency`. The reference
gives no pixels for them.

SGLang's GPU path, the default for Kimi-K3 on a CUDA host, is a different
implementation (float bicubic, straight alpha, a one-step f32 normalize) and
is not mirrored here.

`tests/mm_parity.rs` replays `tests/fixtures/mm_parity/kimi-k3.json`. Its
target is SGLang's Python CPU path, which gives the same values as the
checkpoint's own processor. The decode forms at the shipped limits are rows
of `decode.json`. The `kimi_k3_golden` test checks what the table cannot
hold: the prompt expansion from sizes alone, two drawn images (7300x64 and
9000x4000) at the checkpoint's limits, the configs of both checkpoints, and
a size sweep of the plan and the prompt span. `kimi_k3/generate.py` runs the
table's cases and the two drawn images again on Kimi-K3.1-Pro and stops
unless each value is the same. Regenerate or check with:

```sh
CUDA_VISIBLE_DEVICES="" python multimodal/tests/fixtures/mm_parity/generate.py \
    --family kimi_k3 [--check] --model-dir <moonshotai/Kimi-K3 dir> \
    --vllm-python <python> --llm-mm-harness <binary>
CUDA_VISIBLE_DEVICES="" python multimodal/tests/fixtures/kimi_k3/generate.py \
    [--check] <moonshotai/Kimi-K3 dir> <moonshotai/Kimi-K3.1-Pro dir>
```
## 8. Mistral Large 4 (Pixtral) images

`models::pixtral::PixtralProcessor` (`ProcessorSpec::Pixtral`, JSON
`"family": "pixtral"`) mirrors transformers' `PixtralProcessor` and
`PixtralImageProcessor` (v5.19.0) on the CPU. An engine runs this pipeline
when it serves the checkpoint with the HF tokenizer (`tokenizer.json` and
`chat_template.jinja`), which is SGLang's default mode.

- **Size.** `ratio = max(height, width) / longest_edge`. Above 1, both sides
  are divided by the ratio and floored. Each side then rounds up to a
  multiple of `patch_size * spatial_merge_size` (28 pixels), so a small image
  stretches. `grid(width, height)` returns `(rows, cols)`. An image whose
  short side floors to 0 pixels (3080x1) is `InvalidInput`: HF raises there.
- **Pixels.** `decode_image` is the trait default (PIL `convert("RGB")`).
  `process_item` resizes with the spec's resampler (`aten_u8`, the default,
  or `pil`), normalizes with the rounding of that backend, and returns planar
  `[1, 3, H, W]` f32 plus the model kwarg `image_sizes` (`[1, 2]` i64). Each
  item keeps its own size, which equals its crop of HF's padded batch.
- **Prompt.** The chat template writes one `[IMG]` per image. The layout
  replaces it with `rows` rows of `cols` `[IMG]`; `[IMG_BREAK]` closes each
  row and `[IMG_END]` the last. Only the `[IMG]` positions take embeddings:
  `feature_ranges` holds one range per row and `offsets` the whole span.
- **Counts.** `num_media_tokens` is the prompt span, `rows * (cols + 1)`
  (HF's `_get_num_multimodal_tokens`, which misreads an image 1 or 3 pixels
  high; this crate follows the processor call). `apply_layout` takes the
  number of embeddings, `rows * cols`, not the span: in an engine
  `ProcessedItem::feature_token_count`, in a router the sum of the item's
  `Feature` parts from `layout_from_metadata` (doctest on `PixtralProcessor`).

`spec_from_model_dir` takes this path for a directory without `config.json`
(the NVFP4 repo) or with one that says `mistral_large4`. The token ids and
the geometry come from `params.json` (`vision_encoder`); mean, std and the
knobs come from the nested `processor_config.json`. A pinned knob with
another value, or an unknown knob, is refused. The resolved resampler is
`aten_u8`; select `pil` in the spec JSON. `spec_from_hf_configs` refuses
`mistral_large4`: its two files lack the `[IMG_BREAK]` and `[IMG_END]` ids.

Not implemented:

- mistral-common's grid rule (round, not floor) and its pixel pipeline. vLLM
  counts and preprocesses with mistral-common; SGLang counts with its grid
  rule under `--tokenizer-mode mistral`. The two rules give different grids
  for some oversized images (1600x408: 14 rows here, 15 there), so a router
  in front of such an engine must not use this family.
- GPU resize. An engine that resizes on the GPU gets the same sizes and ids
  as the CPU path that this crate mirrors, but not the same pixel values.
- Samples deeper than 8 bits (PIL clips them) and a prompt with fewer `[IMG]`
  than images: both are `InvalidInput`.
- EXIF orientation. transformers' `load_image` applies it; this decode does
  not, as SGLang's Pixtral path does not.

The `mm_parity` test replays the 41 cases of
`tests/fixtures/mm_parity/mistral-large-4.json` (decode forms, sizes that
upscale or fit into the longest edge on both backends, multi-image prompts
and refusals); its README has the regeneration commands.
`tests/pixtral_planning.rs` checks the resolver, the 140-size sizing sweep
and the refusals against `tests/fixtures/pixtral/geometry.json`, which
`tests/fixtures/mm_parity/generate.py --family pixtral` writes beside the
table from the HF processor.
## 9. Nemotron 3.5 Super VL images

`models::nemotron_vl` serves `model_type = "nemotron_h_omni"`
(NVIDIA-Nemotron-3.5-Super-VL-120B-A12B, BF16 and NVFP4). Each image becomes
one dynamic-resolution tile. Its `<image>` placeholder expands to `<img>`,
one `<image>` per feature token, and `</img>`.

SGLang's Python processor and the checkpoint's HF remote-code processor do
not agree for this model. They use different size formulas and different
resize kernels: a 1280x720 image gives 880 tokens in SGLang and 920 in HF.
The spec field `sizing` names the reference to mirror:

| `sizing` | reference | parity |
| --- | --- | --- |
| `sglang` | SGLang `NanoNemotronVLImageProcessor` on a prompt of token ids | **bitwise**, full pixel path: SGLang's size math and request budget, PIL bicubic on u8, the per-channel normalize table, planar `[1, 3, H, W]` bfloat16 (`TensorData::Bf16`), 1-D positions |
| `hf` | `image_processing_nemotron_h_omni.py` in the checkpoint | exact tile sizes and token counts. `process_items` returns `Unsupported`, because the float32 antialias bicubic kernel is not mirrored. vLLM uses the same size formula but adds a text-aware request budget that this mode does not apply, so counts differ from a vLLM worker when text plus images exceed the context: for 3 x 1920x1080 at context 4096 with 10 text tokens, vLLM 0.28 gives 1344 tokens each (Dynamo's router pins it) and `hf` 2040 |

`spec_from_model_dir` resolves the spec of the HF processor: `sizing = hf`,
and `max_model_len` from the image-processor knobs. As transformers 5.19.0
does, it takes them from the `image_processor` object of
`processor_config.json` when that file has one, else from
`preprocessor_config.json`. It also reads `tokenizer.json`, because the
`<img>` and `</img>` ids are in no config file. `spec_from_hf_configs`
returns `Unsupported` for this model type. A consumer that routes to SGLang
sets two public fields before `build_processor`:
`sizing = NemotronVlSizing::Sglang`, and `max_model_len` = SGLang's
`--context-length`, or 8192 when the flag is not set (0 is refused). SGLang
reads the geometry from `config.json`, so resolution refuses a checkpoint
where `config.json` or a flat `preprocessor_config.json` disagrees with the
knobs that HF loads.

The tile size depends on the whole request. With `sglang` sizing, the images
share `max_model_len` minus the text tokens; each id that is not `<image>` is
a text token. SGLang shrinks the images for at most 10 rounds and then serves
the last sizes. A tile can thus end below `min_num_patches`, and the prompt
can stay longer than the context. With `hf` sizing, the patch budget of each
image depends on the number of images in the call. As a result:

- An engine calls `process_items(input_ids, media)` with the ids of the
  request; a placeholder count that differs from the image count is
  `InvalidInput`. `process_item` returns `Unsupported`: one image alone has
  no request to size it by.
- A router calls `layout_from_metadata`, which is exact for the request. The
  `Feature` part of each segment holds the feature count of that item, and
  `apply_layout` takes those counts. `NemotronVlProcessor::plan_images` gives
  the same counts and the tile sizes, also with no pixel work.
- `num_media_tokens` is the prompt span of one image alone with no text: its
  feature tokens plus the two markers (882 for 1280x720 with `sglang` sizing,
  922 with `hf`). It equals the span in a request only when the request
  budget does not bind.
- `apply_layout` returns `offsets` that span `<img>` to `</img>`. The item
  offsets of SGLang are the `<image>` run, which is `feature_ranges`.

Limits. Video, audio and SGLang's InternVL tile path (`min_num_patches = 0`)
are `Unsupported`, and `downsample_factor` must be 2. `sglang` sizing mirrors
SGLang's processor for a prompt of token ids. That processor also takes the
prompt as text, which SGLang's Python chat serving sends for this model, and
it then counts the text tokens on the prompt with `<image>` removed. The two
counts can differ (31 ids and 30 text tokens for one two-image chat prompt),
and when the budget binds the tiles differ too: at `--context-length 2078`
with 2 x 1024x1024, ids give 961 tokens each and text gives 1024 each. With `hf` sizing, an
image too thin for a tile is `InvalidInput`: there the HF processor raises
`ZeroDivisionError` or asks torch for an empty resize. SGLang turns off
Pillow's decompression-bomb limit for this model. The crate applies the
`DecodeLimits` of the driver, but its PNG, WebP and BMP decoder refuses a
decoded buffer above 512 MiB (178,956,970 RGB pixels); JPEG has no such limit.
GIF has no allocation budget of its own: only the `DecodeLimits` of the driver
bound the canvas that a GIF header asks for.

`tests/fixtures/mm_parity/families/nemotron_vl.py` records both references
from the config, processor and tokenizer files of the checkpoint, with the
library versions. It runs SGLang's processor on CPU, and the HF remote-code
processor. It needs no GPU:

```sh
python multimodal/tests/fixtures/mm_parity/generate.py --family nemotron_vl --model-dir <checkpoint dir> \
    --vllm-python <python> --llm-mm-harness <binary> [--check]
```

`mm_parity` replays the cases of `tests/fixtures/mm_parity/nemotron-3.5-super-vl.json`
against SGLang: the image format sweep of §6, an image already on its tile, a
tall image, request budgets, and rejections. `nemotron_vl_planning` covers spec
resolution, SGLang's requests from sizes alone (sizes below the minimum,
identity, rounding ties, thin images, request budgets), the HF requests and
digested random sweeps for both references, from
`tests/fixtures/nemotron_vl/sizes.json`.

## 10. Roadmap

The crate implements and golden-tests the image pipeline end to end
for `models::qwen_vl`, `models::glm5_next` and `models::pixtral`. For video
it decodes to frames and has the request arithmetic; no family turns the
frames into tensors yet:

- `video_decode` (feature `video-decode`, §2.4): MP4 / WebM / Matroska
  demuxing; H.264, HEVC, VP9 and AV1 through NVDEC, VP8 and VP9 on the CPU
  through libvpx when NVDEC cannot (both loaded at runtime; 64-bit Linux).
  `decode_frames` keeps the frames the caller names and resizes each to the
  size a `video_decode::Fit` gives. The decoder itself knows no model.
- `video`: per-request options. `VideoOptions` (`fps`, `max_frames`,
  `num_frames`, vLLM's `media_io_kwargs.video` keys) and `resolve_num_frames`
  give the frame count; `VideoPixelBudget` (`total_pixels`,
  `max_pixels_per_frame`, specific to this crate) carries pixel budgets.
- `models::qwen_vl::Qwen3VlVideo`: Qwen3-VL video sizing, read from the
  model's `video_preprocessor_config.json` (`Qwen3VLVideoProcessor`, or
  Cosmos3-Edge's `Cosmos3EdgeVideoProcessor`), and `smart_video_resize`,
  golden-tested against HF's own function.

What remains:

1. **`fetch`** — the trusted-source compatibility helper is still a stub;
   the feature currently exposes signatures only, and every call returns
   `MmError::Unsupported`.
2. **Video families** — temporal patchify, timestamp layouts and video
   M-RoPE on decoded frames (`models::qwen_vl` first).
3. **Family coverage** — GLM-5.3-Flash has image preprocessing and metadata-only planning
   (§6), Kimi-K3 has image preprocessing (§7), Mistral Large 4 has image
   preprocessing (§8) and Nemotron 3.5 Super VL has image preprocessing
   with `sglang` sizing and metadata-only planning with `hf` sizing (§9);
   GLM's video expansion is next after `models::qwen_vl`.

### Video and audio: planned layout

```
src/
  image/                 as today
  video.rs               as today: request options, frame count, pixel budgets
  video/
    sample.rs            further sampling policies (Qwen smart_nframes, GLM fps
                         windows) — pure index/timestamp math, no decoders
  audio/                 (feature `audio`)
    decode.rs            container decode + resample to mono f32 (symphonia)
    features.rs          STFT + log-mel filterbank (rustfft), the HF
                         feature-extractor equivalent
  models/
    qwen_vl/             grows into a directory when its video path lands
      mod.rs             image path + spec + registry glue
      video.rs           temporal patchify, timestamp layouts, video M-RoPE
```
