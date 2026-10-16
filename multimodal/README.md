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
      └─ bytes ────────────────────►│  image::decode::decode_rgb
        └─ rgb ────────────────────►│  family.process_item      (per item; the engine drives
                                    │                            the loop, optionally fanning
                                    │                            out on `execution`)
                                    │     │ ProcessedItem
tokenize (if text) ─ ids, items ───►│  family.layout ─► token_layout::apply_layout
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

### 2.1 A router (e.g. dynamo) — accounting and routing, no pixel work

| when | API | in → out |
| --- | --- | --- |
| boot, per model | `registry::spec_from_model_dir` | config dir (hub download is the router's concern) → `ProcessorSpec`; `Err` = model unsupported |
| boot, per model | `registry::build_processor` | `ProcessorSpec` → `Box<dyn MmFamilyProcessor>` |
| per media part | `fetch::MediaFetcher::fetch` *(feature `fetch`)*, or the consumer's own protected fetcher | untrusted media source → raw bytes, under the SSRF policy and a byte cap |
| per trusted media part | `fetch::fetch_bytes` *(feature `fetch`, stub)* | trusted media source → raw bytes once implemented; currently returns `MmError::Unsupported` |
| per decoded image | `content_hash_canonical_image` | shape + dtype + RGB bytes → Dynamo-compatible identity |
| per image part | `image::decode::dimensions` | bytes → `MediaMetadata::Image` dimensions — header probe, no pixel decode |
| per media part | `MmFamilyProcessor::num_media_tokens` | typed lightweight metadata → the item's expanded token count |

Token counts give the expanded prompt length for routing; hashes drive
prefix-cache-affinity routing and travel downstream as `mm_hashes`. The engine
uses supplied hashes unchanged, whether Dynamo derived them from canonical
decoded media or a passthrough URL.

### 2.2 An inference engine (e.g. SGLang) — the full pixel path

| when | API | in → out |
| --- | --- | --- |
| boot, per worker pool | `registry::build_processor` | spec pre-resolved by the engine's gate or config dir → `Box<dyn MmFamilyProcessor>` |
| per image without a supplied hash | `content_hash_bytes` | encoded bytes → SGLang's fallback `u64` identity |
| per image | `image::decode::decode_rgb` | bytes + `DecodeLimits` → `(rgb, h, w)` (8-bit only, PIL-matching, header-checked caps); the engine wraps it as `DecodedMedia::Image` |
| per image | `MmFamilyProcessor::process_item` | `DecodedMedia` → `ProcessedItem` (preserves modality, feature-token count, tensors, and optional geometry) |
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
let mut items: Vec<ProcessedItem> = Vec::with_capacity(images.len());
let mut hashes: Vec<u64> = Vec::with_capacity(images.len());
for (bytes, supplied_hash) in images {                  // fetched + capped by the engine
    let hash = match supplied_hash {
        Some(hash) => hash,
        None => content_hash_bytes(&bytes),
    };
    hashes.push(hash);
    let (rgb, height, width) = image::decode::decode_rgb(&bytes, &limits)?;
    items.push(family.process_item(&DecodedMedia::Image { rgb, height, width })?);
}
let input_ids: Vec<i32> = match request_ids { Some(ids) => ids, None => tokenizer.encode(&text)? };
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
| `MmFamilyProcessor::num_media_tokens` | `_get_num_multimodal_tokens(…)` | exact token counts from typed image/video/audio metadata, no pixel or feature-extraction work |
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
| `models::qwen_vl::mrope_image_only` | `get_rope_index` (in transformers' Qwen model code; image-only branch, identical across Qwen generations) | exact, optional model-input preparation beyond strict `AutoProcessor` parity |

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
   committed fixtures drive the §2.2 composition (`build_processor` →
   `decode_rgb` → `process_item` → `layout` → `apply_layout` → `positions`)
   and compare **every output field bitwise**, plus the resampler stage on
   its own (`resized_<i>.u8`) so a failure names the stage: both resamplers,
   both smart_resize branches, multi-image, and a tall image that crosses
   Pillow's pass-order rule. `tests/fixtures/qwen_vl/generate.py`
   regenerates them from the HF processor and `get_rope_index`, cross-checks
   that the resized buffer reproduces `pixel_values`, and records the
   transformers/torch/torchvision/Pillow versions in each `case.json`.
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

## 6. Roadmap

The crate implements and golden-tests the image pipeline end to end
for `models::qwen_vl`. For video it decodes to frames and has the request
arithmetic; no family turns the frames into tensors yet:

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
3. **Family coverage** — GLM and Kimi are the validated candidates after
   `models::qwen_vl`.

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
