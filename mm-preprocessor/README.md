<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-mm-preprocessor

Model-family multimodal preprocessing for LLM inference routers/engines — a
Rust replacement for the image pipelines behind HF `AutoProcessor`: per-family
decode → resize → normalize → patchify, prompt placeholder expansion, and
position math (M-RoPE), all **bit-exact** against the mirrored HF processor.

| feature    | default | adds |
| ---------- | ------- | ---- |
| `parallel` | **on**  | links rayon; kernels still run inline until `execution::init_pool` arms the crate-owned pool |
| `fetch`    | off     | data:/base64/file/http source resolution with `requests`-parity proxy semantics |

**The boundary.** The crate carries what HF ships, plus what a router and an
engine must compute *identically* or routing keys and token accounting
diverge: spec resolution from HF config files, pixel-free token accounting,
media source resolution (`fetch`), and content-hash identity
(`content_hash_u64`).

## 1. Where the crate sits

An **inference engine** preprocesses each request's multimodal inputs using this crate:

```
engine                              │   dynamo-mm-preprocessor (this crate)
────────────────────────────────────│──────────────────────────────────────
boot: locate model configs ────────►│  registry::spec_from_model_dir (or a
                                    │  pre-resolved spec) ─► build_processor
                                    │           ─► Box<dyn MmFamilyProcessor>
per request:                        │
image sources, caps ───────────────►│  fetch::fetch_bytes         ─► raw bytes
      ├─ bytes ────────────────────►│  content_hash_u64           ─► mm_hashes
      └─ bytes ────────────────────►│  image::decode::decode_rgb
        └─ rgb ────────────────────►│  family.process_item      (per item; the engine drives
                                    │                            the loop, optionally fanning
                                    │                            out on `execution`)
                                    │     │ ProcessedItem { feature, aux, geometry }
tokenize (if text) ─ ids, geoms ───►│  family.layout ─► token_layout::apply_layout
                                    │     │ expanded input_ids + per-item offsets
        └─ offsets, geoms ─────────►│  family.positions          (e.g. M-RoPE)
                                    │
```

A **router** uses this crate for multimodal requests' load accounting and cache-aware routing:

```
router                              │   dynamo-mm-preprocessor (this crate)
────────────────────────────────────│──────────────────────────────────────
boot: locate model configs ────────►│  registry::spec_from_model_dir
                                    │    ─► build_processor ─► Box<dyn MmFamilyProcessor>
per request                         │
   image_url ──────────────────────►│  fetch::fetch_bytes         ─► raw bytes
      ├─ bytes ────────────────────►│  content_hash_u64           ─► cache-affinity key
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
| `fetch` *(feature)* | one media source → raw bytes (data:/base64/file/http, `requests`-parity proxy semantics, streaming byte budgets) |
| `token_layout` | validating placeholder expansion of the *already tokenized* prompt |
| `execution` | the crate's only parallelism seam: inline by default, a runtime-armed crate-owned rayon pool otherwise |


## 2. The API, by consumer

One trait carries the family contract; consumers differ in which methods
they call and how they obtain the spec:

```rust
pub trait MmFamilyProcessor: Send + Sync {
    fn capabilities(&self) -> Capabilities;                       // images-only default
    fn num_media_tokens(&self, width: usize, height: usize) -> Result<usize, String>;
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String>;
    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String>;
    fn positions(&self, input_len: usize, offsets: &[(u32, u32)], items: &[Geometry])
        -> Result<PositionOutput, String>;                        // Rope1D default
}

pub struct ProcessedItem { pub feature: Tensor, pub aux: NamedTensors, pub geometry: Geometry }
pub enum PositionOutput { Rope1D, MRope { positions: Vec<i64>, delta: i64 } }
```

Specs arrive pre-resolved (e.g. by SGLang's Python gate) or from the HF config files. 
Resolution is conservative either way: an unknown
`model_type` or a knob the pipeline cannot honor bit-exactly is an `Err`,
never a silent approximation.

```rust
#[serde(tag = "family", rename_all = "snake_case")]
pub enum ProcessorSpec { QwenVl(QwenVlSpec) }          // one variant per family

pub fn build_processor(spec: ProcessorSpec) -> Result<Box<dyn MmFamilyProcessor>, String>;
pub fn processor_from_spec(json: &str)     -> Result<Box<dyn MmFamilyProcessor>, String>;

// AutoProcessor.from_pretrained parity for Python-free consumers:
pub fn spec_from_hf_configs(config_json: &str, preprocessor_config_json: &str)
    -> Result<ProcessorSpec, String>;
pub fn spec_from_model_dir(dir: &Path) -> Result<ProcessorSpec, String>;
```

### 2.1 A router (e.g. dynamo) — accounting and routing, no pixel work

| when | API | in → out |
| --- | --- | --- |
| boot, per model | `registry::spec_from_model_dir` | config dir (hub download is the router's concern) → `ProcessorSpec`; `Err` = model unsupported |
| boot, per model | `registry::build_processor` | `ProcessorSpec` → `Box<dyn MmFamilyProcessor>` |
| per image part | `fetch::fetch_bytes` *(feature `fetch`)* | media source → raw bytes — or the router's own async connector |
| per image part | `content_hash_u64` | bytes → `u64` cache-affinity key, identical to the engine's |
| per image part | `image::decode::dimensions` | bytes → (h, w) — header probe, no pixel decode |
| per image part | `MmFamilyProcessor::num_media_tokens` | (width, height) → the image's expanded token count |

Token counts give the expanded prompt length for routing; hashes drive
prefix-cache-affinity routing and might travel downstream as `mm_hashes`.

### 2.2 An inference engine (e.g. SGLang) — the full pixel path

| when | API | in → out |
| --- | --- | --- |
| boot, per worker pool | `registry::build_processor` | spec pre-resolved by the engine's gate or config dir → `Box<dyn MmFamilyProcessor>` |
| per image | `content_hash_u64` | bytes → `u64` — same keys as the router |
| per image | `image::decode::decode_rgb` | bytes → `(rgb, h, w)` (8-bit only, PIL-matching); the engine wraps it as `DecodedMedia::Image` |
| per image | `MmFamilyProcessor::process_item` | `DecodedMedia` → `ProcessedItem { feature, aux, geometry }` |
| per request | `MmFamilyProcessor::layout` | input_ids + geometries → `TokenLayout` (a description, not yet applied) |
| per request | `token_layout::apply_layout` | ids + `TokenLayout` → expanded ids + per-item offsets; validates the family contract (full coverage, each item exactly once, no zero-token item) |
| per request | `MmFamilyProcessor::positions` | expanded length + offsets + geometries → `PositionOutput` (e.g. M-RoPE) |

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
for bytes in images {                                  // fetched + capped by the engine
    hashes.push(content_hash_u64(&bytes));             // same keys as the router
    let (rgb, height, width) = image::decode::decode_rgb(&bytes)?;
    items.push(family.process_item(&DecodedMedia::Image { rgb, height, width })?);
}
let input_ids: Vec<i32> = match request_ids { Some(ids) => ids, None => tokenizer.encode(&text)? };
let geometries: Vec<Geometry> = items.iter().map(|i| i.geometry.clone()).collect();
let layout: TokenLayout = family.layout(&input_ids, &geometries)?;
let expanded: ExpandedPrompt = token_layout::apply_layout(&input_ids, &layout, items.len())?;
let positions: PositionOutput =
    family.positions(expanded.input_ids.len(), &expanded.offsets, &geometries)?;
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
   `aux = [("image_grid_thw", [1, 6, 8])]`, `geometry = Grid([1, 6, 8])`.
2. `layout` — the image costs `1·6·8 / 2² = 12` tokens (the ViT merges 2×2
   patches per token): keep text `0..2`, place item 0 as 12 copies of
   `<|image_pad|>`, keep text `3..5`. `apply_layout` executes and validates:
   expanded ids of length 16, `offsets = [(2, 13)]`.
3. `positions(16, …)` — M-RoPE. Text advances the three (t, h, w) rows
   together; the 12 image tokens span a 3×4 grid (`6/2 × 8/2`) at base 2;
   text after resumes at `2 + max(1, 3, 4) = 6`. Returns flat `[3, 16]`
   positions and `delta = 7 + 1 − 16 = −8` (added to the sequence length at
   decode — the image packed 12 tokens into 4 position steps).

The engine scatters the ViT's 12 output embeddings into positions 2..13
(from `offsets`) and feeds `positions` to the model's rotary path.

## 3. Python-parity map

Each item reproduces a specific Python behavior, most of them **bit-exactly**:

| this crate | on-par Python API | parity |
| --- | --- | --- |
| `registry::spec_from_hf_configs` / `spec_from_model_dir` | `AutoProcessor.from_pretrained` (config parsing + processor selection) | selection semantics; unknown knobs → `Err`, never approximation |
| `registry::processor_from_spec` | building the processor from already-resolved kwargs | selection semantics |
| `MmFamilyProcessor::num_media_tokens` | `_get_num_multimodal_tokens(image_sizes=…)` | exact token counts, no pixel work |
| `image::resize::resize_rgb(Pil(_))` | `PIL.Image.resize` (LANCZOS/BICUBIC, u8) | **bitwise** (PIL's i32 fixed-point kernels) |
| `image::resize::resize_rgb(AtenU8)` | `torchvision resize(antialias=True)` on uint8 | **bitwise** (ATen's per-axis i16 weight precision) |
| normalize LUT (family-internal) | slow path `rescale→normalize` vs fast path `_fuse_mean_std_and_rescale_factor` | **bitwise** — the roundings differ on 128 of 256 u8 inputs; the spec selects which to mirror |
| `image::decode::decode_rgb` | `PIL.Image.open(...).convert("RGB")` | same accepted formats; >8-bit samples rejected rather than silently diverging (PIL clips, Rust would rescale) |
| `image::decode::dimensions` | lazy `PIL.Image.open(...).size` | header-only probe |
| `fetch::fetch_bytes` | `transformers.image_utils.load_image` / SGLang `get_image_bytes` (`requests` proxy + `NO_PROXY` semantics, source precedence) | same behavior, plus streaming byte caps Python lacks |
| `content_hash_u64` | the *role* of SGLang `mm_utils.data_hash` | deliberately blake3, one shared definition — router and Rust-engine keys must agree (the Python path's SHA-256 stays a documented divergence) |
| `token_layout::apply_layout` + `layout_by_placeholder` | HF `Qwen2VLProcessor`'s own `<|image_pad|>` expansion / SGLang `_expand_input_ids` + `get_mm_items_offset` | exact ids/offsets, plus full-coverage validation |
| `models::qwen_vl::QwenVlProcessor::process_item` | HF `Qwen2VLImageProcessor(Fast)` / `Qwen2VLImageProcessorPil` `__call__` → `pixel_values`, `image_grid_thw` | **bitwise** |
| `models::qwen_vl::smart_resize` | HF/SGLang `smart_resize` (incl. Python banker's rounding) | exact; also rejects the degenerate 0-side case Python leaves to PIL |
| `models::qwen_vl::mrope_image_only` | `get_rope_index` (in transformers' Qwen model code; image-only branch, identical across Qwen generations) | exact |


## 4. Testing

Three layers, all pinned to bitwise-equality:

1. **Crate-local unit tests** — smart_resize against Python-derived reference
   values (including rounding ties), patchify layout, normalize-LUT
   divergence, layout coverage validation, fetch budgets and `NO_PROXY`
   matching, config-resolution gating, plus a thread-count guard proving
   the crate owns no threads while the pool is unarmed (the default).
2. **Crate-local golden replay** — this repo's CI has no Python/HF, so
   committed fixtures (generated by SGLang tooling from the HF processor and
   `get_rope_index`, cross-checked before writing) drive the §2.2 composition
   (`build_processor` → `decode_rgb` → `process_item` → `layout` →
   `apply_layout` → `positions`) and compare **every output field bitwise**:
   both resamplers, both smart_resize branches, multi-image.
3. **Consumer parity (SGLang CI)** — per-step and end-to-end pytest suites
   compare the Rust path against the live HF/Python processors field-by-field
   with `.tobytes()` equality, plus a GPU e2e test and an MMMU accuracy gate
   (a systematic skew reads as fluent text; only the benchmark catches it
   end-to-end).

## 5. Roadmap

This PR is the skeleton: module layout, public API signatures (`todo!()`
bodies), and this document. A working, fully tested implementation exists
and lands next in two steps:

1. **primitives** — `image`, `token_layout`, `execution`, with unit tests;
   wires the `parallel` feature dep.
2. **registry + `models/qwen_vl`** — the family, golden fixtures + replay
   test, the no-threads guard; flips the crate to publishable.

**Family coverage** grows next — GLM-4V and Kimi are the validated
candidates after `models::qwen_vl`.

### Video and audio: planned layout

```
src/
  image/                 as today
  video/
    sample.rs            frame-sampling policies (Qwen smart_nframes, GLM fps
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
