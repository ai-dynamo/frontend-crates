<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-mm-preprocessor — Design

A Rust replacement for the **image pipelines behind HF `AutoProcessor`**,
for LLM serving routers or engines: per-model-family image preprocessing (decode →
resize → normalize → patchify), prompt placeholder expansion, and
model-family position math (e.g. M-RoPE) — CPU-side, GIL-free, owning no
threads unless asked.

Testing: we test **bit-exactness** against the mirrored HF processor. 

**The boundary.** The crate's scope is exactly what HF ships, nothing more.
On the Python path, the *engine* (e.g. SGLang's `BaseMultimodalProcessor`)
owns request orchestration — fetching sources, content hashing, per-request
caps, failure policy — and calls into HF for the processor math. This crate
keeps that layering: engines write their own driver against the
`MmFamilyProcessor` trait; the crate supplies the processor. Deliberately
**not** in this crate: source fetching (Python: the engine's
`get_image_bytes`), content hashing (Python: the engine's `data_hash`),
request caps and failure semantics, and any async runtime.

**Other non-goals (for now).** Chat-template rendering (that is
`dynamo-renderer`, which deliberately stops at media placeholder markers) and
GPU preprocessing. Video and audio are planned, not implemented — §6 records
their module layout and decode boundary.

## 1. Architecture

The following shows functionalities of this crate, working together with an
inference engine-side pipeline.

```
serving engine (its driver)         │   dynamo-mm-preprocessor (this crate)
────────────────────────────────────│──────────────────────────────────────
boot: resolve HF params ── spec ───►│  registry::build_processor ─► Box<dyn MmFamilyProcessor>
                                    │
per request:                        │
fetch sources, hash, caps           │
   └─ raw bytes ───────────────────►│  image::decode::decode_rgb
        └─ rgb ────────────────────►│  family.process_item      (per item; engine fans out)
                                    │     │ ProcessedItem { feature, aux, geometry }
tokenize (if text) ─ ids, geoms ───►│  family.layout ─► token_layout::apply_layout
                                    │     │ expanded input_ids + per-item offsets
        └─ offsets, geoms ─────────►│  family.positions          (e.g. M-RoPE)
                                    │
drain: pack tensors for scheduler   │
```

| module | responsibility |
| --- | --- |
| `processor` | the model-family seam: `MmFamilyProcessor` trait + data carriers |
| `registry` | family selection from a typed or JSON spec (the `AutoProcessor` entry) |
| `models/` | one module per family — `models::qwen_vl` first |
| `image` | decode (8-bit only, PIL-matching), bit-exact resize kernels, transforms |
| `token_layout` | validating placeholder expansion of the *already tokenized* prompt |
| `execution` | the crate's only parallelism seam: inline by default, a runtime-armed crate-owned rayon pool otherwise |

Cross-cutting decisions:

- **Errors are `Result<T, String>`.** Every `Err` is a request-rejection
  message the engine returns to its client (a 400). There is no fallback
  path and no recoverable error taxonomy to model; `anyhow` is a candidate
  0.2 migration.
- **No environment variables, no implicit threads.** Kernels run inline on
  the caller until a consumer arms the crate-owned rayon pool at runtime
  (`execution::init_pool(n)`; `0` = `min(cores, 8)`) — a server owns its core
  budget, so a library spawning pools behind its back would fight it, while a
  Python extension arms the pool once at startup for intra-call parallelism.
  The default-on `parallel` feature only controls whether rayon is linked;
  `default-features = false` drops it and forces inline at compile time.
  Engine drivers may reuse the `execution` seam for their own per-item
  fan-out so one pool serves both.
- **Expansion never retokenizes.** The prompt is expanded in token-id space,
  so non-media tokens can never drift from a re-encode.
- **Growth without breakage.** `DecodedMedia`, `Geometry`, `TokenPattern`,
  `TensorData`, `PositionOutput`, and `ProcessorSpec` are `#[non_exhaustive]`;
  new families, modalities, and position schemes land as semver-minor
  additions (release-plz runs cargo-semver-checks).

## 2. Key APIs

The family seam (`processor`) — everything an engine's driver programs against:

```rust
pub trait MmFamilyProcessor: Send + Sync {
    fn capabilities(&self) -> Capabilities;                       // images-only default
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String>;
    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String>;
    fn positions(&self, input_len: usize, offsets: &[(u32, u32)], items: &[Geometry])
        -> Result<PositionOutput, String>;                        // Rope1D default
}

pub struct ProcessedItem { pub feature: Tensor, pub aux: NamedTensors, pub geometry: Geometry }
pub enum PositionOutput { Rope1D, MRope { positions: Vec<i64>, delta: i64 } }
```

Family selection (`registry`) — the `AutoProcessor`-shaped entry point. The
consumer resolves processor parameters however it likes (SGLang: from the
already-loaded HF processor, conservatively — unknown knobs mean "no Rust
processor" rather than approximation) and hands them over typed or as JSON:

```rust
#[serde(tag = "family", rename_all = "snake_case")]
pub enum ProcessorSpec { QwenVl(QwenVlSpec) }          // one variant per family

pub fn build_processor(spec: ProcessorSpec) -> Result<Box<dyn MmFamilyProcessor>, String>;
pub fn processor_from_spec(json: &str)     -> Result<Box<dyn MmFamilyProcessor>, String>;
```

Layout application (`token_layout`) — the one piece of mechanics the crate
insists engines share, because it validates the whole family contract (full
source coverage, every item placed exactly once, no zero-token item):

```rust
pub fn apply_layout(src: &[i32], layout: &TokenLayout, n_items: usize)
    -> Result<ExpandedPrompt, String>;    // expanded ids + inclusive per-item offsets
pub fn layout_by_placeholder(ids: &[i32], placeholder_id: i32, counts: &[usize])
    -> Result<TokenLayout, String>;       // the simple qwen-style repeat layout
```

Building blocks families compose (all public, individually testable):
`image::decode::decode_rgb`, `image::resize::resize_rgb` (bit-exact PIL
fixed-point Lanczos/Bicubic and torchvision's uint8-antialias bicubic —
selected per HF processor class), `image::transforms`,
`models::qwen_vl::{smart_resize, mrope_image_only}`.

## 3. Python-parity map

Each item reproduces a specific Python behavior — most of them **bit-exactly**
(the exceptions are called out):

| this crate | on-par Python API | parity |
| --- | --- | --- |
| `registry::processor_from_spec` | `AutoProcessor.from_pretrained` + resolved image-processor kwargs | selection semantics |
| `models::qwen_vl::QwenVlProcessor::process_item` | HF `Qwen2VLImageProcessor(Fast)` / `Qwen2VLImageProcessorPil` `__call__` → `pixel_values`, `image_grid_thw` | **bitwise** |
| `models::qwen_vl::smart_resize` | HF/SGLang `smart_resize` (incl. Python banker's rounding) | exact, plus an explicit reject of the degenerate 0-side case Python leaves to PIL |
| `models::qwen_vl::mrope_image_only` | `get_rope_index` (ships in transformers' Qwen model code; image-only branch, identical across Qwen generations) | exact |
| `image::resize::resize_rgb(Pil(_))` | `PIL.Image.resize` (LANCZOS/BICUBIC, u8) | **bitwise** (PIL's i32 fixed-point kernels) |
| `image::resize::resize_rgb(AtenU8)` | `torchvision resize(antialias=True)` on uint8 | **bitwise** (ATen's per-axis i16 weight precision) |
| normalize LUT (family-internal) | slow path `rescale→normalize` vs fast path `_fuse_mean_std_and_rescale_factor` | **bitwise** — the two roundings differ on 128 of 256 u8 inputs, so the spec selects which to mirror |
| `image::decode::decode_rgb` | `PIL.Image.open(...).convert("RGB")` | same accepted formats; >8-bit samples rejected (PIL clips where Rust would rescale — refuse rather than silently diverge) |
| `token_layout::apply_layout` + `layout_by_placeholder` | HF `Qwen2VLProcessor`'s own `<|image_pad|>` expansion / SGLang `_expand_input_ids` + `get_mm_items_offset` | exact ids/offsets, plus full-coverage validation |

Engine-side counterparts that stay **out** of this crate, matching where they
live on the Python path (SGLang's Rust driver keeps them in `sglang-mm`):

| engine concern | Python home |
| --- | --- |
| request orchestration (fetch → hash → decode → layout control flow, caps, failure→400) | SGLang `BaseMultimodalProcessor.process_mm_data_async` |
| media source resolution (data:/base64/file/http, proxies, byte budgets) | SGLang `get_image_bytes` (≈ `transformers.image_utils.load_image`) |
| content hashing for cache/dedup identity | SGLang `mm_utils.data_hash` |
| scheduler-shaped packing / zero-copy drain | SGLang `wrap_encoded` / `MultimodalDataItem` |

## 4. How a serving engine uses it — SGLang image preprocessing

**Boot — resolve and gate.** Python inspects the already-loaded HF processor
and model type; if and only if every knob is recognized (family known,
processor class known, `do_resize/do_rescale/do_normalize` on,
`rescale_factor == 1/255`), it builds the typed spec and starts Rust MM
workers. Anything unrecognized → launch error, never silent approximation.

```rust
// per worker pool, once at boot
let family: Box<dyn MmFamilyProcessor> =
    registry::build_processor(ProcessorSpec::QwenVl(QwenVlSpec {
        image_token_id, patch_size: 14, merge_size: 2, temporal_patch_size: 2,
        min_pixels, max_pixels, image_mean, image_std, resample: Resampler::AtenU8,
    }))?;
```

**Per request — the engine's driver composes the crate.** SGLang's driver
(in its `sglang-mm` adapter crate) owns fetching (prefetched on its async
runtime so blocking I/O never occupies a fixed CPU worker), hashing, caps,
and the failure contract, then calls the family:

```rust
// SGLang's driver, per request (sketch; error handling elided)
let mut items: Vec<ProcessedItem> = Vec::with_capacity(images.len());
let mut hashes: Vec<u64> = Vec::with_capacity(images.len());
for bytes in images {                                  // fetched + capped by the engine
    hashes.push(engine_content_hash(&bytes));          // engine-owned identity
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

**Drain — hand off zero-copy.** The engine reshapes the tensors into whatever
its scheduler consumes. SGLang packs Qwen's shape (concatenated
`pixel_values`, grids, hashes, offsets, M-RoPE), parks it keyed by request
id, and its Python scheduler wraps the buffers as numpy/torch views without
copying or hashing. Packing and drain are engine-specific and live in SGLang.

SGLang's driver is also exposed to its pytest parity suites through its PyO3
adapter, so the tests exercise the exact server pipeline.

## 5. Testing strategy

Three layers, all pinned to byte-equality:

1. **Crate-local unit tests** — smart_resize against Python-derived reference
   values (including rounding ties), patchify layout, normalize-LUT
   divergence, layout coverage validation, plus a thread-count guard proving
   the crate owns no threads while the pool is unarmed (the default).
2. **Crate-local golden replay** — this repo's CI has no Python/HF, so
   committed fixtures (generated by SGLang tooling from the HF processor and
   `get_rope_index`, cross-checked before writing) drive the §4 composition
   (`build_processor` → `decode_rgb` → `process_item` → `layout` →
   `apply_layout` → `positions`) and compare **every output field bitwise**:
   both resamplers, both smart_resize branches, multi-image.
3. **Consumer parity (SGLang CI)** — per-step and end-to-end pytest suites
   compare the Rust path against the live HF/Python processors field-by-field
   with `.tobytes()` equality, plus a GPU e2e test and an MMMU accuracy gate
   (a systematic skew reads as fluent text; only the benchmark catches it
   end-to-end).

## 6. Roadmap

This PR is the skeleton: module layout, public API signatures (`todo!()`
bodies), and this document. Implementation lands next (a working, fully
tested implementation exists and gets re-homed into this layout):

1. **primitives**: `image` (decode + resize kernels + transforms),
   `token_layout`, `execution`, with their unit tests; wires the `parallel`
   feature dep.
2. **registry + `models/qwen_vl`**: the family implementation, the golden
   fixtures + replay test, the no-threads guard; flips the crate to
   publishable.

Family growth (validated against the GLM-4V and Kimi K2.5/K3 Python
processors, not yet implemented): GLM's `<|begin_of_image|> … <|end_of_image|>`
framing fits `TokenPattern::Explicit` and its M-RoPE variant is a new
`PositionOutput` variant; Kimi's NaViT resize/pad and `(h, w)` merge kernels
are family-internal; Kimi K3 interleaves *tokenized text* inside the media
span, so `layout` gains a defaulted `layout_with(&LayoutContext)` method
(semver-minor) carrying an engine-supplied encode hook.

### Video and audio: planned layout

The trait seam already carries modalities (`Capabilities`; the
`#[non_exhaustive]` `DecodedMedia`/`Geometry`), so growing them is
semver-minor. What is worth planning ahead is the module layout and where
decoding happens:

```
src/
  image/                 as today
  video/
    sample.rs            frame-sampling policies (Qwen smart_nframes, GLM fps
                         windows) — pure index/timestamp math, no decoders
  audio/                 (feature `audio`)
    decode.rs            container decode + resample to mono f32 (symphonia)
    features.rs          STFT + log-mel filterbank, the HF feature-extractor
                         equivalent (rustfft)
  models/
    qwen_vl/             a single-file family grows into a directory when its
      mod.rs             video path lands: image path + spec + registry glue
      video.rs           temporal patchify over real frames, timestamp
                         layouts, video M-RoPE inputs
```

- **Video container decode stays engine-side.** There is no production
  pure-Rust H.264/HEVC path (vLLM's `llm-multimodal` resorts to optional
  OpenCV), and engines already own GPU decode. The crate takes
  already-decoded frames:
  `DecodedMedia::Video { frames /* T·H·W·C u8 */, height, width, timestamps }`.
- **Frame sampling is a planning API, not part of `process_item`.** Which
  frames to decode is processor knowledge that must run *before* decoding so
  the engine never decodes frames it will drop. Families expose it as pure
  functions (`video::sample` + a per-family policy); the engine consults it
  between probing the container and decoding: probe → sample plan → decode
  those frames → `process_item`.
- **Audio decode lives in the crate**, like `image::decode`: symphonia and
  rustfft are pure Rust, and mel/fbank extraction is exactly what HF's
  feature extractors ship (`WhisperFeatureExtractor`), under the same
  bit-exactness contract. Behind an `audio` feature (`dep:symphonia`,
  `dep:rustfft`) so image-only consumers stay lean.
- **Carriers.** `Geometry::Grid` already models video (`t > 1` with real
  frames instead of a still's duplicated temporal copies); audio adds a
  frame-count variant. Timestamp-interleaved token layouts (Qwen3's
  `<t seconds>` text, GLM's per-frame integers) ride the same `layout_with`
  encode hook planned for Kimi K3, and video M-RoPE
  (`tokens_per_second` / `second_per_grid_ts`) stays family-internal behind
  `positions`.
