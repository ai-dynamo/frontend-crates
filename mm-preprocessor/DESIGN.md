<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-mm-preprocessor — Design

A pure-Rust replacement for the **image pipelines behind HF `AutoProcessor`**,
for LLM serving engines. Given one request (prompt tokens or text, plus raw
image sources), the crate fetches, decodes, resizes, normalizes, and patchifies
each image; expands the prompt's media placeholders; and computes position
encodings (e.g. M-RoPE) — CPU-side, GIL-free, owning no threads unless asked.

**Why it exists.** In Python serving stacks, multimodal preprocessing runs
under the GIL through HF processors (PIL/torchvision), competing with the
tokenizer and scheduler loops. Moving it to Rust removes it from the GIL and
from Python worker pools entirely. The hard requirement that shapes every
design decision below: **bit-exactness** against the mirrored HF processor. A
systematically skewed pipeline (wrong resample filter, wrong normalize
rounding, wrong patch order) still produces fluent model output — it just
silently loses accuracy — so "close enough" is not a testable contract;
byte-equality is.

**Non-goals.** Chat-template rendering (that is `dynamo-renderer`, which
deliberately stops at media placeholder markers); GPU preprocessing; video and
audio (the seams exist, no family implements them yet).

## 1. Architecture

One rule organizes the crate: **families produce data, the driver owns control
flow.** A model family never sees the request loop, the thread pool, or the
failure protocol; it turns decoded media into named tensors and *describes*
its prompt geometry as a value the driver applies mechanically.

```
request { text | input_ids, image sources }
   │
   ▼                       driver::process
fetch ─► content hash ─► decode ─► family.process_item   (per item, parallel)
   │                                   │ ProcessedItem { feature, aux, geometry }
   ▼                                   ▼
tokenize (if text) ─► family.layout ─► token_layout::apply_layout
   │                                   │ expanded input_ids + per-item offsets
   ▼                                   ▼
              family.positions ─► Output { input_ids, items, offsets, positions }
```

| module | responsibility |
| --- | --- |
| `pipeline` | the model-family seam: `MmFamilyProcessor` trait + data carriers |
| `driver` | model-independent orchestration, request caps, failure semantics |
| `registry` | family selection from a typed or JSON spec (the `AutoProcessor` entry) |
| `models/` | one module per family — `models::qwen_vl` first |
| `image` | decode (8-bit only, PIL-matching), bit-exact resize kernels, transforms |
| `token_layout` | validating placeholder expansion of the *already tokenized* prompt |
| `fetch` *(feature)* | data:/base64/file/http source resolution, byte budgets |
| `par` *(feature)* | the crate's only parallelism seam: rayon pool or inline |

Cross-cutting decisions:

- **Errors are `Result<T, String>`.** Every `Err` is a request-rejection
  message the engine returns to its client (a 400). There is no fallback
  path and no recoverable error taxonomy to model; `anyhow` is a candidate
  0.2 migration.
- **No environment variables, no implicit threads.** Pool sizing
  (`par::init_pool`) and fetch timeouts (`fetch::FetchOptions`) are explicit
  configuration; without the `parallel` feature rayon is not even linked and
  everything runs inline on the caller (a server owns its core budget — a
  library spawning pools behind its back would fight it).
- **Expansion never retokenizes.** The prompt is expanded in token-id space,
  so non-media tokens can never drift from a re-encode.
- **Growth without breakage.** `DecodedMedia`, `Geometry`, `TokenPattern`,
  `TensorData`, `PositionOutput`, and `PipelineSpec` are `#[non_exhaustive]`;
  new families, modalities, and position schemes land as semver-minor
  additions (release-plz runs cargo-semver-checks).

## 2. Key APIs

The family seam (`pipeline`):

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

The orchestrator (`driver`):

```rust
pub fn process(
    family: &dyn MmFamilyProcessor,
    input: MmInput,                                   // text | input_ids + images
    tokenize: impl Fn(&str) -> Result<Vec<i32>, String>,
) -> Result<Output, String>;
// `process_with(.., &ProcessOptions)` for explicit fetch timeouts.

pub struct Output {
    pub input_ids: Vec<i32>,          // expanded prompt
    pub items: Vec<OutputItem>,       // feature + aux tensors + content hash, prompt order
    pub offsets: Vec<(u32, u32)>,     // per-item inclusive token spans
    pub positions: PositionOutput,
}
```

Family selection (`registry`) — the `AutoProcessor`-shaped entry point. The
consumer resolves processor parameters however it likes (SGLang: from the
already-loaded HF processor, conservatively — unknown knobs mean "no Rust
pipeline" rather than approximation) and hands them over typed or as JSON:

```rust
#[serde(tag = "family", rename_all = "snake_case")]
pub enum PipelineSpec { QwenVl(QwenVlSpec) }          // one variant per family

pub fn build_pipeline(spec: PipelineSpec) -> Result<Box<dyn MmFamilyProcessor>, String>;
pub fn pipeline_from_spec(json: &str)     -> Result<Box<dyn MmFamilyProcessor>, String>;
```

Building blocks families compose (all public, individually testable):
`image::decode::decode_rgb`, `image::resize::resize_rgb` (bit-exact PIL
fixed-point Lanczos/Bicubic and torchvision's uint8-antialias bicubic —
selected per HF processor class), `token_layout::{layout_by_placeholder,
apply_layout}`, `models::qwen_vl::{smart_resize, mrope_image_only}`,
`fetch::{fetch_bytes, fetch_bytes_budgeted_with, ByteBudget}`,
`content_hash_u64` (blake3, first 8 bytes BE).

## 3. Python-parity map

Each item reproduces a specific Python behavior — most of them **bit-exactly**
(the exceptions are called out):

| this crate | on-par Python API | parity |
| --- | --- | --- |
| `registry::pipeline_from_spec` | `AutoProcessor.from_pretrained` + resolved image-processor kwargs | selection semantics |
| `driver::process` | `BaseMultimodalProcessor.process_mm_data_async` orchestration (SGLang) | same stages, same failure→400 semantics |
| `models::qwen_vl::QwenVlProcessor::process_item` | HF `Qwen2VLImageProcessor(Fast)` / `Qwen2VLImageProcessorPil` `__call__` → `pixel_values`, `image_grid_thw` | **bitwise** |
| `models::qwen_vl::smart_resize` | `sglang...qwen_vl.smart_resize` / HF `smart_resize` (incl. Python banker's rounding) | exact, plus an explicit reject of the degenerate 0-side case Python leaves to PIL |
| `models::qwen_vl::mrope_image_only` | `MRotaryEmbedding.get_rope_index` (image-only branch, identical across Qwen generations) | exact |
| `image::resize::resize_rgb(Pil(_))` | `PIL.Image.resize` (LANCZOS/BICUBIC, u8) | **bitwise** (PIL's i32 fixed-point kernels) |
| `image::resize::resize_rgb(AtenU8)` | `torchvision resize(antialias=True)` on uint8 | **bitwise** (ATen's per-axis i16 weight precision) |
| normalize LUT (family-internal) | slow path `rescale→normalize` vs fast path `_fuse_mean_std_and_rescale_factor` | **bitwise** — the two roundings differ on 128 of 256 u8 inputs, so the spec selects which to mirror |
| `image::decode::decode_rgb` | `PIL.Image.open(...).convert("RGB")` | same accepted formats; >8-bit samples rejected (PIL clips where Rust would rescale — refuse rather than silently diverge) |
| `token_layout::apply_layout` + `layout_by_placeholder` | `BaseMultimodalProcessor._expand_input_ids` + `get_mm_items_offset` | exact ids/offsets, plus full-coverage validation |
| `fetch::fetch_bytes` | `sglang...get_image_bytes` (`requests` proxy + `NO_PROXY` semantics, source precedence) | same behavior, plus streaming byte caps Python lacks |
| `content_hash_u64` | the *role* of `mm_utils.data_hash` (SHA-256) | **deliberately different algorithm** (blake3) — hashes are consistent within a path, never comparable across Rust and Python |

## 4. How a serving engine uses it — SGLang image preprocessing

SGLang's integration (the reference consumer) has three parts; only the first
touches this crate's API surface directly.

**Boot — resolve and gate.** Python inspects the already-loaded HF processor
and model type; if and only if every knob is recognized (family known,
processor class known, `do_resize/do_rescale/do_normalize` on,
`rescale_factor == 1/255`), it builds the typed spec and starts Rust MM
workers. Anything unrecognized → launch error, never silent approximation.

```rust
// per worker pool, once at boot
let family: Box<dyn MmFamilyProcessor> =
    registry::build_pipeline(PipelineSpec::QwenVl(QwenVlSpec {
        image_token_id, patch_size: 14, merge_size: 2, temporal_patch_size: 2,
        min_pixels, max_pixels, image_mean, image_std, resample: Resampler::AtenU8,
    }))?;
```

**Per request — drive the pipeline.** SGLang prefetches I/O-backed sources
(URLs, file paths) on its async runtime so blocking I/O never occupies a
fixed CPU worker, then hands bytes plus any CPU-cheap sources (data:/base64
strings) to the driver on an MM worker thread:

```rust
let output: Output = driver::process_with(
    family.as_ref(),
    MmInput { text, input_ids, images },        // images: Bytes (prefetched) or String
    |text: &str| tokenizer.encode(text),        // only used when input_ids absent
    &fetch_options.into(),
)?;                                             // Err => reject request (400)
```

**Drain — hand off zero-copy.** The engine reshapes `Output` into whatever its
scheduler consumes. SGLang packs Qwen's shape (concatenated `pixel_values`,
grids, hashes, offsets, M-RoPE), parks it keyed by request id, and its Python
scheduler wraps the buffers as numpy/torch views without copying or hashing.
That packing is engine-specific and lives in SGLang, not here.

The same `driver::process` is also exposed to SGLang's pytest parity suites
through its PyO3 adapter, so the tests exercise the exact server pipeline.

## 5. Testing strategy

Three layers, all pinned to byte-equality:

1. **Crate-local unit tests** — smart_resize against Python-derived reference
   values (including rounding ties), patchify layout, normalize-LUT
   divergence, layout coverage validation, fetch budgets/NO_PROXY, plus a
   thread-count guard proving the default build owns no threads.
2. **Crate-local golden replay** — this repo's CI has no Python/HF, so
   committed fixtures (generated by SGLang tooling from the HF processor and
   `get_rope_index`, cross-checked before writing) drive
   `pipeline_from_spec → driver::process` and compare **every output field
   bitwise**: both resamplers, both smart_resize branches, multi-image.
3. **Consumer parity (SGLang CI)** — per-step and end-to-end pytest suites
   compare the Rust path against the live HF/Python processors field-by-field
   with `.tobytes()` equality, plus a GPU e2e test and an MMMU accuracy gate
   (a systematic skew reads as fluent text; only the benchmark catches it
   end-to-end).

## 6. Roadmap

This PR is the skeleton: module layout, public API signatures (`todo!()`
bodies), and this document. Implementation lands next (a working, fully
tested implementation exists on the `kan/mm-development` branch and gets
re-homed into this layout):

1. **PR 2 — primitives**: `image` (decode + resize kernels), `token_layout`,
   `par`, `fetch`, with their unit tests; wires the `fetch`/`parallel`
   feature deps.
2. **PR 3 — driver + registry + `models/qwen_vl`**: the full pipeline, the
   golden fixtures + replay test, the no-threads guard; flips the crate to
   publishable.

Family growth (validated against the GLM-4V and Kimi K2.5/K3 Python
processors, not yet implemented): GLM's `<|begin_of_image|> … <|end_of_image|>`
framing fits `TokenPattern::Explicit` and its M-RoPE variant is a new
`PositionOutput` variant; Kimi's NaViT resize/pad and `(h, w)` merge kernels
are family-internal; Kimi K3 interleaves *tokenized text* inside the media
span, so `layout` gains a defaulted `layout_with(&LayoutContext)` method
(semver-minor) carrying the driver's tokenize hook — the `Fn` bound on
`tokenize` is already in place for it. Video/audio grow `DecodedMedia`
variants and `Capabilities` flags.
