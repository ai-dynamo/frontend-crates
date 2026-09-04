<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-mm-preprocessor

Model-family multimodal preprocessing for LLM inference serving — a pure-Rust
replacement for the image pipelines behind HF `AutoProcessor`.

A serving engine hands the driver one request (prompt tokens or text plus raw
image sources); the crate fetches, decodes, resizes, normalizes, and
patchifies each image **bit-exactly** against the mirrored HF image processor,
expands the prompt's media placeholders, and computes position encodings
(e.g. M-RoPE) — all CPU-side, GIL-free, and without owning threads unless
asked to.

The counterpart concern — chat-template rendering down to media placeholder
markers — lives in `dynamo-renderer`; this crate is the "preprocessing
concern owned by the consumer" that the renderer's docs point at.

## Architecture

- **`pipeline`** — the model-family seam. Families implement
  `MmFamilyProcessor` (decoded media → named tensors; prompt geometry as a
  declarative `TokenLayout`; positions). Families produce data, the driver
  owns control flow.
- **`driver`** — the model-independent orchestrator: fetch → hash → decode →
  per-item preprocess (parallel) → layout application → positions, with
  per-request byte/item caps. Every `Err` is a request-rejection message;
  there is no fallback path.
- **`registry`** — family selection: a typed `PipelineSpec` or a JSON spec of
  resolved processor parameters (`{"family": "qwen_vl", ...}`).
- **`image`** — decode (8-bit only, matching PIL), bit-exact resize kernels
  (PIL fixed-point Lanczos/Bicubic and torchvision's uint8 antialias
  bicubic), and reusable transform primitives.
- **`token_layout`** — mechanical, validating expansion of the tokenized
  prompt (no retokenize, so non-media tokens can never drift).
- **`fetch`** (feature `fetch`) — `data:`/base64/`file://`/http(s) source
  resolution with Python-`requests`-compatible proxy/`NO_PROXY` semantics and
  streaming byte budgets.
- **`par`** (feature `parallel`) — the crate's only parallelism seam: a
  crate-owned rayon pool, or fully inline without the feature (the crate then
  owns no threads at all — guarded by an integration test).

## Features

| feature    | default | adds                                                        |
| ---------- | ------- | ----------------------------------------------------------- |
| `fetch`    | off     | string media source resolution (`ureq`, `base64`)           |
| `parallel` | off     | crate-owned rayon pool for intra-request fan-out            |

The crate reads no environment variables: pool sizing (`par::init_pool`) and
fetch timeouts (`fetch::FetchOptions`) are explicit configuration.

## Supported families

| family    | models                                        | output                                  |
| --------- | --------------------------------------------- | --------------------------------------- |
| `qwen_vl` | Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5 VL | `pixel_values`, `image_grid_thw`, M-RoPE |

## Adding a family

1. Implement `MmFamilyProcessor` in `src/<model>.rs`: `process_item` (the HF
   image-processor equivalent), `layout` (how the prompt expands around the
   items), and `positions` if the model has a custom scheme.
2. Add the family's arm to `registry::PipelineSpec`.

The carriers in `pipeline` are `#[non_exhaustive]` and grow with real needs,
validated against the GLM-4V and Kimi K2.5/K3 Python processors:
GLM's `<|begin_of_image|>`-framed spans fit `TokenPattern::Explicit`; Kimi
K3's tokenized-text framing will add a defaulted `layout_with(&LayoutContext)`
trait method (semver-minor) carrying an encode hook; video/audio grow
`DecodedMedia` variants and `Capabilities` flags.

## Bit-exactness

Correctness is defined as byte-identical output against the mirrored HF
processor, not approximate similarity — a systematic skew (wrong resample
filter, fused-vs-unfused normalize rounding, patch order) still reads as
fluent model output while silently costing accuracy. Golden fixtures under
`tests/fixtures/` pin this end to end; consumers (e.g. SGLang) additionally
run per-step and end-to-end byte-parity suites against the live HF
processors.
