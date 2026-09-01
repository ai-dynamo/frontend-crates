<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-mm-preprocessor

Model-family multimodal preprocessing for LLM inference serving — a pure-Rust
replacement for the image pipelines behind HF `AutoProcessor`: fetch → decode
→ resize → normalize → patchify, plus prompt placeholder expansion and
position encodings (M-RoPE), all **bit-exact** against the mirrored HF
processor.

The counterpart concern — chat-template rendering down to media placeholder
markers — lives in `dynamo-renderer`; this crate is the "preprocessing concern
owned by the consumer" that the renderer's docs point at.

**Status: skeleton (design review).** Signatures and module layout are final;
bodies land with the implementation PRs and the crate flips to publishable
then. Start with [`DESIGN.md`](DESIGN.md): architecture, key APIs, the
Python-parity map, how a serving engine (SGLang) uses it, testing strategy,
and the roadmap.

| feature    | default | adds                                              |
| ---------- | ------- | -------------------------------------------------- |
| `fetch`    | off     | string media source resolution (data:/base64/file/http) |
| `parallel` | off     | crate-owned rayon pool for intra-request fan-out (off = inline, zero threads owned) |

Supported families: `models::qwen_vl` (Qwen2-VL / Qwen2.5-VL / Qwen3-VL /
Qwen3.5 VL) — `pixel_values`, `image_grid_thw`, image-only M-RoPE. Adding a
family = one module in `src/models/` + one `registry::PipelineSpec` arm; see
DESIGN.md §6 for the GLM-4V / Kimi growth plan.
