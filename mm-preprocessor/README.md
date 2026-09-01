<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-mm-preprocessor

Model-family multimodal preprocessing for LLM inference serving — a pure-Rust
replacement for the image pipelines behind HF `AutoProcessor`: per-family
decode → resize → normalize → patchify, prompt placeholder expansion, and
model-family position math (M-RoPE), all **bit-exact** against the mirrored
HF processor.

Scope mirrors the Python layering: this crate is the *processor* (what HF
ships); request orchestration — source fetching, content hashing, caps,
failure policy — stays in the serving engine's driver, which composes this
crate through the `MmFamilyProcessor` trait. The adjacent concern of
chat-template rendering down to media placeholder markers lives in
`dynamo-renderer`.

**Status: skeleton (design review).** Signatures and module layout are final;
bodies land with the follow-up implementation and the crate flips to publishable
then. Start with [`DESIGN.md`](DESIGN.md): the boundary, key APIs, the
Python-parity map, an engine-side (SGLang) driver sketch, testing strategy,
and the roadmap.

| feature    | default | adds                                              |
| ---------- | ------- | -------------------------------------------------- |
| `parallel` | **on**  | links rayon; kernels still run inline until `execution::init_pool` arms the crate-owned pool (opt out to drop rayon entirely) |

Supported families: `models::qwen_vl` (Qwen2-VL / Qwen2.5-VL / Qwen3-VL /
Qwen3.5 VL) — `pixel_values`, `image_grid_thw`, image-only M-RoPE. Adding a
family = one module in `src/models/` + one `registry::ProcessorSpec` arm; see
DESIGN.md §6 for the GLM-4V / Kimi growth plan.
