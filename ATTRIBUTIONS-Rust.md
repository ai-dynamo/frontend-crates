<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Attributions

This file lists third-party open-source code that has been incorporated into the crates in this repository, along with the upstream copyright and license terms required for redistribution.

The attributions below are mandatory under the licenses of the upstream projects. They are reproduced here in addition to the inline attribution notices in the affected source files (see e.g. [`protocols/src/lib.rs`](protocols/src/lib.rs) and [`protocols/Cargo.toml`](protocols/Cargo.toml)).

For all *transitive* third-party crates pulled in as `Cargo.toml` `[dependencies]` but not vendored into this repository, license metadata is published as part of each crate's release metadata on crates.io and the standard Cargo-ecosystem tools (`cargo about`, `cargo deny`, `cargo license`) can enumerate them locally. Only code physically present in this tree appears below.

---

## `dynamo-multimodal` — GLM image processing adapted from SGLang

`multimodal/src/models/glm5_next/image.rs`,
`multimodal/src/image/decode/smart.rs`, and the GLM pixel parity tests and
fixtures are adapted from [SGLang PR #46563](https://github.com/sgl-project/sglang/pull/46563)
at commit `916577f5450a0475807c3a07b0b5b8377708935c`, authored by Kan Wu
and Zhihan Guo. The image path mirrors transformers' `Glm5NextImageProcessor`
and SGLang's `smart_to_rgb`. SGLang and transformers are distributed under
Apache License, Version 2.0, reproduced in this repository's `LICENSE`.

---

## `dynamo-multimodal` — the parity image corpus adapted from SGLang

The image recipes of `multimodal/tests/fixtures/mm_parity/corpus.py` for the
files that SGLang's corpus has at the commits below are adapted from SGLang's
`rust/sglang-processor/tests/scripts/generate_mm_parity.py`: the format corpus
at commit `a2ffd214696ea7144d3c237133a15df3315d900b`, and the transparent GIF,
the PNG with an EXIF text profile and the BMP with an alpha bitmask at
[SGLang PR #46563](https://github.com/sgl-project/sglang/pull/46563) commit
`916577f5450a0475807c3a07b0b5b8377708935c`. SGLang is distributed under
Apache License, Version 2.0, reproduced in this repository's `LICENSE`.

---

## `dynamo-multimodal` — DSV4.1 geometry adapted from SGLang

The geometry math in `multimodal/src/models/deepseek_v41.rs` is adapted from
SGLang's `python/sglang/srt/multimodal/deepseek_v41_image_processing.py` at
commit `ffac53d779c08dcdab2d07e5e2a41dba83f0e65c`.

- **Project**: [SGLang](https://github.com/sgl-project/sglang)
- **Source**: [Pinned image processor](https://github.com/sgl-project/sglang/blob/ffac53d779c08dcdab2d07e5e2a41dba83f0e65c/python/sglang/srt/multimodal/deepseek_v41_image_processing.py)
- **License**: [Apache License, Version 2.0](https://github.com/sgl-project/sglang/blob/ffac53d779c08dcdab2d07e5e2a41dba83f0e65c/LICENSE), also reproduced in this repository's `LICENSE`.

The Rust adaptation adds configuration and input validation, checked output
dimensions, typed geometry results, and Python-derived parity fixtures.

---

## `dynamo-protocols` — based on `async-openai`

The `dynamo-protocols` crate is a derivative work of [`async-openai`](https://github.com/64bit/async-openai) by Himanshu Neema. The OpenAI request/response type hierarchy in this crate originated from that project and has been extended and modified for inference-serving use cases.

### Upstream Project

- **Project**: `async-openai`
- **Source**: https://github.com/64bit/async-openai
- **Version originated from**: 0.34
- **Original Copyright**: Copyright (c) 2022 Himanshu Neema
- **License**: MIT

### Upstream License Text

```
MIT License

Copyright (c) 2022 Himanshu Neema

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### NVIDIA Modifications

NVIDIA's modifications are licensed under the Apache License, Version 2.0. They include, but are not limited to:

- Inference-serving extensions (`nvext` request/response fields, agent-context metadata, streaming control hooks).
- Anthropic Messages API request/response types.
- Additional impls for serialization, validation, and error handling tailored to multi-backend inference serving.

The resulting `dynamo-protocols` crate is therefore distributed under the dual license **Apache-2.0 AND MIT** — Apache-2.0 covers the NVIDIA-authored modifications, and MIT covers the portions derived from `async-openai`.

---

## Other Crates

`dynamo-tokenizers` and `dynamo-parsers` are NVIDIA-authored and contain no vendored third-party source. All third-party code consumed by these crates is loaded at build time as standard Cargo dependencies and is governed solely by those upstream crates' own licenses (recorded in each crate's `Cargo.toml` metadata).
