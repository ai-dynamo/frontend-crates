<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Attributions

This file lists third-party open-source code that has been incorporated into the crates in this repository, along with the upstream copyright and license terms required for redistribution.

The attributions below are mandatory under the licenses of the upstream projects. They are reproduced here in addition to the inline attribution notices in the affected source files (see e.g. [`protocols/src/lib.rs`](protocols/src/lib.rs) and [`protocols/Cargo.toml`](protocols/Cargo.toml)).

For all *transitive* third-party crates pulled in as `Cargo.toml` `[dependencies]` but not vendored into this repository, license metadata is published as part of each crate's release metadata on crates.io and the standard Cargo-ecosystem tools (`cargo about`, `cargo deny`, `cargo license`) can enumerate them locally. Only code physically present in this tree appears below.

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
