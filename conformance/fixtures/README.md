<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conformance fixture store (git-lfs)

Shard tarballs, tracked via git-lfs. `conformance/fixtures-manifest.json` pins the active snapshot (sha256 per shard); `extract_fixtures.py` unpacks everything into `~/.cache/dynamo/conformance-fixtures/`. Do not edit these by hand — re-capture and run `package_fixtures.py` (see [`../README.md`](../README.md#fixture-workflows)).

Naming: `<tree>/<impl>-<version>.tar.gz`, where `<version>` is the engine/crate version that produced the outputs. Per impl, the LOWEST version is a full capture (anchor); higher versions are changed-only overlays. `inputs.tar.gz` is the shared test input. The tree suffix names the conformance convention, not the parser implementation: streaming v1 is legacy/non-Unified, while Unified v2 is the ordered reasoning+text+tool protocol. Therefore `dynamo_v2-*` implementation captures legitimately appear inside `fixtures-stream-v1`.

**Runtime context:** Dynamo selects parser names from deployment configuration. `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` enables both experimental routes; routing still checks Unified first. The exact Qwen3 tool+reasoning parser pair uses Unified when the flag is on. Without a matching Unified pair, eligible Qwen3-Coder or DeepSeek V4 requests use the non-Unified incremental parser. Other tool-parsing requests use batch+jail; DeepSeek V4.1 and Muse have separate Unified routes that do not depend on the flag.

| Shard | Implementation | What it holds |
|---|---|---|
| `toolcalling/fixtures-batch-v1/inputs.tar.gz` | — | shared `model_text` + tools for the batch corpus |
| `toolcalling/fixtures-batch-v1/dynamo_v1-3.0.0.tar.gz` | **Dynamo v1** batch (`dynamo-parsers` 3.0.0) | `expected.dynamo_v1` batch expectations |
| `toolcalling/fixtures-batch-v1/vllm_python-{0.23.0,0.24.0}.tar.gz` | vLLM Python batch | `expected.vllm_python` |
| `toolcalling/fixtures-batch-v1/sglang_python-{0.5.12.post1,0.5.14}.tar.gz` | SGLang Python batch | `expected.sglang_python` |
| `toolcalling/fixtures-stream-v1/inputs.tar.gz` | — | shared per-chunk `delta_text` for the stream corpus |
| `toolcalling/fixtures-stream-v1/dynamo_v2-0.1.11.tar.gz` | **Dynamo v2** stream (`dynamo-parsers-v2` 0.1.11) | `expected.dynamo_v2` per-chunk expectations — the v2 anchor the parity test folds |
| `toolcalling/fixtures-stream-v1/dynamo_v1-3.0.0.tar.gz` | **Dynamo v1 JAIL** (`dynamo-parsers` 3.0.0) | v1 jail+batch stream reference for the chart — its own impl namespace, cleanly separate from v2 |
| `toolcalling/fixtures-stream-v1/vllm_python-{0.23.0,0.24.0}.tar.gz` | vLLM Python stream | `expected.vllm_python` |
| `toolcalling/fixtures-stream-v1/vllm_rust-0.23.0.tar.gz` | vLLM Rust stream | `expected.vllm_rust` |
| `toolcalling/fixtures-stream-v1/sglang_python-{0.5.12.post1,0.5.14}.tar.gz` | SGLang Python stream | `expected.sglang_python` |
| `toolcalling/fixtures-batch-on-stream-v1.tar.gz` | all impls, one tree | complete batch text fed through STREAMING parsers; versions live in each fixture's `captured_with` |
| `reasoning/fixtures-v1/inputs.tar.gz` | — | reasoning inputs (v1-era anchor, `captured_with` stamps inside) |
| `reasoning/fixtures-v1/vllm_python-0.24.0.tar.gz` | vLLM Python reasoning | changed-only overlay |
| `reasoning/fixtures-v1/sglang_python-0.5.14.tar.gz` | SGLang Python reasoning | changed-only overlay |

Impl keys are fully explicit and uniform: `dynamo_v1`, `dynamo_v2`, `vllm_python`, `vllm_rust`, `sglang_python` — one namespace per implementation, one version lineage per namespace. Dynamo v1 and v2 never share a key. Legacy spellings (`dynamo`, `dynamo_rust`, `vllm`, `sglang`) remain readable via the alias table in `../utils/src/impls.py`.
