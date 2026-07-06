<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reasoning parsers

Reasoning parsers extract a model's reasoning ("thinking") span and keep it out of the user-visible `content`. They live in `parsers/v1/src/reasoning/`: `base_parser.rs` (`BasicReasoningParser`, the explicit `<think>...</think>` grammar) plus family parsers `granite_parser.rs`, `gpt_oss_parser.rs`, `gemma4_parser.rs`, and `minimax_append_think_parser.rs`. Most families reuse `BasicReasoningParser`; only add a new parser when the grammar actually diverges.

## Goals

Reasoning parsers share the same goals as the tool-call parsers — the full list is in [`../../../v2/README.md`](../../../v2/README.md) under "Parser goals". The ones that bind reasoning most directly:

- **Follow the model's own spec, and record the spec source in the fixture YAML.** The chat template / model card defines the reasoning delimiter (`<think>`, `◁think▷`, `[THINK]`, Harmony channels, ...); parse to that, not to another engine's parser.
- **Never leak reasoning markup into user-visible `content`.** The reasoning span and its delimiters go to the reasoning channel; text outside the span is normal content. Markup must never surface as content.
- **Never leak tool-call markup either** — when a model interleaves reasoning and tool calls, each parser strips only its own markup and leaves the rest intact for the next stage.
- **Make a reasonable, bounded attempt to recover from imperfect / truncated output.** Generation may start already inside a reasoning span (force-reasoning families), or a stream may be cut before the closing delimiter. Recover the span where the grammar allows (e.g. treat marker-free leading text as reasoning for force-reasoning families, or close an open span at end-of-stream) without inventing content, and `tracing::warn!` with a stable `why=` on recovery.
- **Preserve as much of the original output as possible** — only the recognized reasoning markup spans are removed from `content`; surrounding text is kept verbatim.
- **A batch (v1) and a streaming (v2) parse of the same output must always agree** — once a v2 reasoning path exists it differs from v1 only in efficiency (v1 jails/buffers the whole output, v2 is token-incremental), never in result. Divergence from vLLM/SGLang on under-specified recovery, by contrast, is expected and documented with a `reason:`, not "fixed" by matching a peer.

## Case taxonomy

The reasoning corner-case taxonomy — delimiter grammars, family groups, and batch vs stream modes used by the unit tests — is in [`../../../conformance/utils/lib/parsers/REASONING_CASES.md`](../../../conformance/utils/lib/parsers/REASONING_CASES.md). Group families by shared grammar before adding cases, so the same delimiter cases are not copied across many model rows.

## Status

Reasoning is still on the v1 path; there is no `parsers/v2` reasoning crate yet. The v2 streaming-fixture migration for reasoning is tracked as a follow-up in [`../../../v2/README.md`](../../../v2/README.md) under "Reasoning Migration TODO".
