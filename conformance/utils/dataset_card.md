---
license: apache-2.0
pretty_name: Dynamo Parser Conformance Fixtures
task_categories:
  - other
tags:
  - tool-calling
  - parser-conformance
  - reasoning-parsing
  - nvidia-dynamo
  - test-fixtures
language:
  - en
size_categories:
  - n<1K
configs: []
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Parser Conformance Fixtures

## Dataset Description:

Deterministic input / expected-output **test fixtures** for the [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) tool-calling and reasoning text parsers. Each fixture is a small YAML test case that pairs a piece of model-output text (or a sequence of streaming chunks) with the structured output that each parser implementation is expected to produce, so CI can catch parser regressions and cross-engine divergences (leaked delimiters, dropped/duplicated calls, truncated arguments).

Despite HuggingFace's "dataset" namespace, this is a **software test-fixture corpus, NOT a training dataset**: it contains no machine learning content — no model training, no fine-tuning, no model evaluation — and nothing here is, or is used to produce, model weights. It is unit-test material that happens to be hosted on the Hub.

This dataset is ready for commercial or non-commercial uses.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
{{CREATED_PT}} (snapshot `{{STAMP}}`)

## Version:
`{{STAMP}}` — peers: {{PEERS}}; Dynamo parser crates {{CRATES}}. <br>

Previous Version(s): earlier dated snapshots were used internally; no prior public versions are published.

## License/Terms of Use:
Apache-2.0 (`SPDX-License-Identifier: Apache-2.0`), matching the NVIDIA Dynamo project. Some cases are adapted from the vLLM and SGLang test suites (both Apache-2.0) and redistributed under Apache-2.0 with modifications; see [Reference(s)](#references).

## Intended Usage:
For NVIDIA Dynamo developers and contributors to run **parser conformance and regression tests** in CI, and to track cross-engine (Dynamo / vLLM / SGLang) parser divergences. Not intended for training or fine-tuning models, for measuring model quality, or as a general tool-calling benchmark.

## Dataset Characterization
** Data Collection Method<br>
* Hybrid: Manually Collected, Synthetic, Automated <br>
Hand-authored cases; fuzzer/randomizer-generated cases (a chunking-invariance harness); and expected outputs captured automatically by running the parser implementations. Some cases are adapted from the open-source vLLM and SGLang test suites (attributed per fixture). No data is scraped from users; no real user conversations are included.

** Labeling Method (here: expected-output authoring)<br>
* Hybrid: Manually-Labelled, Automated <br>
The "label" for each case is its expected parser output — the structured result a parser implementation must produce for a given input. Labels are produced either by hand-authoring the intended result or by capturing the actual parser output at pinned engine versions.

## Dataset Format
Text. YAML (`.yaml`) test-fixture files, distributed as gzip tarballs (`.tar.gz`) pinned by an in-repo SHA-256 manifest.

## Dataset Quantification
~504 authored fixture files / ~1,967 test cases, plus per-engine-version expected-output overlays. Distributed as versioned `.tar.gz` shards plus a single monolith snapshot. A few megabytes total (well under 100 MB).

Feature Count: N/A — test fixtures, not an ML dataset.

## Reference(s):
* NVIDIA Dynamo — https://github.com/ai-dynamo/dynamo (the `conformance/` fixtures and `parsers/` crates)
* vLLM (Apache-2.0) — https://github.com/vllm-project/vllm
* SGLang (Apache-2.0) — https://github.com/sgl-project/sglang

Upstream-adapted cases carry a per-fixture `ref`/`upstream` pointer to their source file.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
