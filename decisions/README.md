<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# dynamo-decisions

Runtime-independent canonical types and numerical reduction for text decision evaluation. Ordered questions preserve typed choices, structured evidence, original score criteria, and model aliases. `CanonicalRequest::validate_capabilities` checks the selected deployment's semantic limits before execution.

`render_question_prompt` creates prompt-format 1 semantic text. Model chat templating, answer-position token verification, revision binding, and inference remain executor responsibilities. `reduce_vocab_logprobs` accepts complete full-vocabulary-normalized candidate log probabilities, not head logits. Temperature changes conditional probabilities without changing measured vocabulary label mass. Missing cache measurements remain unknown.

`parse_request` performs duplicate-safe JSON decoding and dialect validation. `/v1/decisions` defaults to OpenAI; `nvext.format=sglang_native` selects the native contract. `/v1/systemone` uses Jev without a selector. Unsupported mixed schemas, modalities, and engine controls are rejected before execution. The default text profile allows at most 128 questions and explicitly disabled thinking only.

The transport owns authentication, body limits, model/alias lookup, request-ID headers, admission, deadlines, cancellation, routing, metrics, and cache isolation. The crate does not depend on an HTTP framework, asynchronous runtime, or model engine. Response projection builds on this core in subsequent work.

Run `cargo test -p dynamo-decisions` for canonical capability, numerical, request validation, rendering, and error fixtures. No GPU is required.
