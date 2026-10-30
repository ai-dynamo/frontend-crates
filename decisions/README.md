<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# dynamo-decisions

Runtime-independent text decision evaluation contracts. The crate accepts OpenAI `/v1/decisions` and Jev `/v1/systemone` requests only. Every `nvext` envelope is rejected on both routes; native SGLang wire schemas and format selection are not supported. It preserves typed choice values, ordered questions, structured evidence, model aliases, and original score criteria.

`parse_request` performs duplicate-safe JSON decoding and dialect validation. `CanonicalRequest::validate_capabilities` checks the selected deployment's semantic limits. `render_question_prompt` creates prompt-format 1 semantic text; model chat templating, answer-position token verification, revision binding, and inference remain executor responsibilities. `reduce_vocab_logprobs` accepts complete full-vocabulary-normalized candidate log probabilities, not head logits. `project_response` validates measured usage and projects ordered outcomes to the selected wire contract.

The transport owns authentication, body limits, explicit model/alias lookup, request-ID headers, admission, deadlines, cancellation, routing, metrics, and cache isolation. The crate does not depend on an HTTP framework, asynchronous runtime, or model engine. Unknown model IDs must not silently select the only deployed model. Outcomes must be paired with their original ordinals before response projection.

The initial profile supports text only, at most 128 questions per request by default, non-reasoning execution, and prompt format 1. Custom Jev template controls beyond explicit `thinking=false`/`enable_thinking=false` are rejected. Generic causal execution must prove complete single-operation candidate scoring. A full model qualification is separate from these contract tests.

OpenAI wire types follow `openai-python` revision `4e152cdefe1844c2d5d78653310e9b9c0195c44e`. Internal scoring prompt wording follows SGLang revision `0b635266d4a09f8db2d12bdcb793b085199faa8f`. Jev types follow published System One OpenAPI 0.2.0, with initial profile limits of 1–255 choices and 1–10 score levels. OpenAI/Jev choice and score confidence use the documented Jev reducers; this is not a claim of equivalent hosted-model calibration. OpenAI `cache_write_tokens=0` means not separately accounted, not zero physical writes. Missing cache-read measurements never become fabricated zeros.

Run `cargo test -p dynamo-decisions` for reserved-extension rejection, schema, numerical, projection, and rendering fixtures. No GPU is required.
