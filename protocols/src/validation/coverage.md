# Validation and normalization coverage

Temporary tracker for the incremental implementation. Contracts and sources
live in the module docs under `src/validation/` and in `src/normalization.rs`.
Parsing and builders never call these operations.

## Request trait checks

Each trait runs its covered checks in this order and returns the first failure.
Every check accepts `None`. Success covers only these checks.

| Order / field | OpenAI | vLLM | SGLang |
| --- | --- | --- | --- |
| 1. temperature | [0, 2] | reuse OpenAI | finite, >= 0 |
| 2. top_p | [0, 1] | (0, 1] (shared) | (0, 1] (shared) |
| 3. frequency_penalty | [-2, 2] | reuse OpenAI | reuse OpenAI |
| 4. presence_penalty | [-2, 2] | reuse OpenAI | reuse OpenAI |
| 5. completion logprobs | 0..=5 | 0..=20 (fixed)* | uncovered |
| 6. n | 1..=128 | uncovered | uncovered |
| 7. chat top_logprobs / completion best_of | 0..=20 | uncovered | uncovered |

Row 5 applies only to completions. Row 7 checks the field that exists on the
request type.

\*vLLM itself checks a configurable `max_logprobs` and accepts -1. See
`vllm::validate_completion_logprobs`.

## Other operations

- `min_p` in [0, 1]: vLLM and SGLang, through the shared module. The covered
  request types have no `min_p` field, so the traits do not call it.
- OpenAI `validate_logit_bias_value`: one parsed `f32` value in [-100, 100].
  Whole-map validation is uncovered.
- Normalization: completion `logprobs` is preserved unchanged by all three namespaces.

## Gaps

- Unlisted fields: stop, token limits, required content, tools, stream options,
  guided decoding.
- Engine count rules and all cross-field rules, such as chat logprobs enablement
  and `best_of` relationships.
- Whole `logit_bias` maps. Engines differ here: vLLM
  [clamps values](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/sampling_params.py)
  and SGLang [uses vocabulary context](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/sampling/sampling_params.py).
- Other defaults, greedy transformations, sentinel rewrites, and any
  runtime-dependent admission.

A shared numeric range does not imply full-operation equivalence. For example,
OpenAI specifies `best_of > n`, while Dynamo keeps its own `best_of >= n` rule.
