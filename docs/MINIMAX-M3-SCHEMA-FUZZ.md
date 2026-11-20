<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax M3 schema-fuzz audit

This opt-in CPU audit checks whether MiniMax M3 Unified v2 recovers the exact tool arguments supplied by OpenRouter's pinned `toolcall-schema-fuzz` dataset. It runs no model, renderer, guided-decoding grammar, or peer parser. It does not change parser behavior or the published Unified conformance cases. The vendored dataset's [manifest and attribution](../conformance/utils/data/toolcall-schema-fuzz/README.md) identify the original source and CC-BY-4.0 license.

## Run

Use Python 3.10 or newer, the repository's Rust toolchain, and a clean parser/harness/build-input tree. Install the audit dependency in a virtual environment:

```sh
python3 -m venv .venv-schema-fuzz
.venv-schema-fuzz/bin/python -m pip install -r conformance/utils/schema-fuzz-requirements.in
.venv-schema-fuzz/bin/python conformance/utils/src/audit_toolcall_schema_fuzz.py \
  --output logs/schema-fuzz/baseline
```

The output directory must be new and beneath this checkout's `logs/`. Cargo builds the existing `capture_cross_version` test executable with `--locked`; after dependencies are cached, corpus execution requires no network access. The script runs its `capture_family_jobs` entry point with the native `minimax_m3` parser starting in the response channel. Every request has a finish step.

Use repeated `--case` arguments for a smoke subset; each is an exact original dataset key:

```sh
.venv-schema-fuzz/bin/python conformance/utils/src/audit_toolcall_schema_fuzz.py \
  --output logs/schema-fuzz/smoke \
  --case integer --case 'const<integer>' --case 'root-anyOf/string'
```

Exit status is **0** for an exact pass, **1** when the completed audit contains failures, and **2** for setup or completeness errors. An empty or partial run cannot pass. Existing parser failures are evidence to investigate; they are not expected-failure exemptions. Keep the audit opt-in until its baseline is understood.

## Coverage and oracle

The snapshot contains 1,358 schemas and two independently authored argument objects per schema. Preflight verifies its SHA-256, version, counts, unique keys, schema validity, and all 2,716 expected calls using `jsonschema==4.26.0` with Draft 2020-12 semantics. Decimal JSON numbers stay exact, including during validation. Schemas are passed unchanged; no root-combinator flattening, type backfill, or property renaming occurs.

Every argument object uses nested M3 tags. Strings are literal text, null is `null`, booleans are lowercase, objects have nested property tags, and arrays have `<item>` tags. This is a test encoder, not a reconstruction of checkpoint history rendering; it deliberately preserves null values rather than omitting them. For the 1,488 argument objects with structured parameter values, a second encoding puts compact JSON text inside parameter tags. These representations are reported separately because the latter can bypass nested schema typing.

Each encoding runs as a whole input, Unicode character chunks, complete M3 marker chunks, and fixed-seed variable chunks. This gives **16,816 runs**: 10,864 nested and 5,952 JSON-value runs. Seeds derive from stable case IDs, never Python's randomized hash. IDs append argument index, encoding, and delivery schedule to the unchanged source key.

The scorer requires one completed call with the correct name, valid JSON arguments, and no unexpected events. Arguments must equal the source expectation recursively: object order and whitespace do not matter; array order, string contents, booleans, nulls, and exact numeric values do. A string `"30"` cannot equal number `30`; mathematically equal JSON numbers such as `30` and `30.0` compare equally. Duplicate JSON keys and non-JSON numbers are rejected.

Schema validity is an additional diagnostic. A schema-valid value still fails if it differs from the original expectation. A mismatch is labeled `demonstrated_wire_ambiguity` only when the returned arguments independently validate and re-encode to the identical input bytes. Empty containers, untyped values, and string/number unions can lose information in M3 markup. Names that cannot survive tag framing are labeled `unrepresentable_parameter_name`; names and expectations remain unchanged, and these failures remain in the denominator.

The dataset provides schema coverage. Existing parser tests remain responsible for broader marker-in-string, whitespace, malformed-output, and reasoning behavior.

## Evidence

`provenance.json` records the tested Git revision, Cargo.lock digest, dataset manifest, audit-code digest, validator version, selected keys, expected call count, and expected execution count. The parser, capture harness, and build inputs must be unchanged so that the recorded revision identifies the tested implementation.

`results.jsonl` contains every execution, including its original key, tool schema, encoded input, exact chunk schedule, original arguments, actual arguments, failures, and differing JSON Pointer paths. Each row links its retained batch capture. Batch job/capture JSON and process logs preserve the request bindings and per-chunk deltas. A final empty capture row is the finish step, not another input chunk.

`summary.json` reports pass/fail totals by encoding, schedule, and schema keyword, plus failure kinds, diagnostic labels, and cases whose outcome changes across schedules. Keyword counts overlap. Infrastructure errors are failures, never successful empty outputs. All batches are attempted before returning failure. Repeat into a new directory to compare deterministic `results.jsonl` and `summary.json` bytes.

## Test the audit itself

Install pytest in the audit environment and run:

```sh
.venv-schema-fuzz/bin/python -m pip install pytest
.venv-schema-fuzz/bin/python -m pytest conformance/utils/tests/test_audit_toolcall_schema_fuzz.py -q
```

These tests plant changed values, wrong types, missing/extra calls, malformed JSON, and missing completion/finish evidence. They also check lossless numbers, corpus integrity controls, encodings, deterministic schedules, and ambiguity scoring. The module skips when the optional `jsonschema` dependency is absent; the full corpus is never run automatically during ordinary pytest collection.
