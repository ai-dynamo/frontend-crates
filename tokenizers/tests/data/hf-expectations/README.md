<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frozen Hugging Face expectations

Each `<case>/expected.json` records what the Hugging Face backend returned while it ran on
`tokenizers` 0.23.2, for one `tokenizer.json`. `tests/hf_expectations.rs` replays the recorded
inputs through the current backend and compares every output.

The files were written once, by a generator that needed the 0.23.2 crate and was removed with
it. `provenance` in each file names the `dynamo-tokenizers` version, the `tokenizers` version
and the commit it ran at. The generator is in the history of the pull request that moved the
backend to `tokenizers` 1.x.

## What a case holds

| Key | Contents |
|-----|----------|
| `model_dir` | The directory holding `tokenizer.json`, relative to `tests/data`. A case under `hf-expectations/` carries its own; the others point at `minimal-bpe` or `sample-models/` |
| `texts` | For each text: ids, ids with `add_special_tokens`, decoded text and streamed chunks with `skip_special_tokens` off and on, and the same through `FastTokenizer` where it loads the file |
| `prefixes` | Every id prefix of a text that splits multi-byte characters: decoded text, whether it is partial or unstable, and the stream from each prompt length |
| `segments` | Segmented encodes that mix trusted and ordinary segments, with the error where one is refused |
| `sequences` | Decodes and streams of fixed id sequences, including byte-fallback runs |
| `added_tokens`, `token_to_id` | Each added token's id, encode and decode, and vocabulary lookups |
| the remaining keys | `vocab_size`, `token_id_bound`, `special_token_ids`, `special_tokens`, `num_special_tokens_added`, and whether the tokenizer may be prefix-cached |

Truncation and padding were removed before recording, as the serving path did. A case with
`load_error` is a file the 0.23.2 backend could not load.

## Rules

- Do not edit an `expected.json` to make a test pass. If the backend should differ from the
  recorded output, name the case and field in `tests/hf_expectations.rs` and say why.
- A configuration the 1.x pipeline cannot serve unchanged is listed in `REJECTED` there, with
  the error it must produce at load. Its recorded outputs stay, for the day it loads again.
- A new case cannot be recorded from 0.23.2 any more. Write its expected values by hand, or
  from a reference implementation, and say which in the case's directory.
