---
name: mm-preprocessor-generator
description: Add an image preprocessor for a model family to dynamo-multimodal, gated bit for bit on a parity table whose baseline is the checkpoint's own Hugging Face processor, and compared with SGLang, vLLM and llm-multimodal. Use when adding a family to `multimodal/src/models/` or extending one with a new image path.
license: "Apache-2.0"
---

# Multimodal Preprocessor Generator Skill

Add a family's image path to `dynamo-multimodal` (`multimodal/`). The table
schema, the cell contract and every command are in
[mm_parity/README.md](../../../multimodal/tests/fixtures/mm_parity/README.md).
This file is the checklist.

## 1. Choose the baseline and the target

The baseline is always the checkpoint's own processor in transformers
(`AutoProcessor` with the backend that transformers picks by default, fed by
`transformers.image_utils.load_image`). It writes each case's `expected`. For a
remote-code processor that `frameworks/hf.py` cannot call, the family gives
`ADAPTERS["hf"]`. `load_image` applies the EXIF orientation: a family that
decodes with `decode_rgb` keeps the stored orientation, so declare its EXIF
cases as known gaps.

The target is the column that the crate must equal. Use `hf`. Use the serving
engine's column (for example `sglang`) only when that engine serves other
values for the model, and give the reason in `TARGET_REASON`, which
`PARITY.md` shows. GLM-5.3-Flash is the example: SGLang serves M-RoPE positions and
puts alpha over a background chosen from the image edges; HF counts 1-D
positions and drops alpha.

Pause for human review of the case list and the target before you implement.

## 2. Write the cases

Add `families/<family>.py` with `FILE`, `TARGET`, `ID_MAP`, `CRATE`,
`cases(model_dir)` and `request(case)`. `request(case)` also passes the engine
settings that change the output, such as `max_model_len` (vLLM
`--max-model-len`, SGLang `--context-length`); an adapter applies them for its
column. Add these only when the family needs them:

- `TARGET_REASON`: why the target is not `hf`.
- `FIELDS`: the recorded fields, when they are not the default (for example
  `item_offsets`, `image_sizes` or `feature_dtypes`).
- `spec_shared(model_dir)`: spec fields of every case, such as a token table.
- `ADAPTERS`: `{column: run}` for a column whose module cannot run the
  checkpoint's processor (remote code, an engine class of its own). Like the
  modules, an adapter records a refusal only from the framework's own call.
- `stages(case, images, expected)`: values the family's own test checks per
  case.
- `files(model_dir)`: `{path: text}`, the data files of the family's own tests
  (sizes-only requests, sweeps). The driver writes and checks them with the
  table.

Cover each class that applies to the model, and say in the PR why a class
does not:

| Class | Cases |
| --- | --- |
| Formats | PNG, baseline and progressive JPEG, 4:4:4 JPEG, CMYK, gray, 16-bit, BMP, GIF, lossy and lossless WebP |
| Alpha | RGBA, palette `transparency`, alpha WebP, transparent GIF, bitfields BMP |
| EXIF | an orientation-6 JPEG, and an EXIF chunk in a PNG |
| Sizes | below the minimum, an odd size, above the maximum, an extreme aspect ratio |
| Multi-image | two images in one prompt |
| Request budgets | every per-request override the model takes (token or pixel bounds) |
| Rejections | each input the baseline or the target refuses, with its `rust_error` |
| Backends | each image-processor backend the serving engines select |

Corpus rule: take images from `images/` by name. Add an image only for a new
test idea, with its recipe in `corpus.py`, which also writes its `decode.json`
row. A new decode entry point is a new `decode.json` column, with Pillow's
output for every image. The decode forms at the shipped limits are
`decode.json` rows, not table cases. Inputs with no file (sizes only, pixels
built from a seed) go in the family's own test, with data that `files()`
writes. When the family has a second checkpoint, show in a family check that
it gives the same spec and cases, and say so in the README row.

## 3. Implement

Add the family under `multimodal/src/models/` and its spec in `registry.rs`.
The spec resolver reads the same checkpoint files as the baseline. Reuse the
crate's decode, resize and layout stages; add a stage only for a behavior that
the baseline shows. The replay calls `process_items` for the whole request; a
family that sizes images against the request overrides it and refuses
`process_item`. Features are `TensorData::F32` or `Bf16` (hashed as `<u2`
bits). Check values that `ProcessedItem` does not carry in `tests/<family>_*.rs`, with the helpers in
`tests/common/`.

## 4. Capture, gate and report

1. Run each new adapter on one case first. Then `generate.py --family
   <family> --model-dir <checkpoint>` with `--vllm-python` and
   `--llm-mm-harness` (README.md, "Regenerate", says how to get both). A
   framework that cannot run a case gets `n/a` with the reason. For a batch
   padded to `[N, 3, H, W]`, each item's feature is its `[1, 3, h, w]` crop,
   with `image_sizes` and `item_offsets` in `FIELDS`.
2. `generate.py --family <family> --crate`, after the crate passes.
3. `report.py`, then `report.py --check` and both `generate.py` commands
   with `--check`.
4. `cargo test -p dynamo-multimodal --test mm_parity`. The replay reads the
   new table without a Rust change.
5. Add the table's row to the table at the top of README.md.

Declare `known_gap` and `rust_error` in `CRATE` (README.md, "The cell
contract"); never hand-edit `expected`, a framework cell or `PARITY.md`.

## 5. The PR description

Show the family's summary lines from `PARITY.md` (one per framework, "x of y
cases equal to HF"), the target and its reason, each known gap, and each
difference between frameworks that the tables found.
