# Multimodal parity tables

Each `<model>.json` records, for a set of cases, what the checkpoint's own
Hugging Face processor returns (the baseline) and how each other framework
compares with it. `tests/mm_parity.rs` replays every table through the
public API (`processor_from_spec`, `decode_image`, `process_items`, `layout`,
`apply_layout`, `positions`) and compares the crate with the table's target,
bit for bit. `generate.py` is the one driver that writes the tables; it reads
images only by name from `images/`. `report.py` renders `PARITY.md`, the
side-by-side view of every table.

| Table | Baseline | Target | Provenance |
|---|---|---|---|
| `glm-5.3-flash.json` | `Glm5NextProcessor` of `zai-org/GLM-5.3-Flash` | `sglang` | The target values of every case are byte-identical to `rust/sglang-processor/tests/fixtures/mm_parity/glm-5.3-flash.json` of SGLang PR #46563 at `916577f5`. `generate.py` reproduces them at SGLang `a2ffd21469`. |
| `nemotron-3.5-super-vl.json` | `NemotronH_Omni_Reasoning_V3Processor`, the remote code of `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` (`trust_remote_code`), with the image processor's `max_model_len` set to the case's `context_length` | `sglang` | The target values of the 26 cases on the former inputs of `tests/fixtures/nemotron_vl/` (SGLang `a2ffd21469`) are byte-identical to that fixture. The other 7 cases use corpus images in place of generated ones, so their values are regenerated. `families/nemotron_vl.py` also writes `tests/fixtures/nemotron_vl/sizes.json`, the sizes-only SGLang requests and sweeps of `tests/nemotron_vl_planning.rs`. |
| `qwen2-vl.json` | `Qwen2VLProcessor` of `Qwen/Qwen2-VL-2B-Instruct`, with each case's pixel bounds and its ids mapped (`ID_MAP`) | `hf` | Each digest is the SHA-256 of the former `tests/fixtures/qwen_vl/` golden file (transformers 5.12.1). `aten_upscale` uses `rgb_40x40.png` from `aten_multi`, so its pixel values are regenerated. |

## The image corpus

`images/` is the one image corpus of the crate's parity tests. Each file
tests one idea: a format, a mode, a transparency form, an EXIF form, or a
form whose decode is known to differ from Pillow. A file that SGLang's corpus
(`rust/sglang-processor/tests/fixtures/mm_parity/images/`) also has keeps
SGLang's name and bytes. Other names are `<idea>_<w>x<h>.<ext>`.

`corpus.py` builds every image except the files in its `COMMITTED` table.
The five inputs of the former Qwen-VL goldens (`rgb_300x200.png`,
`rgb_40x40.png`, `rgb_56x56.png`, `rgb_100x76.png`, `tall_10x1100.png`) have
no recorded recipe. The three `libvpx_filter_*_96x80.webp` files are bytes
that no Pillow option writes. These are VP8 key frames from libvpx
1.14.0 (`vpx_codec_encode` at quantizer 60 on a 96x80 picture, with a
region-of-interest map of 4 segments), in the simple RIFF container. The name
gives the filter levels of the segments before the loop-filter deltas: the
frame level plus the map's deltas (55 + [0, 12, -12, -63] for `below0`,
44 + [0, 5, 20, 63] for `above63`, 63 + [0, -10, -30, -63] for `in_range`).
`corpus.py` checks their SHA-256.

## The decode table

`decode.json` has one row per corpus image and one column per decode entry
point of the crate. `corpus.py` records each column from Pillow 12.3.0:

| column | Pillow's output |
| --- | --- |
| `decode_rgb` | `Image.open(f).convert("RGB")` |
| `decode_rgb_exif_transposed` | `ImageOps.exif_transpose(Image.open(f)).convert("RGB")` |
| `decode_rgb_smart` | SGLang's `smart_to_rgb(Image.open(f))` (GLM) |
| `decode_native` | `Image.open(f).convert("RGBA")` when the mode is not RGB and has an alpha band or a `transparency` entry, else `convert("RGB")`; no EXIF transpose (Kimi-K3) |

A cell is one of:

- `{"rgb_sha256" | "rgba_sha256", "width", "height"}`: Pillow's output. The
  crate must give the same pixels, channels and size. A `decode_native` cell
  also holds `mode`, the mode of `Image.open`. The crate's `exact_resize` must
  be false for `P`, `1`, `CMYK` and an `L` with a `transparency` entry, else true.
- `{"rejected": "<MmError kind>", "pillow_error"}`: both sides refuse. The
  crate must refuse with this kind.
- `{"rejected": "<MmError kind>", "known_gap"}`: Pillow decodes and the crate
  refuses with this kind, for the reason in `known_gap`.
- `{"rejected": "<MmError kind>", "known_gap", "rgb_sha256", "width", "height"}`:
  the crate decodes this format only when the build has the `image` crate's
  decoder for it. A workspace build has every `image` decoder, because another
  crate turns on the `image` default features. Without the decoder, the crate
  must refuse with this kind. With the decoder, the crate must give Pillow's
  pixels. `corpus.py` lists these cells in `DECODER_GAPS`.
- `{"diverges", "rgb_sha256", "width", "height"}`: the crate decodes other
  pixels than Pillow at Pillow's size, with no error, for the reason in
  `diverges`. The test fails when the crate starts to match Pillow: then remove
  the entry from `DIVERGES`.

`resize_rgba` rows hold Pillow's `resize(size, BICUBIC).convert("RGBA")` of an
RGBA or LA corpus image. Pillow resizes RGBA and LA premultiplied. The crate's
`image::resize::resize_rgba` must give the same bytes from the
`decode_native` pixels.

`corpus.py` never runs the crate. The crate's error kinds and the reasons are
in its `REJECTED`, `KNOWN_GAPS` and `DIVERGES` tables. The test
`image::decode_table` in `src/image/decode_table.rs` replays the table, and
fails when an image in `images/` has no row.

## Schema

The SGLang mm_parity schema: `{model, revision, pillow, versions, spec_shared?,
cases}`. A case has `name`, `images`, `spec`, `input_ids` (the prompt before
expansion), and `expected` or `error`. `expected` holds the core fields
`num_input_ids`, `input_ids_sha256`, `offsets` (each item's one inclusive
feature span) or `item_offsets` (all of its spans), `feature_shapes` and
`feature_sha256`, and the family's named extras: model kwargs such as
`image_grid_thw` or `image_sizes`, `feature_dtypes`, `mrope_sha256` and
`mrope_delta`. Digests use the reference's byte order: `<i4` ids, `<f4`
features, `<i8` positions, bfloat16 bits as `<u2`. The replay compares only
the fields that `expected` holds. `spec_shared` holds the spec fields of every
case, such as a token table; the replay merges it into each case's `spec`.

Unlike SGLang's tables, `expected` holds the HF baseline, not the gate: the
crate is gated on the `target` column. These keys add the framework
comparison:

- `baseline` (table): the HF processor class, its default backend and the
  decode. `expected` and `error` are the baseline's. `pillow` and `versions`
  are the libraries of the driver's environment, where `hf` runs.
- `target` (table): the column the crate must equal, `hf` or a framework.
  `target_reason` says why, when the target is not `hf`.
- `frameworks` (table): per column, `version`, `source` (the code path run),
  `omits` (fields the framework does not compute) and `versions` (the
  libraries of the environment the column ran in).
- `frameworks` (case): one cell per column but `hf`:
  - `{"=": true}`: equal to the baseline on every recorded field;
  - `{"differs": [fields], <field>: <value>, ...}`: the framework's values of
    the fields that differ;
  - `{"omits": [fields]}`: recorded fields that the framework does not give;
    alone or with `differs`, never with `=`;
  - `{"rejects": "<exception or MmError kind>"}`: the framework refuses the
    input;
  - `{"n/a": "<reason>"}`: the framework cannot run the case.

`stages` holds values that `ProcessedItem` does not carry. The family's own
test checks them (`tests/qwen_vl_golden.rs`: the content hash and the resized
buffer of each image). A case may also hold an engine setting that the
family's `request(case)` passes on, such as the context length of a family
that sizes images against the whole request.

## The cell contract

- The crate's reference is the target: `expected` with the target cell's
  `differs` values, or the target's `rejects` as the `error`. When the baseline
  refuses and the target accepts, the target cell holds every field. For a
  field that the target omits, the crate must give the baseline's value.
- A case that the target refuses: the crate must refuse it with the `MmError`
  kind in `rust_error`.
- A case with `known_gap`: the crate and the target differ, for the reason
  given. The case must keep failing, and the test fails when it starts to
  match. With `rust_error`, the gap is a refusal of that kind, and the crate
  must keep that refusal.
- The `dynamo-multimodal` cell must be what the crate gives against the
  baseline. The test fails when the crate changes; `generate.py --crate`
  records the column again.

`generate.py` takes `expected`, `error` and the framework cells from the
frameworks only, and the `dynamo-multimodal` cells from the crate only. The
family's `CRATE` table supplies `known_gap` and `rust_error` by case or image
name. The driver stops when the target refuses a case that has no
`rust_error`, and when a case that the target accepts has a `rust_error`
without a `known_gap`.

## The framework columns

Every column runs the case's images from their bytes, with the framework's
own decode. A framework module in `frameworks/` takes the request and returns
the response of one protocol, so a worker in another environment runs the same
way. A family whose processor a module cannot run (remote code, or an engine
class of its own) gives its own `run` for that column in `ADAPTERS`. A module
records a refusal only from the framework's own call (its decode and its
processor). An error anywhere else, for example in an import, in the
environment or in the module's own conversion, stops the run.

| Column | What runs |
|---|---|
| `hf` | `frameworks/hf.py`: `AutoProcessor` on the token ids decoded to text, images from `load_image`, M-RoPE from Qwen2-VL's `get_rope_index` when `config.json` has an `mrope_section`, else `null` (1-D positions). The module stops on an M-RoPE checkpoint that is not Qwen2-VL. A batch padded to `[N, 3, H, W]` gives each image its crop of `image_sizes`, and the image end token closes its rows. |
| `sglang` | `frameworks/sglang.py`: SGLang's `process_mm_data_async` on the CPU with PIL decode, the case's backend and `--mm-process-config` image overrides. `n/a` when SGLang's config loader (`get_config`) refuses the checkpoint. A family adapter passes its subclass of `SglangPath` to `run`. |
| `vllm` | `frameworks/vllm.py` runs itself as a worker under `--vllm-python`: `MediaConnector.fetch_image` on the bytes, then `MULTIMODAL_REGISTRY.create_processor(...).apply` on the token prompt with per-request `mm_processor_kwargs` (`use_fast=False` selects `pil`), then the model class's `get_mrope_input_positions`, or `null` when `ModelConfig.uses_mrope` is false. By default vLLM sends `uint8` patches and normalizes in the ViT (`x * w + b`), which gives other `float32` values; the worker sets `mm_device_do_normalize=False` and `dtype="float32"` to compare the processor's values. It trusts remote code, makes one processor per `max_model_len`, gives an item of a batched field its batch axis, and turns each placeholder id into the ids of `image_placeholder` when the config names one (Kimi-K3). |
| `llm-multimodal` | `frameworks/llm_multimodal.py` runs the harness binary of `llm-mm-harness/` (see below) on the published crate `llm-multimodal = "=1.15.0"`: its model registry picks the spec from the checkpoint, `MediaConnector` decodes the bytes, the spec's vision processor gives the pixel values and grid, and the spec's prompt replacements expand the ids. The crate has no position code, so the column omits `mrope`. |
| `dynamo-multimodal` | `tests/mm_parity.rs` with `MM_PARITY_RECORD_CRATE=<dir>`: the crate's cell for each case, which `generate.py --crate` copies into the table. |

Request (on stdin for a worker): `{family, model_dir, images_dir, cases:
[{name, images, input_ids, backend, image_kwargs, max_model_len?}]}`. The ids
are in the checkpoint's id space (the family's `ID_MAP`); `backend` is
`torchvision` or `pil`, and a framework that cannot select it runs the backend
it serves, which its `source` names; `image_kwargs` are the case's
image-processor overrides, and a framework without a way to apply them gives
`n/a`; `max_model_len` is the engine's context length (`--max-model-len`,
`--context-length`) when the case sets one. Response (on stdout): `{version,
source, omits, versions?, cases: {name: {"output": {...}} | {"rejects": ...} |
{"n/a": ...}}}`. A worker must give `versions`, the libraries of its own
environment; the driver records its own for a column that runs in its
process. `output` holds `input_ids` (expanded, not hashed), `offsets` or
`item_offsets`, `image_grid_thw` or `image_sizes`, `feature_shapes`,
`feature_dtypes`, `feature_sha256`, `mrope_sha256` and `mrope_delta`. The
driver maps the ids back and hashes them, and keeps the fields in the family's
`FIELDS` (by default `offsets`, `image_grid_thw`, the features and M-RoPE). A
reason names the checkpoint `<model_dir>` and holds no object address
(`frameworks.reason`), so that the table does not depend on the host.

## Regenerate

Use an environment with SGLang `a2ffd21469` installed editable, transformers
5.19.0, torch 2.14.1, torchvision 0.29.2, Pillow 12.3.0, NumPy 2.3.5 and
blake3. The table's `pillow` and `versions` record it. The `sglang` column
runs SGLang's processor on the CPU, but it needs a visible CUDA device:
`frameworks/sglang.py` leaves the server argument `device` unset, and SGLang
then picks an accelerator or stops. Also, `sglang.test.test_utils` fails at
import when `CUDA_VISIBLE_DEVICES` is the empty string. The other columns need
no GPU.

```bash
python multimodal/tests/fixtures/mm_parity/corpus.py --check
cargo test -p dynamo-multimodal --lib image::decode_table
# Each table: <family> is its module in families/, <checkpoint> the config,
# processor and tokenizer files of the table's `model` at its `revision`.
python multimodal/tests/fixtures/mm_parity/generate.py --family <family> --check \
    --model-dir <checkpoint> --vllm-python <python> --llm-mm-harness <binary>
python multimodal/tests/fixtures/mm_parity/generate.py --family <family> --crate --check
cargo test -p dynamo-multimodal --test mm_parity
python multimodal/tests/fixtures/mm_parity/report.py --check
```

A family module states what else its run needs next to `TARGET` (for
example `CUDA_VISIBLE_DEVICES=""`).

`--vllm-python` is the interpreter of an environment with vLLM `4e3463c87`
(the per-commit wheel `vllm-0.34.0rc1.dev156+g4e3463c87`), transformers
5.22.2, torch 2.14.0, torchvision 0.29.0, Pillow 13.0.0 and NumPy 2.3.5; no
GPU is necessary. The `vllm` column's `versions` records them.

`llm-mm-harness/` is the llm-multimodal worker, a binary crate with the
dependency `llm-multimodal = { version = "=1.15.0", default-features = false }`.
It is a workspace of its own with its own `Cargo.lock`, so the repository's
crates do not depend on it. Its `main` reads the request on stdin, runs each
case through the API above, and prints the response. Build it with
`cargo build --release --locked` in that directory and pass the binary as
`--llm-mm-harness`. The binary reports the SHA-256 of its `Cargo.toml`,
`Cargo.lock` and `src/main.rs`; the driver stops when it differs from the
committed files. At run time the crate loads `libturbojpeg.so.0` and
`libwebp` when the loader finds them, and decodes JPEG with the `image` crate
otherwise; the column's `source` names the decoders that ran.

Without `--check`, each script writes its files. `--frameworks sglang,vllm`
runs only the named columns (and `hf`); the others keep their committed cells.
The driver stops when the baseline's result for a case changed or the case is
new, because a kept cell is then not valid: run that column again.
`corpus.py` needs SGLang only for `smart_to_rgb`.

## Add an image preprocessor

Follow `.agents/skills/mm-preprocessor-generator/SKILL.md`.

## Credit

`generate.py` and `tests/mm_parity.rs` adapt SGLang's
`rust/sglang-processor/tests/scripts/generate_mm_parity.py` and
`rust/sglang-processor/tests/mm_parity.rs`. The original generator and
image-pipeline authors are Kan Wu and Zhihan Guo. SGLang is licensed under
Apache-2.0.
