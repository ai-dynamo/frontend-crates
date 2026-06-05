# conformance/utils

Validate parser conformance, update frontend-crates-owned v2 fixtures, and render the v2 conformance table. Migration and ownership are documented in [`../README.md`](../README.md).

## Validate V2 Changes

Run this after changing `parsers_v2/`, v2 fixtures, fixture tests, or the v2 table renderer.

```bash
cargo fmt
cargo test --locked -p dynamo-parsers-v2 -- --nocapture
cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture
conformance/utils/render_table_v2.sh
git diff --check
```

What each step proves:

| Step | Why it is needed |
|---|---|
| `cargo fmt` | Formats Rust changes. |
| `cargo test --locked -p dynamo-parsers-v2 -- --nocapture` | Runs Rust unit tests for the v2 parser implementation. |
| `cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture` | Runs fixture-based tests against committed YAML fixtures. |
| `conformance/utils/render_table_v2.sh` | Generates `conformance/CONFORMANCE_v2.html` from the staged v2 fixture view. |
| `git diff --check` | Checks for whitespace errors and conflict markers. |

## Update V2 Fixtures

Use this when v2 parser behavior changes or when adding a v2 stream case. This updates only `conformance/toolcalling/fixtures-stream-v2/`; it does not update v1.

1. Edit or add the v2 stream fixture YAML under `conformance/toolcalling/fixtures-stream-v2/`.
2. If a token-id stream fixture's `delta_text` changed, refresh token IDs:

```bash
conformance/utils/record_v2.sh tokens
```

3. Record new Dynamo per-chunk output for the affected fixture:

```bash
conformance/utils/record_v2.sh stream conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.stream.1.yaml
conformance/utils/record_v2.sh stream conformance/toolcalling/fixtures-stream-v2/harmony_text/TOOLCALLING.stream.1.yaml --text
```

4. Copy the printed JSON deltas into the fixture's `chunks[].expected.dynamo` blocks.
5. Run the validate flow above.

| Script | Purpose |
|---|---|
| `render_table_v2.sh` | Builds `.stage-v2/` and writes `conformance/CONFORMANCE_v2.html`. |
| `render_parity_v1.sh` | Builds the v1 `.stage/` and writes `.stage/tests/parity/PARITY_v1.html` with old Dynamo `generate_parity_table.py`. |
| `check_v2.sh` | Runs Dynamo, vLLM, and SGLang checks against staged fixtures. |
| `record_v2.sh` | Records Dynamo v2 streaming fixture data. |

---

## Which Parser Runs Where

The HTML table has three tool-calling views. Each view exercises a different Dynamo parser; the source path is what the Dynamo column runs.

| Tab | Dynamo parser | Source | What it checks |
|---|---|---|---|
| `TC batch` | batch tool-calling parser | `parsers/src/tool_calling/` | parse one complete model output |
| `TC stream` | v2 token-incremental streaming parser | `parsers_v2/src/lib.rs` | emit tool-call deltas as tokens arrive |
| `TC batch-on-stream` | v2 streaming parser, fed batch text | `parsers_v2/src/lib.rs` | does streaming a complete output reconstruct what the batch parser gives? |
| `Reasoning batch` / `Reasoning stream` | vendored Dynamo reasoning renderer data | `conformance/reasoning/fixtures/` | compare reasoning extraction output across engines |

`TC batch-on-stream` means batch data on the streaming parser: each batch fixture's full text is run through the v2 streaming parser, and the assembled result is compared to that engine's own batch parser. `=` means consistent; otherwise the letters name the engines that diverge (`D`=Dynamo, `V`=vLLM, `S`=SGLang).

The v2 streaming parser (`parsers_v2/src/lib.rs`) consumes token chunks directly and also supports text chunks through a small held-suffix tokenizer bridge. It emits deltas before finish, with no jail and no buffer-then-release. The old streaming path buffered or jailed markup until a tool call was complete, then ran the batch parser on the assembled text; that code lives in the Dynamo repo, not in frontend-crates.

The v1/v2 output locations and source views are listed in [`../README.md`](../README.md#render-outputs).

---

## render_table_v2.sh

Render the conformance table. No engine containers are needed.

```bash
conformance/utils/render_table_v2.sh
conformance/utils/render_table_v2.sh --dry-run
```

Output: `conformance/CONFORMANCE_v2.html`. Open it in a browser.

---

## render_parity_v1.sh

Render the old Dynamo parity table. No engine containers are needed.

```bash
conformance/utils/render_parity_v1.sh
conformance/utils/render_parity_v1.sh --dry-run
```

Output: `conformance/utils/.stage/tests/parity/PARITY_v1.html`. Open it in a browser.

---

## check_v2.sh

Run a parser against the committed fixtures. The checks are read-only.

```bash
conformance/utils/check_v2.sh <dynamo|vllm|sglang|all> [options]
```

| Command | What runs |
|---|---|
| `check_v2.sh dynamo [batch|stream|all]` | `cargo test -p dynamo-conformance-fixtures-v2` against the Rust fixture tests. |
| `check_v2.sh vllm --container NAME` | vLLM parser inside a Docker container. |
| `check_v2.sh vllm --pip` | vLLM parser in the current Python interpreter. |
| `check_v2.sh sglang --container NAME` | SGLang parser inside a Docker container. |
| `check_v2.sh sglang --pip` | SGLang parser in the current Python interpreter. |
| `check_v2.sh all --container-vllm NAME --container-sglang NAME` | Dynamo batch+stream checks, then vLLM and SGLang checks. |

Options:

| Flag | Applies to | Meaning |
|---|---|---|
| `--container NAME` | `vllm`, `sglang` | Run the engine inside Docker container `NAME` via `docker exec`. |
| `--pip` | `vllm`, `sglang` | Run the engine in-process; the engine must be importable in this interpreter. |
| `--container-vllm NAME` | `all` | Container for the vLLM check when running `all`. |
| `--container-sglang NAME` | `all` | Container for the SGLang check when running `all`. |
| `--dry-run` / `--dryrun` | all commands | Print what would run, without executing it. |

Examples:

```bash
conformance/utils/check_v2.sh dynamo
conformance/utils/check_v2.sh dynamo stream
conformance/utils/check_v2.sh sglang --container sglang-localdev
conformance/utils/check_v2.sh vllm --container vllm-localdev
conformance/utils/check_v2.sh vllm --pip
conformance/utils/check_v2.sh all --container-vllm vllm-localdev --container-sglang sglang-localdev
conformance/utils/check_v2.sh dynamo --dry-run
```

Dependencies:

| Check | Needs |
|---|---|
| `dynamo` | `cargo`; the workspace must build. |
| `vllm` / `sglang` | a running container (preferred) or the engine pip-installed. |

If the default cargo is too old for edition 2024 / resolver `3`, prefix the command with `CARGO='cargo +1.93.1'` or run inside the devcontainer.

---

## record_v2.sh

Regenerate Dynamo v2 fixture data. `record stream` and `record batch` print JSON to stdout; `record tokens` writes token IDs into the overlay fixtures in place.

```bash
conformance/utils/record_v2.sh <stream <fixture.yaml> [--text] | batch | tokens> [--dry-run]
```

| Command | What runs |
|---|---|
| `record_v2.sh stream <fixture.yaml> [--text]` | `record_dynamo_stream`; prints per-chunk `expected.dynamo` JSON for one stream fixture. |
| `record_v2.sh batch` | `record_batch_via_stream`; prints Dynamo stream-on-batch JSON over the harmony batch samples. |
| `record_v2.sh tokens` | `stamp_stream_token_ids`; stamps `delta_token_ids` into harmony stream fixtures. |

Examples:

```bash
conformance/utils/record_v2.sh stream conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.stream.1.yaml
conformance/utils/record_v2.sh stream conformance/toolcalling/fixtures-stream-v2/harmony_text/TOOLCALLING.stream.1.yaml --text
conformance/utils/record_v2.sh batch
conformance/utils/record_v2.sh tokens --dry-run
```

## validate.py

`validate.py` runs a Python engine parser against the tool-calling fixture corpus and diffs the output against the `expected.<impl>` blocks. `check_v2.sh` calls it after building `.stage-v2/`; direct use is only for debugging a specific case.

```bash
validate.py --impl <vllm|sglang> --fixtures <dir> (--container NAME | --pip)
```

On startup, `validate.py` prints the live engine version alongside the version Dynamo pinned the fixtures against in `pyproject.stub.toml`. If they differ, it prints a warning. A mismatch does not abort; it means failures may be version drift rather than parser regressions.

Example:

```bash
validate.py --impl sglang --container sglang-localdev --fixtures conformance/utils/.stage-v2/tests/parity/toolcalling/fixtures
```

---

## How It Works

The Python v1 generator and vLLM/SGLang adapters under `conformance/utils/tests/parity/` are vendored from Dynamo and updated manually. The v2 conformance generator lives under `conformance/utils/` and is copied into `.stage-v2/tests/parity/` at render time. See [`PARSERS-SYNC.md`](../../PARSERS-SYNC.md) for the upstream mapping and sync instructions.

Those vendored files hard-code Dynamo's repo layout, so the scripts build an ephemeral stage tree before Python runs. The Python package and selected fixture view are copied so `__file__.resolve()` stays in-stage; `parsers/src/tool_calling`, `*_CASES.md`, and `pyproject.stub.toml` are symlinked to this repo's real paths. Python runs with the selected stage as `PYTHONPATH`.

`validate.py` in container mode ships a minimal adapter bundle and worker into the engine container via `docker exec`, then reads results back from a temp file.

## Notes

- `.stage*/` is a gitignored artifact.
- frontend-crates CI runs the Rust conformance checks and a table-render smoke check. vLLM and SGLang checks are local/on-demand because engine validation runs in Dynamo.
- `expected.{vllm,sglang}` were captured against the engine versions pinned in `pyproject.stub.toml`; re-capture them when those pins change.
