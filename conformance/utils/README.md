# conformance/utils

Validate parser conformance, update frontend-crate-owned v2 fixtures, and render the v2 conformance table. Migration and ownership are documented in [`../../PARSERS-V2-MIGRATION-PLAN.md`](../../PARSERS-V2-MIGRATION-PLAN.md).

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

### Record Dynamo parser v2 output

1. Edit or add the v2 stream fixture YAML under `conformance/toolcalling/fixtures-stream-v2/`.
2. If a token-id stream fixture's `delta_text` changed, refresh token IDs:

```bash
conformance/utils/record_v2.sh tokens
```

3. Record new Dynamo parser v2 per-chunk output for the affected fixture. The fixture block is still named `expected.dynamo` because that is the existing local-parser key:

```bash
conformance/utils/record_v2.sh stream conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.stream.1.yaml
conformance/utils/record_v2.sh stream conformance/toolcalling/fixtures-stream-v2/harmony_text/TOOLCALLING.stream.1.yaml --text
```

4. Copy the printed JSON deltas into the fixture's `chunks[].expected.dynamo` blocks.
5. Run the validate flow above.

### Capture vLLM/SGLang stream output

Use this when a `TC stream (v2)` fixture exists for a family but the Dynamo parser v2 is still TODO. The fixture should still contain vLLM/SGLang output because those are peer targets for the future Dynamo parser v2 implementation. Keep the case-level `unavailable.dynamo` block, and add per-chunk `expected.vllm` / `expected.sglang` plus `normal_text.<impl>` when a peer emits non-tool text.

For one fixture, copy the capture helper and fixture into the engine containers:

```bash
docker cp conformance/utils/capture_stream.py vllm-localdev:/tmp/capture_stream.py
docker cp conformance/utils/capture_stream.py sglang-localdev:/tmp/capture_stream.py
docker cp conformance/toolcalling/fixtures-stream-v2/deepseek_v4/TOOLCALLING.stream.1.yaml vllm-localdev:/tmp/TOOLCALLING.stream.1.yaml
docker cp conformance/toolcalling/fixtures-stream-v2/deepseek_v4/TOOLCALLING.stream.1.yaml sglang-localdev:/tmp/TOOLCALLING.stream.1.yaml
```

Capture peer parser output. Use the parser names from `capture_all_families_driver.py` for the family; for DeepSeek V4 they are `deepseek_v4` for vLLM and `deepseekv4` for SGLang.

```bash
docker exec vllm-localdev python3 /tmp/capture_stream.py --impl vllm --fixture /tmp/TOOLCALLING.stream.1.yaml --parser deepseek_v4 > /tmp/deepseek_v4.stream.1.vllm.json
docker exec sglang-localdev python3 /tmp/capture_stream.py --impl sglang --fixture /tmp/TOOLCALLING.stream.1.yaml --parser deepseekv4 > /tmp/deepseek_v4.stream.1.sglang.json
```

Extract the case maps that `build_stream_fixtures.py` consumes, and note the printed versions for `--captured`.

```bash
python3 - <<'PY'
import json
for impl in ("vllm", "sglang"):
    src = f"/tmp/deepseek_v4.stream.1.{impl}.json"
    dst = f"/tmp/deepseek_v4.stream.1.{impl}.cases.json"
    data = json.load(open(src))
    json.dump(data["cases"], open(dst, "w"), ensure_ascii=False)
    print(impl, data["version"])
PY
```

Rewrite the YAML with the peer outputs while keeping Dynamo parser v2 unavailable:

```bash
python3 conformance/utils/build_stream_fixtures.py \
  --source conformance/toolcalling/fixtures-stream-v2/deepseek_v4/TOOLCALLING.stream.1.yaml \
  --out conformance/toolcalling/fixtures-stream-v2/deepseek_v4/TOOLCALLING.stream.1.yaml \
  --vllm /tmp/deepseek_v4.stream.1.vllm.cases.json \
  --sglang /tmp/deepseek_v4.stream.1.sglang.cases.json \
  --captured vllm=<version printed above> \
  --captured sglang=<version printed above> \
  --unavailable "dynamo=Dynamo parser v2 TC streaming not yet implemented for this family; vLLM/SGLang per-chunk output is the target to match."
```

The YAML should then have this shape:

```yaml
captured_with:
  vllm: '0.22.0'
  sglang: '0.5.12.post1'
cases:
  TOOLCALLING.stream.1.a:
    unavailable:
      dynamo: 'Dynamo parser v2 TC streaming not yet implemented for this family; vLLM/SGLang per-chunk output is the target to match.'
    chunks:
    - delta_text: '...'
      expected:
        vllm:
        - {index: 0, name: 'get_weather'}
        sglang: []
      normal_text:
        sglang: '...'
```

For all non-Harmony families, use the bulk capture wrapper:

```bash
conformance/utils/capture_all_families.sh vllm-localdev sglang-localdev
```

Review the YAML diff after the bulk command. It rewrites `conformance/toolcalling/fixtures-stream-v2/<family>/TOOLCALLING.stream.*.yaml` with `captured_with`, `expected.vllm`, `expected.sglang`, and any `normal_text` evidence.

### Capture batch-on-stream peer output

Use this when refreshing the `TC batch-on-stream (v2)` tab. It mirrors every v1 batch YAML into `conformance/toolcalling/fixtures-batch-on-stream-v2/<family>/`, captures vLLM/SGLang by feeding each batch `model_text` to the engine streaming parser, and records Dynamo parser v2 output only for Harmony.

```bash
conformance/utils/record_v2.sh batch > /tmp/harmony_batch_on_stream_dynamo.json
python3 conformance/utils/capture_batch_on_stream_all.py \
  --root "$PWD" \
  --work /tmp/batch_on_stream_capture \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --dynamo-harmony-json /tmp/harmony_batch_on_stream_dynamo.json
```

| Script | Purpose |
|---|---|
| `render_table_v2.sh` | Builds `.stage-v2/` and writes `conformance/CONFORMANCE_v2.html`. |
| `render_parity_v1.sh` | Builds the v1 `.stage/` and writes `.stage/tests/parity/PARITY_v1.html` with old Dynamo `generate_parity_table.py`. |
| `check_v2.sh` | Runs local-parser, vLLM, and SGLang checks against staged fixtures. |
| `record_v2.sh` | Records Dynamo parser v2 streaming fixture data. |
| `capture_stream.py` | Captures vLLM/SGLang per-chunk streaming output inside an engine container. |
| `capture_all_families.sh` | Captures and rewrites v2 stream fixtures for all non-Harmony families. |
| `capture_batch_on_stream.py` | Captures one engine's streaming parser over batch fixture text inside an engine container. |
| `capture_batch_on_stream_all.py` | Captures and rewrites all batch-on-stream overlay YAMLs for vLLM/SGLang, plus Dynamo parser v2 Harmony when provided. |
| `build_stream_fixtures.py` | Rebuilds one v2 stream fixture from source chunks, captured peer output, and unavailable markers. |

---

## Which Parser Runs Where

The HTML table is a bridge table. Read parser version and fixture version separately. `v1` means Dynamo-synced parser code or fixtures. `v2` means Dynamo parser v2 behavior owned in frontend-crates during the bridge. The fixture schema and some commands still use `dynamo` as the local-parser key.

| Tab | Parser version | Parser path | Fixture version | What it checks |
|---|---|---|---|---|
| `TC batch (v1)` | v1 Dynamo-synced batch parser | `parsers/src/tool_calling/` | v1 batch fixtures in `conformance/toolcalling/fixtures/` | Parse one complete model output with the v1 batch parser. |
| `TC batch-on-stream (v2)` | Dynamo parser v2 (streaming) | `parsers_v2/src/tool_calling/*`; current Harmony implementation in `parsers_v2/src/tool_calling/harmony.rs` | v1 batch fixtures in `conformance/toolcalling/fixtures/` | Feed complete batch text into the v2 stream parser and compare assembled output to v1 batch expected output. |
| `TC stream (v2)` | Dynamo parser v2 (token-incremental streaming) | `parsers_v2/src/tool_calling/*`; current Harmony implementation in `parsers_v2/src/tool_calling/harmony.rs` | v2 stream fixtures in `conformance/toolcalling/fixtures-stream-v2/` | Emit tool-call deltas as tokens or text chunks arrive. |
| `Reasoning batch (v1)` / `Reasoning stream (v1)` | v1 Dynamo-synced reasoning parser | `parsers/src/reasoning/` | v1 reasoning fixtures in `conformance/reasoning/fixtures/` | Compare reasoning extraction output across engines. |

`TC batch-on-stream` means v1 batch data on Dynamo parser v2: each batch fixture's full text is run through the v2 streaming parser, and the assembled result is compared to that engine's own batch parser. `=` means consistent; otherwise the letters name the engines that diverge (`D`=local parser fixture key, `V`=vLLM, `S`=SGLang). For this tab, `D` compares Dynamo parser v2 stream output against the v1 Dynamo batch expected output stored under `expected.dynamo`.

Dynamo parser v2 tool-calling streaming code lives under `parsers_v2/src/tool_calling/*`; the current Harmony implementation is `parsers_v2/src/tool_calling/harmony.rs`. It consumes token chunks directly and also supports text chunks through a small held-suffix tokenizer bridge. It emits deltas before finish, with no jail and no buffer-then-release. The old Dynamo streaming path buffered or jailed markup until a tool call was complete, then ran the batch parser on the assembled text; that code lives in Dynamo, not in frontend-crates.

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
| `check_v2.sh dynamo [batch|stream|all]` | `cargo test -p dynamo-conformance-fixtures-v2` against the Rust fixture tests. The subcommand name is the legacy local-parser key; v2 stream tests run Dynamo parser v2 code. |
| `check_v2.sh vllm --container NAME` | vLLM parser inside a Docker container. |
| `check_v2.sh vllm --pip` | vLLM parser in the current Python interpreter. |
| `check_v2.sh sglang --container NAME` | SGLang parser inside a Docker container. |
| `check_v2.sh sglang --pip` | SGLang parser in the current Python interpreter. |
| `check_v2.sh all --container-vllm NAME --container-sglang NAME` | Dynamo parser fixture checks, then vLLM and SGLang checks. |

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

Regenerate Dynamo parser v2 fixture data. `record stream` and `record batch` print JSON to stdout; `record tokens` writes token IDs into the overlay fixtures in place. Output is still pasted under `expected.dynamo` because that is the existing local-parser fixture key.

```bash
conformance/utils/record_v2.sh <stream <fixture.yaml> [--text] | batch | tokens> [--dry-run]
```

| Command | What runs |
|---|---|
| `record_v2.sh stream <fixture.yaml> [--text]` | `record_dynamo_stream`; prints per-chunk `expected.dynamo` JSON for one stream fixture. The binary name is legacy; the parser code is Dynamo parser v2. |
| `record_v2.sh batch` | `record_batch_via_stream`; prints Dynamo parser v2 stream-on-batch JSON over the Harmony batch samples. |
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

On startup, `validate.py` prints the live engine version alongside the fixture engine version pinned in `pyproject.stub.toml`. If they differ, it prints a warning. A mismatch does not abort; it means failures may be version drift rather than parser regressions.

Example:

```bash
validate.py --impl sglang --container sglang-localdev --fixtures conformance/utils/.stage-v2/tests/parity/toolcalling/fixtures
```

---

## How It Works

The Python v1 generator and vLLM/SGLang adapters under `conformance/utils/tests/parity/` are vendored from Dynamo and updated manually. The v2 conformance generator lives under `conformance/utils/` and is copied into `.stage-v2/tests/parity/` at render time. See [`PARSERS-V2-MIGRATION-PLAN.md`](../../PARSERS-V2-MIGRATION-PLAN.md) for the upstream mapping and sync instructions.

Those vendored files hard-code Dynamo's repo layout, so the scripts build an ephemeral stage tree before Python runs. The Python package and selected fixture view are copied so `__file__.resolve()` stays in-stage; `parsers/src/tool_calling`, `*_CASES.md`, and `pyproject.stub.toml` are symlinked to this repo's real paths. Python runs with the selected stage as `PYTHONPATH`.

`validate.py` in container mode ships a minimal adapter bundle and worker into the engine container via `docker exec`, then reads results back from a temp file.

## Notes

- `.stage*/` is a gitignored artifact.
- frontend-crates CI runs the Rust conformance checks and a table-render smoke check. vLLM and SGLang checks are local/on-demand because engine validation runs in Dynamo.
- `expected.{vllm,sglang}` were captured against the engine versions pinned in `pyproject.stub.toml`; re-capture them when those pins change.
