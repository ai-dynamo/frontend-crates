# parsers and conformance/utils: v1 sync runbook

This file documents the manual sync boundary for the temporary v1 Dynamo mirror. The migration plan and v1/v2 ownership model live in [`conformance/README.md`](conformance/README.md).

## Sync Boundary

The syncable v1 files are resettable to Dynamo content. Do not put frontend-crates v2 parser or renderer work in those paths.

| Area | Sync rule |
|---|---|
| `parsers/src/` | Sync from Dynamo only for deliberate v1 mirror refreshes. |
| `parsers/tests/` | Sync from Dynamo when present upstream. |
| `conformance/utils/tests/parity/` | Sync the old Dynamo parity generator package. Keep `generate_parity_table.py`, `parity_table.html.j2`, `common.py`, `markup.py`, and the `toolcalling/` / `reasoning/` renderers unchanged from Dynamo. |
| `conformance/utils/lib/parsers/TOOLCALLING_CASES.md` and `REASONING_CASES.md` | Sync with the v1 mirror so old `generate_parity_table.py` renders the old table. |

The v2-owned paths are not sync targets: `parsers_v2/`, `parsers_v2-py/`, `conformance/toolcalling/fixtures-stream-v2/`, `conformance/utils/generate_conformance_table_v2.py`, and `conformance/utils/conformance_table_v2.html.j2`.

## Sync Commands

```bash
scripts/manual-sync-parsers.sh /path/to/dynamo          # dry-run: shows what would change
scripts/manual-sync-parsers.sh --apply /path/to/dynamo  # apply
```

After applying a sync, verify both renderers:

```bash
conformance/utils/render_parity_v1.sh
conformance/utils/render_table_v2.sh
```

`pyproject.stub.toml` is a local vLLM/SGLang version pin file; bump it manually when engine pins change. `parsers/Cargo.toml` is intentionally diverged for standalone publishing; merge dependency changes manually.

## Manual version pins (check on every sync)

`sync-from-dynamo.sh` syncs `src/`/`tests/`/fixtures but never dependency versions (it lists `Cargo.toml` as manual-review and never auto-applies). So no version below is ever synced automatically; check this table each sync. "last-synced" is the value verified against dynamo `main` on 2026-06-04; re-verify against current `main`, not a stale local checkout.

| Pin | frontend-crates file | dynamo file | last-synced value | notes |
|---|---|---|---|---|
| `openai-harmony` (Rust crate) | root `Cargo.toml` `[workspace.dependencies]` | `lib/parsers/Cargo.toml` | `0.0.3` (both) | Build matches. The real risk is the **runtime** gap below. |
| `openai_harmony` (Python, in the engine containers) | recorded as `captured_with` in `conformance/toolcalling/fixtures-stream-v2/harmony*/` | n/a (engine container) | vLLM container `0.0.8`, SGLang container `0.0.4` | The gpt-oss/Harmony parser's behavior is defined by the Harmony grammar; a Rust-`0.0.3`-vs-Python-`0.0.8` gap is the most likely source of a harmony conformance mismatch. Re-check the in-container version after any vllm/sglang bump. Consider bumping the Rust crate to match. |
| `fastokens` (Rust) | root `Cargo.toml` | root `Cargo.toml` | FC `0.1.0` vs dynamo `0.2.0` (**skew**) | Tokenizer backend; low parser conformance impact but the one hard Rust skew. Bump to `0.2.0` to stay honest. |
| `vllm` / `sglang` (Python engine pins) | `conformance/utils/pyproject.stub.toml` | `pyproject.toml` | `vllm==0.22.0`, `sglang==0.5.12.post1` | Matches current `main`. After bumping, re-capture peer streaming data and update `captured_with`. |
| Shared crate versions + parser deps | `parsers/`,`tokenizers/`,`protocols/`,`renderer/` `Cargo.toml` + root | `lib/*/Cargo.toml` + root | all `1.3.0`; async-openai `0.34`, tokenizers `0.21.4`, tiktoken-rs `0.9`, rustpython-parser `0.4.0`, minijinja `2.20.0`; Rust `1.93.1` | Should always match the dynamo workspace; verify on sync. |

## Files unique to this repo

These have no upstream counterpart. Never overwrite during a sync.

| File | Purpose |
|---|---|
| `conformance/utils/_common.sh` | Shared stage builder for the conformance scripts |
| `conformance/utils/check_v2.sh` | Runs Dynamo, vLLM, and SGLang checks against staged fixtures |
| `conformance/utils/render_table_v2.sh` | Renders `conformance/CONFORMANCE_v2.html` with the v2 conformance generator |
| `conformance/utils/render_parity_v1.sh` | Renders `.stage/tests/parity/PARITY_v1.html` with old Dynamo `generate_parity_table.py` |
| `conformance/utils/record_v2.sh` | Records Dynamo v2 stream fixture data |
| `conformance/utils/validate.py` | Cross-impl validation via `docker exec` or pip |
| `conformance/utils/README.md` | Usage docs |
| `conformance/utils/.gitignore` | Excludes `.stage*/`, old local `CONFORMANCE*.html` outputs, and Python bytecode |
| `conformance/utils/tests/__init__.py` | Empty package root for `.stage/` imports |
| `conformance/utils/generate_conformance_table_v2.py` | frontend-crates-owned conformance renderer; staged into `tests/parity/` at render time |
| `conformance/utils/conformance_table_v2.html.j2` | frontend-crates-owned conformance HTML template; staged into `tests/parity/` at render time |
| `parsers/Cargo.toml` | Inlined for standalone publishing |
