# conformance/utils

Run the parser parity matrix and conformance lanes against the vendored fixtures (`conformance/{toolcalling,reasoning}/fixtures`). Three impls, two languages:

| Lane | Validates | How |
|---|---|---|
| `table` | — (renders the matrix) | dynamo's Python generator over the fixtures; no engines |
| `dynamo` | Dynamo Rust parser vs `expected.dynamo` | `cargo test -p dynamo-conformance` |
| `vllm` / `sglang` | vLLM / SGLang parser vs `expected.{vllm,sglang}` | the engine's Python parser, in a container or pip env |

---

## run.sh

Orchestrator for all lanes. Builds the ephemeral `.stage/` layout before each Python run, then delegates to `cargo test` or `validate.py`.

```
conformance/utils/run.sh <lane> [options]
```

**Lanes**

| Lane | What runs |
|---|---|
| `table` | Renders `conformance/utils/PARITY.html` — the full parity matrix. No engines needed. |
| `dynamo` | `cargo test -p dynamo-conformance --test parity_toolcalling` — Rust parser vs fixtures. |
| `vllm` | `validate.py --impl vllm` against the toolcalling fixtures. |
| `sglang` | `validate.py --impl sglang` against the toolcalling fixtures. |
| `all` | `dynamo` + `vllm` + `sglang` + `table` in sequence. |

**Options**

| Flag | Applies to | Meaning |
|---|---|---|
| `--container NAME` | `vllm`, `sglang`, `all` | Run the engine inside docker container `NAME` via `docker exec`. Preferred — no local install needed. |
| `--pip` | `vllm`, `sglang` | Run the engine in-process (engine must be pip-installed in this interpreter). |
| `--container-vllm NAME` | `all` | Container for the vLLM lane when running `all`. |
| `--container-sglang NAME` | `all` | Container for the SGLang lane when running `all`. |

**Examples**

```bash
conformance/utils/run.sh table
conformance/utils/run.sh dynamo
conformance/utils/run.sh sglang --container sglang-localdev
conformance/utils/run.sh vllm   --container vllm-localdev
conformance/utils/run.sh vllm   --pip
conformance/utils/run.sh all --container-vllm vllm-localdev --container-sglang sglang-localdev
```

**Dependencies**

- `table`: `python3`, `pyyaml`, `jinja2`
- `dynamo`: `cargo`, the workspace must build
- `vllm` / `sglang`: a running container (preferred) or the engine pip-installed

**Output**

- `table` writes `conformance/utils/PARITY.html` (gitignored). Open in a browser.
- `dynamo` prints cargo test output; exits non-zero on any failure.
- `vllm` / `sglang` print a per-case pass/fail summary and a final count line, e.g. `vllm parity: 624/624 cases passed`. Exits non-zero on any mismatch.

---

## validate.py

Runs a Python engine's parser (vLLM or SGLang) against the toolcalling fixture corpus and diffs the output against the `expected.<impl>` blocks. Called by `run.sh`; can also be run directly.

```
validate.py --impl <vllm|sglang> --fixtures <dir> (--container NAME | --pip)
```

**Flags**

| Flag | Required | Meaning |
|---|---|---|
| `--impl` | yes | Which engine: `vllm` or `sglang`. |
| `--fixtures` | yes | Path to the staged toolcalling fixtures dir. `run.sh` builds and passes this automatically. |
| `--container NAME` | one of | Run the engine inside docker container `NAME`. Ships a minimal worker bundle in via `docker exec`; results come back via a temp file. No local engine install needed. |
| `--pip` | one of | Import and run the engine in-process. Engine must be importable in the current interpreter. |

**Version check**

On startup, `validate.py` prints the live engine version alongside the version dynamo pinned the fixtures against (`pyproject.stub.toml`). If they differ it prints a warning:

```
WARNING: engine 0.8.5 != pin 0.7.3 — diffs below may be version drift, not parser bugs.
```

A mismatch doesn't abort — it flags that failures may be pin drift, not real regressions.

**Output**

Prints one line per failing case (`FAIL <key> [<mode>] ...`) then a summary:

```
sglang parity: 546/546 cases passed
```

Exits 0 if all cases pass, 1 if any fail.

**Direct use**

`run.sh` handles staging and passes `--fixtures` automatically. Run directly only when debugging a specific case:

```bash
validate.py --impl sglang --container sglang-localdev \
  --fixtures conformance/utils/.stage/tests/parity/toolcalling/fixtures
```

---

## How it works

The Python generator and the vLLM/SGLang adapters under `tests/parity/` are **vendored from dynamo** and updated manually — they are not auto-synced and sync-check does not gate on them. See [`PARSERS-SYNC.md`](../PARSERS-SYNC.md) for the exact upstream mapping and sync instructions. They hard-code dynamo's repo layout, so `run.sh` builds an ephemeral `.stage/` (gitignored) that presents that layout: the Python package is copied (so `__file__.resolve()` stays in-stage) and the data — fixtures, `parsers/src/tool_calling`, `*_CASES.md`, the peer-version pin stub — is symlinked to this repo's real paths. The unmodified tools then run with `PYTHONPATH=.stage`.

`validate.py` (container mode) ships a minimal adapter bundle + worker into the engine container via `docker exec` and reads results back from a file — mirroring dynamo PR #10296's capture pattern.

## Notes

- **Version pinning.** `expected.{vllm,sglang}` were captured against the versions dynamo pins (`pyproject.stub.toml`). `validate.py` prints the live engine version and warns on mismatch — then diffs are version drift, not parser bugs. Re-capturing `expected.*` is the dynamo capture tool's job, not this harness.
- **Engine validation in CI** lives in dynamo's vLLM/SGLang lanes; here the engine lanes are on-demand/local. frontend-crates CI runs the Rust lane + a `table` render smoke only.
- `PARITY.html` and `.stage/` are gitignored artifacts.
