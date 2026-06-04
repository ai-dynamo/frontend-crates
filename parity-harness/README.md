# parity-harness

Run the parser parity matrix and conformance lanes against the vendored fixtures (`conformance/{toolcalling,reasoning}/fixtures`). Three impls, two languages:

| Lane | Validates | How |
|---|---|---|
| `table` | — (renders the matrix) | dynamo's Python generator over the fixtures; no engines |
| `dynamo` | Dynamo Rust parser vs `expected.dynamo` | `cargo test -p dynamo-conformance` |
| `vllm` / `sglang` | vLLM / SGLang parser vs `expected.{vllm,sglang}` | the engine's Python parser, in a container or pip env |

## Run

```bash
parity-harness/run.sh table                          # -> parity-harness/PARITY.html (open in a browser)
parity-harness/run.sh dynamo                          # Rust lane (606/606 batch)
parity-harness/run.sh sglang --container sglang-localdev   # SGLang vs expected.sglang
parity-harness/run.sh vllm   --container vllm-localdev     # vLLM vs expected.vllm
parity-harness/run.sh vllm   --pip                    # if vLLM is importable in this env
parity-harness/run.sh all --container-vllm vllm-localdev --container-sglang sglang-localdev
```

`table` needs only `python3` + `pyyaml` + `jinja2`. The engine lanes need either a running vLLM/SGLang container (preferred — no local install) or the engine pip-installed.

## How it works

The Python generator and the vLLM/SGLang adapters under `tests/parity/` are **vendored verbatim from dynamo** (kept current by `scripts/sync-from-dynamo.sh`; sync-check gates drift). They hard-code dynamo's repo layout, so `run.sh` builds an ephemeral `.stage/` (gitignored) that presents that layout: the Python package is copied (so `__file__.resolve()` stays in-stage) and the data — fixtures, `parsers/src/tool_calling`, `*_CASES.md`, the peer-version pin stub — is symlinked to this repo's real paths. The unmodified tools then run with `PYTHONPATH=.stage`.

`validate.py` (container mode) ships a minimal adapter bundle + worker into the engine container via `docker exec` and reads results back from a file — mirroring dynamo PR #10296's capture pattern.

## Notes

- **Version pinning.** `expected.{vllm,sglang}` were captured against the versions dynamo pins (`pyproject.stub.toml`). `validate.py` prints the live engine version and warns on mismatch — then diffs are version drift, not parser bugs. Re-capturing `expected.*` is the dynamo capture tool's job, not this harness.
- **Engine validation in CI** lives in dynamo's vLLM/SGLang lanes; here the engine lanes are on-demand/local. frontend-crates CI runs the Rust lane + a `table` render smoke only.
- `PARITY.html` and `.stage/` are gitignored artifacts.
