# parsers and conformance/utils: manual sync

`parsers/` and `conformance/utils/` are permanently detached from the automated `sync-check` CI pipeline. Updates are manual only.

## Why

`dynamo` depends on this repo's `parsers` crate as an upstream library. Auto-syncing `dynamo/lib/parsers/` back into `parsers/` would create a circular dependency: dynamo → frontend-crates/parsers → synced from dynamo → dynamo. Cutting the rsync breaks the cycle. `parsers/` is now an independent crate that dynamo consumes; changes originate here and flow outward, not inward. `conformance/utils/` is detached for the same reason — parser test expectations evolve on this repo's schedule.

## How to sync

```bash
scripts/manual-sync-parsers.sh /path/to/dynamo          # dry-run: shows what would change
scripts/manual-sync-parsers.sh --apply /path/to/dynamo  # apply
```

The script covers `parsers/src/`, `parsers/tests/` (when present), the 15 Python files under `conformance/utils/tests/parity/`, and the two `*_CASES.md` docs. After applying, verify with `conformance/utils/run.sh table`.

## What the script does NOT touch

`pyproject.stub.toml` (vllm/sglang version pins) is not updated by the script — bump it manually when dynamo changes those pins. `parsers/Cargo.toml` is intentionally diverged (inlined for standalone publishing) — merge changes manually.

## Files unique to this repo

These have no upstream counterpart. Never overwrite during a sync.

| File | Purpose |
|---|---|
| `conformance/utils/run.sh` | Orchestrator — builds `.stage/`, routes lanes |
| `conformance/utils/validate.py` | Cross-impl validation via `docker exec` or pip |
| `conformance/utils/README.md` | Usage docs |
| `conformance/utils/.gitignore` | Excludes `.stage/` and `PARITY.html` |
| `conformance/utils/tests/__init__.py` | Empty package root for `.stage/` imports |
| `parsers/Cargo.toml` | Inlined for standalone publishing |
