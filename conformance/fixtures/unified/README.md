# Unified conformance surface (reasoning + content + tool calls)

A third conformance surface that measures the whole assistant output as ONE ordered event stream (`reasoning` / `text` / `tool_call`), alongside the existing tool-only (`conformance/toolcalling/`) and reasoning suites. It exists because those two compare `{normal_text, calls}` / reasoning-only shapes that cannot express the ORDER between reasoning and tool calls — which is exactly where the split parser pipeline breaks.

Status: **U0 spike** (schema + golden corpus + round-trip test). Capture tooling, the parity harness, and the CONFORMANCE_v2.html tab are not built yet — see `DOIT.unifiedparsers_capture.md`.

## Columns (when built)

`GOLDEN | vLLM 0.25.x (Rust) | Dynamo (Rust)` — the golden is the authored oracle; both engines are diffed against it and both can be red.

- **GOLDEN** — authored by `../utils/src/gen_unified_golden.py` from one scenario spec, reasoned from the invariants/policies in `../utils/lib/parsers/UNIFIED_CASES.md`. Never captured from an implementation. Shipped as the versioned `golden.tar.gz` shard here (derived from the build-tree `conformance/unified/golden_spec/<family>.yaml`).
- **vLLM 0.25.x (Rust)** — gemma4 via the native `Gemma4UnifiedParser`; other families via `CombinedParser(reasoning, tool)`. (U1, not built.)
- **Dynamo (Rust)** — `parsers/v1` reasoning + `parsers/v2` tool, composed and stitched to how Dynamo serves today. The red comes from the SPLIT, not from the v2 tool parser being wrong. (U2, not built.)

## Layout

Every fixture ships as a per-version LFS shard here, same convention as the toolcalling/reasoning trees (no loose YAML):

- `inputs.tar.gz` — the shared raw streamed model text per case/family.
- `golden.tar.gz` — the authored oracle (spec-derived event list), derived from `gen_unified_golden.py`.
- `<impl>-<version>.tar.gz` — one shard per engine version (`dynamo_v2-*`, `vllm_python-*`, `vllm_rust-*`, `sglang_python-*`).
- `../utils/lib/parsers/UNIFIED_CASES.md` — schema, invariants, policies, divergence classes, case taxonomy.
- `../tests/unified_schema_roundtrip.rs` — proves every authored golden case parses and round-trips through the event schema.

### Pre-unified columns (`dynamo_v2-0.1.22`, `dynamo_v2-0.1.23`)

`0.1.23` is the last release with NO `unified` module at all — the unified parser first shipped in `0.1.24`, verified with `git ls-tree -d <tag> parsers/v2/src/unified` across `0.1.22`..`0.1.26`. **BOTH `0.1.22` and `0.1.23`** are therefore the SPLIT path by definition (v1 reasoning + v2 tool), and they are what show the argument-integrity divergences the unified parser fixes (`UNIFIED.12.a`, `UNIFIED.7.b`). (An earlier revision of this file said unified shipped in `0.1.23` and scoped this section to `0.1.22` alone; both were wrong.)

Their captures record `parser_path: split`, and the table labels the column `SPLIT ONLY — no unified parser in this build`. That label is not cosmetic: an empty result under a `Combined & Unified` heading reads as "the unified parser returned nothing", when in fact there was never a unified parser to run. On a native case `0.1.22` returns a real `tool_call`; on the guided cases it returns nothing at all, and THAT is the finding.

**Reading the diff counts.** The cross-version harness drives `push`/`finish` only — it has to compile against builds with no `initialize` / output-mode API — so it cannot apply a case's `init:`. Every case therefore runs in that build's ONLY mode. For a pre-request-mode build that is not a mis-measurement (it has one mode, so "what it does" is "what it would have done"), but it does mean part of any diff count against a modern column is missing capability rather than changed behaviour in a comparable mode. The group 30/31/40/41/50/51 cases are the affected ones.

`capture_cross_version.rs` cannot be used unmodified against them: that harness falls back to the split path when a family has no native unified parser, but it still needs `UnifiedDelta`/`assemble` to EXIST at compile time. To re-capture, copy it into the old worktree, drop the unified imports, delete the `native` branch and its `ev_to_yaml`/`delta_to_yaml` helpers, and pin `let native = false`.

### Back-capturing a NEW case into the older columns (MUST, every time)

Adding a corpus case only writes the CURRENT build's column. Every older shard holds just the cases that existed when it was taken, so a new case renders `not captured at <ver> — this case postdates that build` on every historical column. **A row with data in exactly one column shows NO difference, and the difference is the entire point of this table** — a reviewer cannot tell a fixed regression from a case nobody ever ran. Back-capture in the SAME change:

```bash
git worktree add --detach /tmp/old-<ver> dynamo-parsers-v2-v<ver>
# pre-unified (<= 0.1.23): apply the split-only edits above before running
cp conformance/tests/capture_cross_version.rs /tmp/old-<ver>/conformance/tests/
cd /tmp/old-<ver> && \
  XVER_INPUTS=<repo>/conformance/unified/inputs \
  XVER_FAMILIES=<repo>/conformance/utils/src/parser_families.yaml \
  XVER_OUT=/tmp/xver-<ver> XVER_LABEL=<ver> \
  cargo test -p dynamo-conformance-fixtures-v2 --test capture_cross_version -- --nocapture
```

Then merge **only files absent from the released tarball** into `conformance/unified/dynamo_v2-<ver>/` — never overwrite a shipped entry, that is the rewrite this file forbids — and run `package_fixtures.py`, `extract_fixtures.py`, `render_table_v2.sh`.

**Done means the whole chain, in every worktree that has the corpus.** A stacked PR and its base are two separate renders, and their `inputs/` can legitimately differ, so each needs its OWN capture — never copy one branch's shards into the other. Verify per worktree: every `dynamo_v2-*` dir has the same case count as `inputs/`, and the rendered HTML greps **0** for `postdates that build`.

## Golden case file format (authored spec, `conformance/unified/golden_spec/<family>.yaml`)

```yaml
version: 1
family: <family>
cases:
  UNIFIED.<scenario>.<family>:
    description: <one line>
    policy: [P1]            # optional: policy decisions this case depends on
    input: |-              # raw streamed model text
      ...
    golden:                # spec-derived correct event list (the oracle)
      - {kind: reasoning, text: "..."}
      - {kind: tool_call, name: "...", arguments: {...}}
      - {kind: text, text: "..."}
    expect:                # PROVISIONAL documentation of expected engine verdicts (not asserted in U0)
      vllm:   {verdict: match | diverge, class?: <CLASS>, note?: "..."}
      dynamo: {verdict: match | diverge, class?: <CLASS>, note?: "..."}
```
