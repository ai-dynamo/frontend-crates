# Unified conformance surface (reasoning + content + tool calls)

A third conformance surface that measures the whole assistant output as ONE ordered event stream (`reasoning` / `text` / `tool_call`), alongside the existing tool-only (`conformance/toolcalling/`) and reasoning suites. It exists because those two compare `{normal_text, calls}` / reasoning-only shapes that cannot express the ORDER between reasoning and tool calls — which is exactly where the split parser pipeline breaks.

Status: the capture tooling, parity harness, and `CONFORMANCE_v2.html` Unified tab are active. The current Dynamo column is validated against the authored GOLDEN corpus.

## Columns

`GOLDEN | vLLM 0.25.x (Rust) | Dynamo (Rust)` — the golden is the authored oracle; both engines are diffed against it and both can be red.

- **GOLDEN** — authored by `../utils/src/gen_unified_golden.py` from one scenario spec, reasoned from the invariants/policies in `../utils/lib/parsers/UNIFIED_CASES.md`. Never captured from an implementation. Stored in each family's `inputs_and_golden.yaml` under `conformance/fixtures-unified-v2/families/`.
- **vLLM Rust** — a captured peer implementation, shown by version.
- **Dynamo Rust** — the native Unified parser where the family has one, otherwise the historical split reasoning-plus-tool path. Current native families must pass the zero-red/zero-empty gate below.

## Layout

Unified uses one directory per family with one canonical input/golden document and one sparse capture YAML per captured implementation version:

- `conformance/fixtures-unified-v2/families/<family>/inputs_and_golden.yaml` — immutable case identity, current display ID, historical aliases, exact request, and authored golden.
- `conformance/fixtures-unified-v2/families/<family>/<implementation>-<runtime_version>.yaml` — one explicit capture node with provenance, a parent link, changed observations, and tombstones. Unchanged versions remain as empty deltas so the tested version is preserved without copying output.
- `../utils/lib/parsers/UNIFIED_CASES.md` — schema, invariants, policies, divergence classes, case taxonomy.
- `../tests/unified_schema_roundtrip.rs` — proves every authored golden case parses and round-trips through the event schema.

### TODO: delete archive duplicates of YAML lineage

The following inactive legacy archives have their captured data normalized into the current YAML lineage. Delete them with their `inactive_shards` manifest entries after package validation no longer requires the original archive bytes as import evidence:

```
dynamo_v2-0.6.0+source.06847bad0fee2c369c44d7fef7a16ac976af4e1ea606882fcc3cbe5a3a402518.patch1.tar.gz
dynamo_v2-0.6.0+source.06847bad0fee2c369c44d7fef7a16ac976af4e1ea606882fcc3cbe5a3a402518.patch2.tar.gz
dynamo_v2-0.6.0+source.06847bad0fee2c369c44d7fef7a16ac976af4e1ea606882fcc3cbe5a3a402518.tar.gz
dynamo_v2-0.6.0.patch3.tar.gz
dynamo_v2-0.6.0.patch4.tar.gz
dynamo_v2-0.6.0.patch5.tar.gz
dynamo_v2-0.6.0.patch6.tar.gz
dynamo_v2-0.6.1+source.65b6e028637ed1b7dbd47d4ff52b989be343ba94d44becca626d5292de62078d.tar.gz
sglang_python-0.5.14.tar.gz
sglang_python-0.5.16.tar.gz
vllm_python-0.25.1.patch1.tar.gz
vllm_python-0.25.1.tar.gz
vllm_python-0.26.0.tar.gz
vllm_rust-0.25.1.patch1.tar.gz
vllm_rust-0.25.1.tar.gz
vllm_rust-0.26.0.tar.gz
```

### Pre-unified columns (`dynamo_v2-0.1.22`, `dynamo_v2-0.1.23`)

`0.1.23` is the oldest retained active release with NO `unified` module at all — the unified parser first shipped in `0.1.24`, verified with `git ls-tree -d <tag> parsers/v2/src/unified` across `0.1.22`..`0.1.26`. It is the SPLIT path by definition (v1 reasoning + v2 tool) and shows the argument-integrity divergences the unified parser fixes (`UNIFIED.12-1`, `UNIFIED.7-2`). The `0.1.22` capture had the same rendered outcomes and was removed from the active history.

Some released capture rows retain legacy `parser_path` metadata, but the capture producer does not own that field consistently and the renderer does not treat it as authoritative. `unified_parser_path()` derives the label from the tested release boundary: `0.1.22` and `0.1.23` map to `split`, while `0.1.24` and later map to `unified`; `test_unified_parser_path_uses_the_release_boundary_not_fixture_metadata` pins that mapping. The table labels the older columns `SPLIT ONLY — no unified parser in this build`. Native cases can return real calls through that split path; guided cases are unavailable because the build cannot apply their request mode.

**Reading the diff counts.** The cross-version harness applies each case's serialized `init:` when the build supports request initialization. For older builds, compile with `--cfg conformance_legacy_init`; cases requiring an unsupported initialization are recorded as unavailable, not executed under a different mode. Missing capability must not appear as an empty successful result or as a comparable parser regression.

For builds before UnifiedParser, compile the copied harness with `--cfg conformance_split_only`. For UnifiedParser builds without request initialization, use `--cfg conformance_legacy_init`. Add `--cfg conformance_legacy_terminal` when `ToolCallDelta` has no `complete` field; the capture then records that terminal metadata is unavailable rather than inventing it. These flags select compatibility branches in the same harness; no manual source edits are needed.

### Back-capturing a NEW case into the older columns (MUST, every time)

Adding a corpus case only writes the CURRENT build's column. Every older capture node holds just the cases that existed when it was taken, so a new case renders `not captured at <ver> — this case postdates that build` on every historical column. **A row with data in exactly one column shows NO difference, and the difference is the entire point of this table** — a reviewer cannot tell a fixed regression from a case nobody ever ran. Back-capture in the SAME change:

```bash
git worktree add --detach /tmp/old-<ver> dynamo-parsers-v2-v<ver>
\cp -f conformance/tests/capture_cross_version.rs /tmp/old-<ver>/conformance/tests/
\cp -f conformance/tests/common/mod.rs /tmp/old-<ver>/conformance/tests/common/mod.rs
\cp -f conformance/utils/src/unified_tools.json /tmp/old-<ver>/conformance/utils/src/unified_tools.json
cd /tmp/old-<ver> && \
  CONFORMANCE_DYNAMO_PROVENANCE_SCRIPT=<repo>/conformance/utils/src/dynamo_version.py \
  RUSTFLAGS='<compatibility flags for this release, or empty>' \
  XVER_INPUTS=<repo>/conformance/unified/inputs \
  XVER_FAMILIES=<repo>/conformance/utils/src/parser_families.yaml \
  XVER_OUT=/tmp/xver-<ver> XVER_LABEL=<ver> \
  cargo test -p dynamo-conformance-fixtures-v2 --test capture_cross_version -- --nocapture
```

Write new cases and corrections into a new loose capture directory. The source-identity checker must verify the historical checkout against its tag before accepting `<ver>`. Corrections must be recaptured from that source with the current input, initialization, tool schemas, and chunk schedule, never copied from the current parser. Then run `package_fixtures.py`, which adds sparse capture files under the affected family directory, followed by `extract_fixtures.py` and `render_table_v2.sh`.

Rust and peer capture harnesses read the same tool declarations from `conformance/utils/src/unified_tools.json`. Each new capture binds its output to the request it executed. Historical records with missing or different request metadata remain preserved but are unavailable for comparison with the displayed request; matching case IDs alone do not establish matching inputs.

Historical capture identities and import lineage remain recorded in each capture YAML. A capture without request bindings cannot prove that its output belongs to the displayed request, so its stimulus is explicitly unavailable rather than silently rebound to the current request.

**Done means the whole chain, in every worktree that has the corpus.** A stacked PR and its base are two separate renders, and their `inputs/` can legitimately differ, so each needs its OWN capture — never copy one branch's capture history into the other. Verify per worktree that each Dynamo release, after folding its append-only capture nodes, covers every current input and has no rendered `postdates that build` cells. Hash-pinned inactive archives remain unchanged as retained evidence; peer-engine captures have separate coverage and are not evidence of a Dynamo backfill.

## Unified zero-red/zero-empty gate

Unified work is complete only when the affected family's selected current Dynamo column has **zero empty cells and zero red cells**.

## Required conversion collection

For every family converted to `UnifiedParser`, collect the current source in the same change. Unpublished source uses `<crate-version>+source.<sha256>`; a plain version requires source equality with its release tag. The collection is: generate the authored Unified golden inputs, run `unified_render` to capture live Dynamo output, run `explode_unified_fixtures.py`, run `package_fixtures.py`, run `extract_fixtures.py --full-refresh`, then render `conformance/CONFORMANCE_v2.html`. Commit `conformance/fixtures/`, `conformance/fixtures-unified-v2/`, and `conformance/fixtures-manifest.json` together; the self-contained family and capture YAML files plus the manifest pin are the evidence that the family is collected. Files under the gitignored `conformance/unified/` build tree are not.

Use only `conformance/utils/render_table_v2.sh --output conformance/CONFORMANCE_v2.html` for the report. Do not generate `CONFORMANCE_unified.html`. The required final gate is `conformance/utils/check.sh status --model <family> --tab unified`, which must show zero current Dynamo red cells and zero current Dynamo empty cells.

- **Empty current cell:** the current capture is missing the case. Repair the capture pipeline and regenerate the qualified current shard.
- **Red current cell:** current Dynamo output differs from GOLDEN. Reproduce the popup's exact input, initialization, and chunks, then fix the parser unless the authored GOLDEN is demonstrably wrong.
- **Not a fix:** adding `reason:`, marking the current parser unavailable, selecting a historical column, serving stale HTML, or editing GOLDEN only to match current output.

Follow this sequence until the rendered counts are both zero:

1. Write the parser or capture change.
2. Read every affected popup: input, initialization, chunks, GOLDEN events, and current Dynamo events.
3. Fix the owning parser or capture path.
4. Regenerate the qualified current capture, run `package_fixtures.py`, and keep all three published fixture paths in the same commit.
5. Render `conformance/CONFORMANCE_v2.html` from the same worktree and read the current Unified column again.
6. Run `conformance/utils/check.sh status --model <family> --tab unified`. This standard gate renders first, prints every empty/red case, and exits nonzero until the selected row is clear. Every render also writes the complete machine-readable report to `conformance/CONFORMANCE_v2.json`.
7. Run `cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture` and `cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity -- --nocapture`.

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
