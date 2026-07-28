# Unified conformance surface (reasoning + content + tool calls)

A third conformance surface that measures the whole assistant output as ONE ordered event stream (`reasoning` / `text` / `tool_call`), alongside the existing tool-only (`conformance/toolcalling/`) and reasoning suites. It exists because those two compare `{normal_text, calls}` / reasoning-only shapes that cannot express the ORDER between reasoning and tool calls — which is exactly where the split parser pipeline breaks.

Status: **U0 spike** (schema + golden corpus + round-trip test). Capture tooling, the parity harness, and the CONFORMANCE_v2.html tab are not built yet — see `DOIT.unifiedparsers_capture.md`.

## Columns (when built)

`GOLDEN | vLLM 0.25.x (Rust) | Dynamo (Rust)` — the golden is the authored oracle; both engines are diffed against it and both can be red.

- **GOLDEN** — authored (`golden/<family>.yaml`), reasoned from the invariants/policies in `../utils/lib/parsers/UNIFIED_CASES.md`. Never captured from an implementation.
- **vLLM 0.25.x (Rust)** — gemma4 via the native `Gemma4UnifiedParser`; other families via `CombinedParser(reasoning, tool)`. (U1, not built.)
- **Dynamo (Rust)** — `parsers/v1` reasoning + `parsers/v2` tool, composed and stitched to how Dynamo serves today. The red comes from the SPLIT, not from the v2 tool parser being wrong. (U2, not built.)

## Layout

- `golden/<family>.yaml` — authored golden cases (input + spec-derived event list + provisional per-engine `expect`).
- `../utils/lib/parsers/UNIFIED_CASES.md` — schema, invariants, policies, divergence classes, case taxonomy.
- `../tests/unified_schema_roundtrip.rs` — proves every golden file parses and round-trips through the event schema.

## Golden case file format

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
