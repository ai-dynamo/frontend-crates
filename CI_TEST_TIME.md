# Rust CI test-time campaign

Started 2026-10-14 on `rmccormick/speedup-ci`, based on main
`dcdddc8e` (includes PR #409). Preserve test assertions, fixture inputs,
all registered families, and debug/overflow checks. Measure each change before
claiming a speedup; local timings are not GitHub runner timings.

## Reference job

[Run 38661627008, rust job 116183991984](https://github.com/ai-dynamo/frontend-crates/actions/runs/38661627008/job/116183991984)
at `6a7bc68c88cbbce1ea147a6f3a47efa15c00828a`, Ubuntu hosted runner,
Rust 1.96.1. Job succeeded in 32m58s. `Run tests` took 26m12s.

| Component inside Run tests | Seconds | Evidence |
| --- | ---: | --- |
| Compilation | 207.4 | Cargo finish timestamp |
| Parser-v2 unit binary | 611.91 | libtest summary |
| Stream chunk split sweep | 397.47 | libtest summary |
| Interleaved stream isolation | 123.14 | libtest summary |
| MiniMax reasoning fixture suite | 92.96 | libtest summary; includes cold fixture setup |
| Unified parity | 31.26 | libtest summary |
| XML schema coercion | 25.19 | libtest summary |
| Unified render | 21.56 | libtest summary |
| Everything else and process overhead | ~61 | Remainder |

Execution alone is 22m45s. Cargo runs test binaries sequentially; libtest
parallelizes individual test functions, but several expensive functions contain
entire serial matrices. The split sweep, interleave suite, and parser-v2 unit
binary account for 83% of execution wall time.

## Investigation and experiment ledger

- Hosted baseline dispatched from unchanged `dcdddc8e`:
  [run 38665623886](https://github.com/ai-dynamo/frontend-crates/actions/runs/38665623886),
  Rust job `116196807044`. This controls for changes on main after the reported job.
- Local default-profile run was deliberately stopped during the split sweep once
  the hosted baseline started. Earlier completed results: interleave 114.86s
  (16,580 pair-schedules), MiniMax 1.24s (warm fixtures), stream parity 7.46s.
  This is NOT a completed full-suite baseline. Baseline executables are retained
  for controlled targeted comparisons.
- First candidate: optimize only `regex`, `regex-automata`, and `regex-syntax`
  in the test profile. No source, cases, or assertion changes.
- Second candidate, if needed: additionally optimize `dynamo-parsers-v2` at
  level 1. Avoid paying for optimized code generation across unrelated networking
  and model-download dependencies.
- Profile compilation versus execution separately. Experiment with optimizing
  hot dependencies and/or test code while retaining debug assertions and overflow
  checks. Include compile cost in the decision.
- Inspect redundant setup and repeated work before reducing test coverage.
- Evaluate runner parallelism only after identifying the expensive inner loops;
  nextest alone cannot split a monolithic test function.
- Measure cold fixture extraction independently from warm tests; do not credit
  an extraction step moved elsewhere as a job speedup.

Local raw evidence: `/home/rmccormick/reports/frontend-ci-2026-10-14/`.
Isolated Cargo target: `/home/rmccormick/.cache/frontend-ci-campaign/`.
Local host: 12 logical CPUs, 62 GiB RAM. Python 3.12 is required for fixture
helpers (`tomllib`); the system Python 3.10 cannot run them.

## Coverage/redundancy audit

- Captured-stream parity pins per-chunk deltas at the recorded schedule;
  split sweeps check chunk-size invariance; interleave tests check independent
  request state against captured oracles in both role assignments. These are
  different regression claims, not interchangeable copies.
- Parser-v2 header tests additionally check completion before EOF, native IDs,
  and reset/reuse. Those assertions must survive any scheduling change.
- The guided competing-marker matrix covers seven grammars, three initial
  states, three invalid-payload policies, three output modes, every UTF-8 split,
  and scalar-at-a-time streams. No coverage reduction has been approved.
- Multimodal all-features and default-feature runs check different configurations
  (including no-worker-thread behavior); combining them loses default coverage.
- `test-utils` enables five extra Gemma recovery work-bound assertions; the
  ordinary integration run cannot replace it.
- Protocol schema feature tests repeat inexpensive base tests. Their measured
  execution cost is under one second; most of that step is compilation.
- Shared `common` helper tests are compiled into multiple integration binaries.
  Their execution is negligible; consolidating them would not address the hotspot.
- `cargo check --workspace --all-targets --locked` repeats the type-checking that
  the subsequent default-feature `cargo test --workspace --all-targets --locked`
  already performs. The original check step is 106s, but that is NOT all net
  savings: shared build scripts/proc macros must still be built by the test step.
  Measure job time before attributing a saving.

## Hot paths and follow-up candidates

- Kimi K2's typing path invokes a config-specific regex compiler for each parsed
  block (`parsers/v2/src/tool_calling/v1core/xml/kimi_k2_parser.rs`). Its header
  test crosses 144 header variants with every split and two reuse cycles.
  XML and GLM batch typing also compile config-specific regexes per block.
  Test-profile dependency optimization targets this cost without changing
  production parser ownership or custom-config semantics. A production cache
  would need config-keyed ownership and separate validation.
- The competing-marker test crosses 15,309 combinations before splitting each
  input: seven families × three heads × nine bodies × three tails × three
  starting states × three policies × three output modes. It is one libtest
  scheduling unit. Independent family tests could expose parallelism while
  retaining the complete matrix.
- The split sweep and interleave sweep are also single scheduling units. Merely
  raising `--test-threads` or installing nextest cannot parallelize their loops.
  Family-level partitioning must retain registry completeness and strict
  missing/stale divergence-allowlist checks.
- Cold extraction materializes versioned YAML with Python workers; the MiniMax
  suite is the first test requiring that snapshot in the reference run. A warm
  local verifier profile takes 0.256s, mostly imports/YAML taxonomy parsing.
  Moving extraction to its own CI step improves attribution, not total runtime.
- Build caching remains unmeasured. Cache transfer/save cost matters: local debug
  build artifacts exceed 10 GiB. Do not equate a local warm rebuild with a hosted
  cache speedup.

## Results

Regex-only experiment: all 2,692 default-feature workspace tests pass, 12 ignored,
zero failures. Total binary execution is 384.04s; the incremental rebuild is
124s (not a cold-build comparison). All 1,051,218 competing-marker schedules pass.

| Suite | Local default | Local regex-only | Attribution |
| --- | ---: | ---: | --- |
| Interleave (16,580 pair-schedules) | 114.86s | 36.60s | 78.26s saved, 3.14× |
| Captured stream parity | 7.46s | 4.86s | 2.60s saved, 1.53× |
| Split sweep (1,647 cases; 237,754 chunkings) | Not completed | 73.89s | Reference CI: 397.47s; cross-host, not a controlled speedup |
| Parser-v2 unit binary (740 tests) | Not run | 195.14s | Reference CI: 611.91s; cross-host, not a controlled speedup |
| Unified parity | Not run | 7.69s | Reference CI: 31.26s |
| XML schema coercion | Not run | 19.04s | Reference CI: 25.19s |

Second experiment and hosted baseline/candidate comparisons are in progress.
