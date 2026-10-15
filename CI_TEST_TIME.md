# Rust CI test-time campaign

Started 2026-10-14 on `rmccormick/speedup-ci`, based on main
`dcdddc8e` (includes PR #409). Preserve test assertions, fixture inputs,
all registered families, and debug/overflow checks. Measure each change before
claiming a speedup; local timings are not GitHub runner timings.

## Validated outcome

The selected change cuts the hosted Rust job from **32m59s to 13m06s (60.3%)**
and `Run tests` from **26m04s to 7m27s (71.4%)**, against a fresh unchanged-main
baseline. All 2,692 tests pass, with the same 12 ignored tests. No test inputs,
assertions, production source, or feature coverage were removed. The reference
PR job had 2,677 tests; main added 15 tests before this campaign's frozen baseline.

Final configuration: optimize `regex`, `regex-automata`, and `regex-syntax` at
level 3 and `dynamo-parsers-v2` at level 2 in the test profile; retain inherited
debug assertions and overflow checks. Remove the duplicate workspace check.
Dev/release profiles are unchanged. Feature tests, doctests, all-feature Clippy,
and publish checks all pass in the final hosted job.

### Hosted experiment ledger

Each row is one fresh hosted job with no target-directory cache. These are
observations, not repeated-run confidence intervals; fixture extraction and
runner variation affect attribution. All five default workspace runs pass
2,692 tests and ignore 12.

| Candidate / commit | Run tests | Test compilation | Binary execution | Whole Rust job |
| --- | ---: | ---: | ---: | ---: |
| [Baseline](https://github.com/ai-dynamo/frontend-crates/actions/runs/38665623886/job/116196807044), `dcdddc8e` | 26m04s | 3m27s | 1,356.54s | 32m59s |
| [Regex only](https://github.com/ai-dynamo/frontend-crates/actions/runs/38666562731/job/116199754395), `117502e2` | 13m43s | 3m35s | 607.44s | 20m39s |
| [Regex + parser level 1, check removed](https://github.com/ai-dynamo/frontend-crates/actions/runs/38667666660/job/116203218429), `5fcb493c` | 8m53s | 3m20s | 331.93s | 14m35s |
| [Broad test level 1, rejected](https://github.com/ai-dynamo/frontend-crates/actions/runs/38667805053/job/116203645445), `2687303a` | 13m45s | 9m36s | 248.69s | 23m27s |
| [Regex + parser level 2, selected](https://github.com/ai-dynamo/frontend-crates/actions/runs/38668633673/job/116206260240), `84d2c56e` | **7m27s** | 3m54s | **212.67s** | **13m06s** |

- Regex optimization alone saves **12m21s / 47.4%** of `Run tests` and
  **12m20s / 37.4%** of the job against the frozen baseline.
- Adding parser level 1 and removing the check saves another **4m50s** in
  `Run tests` and **6m04s** in the job. These changes were combined in the hosted
  arm; their individual net job contributions are not isolated. Local parser-only
  execution comparisons below provide separate evidence for that optimization.
- Parser level 2 saves another **1m26s** in `Run tests` and **1m29s** overall
  versus the level-1 candidate. Cold compilation increases by 34s, but execution
  falls by 119.26s. MiniMax setup is similar in these two arms (66.57s / 66.98s).
- Broad optimization loses: its 9m36s cold test compilation outweighs execution
  gains and also increases later feature-build costs. It is not the final setting.

### Where the execution saving comes from

| Suite | Fresh baseline | Selected | Reduction |
| --- | ---: | ---: | ---: |
| Parser-v2 units (740 tests) | 604.49s | 49.51s | 554.98s |
| Split sweep | 395.53s | 31.44s | 364.09s |
| Interleave | 124.50s | 16.46s | 108.04s |
| MiniMax including cold fixture setup | 94.45s | 66.98s | 27.47s; not isolated as a compiler-profile effect |

The first three account for 1,027.11s of the observed 1,143.87s execution
reduction. The selected step now spends 3m54s compiling and 3m33s executing;
additional execution-only improvements have a smaller whole-job ceiling.

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
- Second candidate: additionally optimize `dynamo-parsers-v2` at
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
- Legacy v1 and v2's embedded `v1core` share examples, but exercise separately
  compiled implementations that can drift. The entire v1 unit binary costs only
  8.29s in the reference job. Deleting those checks is a poor tradeoff.
- `cargo check --workspace --all-targets --locked` repeats the type-checking that
  the subsequent default-feature `cargo test --workspace --all-targets --locked`
  already performs. The original check step is 106s, but that is NOT all net
  savings: shared build scripts/proc macros must still be built by the test step.
  Measure job time before attributing a saving.

## Hot paths and follow-up candidates

- Kimi K2's typing path invokes a config-specific regex compiler for each parsed
  block (`parsers/v2/src/tool_calling/v1core/xml/kimi_k2_parser.rs:43`). Its header
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
- Splitting CI into jobs would overlap independent work but duplicate compilation
  without a build-artifact handoff. Nextest can overlap binaries and expose
  per-test timings, but adds process startup and repeats process-local caches.
  The controlled nextest experiment below measures this tradeoff without
  changing the CI runner or shrinking the matrices.

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

Adding parser-v2-only `opt-level = 1` passes the same 2,692 tests and reduces
binary execution to 268.10s: another 115.94s (30.2%) below regex-only. Its
incremental rebuild is 26.39s. Per-suite times: parser-v2 units 113.20s, split
sweep 56.26s, interleave 28.87s, Unified parity 5.88s, XML schema 14.59s.
This was committed as `c8497574`; the hosted ledger compares its combined
build-plus-test cost against the other candidates.

The first committed candidate is `117502e2` (regex-only). Its hosted run is
[38666562731](https://github.com/ai-dynamo/frontend-crates/actions/runs/38666562731),
Rust job `116199754395`.

The broader test-profile `opt-level = 1` trial tested whether optimizing the
remaining dependencies repays the extra build cost. It did not in hosted CI.
Debug assertions and overflow checks stay enabled in every experiment.

The duplicate workspace check is removed starting with `5fcb493c`. `cargo test`
already compiles exactly the same workspace, targets, lockfile, and default
features; the separate all-feature Clippy and feature-specific tests remain.
This removes redundant work, but its net saving must not be equated to the
baseline check step's duration because some compilation moves into `Run tests`.

Hosted narrow candidate: `5fcb493c`,
[run 38667666660](https://github.com/ai-dynamo/frontend-crates/actions/runs/38667666660),
Rust job `116203218429` (regex + parser-v2 optimization, redundant check removed).

The broad profile also passes all 2,692 tests. Its rebuild is 273s. Execution
totals 217.85s, including an anomalous 71.88s in the Git-based capture-selector
fixture test (normally about 2s). A regex/broad reversed-order repeat passed
four times in 1.18–1.21s; the outlier did not reproduce. Keep the raw aggregate
rather than subtracting it; its cause was not established. Parser-v2 units take
54.94s, split sweep 42.05s, interleave 14.62s,
Unified parity 3.78s, and XML schema coercion 3.37s.

Hosted broad candidate: `2687303a`,
[run 38667805053](https://github.com/ai-dynamo/frontend-crates/actions/runs/38667805053),
Rust job `116203645445`. It differs from the narrow candidate only in the scope
of test-profile optimization. Select on total CI job time, not execution alone.

### Ordered repeats and stack sampling

Standalone saved test executables, fresh process per measurement, CPU affinity
0–3, order default → regex → parser-only → broad → broad → parser-only → regex
→ default. Every invocation passed its original assertions. Means of two runs:

| Test | Default | Regex | Parser-only level 1 | Broad level 1 |
| --- | ---: | ---: | ---: | ---: |
| Kimi header empty-suffix split/reuse | 1.8973s | 0.2585s | 0.2515s | 0.2381s |
| Gemma native-looking one-byte stream | 32.9435s | 32.5631s | 23.5471s | 5.5422s |

Six debugger stack samples of the default Gemma test caught whole-buffer
character searches in `GuidedState::drain` (`unified/mod.rs:4245`), reverse
character scans in `partial_bare_opener_suffix_len` (`tool_calling/gemma4.rs:906`),
and control-marker search. The existing linear-work assertions count selected
boundary scans, not every search in these paths. Preserve this test; investigate
those scans as a production-performance follow-up rather than shrinking its
input to make CI green. Sampling was diagnostic only, separate from timings;
the debugger-owned process was terminated after six samples.

[Cargo's generic-code sharing rules](https://doc.rust-lang.org/cargo/reference/profiles.html#overrides-and-generics)
provide a reason to try parser-only level 2: it avoids importing generic code
instantiated by unoptimized crates. The final narrow level-2 trial passes all
2,692 tests locally with 189.97s binary execution and a 39.67s incremental
rebuild. Parser-v2 units take 53.31s, split sweep 50.47s, and interleave 23.76s.
Its hosted result wins without optimizing the whole dependency graph. The
experiment measures the complete setting; it does not attribute all gains
specifically to generic-code sharing.

### Controlled runner parallelism experiment

Use the selected profile, warm compiled artifacts and the same verified immutable
fixture snapshot. Pin both runners to CPUs 0–3 (four physical cores) and four test
threads. Run fresh processes in Cargo → nextest → nextest → Cargo order.
Nextest 0.9.140 and Cargo 1.96.1 each discover 2,704 tests; every run passes all
2,692 enabled tests and skips the same 12.

| Runner | First wall time | Reversed-order wall time | Mean |
| --- | ---: | ---: | ---: |
| Cargo/libtest | 201.04s | 242.52s | 221.78s |
| nextest | 136.11s | 131.31s | 133.71s |

Nextest saves **88.07s / 39.7%** on these local means. Both ordered pairs favor
it; the Cargo spread is substantial, so retain both samples. This excludes cold
compilation, fixture extraction, and nextest installation. It is promising
follow-up evidence, not a measured hosted-job speedup. The existing fixture
lock/verifier handled this warm concurrent run; cold extraction concurrency
still needs validation before changing CI.

Reproduce with `CARGO_TARGET_DIR` pointing to the isolated build directory,
`RUSTC_WRAPPER=` and `CONFORMANCE_FIXTURES_ROOT` pointing to the verified snapshot:

```bash
taskset -c 0-3 cargo test --workspace --all-targets --locked -- --test-threads=4
taskset -c 0-3 cargo nextest run --workspace --all-targets --locked --test-threads=4
```

### Ranked continuation work

1. **Trial nextest in hosted CI:** compare complete cold jobs, including installer
   and fixture materialization, against the selected baseline. Keep separate
   doctests and feature configurations. Local evidence above establishes a useful
   candidate but does not establish its hosted net gain.
2. **Partition the expensive matrices by family:** the competing-marker test
   (`parsers/v2/src/unified/mod.rs:5694`) remains one scheduling unit. Preserve all
   1,051,218 schedules, all 1,647 split cases / 237,754 chunkings, all 16,580
   interleave pair-schedules, registry completeness, and allowlist checks.
   Measure partitioning on top of the chosen runner; do not add overlapping gains.
3. **Optimize cold fixture materialization:** the selected MiniMax binary now
   accounts for 66.98s / 31% of execution, versus 1.24s in a warm local run.
   Profile YAML decoding/encoding and filesystem work before choosing a fix;
   compare the materialized outputs and complete verification results. No emitter
   replacement or cold extraction speedup was measured in this campaign.
4. **Measure build caching:** test transfer/save time and cache invalidation across
   representative changes. Compilation is now over half of `Run tests`; neither
   the artifact size nor a warm local build justifies an assumed CI saving.
5. **Investigate production parser hot paths:** config-keyed regex reuse and
   Gemma whole-buffer scans have evidence above, but need independent correctness
   and performance experiments. Do not delete their tests to hide the costs.

### Evidence and validation

The raw evidence directory contains `hosted-{baseline,regex,narrow,broad,final}`
job metadata, logs, and parsed binary timings; `paired-tests.json` and sampling
logs; local profile run logs; and `runner-comparison.json`, its driver script,
and all four runner logs. Incremental local rebuild times use different scopes
and must not be interpreted as cold-build comparisons.

Final hosted Rust and conformance-table jobs succeed; branch-only Pages deployment
is skipped as expected. Local formatting and whitespace checks pass. Final code
is the tested `84d2c56e`; subsequent campaign-report updates do not change the
profile or workflow. The original checkout is preserved in a separate worktree.
