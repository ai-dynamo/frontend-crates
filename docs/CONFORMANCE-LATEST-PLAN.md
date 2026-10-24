# PR #470: latest captures without historical migration

## Scope

Add 76 complete, versionless latest YAML files across the 11 existing Unified families at `conformance/fixtures-unified-v2/families/<family>/<implementation>.yaml`. Keep batch and jail variants. Each latest file must contain complete records, sealed request bindings, and original producer provenance separate from selection metadata. Re-capture local parser outputs from the current source; retain recorded producer identities for peer implementations.

No backfill is needed. Preserve all 634 existing versioned captures, all 11 authored input/GOLDEN files, and every existing legacy archive byte for byte at their existing paths relative to base `0259355e5`. Keep the existing versioned v2 files in this PR; remove them only in later PRs. Historical relocation, archive expansion, new compression, and branch-history rewriting are outside this change.

Replace the candidate's expanded historical tree, 482-file capture layout, and 833 added legacy input files with this smaller storage change. Preserve unrelated viewer/publisher changes and required evidence. Retain pushed candidate `184fe90141ce4d21f44eb273f3cf881b47174413` as a recovery reference. Keep PR #470 in draft.

## Readers and generation

Use the shared path resolver. Latest parser tests discover the versionless files directly without sorting historical releases. Latest generation and parser tests must not read historical captures or use network sockets, and generation must not write capture YAML.

Default generation produces `conformance/CONFORMANCE_v2.html` with latest selections and required JSON/assets only. `--include-history` adds historical selections from unchanged versioned YAML and existing legacy archives; `--history-source` remains available for alternate datasets. Both modes use the same HTML template and JavaScript. The generated catalog controls available releases and comparisons.

Reuse the complete-publication cache first. On a miss, restore compatible converted fragments before assembly. Convert historical YAML only when history was requested and the restored JSON is absent, incompatible, or corrupt. Validate source and payload hashes. Conversion identities must survive commit metadata, renderer changes, and physical relocation; rebuild publication envelopes and presentation as needed. Ordinary CI generates latest only, with an explicit history workflow input. Generated JSON and cache contents remain outside Git.

## Acceptance checks

- [x] Exactly 76 new latest YAML files across 11 families; all existing captures, authored input/GOLDEN files, and archives remain byte identical at their original paths.
- [x] Local parser outputs re-captured from current source; peer producer provenance retained.
- [x] Latest generation and parser tests pass with historical capture access and network sockets denied; capture YAML remains unchanged.
- [x] Warm cache performs zero conversions; a changed family converts independently; historical-only changes cause zero latest conversions; corrupt entries rebuild safely.
- [x] Cold and cached report data are equivalent.
- [x] Latest and history reports pass Chromium release selection, comparison, reload, and download checks.
- [x] Final aggregate diff, including added, deleted, and renamed paths, is measured from the real merge base and remains under 1,000 changed paths.
- [x] Current Markdown instructions describe this scope and explicitly defer backfill, versioned v2 removal, historical moves, and compression.

## Measured implementation results

The local candidate adds exactly 76 latest YAML files across 11 families. All 634 versioned captures, 11 authored input/GOLDEN files, and 74 existing archives match base `0259355e5313fdf9a54eade967d6b19ed5a6029c` byte for byte. No backfill was performed. Versioned v2 removal remains deferred to later PRs.

Re-captured 31 local Dynamo snapshots containing 7,841 records from current local source, with producer source identities retained separately from latest-selection labels. Peer snapshots retain their recorded producer identities. For 258 peer records whose original requests were not retained, the sealed bindings explicitly preserve `original_request_not_retained`; no request was invented or backfilled.

Latest generation and parser tests passed with historical capture reads and network sockets denied. The Rust parser suites passed 20 parity tests and 18 render tests, including captured-versus-live output checks. Capture hashes remained unchanged by generation. The publisher suite passed 70 tests, with two intentionally ignored tests. The final Python and Chromium qualification passed all 230 selected tests, covering canonical latest and history reports, release selection, comparisons, URL reloads, and downloads.

Both warm publications performed zero conversions. Cache regressions cover independent family changes, historical-only and renderer-only identity independence, reordered JSON metadata, and corrupt-entry recovery. Cold and warm latest reports have 220 byte-identical catalog/data resources; history reports have 860. Publication-envelope differences are limited to output path, command, and generation time. All four publications match the current 76 YAML hashes; historical publications also match the retained-history pin.

A temporary Git index measured the complete candidate against the real merge base: 417 changed paths, comprising 314 additions, 91 modifications, and 12 deletions, with no renames. This includes the preserved viewer/publisher work and stays below the 1,000-path acceptance target. The real index remains unstaged. The independent source audit reported no unresolved findings.

Shared owners remain `capture_sources.py` for latest/history path resolution and `codec::canonical_json_bytes` for conversion/publication JSON identities. Searches for `latest_unified_paths`, `--has-latest`, `include_history`, and `canonical_json_bytes` verified the Python, Rust, Node, and CI consumers. In-place historical archival now rejects mutation before writing because historical additions, compression, and backfills are deferred.

Evidence is in `../artifacts/pr470-latest/` beside the worktree container: `inventory.json`, `review-diff.json`, `capture-hashes.json`, `recapture.json`, and `cache-equivalence.json`. The generated latest report is `site/conformance/CONFORMANCE_v2.html`; the history report is `history-site/conformance/CONFORMANCE_v2.html`. The latest canonical report was also regenerated at the worktree's `conformance/CONFORMANCE_v2.html` and remains unstaged.

PR #470 remains draft at pushed recovery candidate `184fe90141ce4d21f44eb273f3cf881b47174413`. Implementation changes are local and uncommitted; no push, historical relocation, new compression, or branch-history rewriting was performed.
