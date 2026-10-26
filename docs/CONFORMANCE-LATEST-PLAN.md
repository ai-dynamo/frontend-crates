# PR #470: latest captures and Unified-only conformance

## Baseline and preserved evidence

The base is `4a161dffe7749dbde417df81614bc018e627d53d`. The latest store contains 77 complete versionless YAML files across 12 families. Each family/implementation file contains complete records, sealed request bindings, and original producer provenance separate from selection metadata. Stream, batch, and jail-buffer identities remain distinct; missing measurements remain explicit.

Direct Git-blob comparison preserves all 804 baseline evidence paths: 713 versioned capture YAMLs, 12 authored input/GOLDEN documents, 77 archives, and two Beam YAMLs. Existing evidence stays at its original path. Backfill, versioned-v2 removal, historical moves, archive expansion, and new compression remain deferred.

## Active workflow

Unified v2 is the only active conformance report and executor. Deprecated batch, stream, batch-on-stream, and Reasoning generators, runners, renderers, and tests are retired. Existing Dynamo v1 observations measured against Unified requests remain selectable comparisons. The retained split scenarios preserve their IDs and observations; the old archive-import writer is retired. Parser-library APIs and unrelated workspace checks are unchanged.

The viewer has seven runtime owners plus the existing DOM/detail/coloring boundaries. Development YAML stays outside production. One Node qualification CLI owns status, model, snapshot, and marker checks; six Node suites replace 25 test files. Shared Python owners cover request normalization, peer compatibility, inventory/coverage, migrations, and evidence verification. Historical readers and independent publication validation remain separate from production publishing.

## Publication contract

Default output is one self-contained `index.html`, with latest data and downloads embedded. Full history is a user option to add sidecar JSON beside byte-identical HTML. HTTP viewing automatically discovers a matching history publication. Missing sidecars leave latest usable; local files make zero external reads. Historical results load on demand, preserve request/provenance bindings, and cannot overwrite a newer selection.

Reuse complete publications before staging, then validated family fragments before conversion. Latest generation and parser tests must pass with history reads and network sockets denied, and generation must preserve capture YAML. Warm conversion reuse, independent family invalidation, historical-only isolation, and corrupt-cache recovery remain required.

## Rebase qualification

The new base includes the single-family generic-case audit from #469. Its generator variants and independent request/schema assertions remain active. Generic displayed columns must span multiple families; the two default-reasoning cases retain their IDs in the named Nemotron v3 section. Both null-reference probes and mixed-null fields apply across all native families. The shared JavaScript model preserves every applicable leaf through folding, and retained manifest pins remain independent of latest-only generation.

Current Dynamo latest snapshots are refreshed against the rebased authored requests. Existing peer producer identities and request bindings remain unchanged; absent peer measurements stay explicit. Final report sizes, timings, test results, exact-tree audit and hosted CI belong to the final rebased delivery receipt. Results from the previous head do not establish current-head validation.
