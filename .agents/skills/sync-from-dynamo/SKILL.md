---
name: sync-from-dynamo
description: Resolve sync drift between frontend-crates and ai-dynamo/dynamo. Pulls lib/{protocols,renderer} src+tests up to dynamo main, fixes the diverged-Cargo.toml breakage the new code introduces, verifies the full CI gate, and opens a sync PR. Use when sync-check fails, a `sync-drift` issue is open, or someone asks to "sync from dynamo". NOTE: parsers/, tokenizers/, and conformance/utils/ are NOT in scope (frontend-crates-owned) — see PARSERS-SYNC.md.
---

# Skill: Sync from dynamo

## Purpose

`frontend-crates` mirrors two dynamo library crates (`lib/protocols`, `lib/renderer`). (`tokenizers/` and `parsers/` were detached and are now frontend-crates-owned.) An hourly `sync-check` workflow opens a `sync-drift` issue when dynamo `main` advances past what this repo mirrors. This skill restores the mirror: apply the sync, fix whatever the new upstream code needs to build/test standalone, prove the CI gate passes, and open a PR.

**`parsers/` and `conformance/utils/` are not part of this automated sync.** Do not include them here. For the temporary v1 parser/harness sync and the post-release migration plan, follow `PARSERS-SYNC.md`.

## When to Use

- The `sync-check` CI job is red, or a `sync-drift` issue is open.
- Someone asks to "sync from dynamo" / "fix the sync" / "pull latest dynamo into frontend-crates".

**Not this skill:** someone asks to sync the parser crate or conformance/utils → use `PARSERS-SYNC.md` instead.

## Key facts

- **Only `src/` and `tests/` are synced** for both crates. Each crate's `Cargo.toml`, `README.md`, `CLAUDE.md`, `AGENTS.md` are **intentionally diverged** (inlined for standalone publishing) — the script flags them as `NOTE:` lines and never overwrites them.
- **New upstream code is the usual source of breakage.** Synced `src/` may reference a dependency or an `async-openai` feature the diverged `Cargo.toml` doesn't enable yet. The script syncs code verbatim; you patch the diverged `Cargo.toml`s so it builds.
- **Never hand-edit synced files** (`src/`, `tests/`) to make them build — they'll just drift again next sync. Fix the local-owned side instead (workspace + crate `Cargo.toml`).

## Workflow

### 1. Get a fresh dynamo checkout

```bash
git -C /ephemeral clone --depth 1 https://github.com/ai-dynamo/dynamo.git dynamo-sync
DYN=/ephemeral/dynamo-sync
git -C "$DYN" rev-parse --short HEAD   # note the SHA for the branch/commit/PR
```

### 2. See the drift (dry run)

```bash
scripts/sync-from-dynamo.sh "$DYN"   # exit 1 = code drift; 0 = in sync
```

Read the itemized changes. `NOTE:` lines are intentional divergence — ignore them. `>f`, `+++`, `cd+++` lines under `src/`/`tests/` are real code/file changes that will be applied.

### 3. Branch and apply

```bash
git checkout -b keivenchang/sync-dynamo-<short-sha>
scripts/sync-from-dynamo.sh --apply "$DYN"
git status   # new files (e.g. new modules, fixtures) show as untracked
```

### 4. Build — fix the diverged Cargo.toml / fixtures

```bash
cargo build --workspace
```

Resolve failures by patching the **local-owned** side, not synced code:

- **`unresolved import async_openai::types::<x>`** — dynamo enabled a new `async-openai` feature. Diff the feature lists and add the missing one to `protocols/Cargo.toml`: `diff "$DYN/lib/protocols/Cargo.toml" protocols/Cargo.toml`
- **`unresolved import <crate>` / `unlinked crate <x>`** — new code pulled in a dependency. Add it to the workspace `[workspace.dependencies]` in `./Cargo.toml` (match dynamo's version: `grep '^<crate>' "$DYN/Cargo.toml"`) and to the consuming crate's `Cargo.toml` (`<crate>.workspace = true`).

### 5. Run the full CI gate locally

Mirror `.github/workflows/ci.yml` exactly — these are the merge gates:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo check --workspace --all-targets --locked
cargo test --workspace
cargo deny --all-features check bans licenses
cargo machete                       # cargo install cargo-machete@0.9.2 --locked
scripts/sync-from-dynamo.sh "$DYN"  # must now exit 0
```

`--locked` matters: the build updates `Cargo.lock`, which is committed. New deps must pass `cargo deny` (license/bans) and be actually used (`cargo machete`).

### 6. Commit and PR

```bash
git add -A
git commit -m "chore: sync from dynamo @ <short-sha>"   # follow prior sync commits
git push -u origin HEAD
gh pr create --base main --title "chore: sync from dynamo @ <short-sha>" --body "Resolves #<issue>. ..."
```

The `sync-drift` issue auto-closes on the next hourly `sync-check` run once `main` is back in sync (after merge).

## Notes

- The drift issue's SHA may be stale by the time you run — sync to current dynamo `main` HEAD, not the SHA in the issue.
- No `Co-Authored-By` lines in commits/PRs for this repo.
- Commit message convention from git history: `chore: sync from dynamo @ <short-sha>`.
- If a `NOTE:`-flagged file (README/CLAUDE/AGENTS) has a substantive upstream change worth carrying over, that's a separate manual review — not part of the automated sync. Diff it and decide deliberately.
