<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing

Releases of `dynamo-protocols`, `dynamo-parsers`, `dynamo-parsers-v2`,
`dynamo-tokenizers`, and `dynamo-renderer` to crates.io are automated by
`.github/workflows/release.yml`. This document covers what the workflow does,
what one-time setup it requires, and how to recover when it goes wrong.

## What happens on every push to `main`

1. The `release` workflow runs.
2. **A single blanket `release-plz update`** decides every crate at once. This works — without the version churn it used to cause — because the workspace pins internal crates at **major precision** (`dynamo-protocols = { version = "4" }`, not `"4.0.0"`; see "Why the pins are `"4"`" below). release-plz then:
   - bumps a crate whose **own packaged source** changed, sized from Conventional Commits (only `release_commits` types count: `feat:`/`fix:`/`perf:`/`refactor:`/`sync:`, plus — temporarily, for the dynamo→frontend-crates transition — `chore:`) and validated by `cargo-semver-checks`;
   - re-releases a **dependent** ONLY when a workspace dependency has a **breaking (major)** release. A compatible dependency bump leaves the `"4"` requirement text unchanged, so no dependent is touched (this is what removed the version churn; the old `-p` path-scoped loop existed only because full-precision pins made release-plz cascade on *every* bump). A breaking bump rewrites the pin to `"5"` and cascades a **patch** bump to the dependent — just enough to move its requirement;
   - respects a **manual peg**: if `Cargo.toml`'s version is already ahead of the last published release, release-plz reports *"local version > registry version, only changelog will be updated"* and the publish step ships exactly that version (the fixture-synced release flow). A **first release** (untagged crate) stays at its current version.
3. **Heavy-dependent escalation** (`Escalate heavy dependents on a breaking dependency major` step). release-plz only ever *patch*-cascades a dependent, because it cannot detect that a dependency's types appear in the dependent's **own** public API — "dependency-surface drift", which `cargo-semver-checks` also misses. So for crates listed as **HEAVY** in the workflow, a **major** bump of a listed dependency forces a **major** bump of the dependent (via `release-plz set-version`). Currently the only entry is `dynamo-renderer:dynamo-protocols,dynamo-tokenizers` — renderer exposes `dynamo-protocols` types (`OAIChatLikeRequest::typed_messages`) and re-exports `dynamo-tokenizers` (`renderer/src/lib.rs`). **Add a crate to the HEAVY list when it starts exposing another workspace crate's types in its public API**, or its consumers will get a semver-wrong patch.
4. If any versions changed, the workflow commits them with an informative
   subject listing every crate whose version changed this run — e.g.
   `chore: release dynamo-protocols v5.0.0, dynamo-renderer v3.0.0` (with
   `Signed-off-by:` for DCO) — and pushes to `main`.
5. `release-plz release` always runs (not gated on step 4 — a merge that only
   manually pegged a version produces no bump commit but must still publish).
   It publishes every crate whose `Cargo.toml` version isn't on crates.io yet
   and creates the per-crate git tags; it's a no-op otherwise. GitHub Releases
   are disabled (`git_release_enable = false`); the per-crate `CHANGELOG.md`
   files plus the `*-v*` tags are the release record.
6. The push from step 4 retriggers the workflow. A recursion guard on the
   `chore: release` commit message short-circuits the second run.

## Manual version peg (fixture-synced releases)

To release a crate at a **specific, deliberate version** — e.g. so a conformance fixture snapshot (`conformance/fixtures/`, git-lfs) and the crates.io release carry the same pegged number — edit the crate's `version` in its `Cargo.toml` in your PR (any jump you want: patch, minor, major). On merge, `release-plz update` sees the local version is already ahead of the last published release (*"local version > registry version, only changelog will be updated"*), leaves the number alone, and the publish step ships exactly that version. `Cargo.toml` is the single source of truth: fixture provenance embeds the built crate's version, so both artifacts stay in sync by construction.

The same mechanism covers a crate's **first release**: a crate with no release tag yet stays at its current version — set it manually and merge.

Note: release-plz writes a changelog entry for a pegged version automatically; add extra `CHANGELOG.md` prose in the same PR if the release warrants it.

## Bump policy

release-plz applies SemVer to the Conventional Commits since each crate's last
tag. SemVer treats `0.x` versions specially (the minor slot is the de-facto
breaking position), so the bump depends on whether a crate has reached `1.0.0`.

`dynamo-protocols`, `dynamo-parsers`, `dynamo-tokenizers`, and
`dynamo-renderer` are all at `1.x`+ (standard SemVer):

| Commit                          | Bump from 1.3.0 |
| ------------------------------- | --------------- |
| `fix:`, `perf:`, `refactor:`    | 1.3.1 (patch)   |
| `feat:`                         | 1.4.0 (minor)   |
| `feat!:` / `BREAKING CHANGE:`   | 2.0.0 (major)   |
| `chore:`, `ci:`, `build:`, etc. | no bump         |

`dynamo-parsers-v2` is at `0.x`, where the minor slot is the breaking position (cargo treats `0.1.21 -> 0.2.0` as breaking, `0.1.21 -> 0.1.22` as compatible), so compatible changes bump the patch slot and breaking changes bump the minor slot. Lifecycle note: v1 (`dynamo-parsers`) is interim and will be removed outright once v2 reaches parity; v2 is the ultimate implementation (WIP), so expect its `0.x` line to keep moving while v1 stays quiet. Downstream exact pins like vLLM's `dynamo-parsers-v2 = "=0.1.x"` only move when their owners update them.

## Why the internal pins are `"4"`, not `"4.0.0"`

The `[workspace.dependencies]` entries pin sibling crates at **major precision** (`dynamo-protocols = { path = "protocols", version = "4" }`). This is load-bearing, not a style choice, and it is verified against release-plz's source (`cargo_utils::upgrade_requirement`):

release-plz re-releases a dependent **iff a dependency bump forces that requirement string to be rewritten**, at the precision it is written. With `"4"`, a compatible `4.x` bump leaves the text `"4"` unchanged → the dependent is **not** re-released (no churn). A breaking `4.x → 5.0` rewrites it to `"5"` → the dependent **is** re-released. So major-precision pins express exactly "re-release dependents only on a breaking dependency bump".

**Do not tighten these to `"4.0.0"`.** Full precision rewrites the requirement on *every* dependency bump (patch/minor included), so release-plz re-releases every dependent every time — churning versions of crates whose code never changed. `0.x` crates, if any are ever depended on internally, would use minor precision (`"0.1"`) since `0.x`'s breaking slot is the minor.

Note the tradeoff this accepts: a published dependent's requirement floor is only the major (`^4`), so it no longer auto-tightens to a specific minimum minor. In-workspace builds always use the current sibling via the `path`, and consumers normally take matched versions, so this is a deliberate, low-risk trade for churn-free releases.

## Cascade and heavy-dependent escalation (worked example)

When `dynamo-protocols` has a **breaking** release (`3.x → 4.0.0`):

- `dynamo-parsers` depends on protocols but exposes **none** of its types publicly → release-plz patch-cascades it (`5.1.1 → 5.1.2`) only to move its requirement to `^4`. Correct and sufficient.
- `dynamo-renderer` **exposes** protocols types in its public API and re-exports `dynamo-tokenizers` → it is on the HEAVY list, so the escalation step upgrades release-plz's patch cascade to a **major** (`2.0.0 → 3.0.0`). Without this it would ship a semver-wrong patch, because neither release-plz nor `cargo-semver-checks` can see that a dependency's type change broke renderer's own API (this was validated empirically).

A **compatible** protocols release (`4.0.0 → 4.1.0`) re-releases neither: `^4` still admits it.

### What `cargo-semver-checks` validates

After release-plz picks a bump, it runs `cargo-semver-checks` against the
last published version on crates.io. If the proposed bump is too small for
the API change it sees, the workflow fails before publishing anything.

- **Catches:** removed/renamed pub items, signature changes, narrowed
  visibility, removed trait method defaults, and similar structural breaks
  that were labeled `fix:`/`chore:` instead of `feat!:`.
- **Does NOT catch:** behavioral changes with unchanged signatures, new
  panic conditions, MSRV bumps, dependency-surface drift, some trait
  default-impl subtleties. For these, the contributor must label the commit
  `feat!:` / `fix!:` themselves.

When `cargo-semver-checks` fails, fix the underlying commit: open a PR that
amends or adds a follow-up commit with the correct label (e.g. add a
`feat!:` or `fix!:` with a `BREAKING CHANGE:` trailer), merge, and the next
workflow run will pick up the right bump.

## One-time setup (bootstrap)

These admin actions must be done before the workflow can run end-to-end.

1. **Create the `automated-release` GitHub Environment.** Repo Settings →
   Environments → New environment → name `automated-release`. Configure:
   - **Deployment branches:** Selected branches → `main` only. Prevents
     publishes from PR branches even if a workflow file is added there.
   - **Required reviewers:** leave empty. (Adding reviewers would force a
     manual click on every release, defeating the direct-release flow.)
   - **Wait timer:** leave at 0.
   - **Environment secrets:** the app ID and private key provisioned in step
     3 below should live here, not at repo level, so other workflows can't
     read them.

2. **Configure trusted publishing on crates.io.** For each published crate
   (`dynamo-protocols`, `dynamo-parsers`, `dynamo-parsers-v2`,
   `dynamo-tokenizers`, `dynamo-renderer`), go to
   crates.io → crate Settings → Trusted Publishers → add a GitHub trusted
   publisher with:
   - Repository: `ai-dynamo/frontend-crates`
   - Workflow filename: `release.yml`
   - **Environment: `automated-release`** — binds the OIDC claim to this
     environment so requests from anywhere else are rejected.

3. **Provision release GitHub App credentials (in the `automated-release`
   environment).** The default `GITHUB_TOKEN` cannot push to a protected
   branch nor trigger downstream workflows (CI on the release commit).
   Create a small GitHub App with `Contents: write` permissions and install
   it on only this repo. Then add:
   - **Environment secret:** `RELEASE_APP_ID` with the app's Client ID.
   - **Environment secret:** `RELEASE_APP_PRIVATE_KEY` with the full private
     key file contents, including the begin/end lines.
   The workflow uses `actions/create-github-app-token` to mint a short-lived
   installation token for checkout, pushing release commits, and release-plz
   GitHub API calls.

4. **Branch protection on `main`:** require CI status checks, require
   linear history, require DCO. Add the release GitHub App to the ruleset
   bypass list with "Always allow" so it can push the `chore: release`
   commit directly. Human commits still go through PRs.

5. **Tag protection:** add a tag protection rule matching `*-v*` to
   prevent updates and deletes.

## Recovery: partial publish failure

If the workflow publishes `dynamo-protocols` but fails on
`dynamo-tokenizers` (e.g. crates.io throttling, transient network error,
late-failing `cargo-semver-checks`):

1. The `chore: release` commit is already on `main` and the
   `dynamo-protocols-v<version>` tag exists.
2. Rerun the workflow via **Actions → release → Run workflow**. The
   `workflow_dispatch` trigger bypasses the recursion guard. `release-plz
   release` is idempotent: it checks crates.io and skips already-published
   crates, then retries the failed one.
3. If the rerun still fails, publish manually from a local checkout:
   ```
   cargo publish -p dynamo-tokenizers --token "$CARGO_REGISTRY_TOKEN"
   git tag -s -m "release dynamo-tokenizers <version>" \
       dynamo-tokenizers-v<version>
   git push origin dynamo-tokenizers-v<version>
   ```
   You'll need to recreate the `CARGO_REGISTRY_TOKEN` secret temporarily
   for this, then remove it again.

## Manual hotfix

To ship a fix outside the normal merge flow (e.g. a security patch on a
single crate):

1. Cut a branch from the latest release tag of the affected crate.
2. Make the fix, commit with the appropriate Conventional Commit type
   (`fix:` for a patch, `feat!:` for a breaking emergency).
3. Open a PR to `main`, merge.
4. The workflow handles the rest.

If `main` has unreleased changes you don't want to ship yet, that's a
sign the normal flow is broken — investigate before reaching for branch
gymnastics.

## Upstream sync commits

`sync(<crate>):` commits modify packaged source under `<crate>/src/`, so
they trigger releases. By default this maps to a patch bump. If an upstream
sync introduces a breaking API change, the human running `sync-from-dynamo.sh`
must amend the commit message to `sync!(<crate>):` or add a
`BREAKING CHANGE:` trailer so the right bump is chosen. If
`cargo-semver-checks` catches a structural break that wasn't labeled
correctly, the publish will fail and you'll need to follow up with a
correctly-labeled commit (see "What `cargo-semver-checks` validates"
above).
