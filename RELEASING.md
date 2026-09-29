<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing

Releases of `dynamo-protocols`, `dynamo-parsers`, `dynamo-parsers-v2`, `dynamo-tokenizers`, and `dynamo-renderer` to crates.io are automated by `.github/workflows/release.yml`. `dynamo-mm-preprocessor` is publishable by Cargo but excluded from release-plz until its manual 0.1.0 release is complete. This document covers the workflow, one-time setup, and recovery.

## What happens on every push to `main`

1. The `release` workflow runs.
2. The workflow decides **per crate, path-scoped** (DIS-2414). For each publishable workspace crate it compares the crate's `Cargo.toml` version against the crate's latest release tag (e.g. `dynamo-protocols-v2.0.2`):
   - **version ≠ last tag** → the version was **manually pegged** in a PR. The workflow does not touch it; the publish step ships exactly that version. See "Manual version peg" below.
   - **version == last tag, no changes under the crate's own directory** since that tag → nothing happens. In particular, a change to one crate never re-releases its dependents: all inter-crate deps are caret reqs, so a compatible bump never requires a dependent re-release (the old blanket `release-plz update` cascaded these, churning versions of crates whose code never changed).
   - **version == last tag, changes under the crate's own directory** → `release-plz update -p <crate>` bumps that crate only. Within the crate, release-plz still applies its own two filters: only **packaged contents** count (the `include = [...]` list in the crate's `Cargo.toml`: `src/**/*`, `Cargo.toml`, `README.md` — not `tests/` etc.), and only commits matching `release_commits` in `release-plz.toml` count (`feat:`, `fix:`, `perf:`, `refactor:`). If neither filter passes, no bump is proposed even though the directory changed.
   - **no release tag yet** → the bump step skips this crate. This includes `dynamo-mm-preprocessor` before 0.1.0.
3. If any bumps were proposed, the workflow commits them with an informative
   subject listing every crate whose version changed this run — e.g.
   `chore: release dynamo-protocols v1.4.0, dynamo-renderer v1.3.1` (with
   `Signed-off-by:` for DCO) — and pushes to `main`.
4. `release-plz release` always runs (not gated on step 3 — a merge that only manually pegged a version produces no bump commit but must still publish). It publishes every release-plz-managed crate whose `Cargo.toml` version isn't on crates.io yet and creates the per-crate git tags; it's a no-op otherwise. GitHub Releases are disabled (`git_release_enable = false`); the per-crate `CHANGELOG.md` files plus the `*-v*` tags are the release record.
5. The push from step 3 retriggers the workflow. A recursion guard on the
   `chore: release` commit message short-circuits the second run.

## Manual version peg (fixture-synced releases)

To release a crate at a **specific, deliberate version** — e.g. so a conformance fixture snapshot (`conformance/fixtures/`, git-lfs) and the crates.io release carry the same pegged number — edit the crate's `version` in its `Cargo.toml` in your PR (any jump you want: patch, minor, major). On merge, the workflow sees the version differs from the last release tag, skips any auto-bump for that crate, and publishes exactly that version. `Cargo.toml` is the single source of truth: fixture provenance embeds the built crate's version, so both artifacts stay in sync by construction.

For a crate's **first release**, set its intended version manually: the bump step skips crates with no release tag. See the bootstrap steps below for `dynamo-mm-preprocessor`.

Note: a manual peg skips the auto-changelog; add a `CHANGELOG.md` entry in the same PR if the release warrants one.

## First release of `dynamo-mm-preprocessor`

Cargo permits publication after the enabling PR merges. Release-plz remains disabled for this crate (`release = false`), so the release workflow can run without trying to create it. [crates.io trusted publishing](https://blog.rust-lang.org/2025/07/11/crates-io-development-update-2025-07/) requires a manual first release. An authorized publisher must publish 0.1.0 from a clean checkout at the exact merged `main` commit, using a short-lived API token that permits crate creation. Set `CARGO_REGISTRY_TOKEN` in the environment without adding it to the repository or command line. The tag push needs a bypass of the "Disallow publication of tags except release managers" ruleset. After the publish, add the owners of the other crates with `cargo owner --add <login> dynamo-mm-preprocessor`. Then run:

```bash
cargo publish -p dynamo-mm-preprocessor --dry-run --locked
cargo publish -p dynamo-mm-preprocessor --locked
# Confirm 0.1.0 on crates.io, configure the trusted publisher, and merge
# the PR that removes `release = false`. Then tag the published commit:
git tag -s -m "release dynamo-mm-preprocessor 0.1.0" \
  dynamo-mm-preprocessor-v0.1.0 <published-sha>
git push origin dynamo-mm-preprocessor-v0.1.0
```

Configure the trusted publisher as described below. A separate follow-up PR removes the `release = false` override; subsequent releases are then automated. The tag gives the bump step its 0.1.0 baseline, so push it last. The bump step does not read `release = false`: with the tag pushed and the override still set, the next push that edits the root `Cargo.toml` (every `chore: release` does) runs `release-plz update -p dynamo-mm-preprocessor`, which fails the release job for every crate.

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

`dynamo-parsers-v2` and, once automation is enabled, `dynamo-mm-preprocessor` are at `0.x`, where the minor slot is the breaking position (cargo treats `0.1.21 -> 0.2.0` as breaking, `0.1.21 -> 0.1.22` as compatible), so compatible changes bump the patch slot and breaking changes bump the minor slot. Lifecycle note: v1 (`dynamo-parsers`) is interim and will be removed outright once v2 reaches parity; v2 is the ultimate implementation (WIP), so expect its `0.x` line to keep moving while v1 stays quiet. Downstream exact pins like vLLM's `dynamo-parsers-v2 = "=0.1.x"` only move when their owners update them — another reason auto-bumps stay scoped to crates whose own code changed.

### What `cargo-semver-checks` validates

After release-plz picks a bump, it runs `cargo-semver-checks` against the
last published version on crates.io. If the proposed bump is too small for
the API change it sees, the workflow fails before publishing anything.

- **Catches:** removed/renamed pub items, signature changes, narrowed
  visibility, removed trait method defaults, and similar structural breaks
  that were labeled `fix:` instead of `feat!:`.
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
   `dynamo-tokenizers`, `dynamo-renderer`, and `dynamo-mm-preprocessor` after
   its manual 0.1.0 release), go to
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
