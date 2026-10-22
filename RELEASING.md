<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing

Releases of `dynamo-protocols`, `dynamo-parsers`, `dynamo-parsers-v2`, `dynamo-tokenizers`, `dynamo-renderer`, and `dynamo-multimodal` to crates.io are automated by `.github/workflows/release.yml`. This document covers the workflow, one-time setup, and recovery.

## Before merging a PR

After a maintainer runs `/ok to test <sha>`, the `pull-request/<number>` mirror push runs the `release / preview` job. It builds the squash commit this PR would land (current `main` plus the PR diff, titled with the PR title) and runs the same bump script as publishing (`.github/scripts/bump-versions.sh`). It has read-only access, no release environment or credentials, and never commits, pushes, or publishes.

If release-plz would generate a major-number change, the job fails and names each affected manifest. Either remove the breaking marker from the PR title, or explicitly edit the intended version and affected dependency requirements in the PR. A committed version is treated as pegged, so release-plz does not touch it. Editing only the PR title does not push to the mirror; rerun the job after changing the title.

The preview assumes squash merges whose commit message is the PR title with a blank body. After merge, the release job publishes whatever was merged.

## What happens on every push to `main`

1. The `release` workflow runs.
2. The workflow decides **per crate, path-scoped** (DIS-2414). For each publishable workspace crate it compares the crate's `Cargo.toml` version against the crate's latest release tag (e.g. `dynamo-protocols-v2.0.2`):
   - **version ≠ last tag** → the version was **manually pegged** in a PR. The workflow does not touch it; the publish step ships exactly that version. See "Manual version peg" below.
   - **version == last tag, no changes under the crate's own directory or root `Cargo.toml`** since that tag → nothing happens. In particular, a change to one crate never re-releases its dependents: all inter-crate deps are caret reqs, so a compatible bump never requires a dependent re-release (the old blanket `release-plz update` cascaded these, churning versions of crates whose code never changed).
   - **version == last tag, changes under the crate's own directory or root `Cargo.toml`** → `release-plz update -p <crate>` bumps that crate only. Within the crate, release-plz still applies its own two filters: only **packaged contents** count (the `include = [...]` list in the crate's `Cargo.toml`: `src/**/*`, `Cargo.toml`, `README.md` — not `tests/` etc.), and only commits matching `release_commits` in `release-plz.toml` count (`feat:`, `fix:`, `perf:`, `refactor:`). If neither filter passes, no bump is proposed even though the directory changed.
   - **no release tag yet** → the bump step skips this crate.
3. If any bumps were proposed, the workflow commits them with an informative
   subject listing every crate whose version changed this run — e.g.
   `chore: release dynamo-protocols v1.4.0, dynamo-renderer v1.3.1` (with
   `Signed-off-by:` for DCO) — and pushes to `main`.
4. `release-plz release` always runs (not gated on step 3 — a merge that only manually pegged a version produces no bump commit but must still publish). It publishes every release-plz-managed crate whose `Cargo.toml` version isn't on crates.io yet and creates the per-crate git tags; it's a no-op otherwise. GitHub Releases are disabled (`git_release_enable = false`); the per-crate `CHANGELOG.md` files plus the `*-v*` tags are the release record.
5. The push from step 3 retriggers the workflow. A recursion guard on the
   `chore: release` commit message short-circuits the second run.

## Manual version peg (fixture-synced releases)

To release a crate at a **specific, deliberate version** — e.g. so a conformance fixture snapshot (`conformance/fixtures/`, git-lfs) and the crates.io release carry the same pegged number — edit the crate's `version` in its `Cargo.toml` in your PR (any jump you want: patch, minor, major). On merge, the workflow sees the version differs from the last release tag, skips any auto-bump for that crate, and publishes exactly that version. `Cargo.toml` is the single source of truth: fixture provenance embeds the built crate's version, so both artifacts stay in sync by construction.

For a crate's **first release**, set its intended version manually: the bump step skips crates with no release tag. See the bootstrap steps below.

Note: a manual peg skips the auto-changelog; add a `CHANGELOG.md` entry in the same PR if the release warrants one.

## First release of a new crate

After the PR that makes a new crate publishable merges, keep its `release = false` entry in `release-plz.toml` until the first version is on crates.io. [crates.io trusted publishing](https://blog.rust-lang.org/2025/07/11/crates-io-development-update-2025-07/) requires a manual first release. An authorized publisher must publish from a clean checkout at the exact merged `main` commit and record that commit as `<published-sha>`. Use a short-lived API token that permits crate creation. Set `CARGO_REGISTRY_TOKEN` in the environment without adding it to the repository or command line, then run:

```bash
cargo publish -p <crate> --dry-run --locked
cargo publish -p <crate> --locked
```

Confirm the first version is available on crates.io, add the other crate owners with `cargo owner --add <login> <crate>`, and configure its trusted publisher as described below. Then merge the separate PR that removes its `release = false` entry. Only after that PR merges, push the signed tag for the published commit (the tag push needs a bypass of the "Disallow publication of tags except release managers" ruleset):

```bash
git tag -s -m "release <crate> <version>" <crate>-v<version> <published-sha>
git push origin <crate>-v<version>
```

The tag gives the bump step its first-version baseline; subsequent releases are automated. Push it last because the bump step does not read `release = false`: with the tag pushed and the override still set, the next push that edits the root `Cargo.toml` runs `release-plz update -p <crate>`, which fails the release job for every crate.

## Bump policy

Major versions are maintainer-controlled. To request a major release, explicitly edit the crate version and any affected workspace dependency requirements in a PR. The workflow accepts that committed version. The `release / preview` job rejects any major-number change generated by release-plz before merge.

For crates at `1.x` or later:

| Commit | Automatic result from 1.3.0 |
| --- | --- |
| `fix:`, `perf:`, `refactor:` | 1.3.1 (patch) |
| `feat:` | 1.4.0 (minor) |
| `feat!:` / `BREAKING CHANGE:` | Preview fails: drop the marker or manually peg the intended version |
| `chore:`, `ci:`, `build:`, etc. | No bump |

Use ordinary `feat:` commits for features intended for a minor release, even when Rust API changes require consumers to migrate. Document those migrations in the PR and release notes. A breaking commit marker still makes release-plz propose a major bump, which the preview rejects; it does not silently convert that proposal to a minor bump. A manual minor peg also bypasses automatic bump selection.

For `0.x` crates, existing release-plz behavior is unchanged: compatible changes bump the patch slot and breaking commit markers bump the minor slot. Moving from `0.x` to `1.x` requires a manual peg.

### No API checks

Release-plz's automatic API check is disabled in `release-plz.toml`, and the workflow does not run `cargo-semver-checks`. This policy intentionally permits Rust API breaks in minor releases. Consumers using caret requirements may receive an update that requires source changes; consumers needing explicit upgrade control should use exact version pins.

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
   `dynamo-tokenizers`, `dynamo-renderer`, and `dynamo-multimodal`), go to
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

4. **Branch protection on `main`:** require `rust` and `release / preview`, require branches to be up to date, require linear history, and require DCO. Add the release GitHub App to the ruleset bypass list with "Always allow" so it can push the `chore: release` commit directly. Human commits still go through PRs.

5. **Tag protection:** add a tag protection rule matching `*-v*` to
   prevent updates and deletes.

## Recovery: partial publish failure

If the workflow publishes `dynamo-protocols` but fails on
`dynamo-tokenizers` (e.g. crates.io throttling, transient network error):

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
   (`fix:` for a patch; explicitly peg the version for a breaking emergency).
3. Open a PR to `main`, merge.
4. The workflow handles the rest.

If `main` has unreleased changes you don't want to ship yet, that's a
sign the normal flow is broken — investigate before reaching for branch
gymnastics.
