<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing

Releases of `dynamo-protocols`, `dynamo-parsers`, `dynamo-tokenizers`, and
`dynamo-renderer` to crates.io are automated by `.github/workflows/release.yml`. This document
covers what the workflow does, what one-time setup it requires, and how to
recover when it goes wrong.

## What happens on every push to `main`

1. The `release` workflow runs.
2. `release-plz update` inspects commits since the last per-crate release tag
   (e.g. `dynamo-protocols-v0.1.0`) and proposes version bumps for any crate
   whose **packaged contents** changed. Packaged contents are spelled out
   via `include = [...]` in each crate's `Cargo.toml`: `src/**/*`,
   `Cargo.toml`, and `README.md`. Anything outside that list — `tests/`,
   root `README.md`, `.github/`, `scripts/`, `deny.toml`, the
   `examples/dynamo-demo-server/` crate (`publish = false`), and contributor
   docs outside the packaged crate include list — does not trigger a
   release. A second filter, `release_commits` in `release-plz.toml`, only
   considers commits whose messages start with `feat:`, `fix:`, `perf:`,
   `refactor:`, or `sync:`. That filter keeps `chore:` / `ci:` / `build:` /
   `test:` / `docs:` commits — including dependabot's `chore(deps):` bumps
   that only touch `Cargo.lock` — from triggering releases.
3. If any bumps were proposed, the workflow commits them with an informative
   subject listing every crate whose version changed this run — e.g.
   `chore: release dynamo-protocols v1.4.0, dynamo-renderer v1.3.1` (with
   `Signed-off-by:` for DCO) — pushes to `main`, then runs `release-plz
   release` to publish to crates.io and create per-crate git tags. GitHub
   Releases are disabled (`git_release_enable = false`); the per-crate
   `CHANGELOG.md` files plus the `*-v*` tags are the release record.
4. The push from step 3 retriggers the workflow. A recursion guard on the
   `chore: release` commit message short-circuits the second run.

## Bump policy

release-plz applies SemVer to the Conventional Commits since each crate's last
tag. SemVer treats `0.x` versions specially (the minor slot is the de-facto
breaking position), so the bump depends on whether a crate has reached `1.0.0`.

`dynamo-protocols`, `dynamo-parsers`, `dynamo-tokenizers`, and
`dynamo-renderer` are all at `1.x` (standard SemVer):

| Commit                          | Bump from 1.3.0 |
| ------------------------------- | --------------- |
| `fix:`, `perf:`, `refactor:`    | 1.3.1 (patch)   |
| `feat:`                         | 1.4.0 (minor)   |
| `feat!:` / `BREAKING CHANGE:`   | 2.0.0 (major)   |
| `chore:`, `ci:`, `build:`, etc. | no bump         |

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
   - **Environment variables/secrets:** the app client ID and private key
     provisioned in step 3 below should live here, not at repo level, so
     other workflows can't read them.

2. **Configure trusted publishing on crates.io.** For each published crate
   (`dynamo-protocols`, `dynamo-parsers`, `dynamo-tokenizers`,
   `dynamo-renderer`), go to
   crates.io → crate Settings → Trusted Publishers → add a GitHub trusted
   publisher with:
   - Repository: `ai-dynamo/frontend-crates`
   - Workflow filename: `release.yml`
   - **Environment: `automated-release`** — binds the OIDC claim to this
     environment so requests from anywhere else are rejected.

3. **Provision release GitHub App credentials (in the `automated-release`
   environment).** The default `GITHUB_TOKEN` cannot push to a protected
   branch nor trigger downstream workflows (CI on the release commit).
   Create a small GitHub App with `Contents: write` + `Pull requests: write`
   permissions and install it on only this repo. Then add:
   - **Environment variable:** `RELEASE_APP_CLIENT_ID` with the app's
     Client ID.
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
