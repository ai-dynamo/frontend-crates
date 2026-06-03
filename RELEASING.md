<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing

Releases of `dynamo-protocols`, `dynamo-parsers`, `dynamo-tokenizers`, and
`dynamo-renderer` to crates.io are automated by `.github/workflows/post-merge.yml`. This document
covers what the workflow does, what one-time setup it requires, and how to
recover when it goes wrong.

## What happens on every push to `main`

1. The `post-merge` workflow runs.
2. `release-plz update` inspects commits since the last per-crate release tag
   (e.g. `dynamo-protocols-v0.1.0`) and proposes version bumps for any crate
   whose **packaged contents** changed. Packaged contents are spelled out
   via `include = [...]` in each crate's `Cargo.toml`: `src/**/*`,
   `Cargo.toml`, and `README.md`. Anything outside that list — `tests/`,
   root `README.md`, `.github/`, `scripts/`, `deny.toml`, the
   `examples/dynamo-demo-server/` crate (`publish = false`), and per-crate
   contributor docs like `CLAUDE.md` / `PARSER_CASES.md` — does not trigger a
   release. A second filter, `release_commits` in `release-plz.toml`, only
   considers commits whose messages start with `feat:`, `fix:`, `perf:`,
   `refactor:`, or `sync:`. That filter keeps `chore:` / `ci:` / `build:` /
   `test:` / `docs:` commits — including dependabot's `chore(deps):` bumps
   that only touch `Cargo.lock` — from triggering releases.
3. If any bumps were proposed, the workflow commits them with an informative
   subject listing every crate whose version changed this run — e.g.
   `chore: release dynamo-protocols v1.4.0, dynamo-renderer v0.1.1` (with
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

`dynamo-protocols`, `dynamo-parsers`, and `dynamo-tokenizers` are at `1.x`
(standard SemVer):

| Commit                          | Bump from 1.3.0 |
| ------------------------------- | --------------- |
| `fix:`, `perf:`, `refactor:`    | 1.3.1 (patch)   |
| `feat:`                         | 1.4.0 (minor)   |
| `feat!:` / `BREAKING CHANGE:`   | 2.0.0 (major)   |
| `chore:`, `ci:`, `build:`, etc. | no bump         |

`dynamo-renderer` is still at `0.x` (pre-1.0 rules):

| Commit                                | Bump from 0.1.0 |
| ------------------------------------- | --------------- |
| `feat:`, `fix:`, `perf:`, `refactor:` | 0.1.1 (patch)   |
| `feat!:` / `BREAKING CHANGE:`         | 0.2.0 (minor)   |
| `chore:`, `ci:`, `build:`, etc.       | no bump         |

`renderer` moves to the standard-SemVer column once it's bumped to `1.0.0`.

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

Coordination with the upstream `ai-dynamo/dynamo` monorepo: dynamo has
**already published 1.2.0** of `dynamo-protocols`, `dynamo-parsers`, and
`dynamo-tokenizers` to crates.io. This repo's local Cargo.toml versions are
**1.3.0**, so the first publish from this repo is a clean minor bump over that
1.2.0 baseline. Going forward, **the monorepo must stop publishing these crate
names** — once this repo is the publisher, two publishers would collide on the
same names/versions.

Crate-existence note for trusted publishing: `dynamo-protocols`,
`dynamo-parsers`, and `dynamo-tokenizers` already exist on crates.io, so trusted
publishing works for them as soon as it's configured. **`dynamo-renderer` does
NOT yet exist on crates.io**, and trusted publishing cannot create a brand-new
crate — it must be first-published manually (see step 3a).

1. **Create the `automated-release` GitHub Environment.** Repo Settings →
   Environments → New environment → name `automated-release`. Configure:
   - **Deployment branches:** Selected branches → `main` only. Prevents
     publishes from PR branches even if a workflow file is added there.
   - **Required reviewers:** leave empty. (Adding reviewers would force a
     manual click on every release, defeating the direct-release flow.)
   - **Wait timer:** leave at 0.
   - **Environment secrets:** the `RELEASE_PLZ_TOKEN` provisioned in step
     5 below should live here, not in repo-level secrets, so other
     workflows can't read it.

2. **Configure trusted publishing on crates.io.** For each of the four crates
   (`dynamo-protocols`, `dynamo-parsers`, `dynamo-tokenizers`, and —
   after its manual first publish, step 3a — `dynamo-renderer`), go to
   crates.io → crate Settings → Trusted Publishers → add a GitHub trusted
   publisher with:
   - Repository: `ai-dynamo/frontend-crates`
   - Workflow filename: `post-merge.yml`
   - **Environment: `automated-release`** — binds the OIDC claim to this
     environment so requests from anywhere else are rejected.

3. **1.2.0 baseline (already satisfied).** release-plz compares local
   Cargo.toml versions against the crates.io registry, so the registry must be
   at the "previous" version before this repo's first publish. dynamo has
   already published `dynamo-protocols`/`dynamo-parsers`/`dynamo-tokenizers` at
   1.2.0, so this prerequisite is met and the first publish here will be 1.3.0.

3a. **Bootstrap `dynamo-renderer` (manual first publish).** renderer is new to
   crates.io and stays at `0.1.0` (not bumped). Trusted publishing can't create
   a new crate, so publish it once by hand, then wire its trusted publisher:
   - Ensure `dynamo-protocols@1.3.0` and `dynamo-tokenizers@1.3.0` are live
     first — renderer's published manifest pins them at `^1.3.0`, so its verify
     build needs them present. (They publish via OIDC on the first automated
     run, or publish them manually up front.)
   - `cargo publish -p dynamo-renderer` with a crates.io token that has the
     `publish-new` scope.
   - Add renderer's trusted publisher (same fields as step 2).
   - To keep the *first* automated run from failing on the not-yet-published
     renderer, either finish this bootstrap before the pipeline runs, **or**
     temporarily add `[[package]]` / `name = "dynamo-renderer"` /
     `release = false` to `release-plz.toml` for that run and remove it after.

4. **Seed baseline release tags at the 1.2.0-sync commit.** release-plz
   uses per-crate git tags to scope "what commits are new since the last
   release" when generating the changelog. Tag the commit in this repo's
   history that corresponds to the 1.2.0 sync state from dynamo (i.e. the
   most recent `sync(...)` or `chore: sync from dynamo @ <sha>` commit
   whose source matches what dynamo published as 1.2.0). For example:
   ```
   BASELINE_SHA=<commit-sha>   # the 1.2.0-sync commit
   git tag -s -m "baseline dynamo-protocols 1.2.0"  -a dynamo-protocols-v1.2.0  "$BASELINE_SHA"
   git tag -s -m "baseline dynamo-parsers 1.2.0"    -a dynamo-parsers-v1.2.0    "$BASELINE_SHA"
   git tag -s -m "baseline dynamo-tokenizers 1.2.0" -a dynamo-tokenizers-v1.2.0 "$BASELINE_SHA"
   git push origin --tags
   ```
   If no clean 1.2.0-sync commit exists (e.g. main has already drifted
   past 1.2.0 by mixed source + non-source changes), pragmatic fallback:
   tag the closest sync commit and accept that the first changelog will
   include a small amount of extra context. Don't tag at HEAD — that
   would suppress the 1.3.0 release entirely.

5. **Provision a `RELEASE_PLZ_TOKEN` secret (in the `automated-release`
   environment).** The default `GITHUB_TOKEN`
   cannot push to a protected branch nor trigger downstream workflows
   (CI on the release commit). Two options:
   - **GitHub App (preferred for a public repo):** create a small app with
     `Contents: write` + `Pull requests: write` permissions, install it on
     this repo, and use `actions/create-github-app-token` at the top of the
     workflow to mint a short-lived token. Replace `secrets.RELEASE_PLZ_TOKEN`
     references with the app-token output.
   - **Classic PAT:** scope `repo` + `workflow`, owned by a service
     account, stored as `RELEASE_PLZ_TOKEN`. Rotate periodically.

6. **Branch protection on `main`:** require CI status checks, require
   linear history, require DCO. Add the bot identity to the bypass list so
   it can push the `chore: release` commit directly. Human commits still
   go through PRs.

7. **Tag protection:** add a tag protection rule matching `*-v*` to
   prevent updates and deletes.

## Recovery: partial publish failure

If the workflow publishes `dynamo-protocols` but fails on
`dynamo-tokenizers` (e.g. crates.io throttling, transient network error,
late-failing `cargo-semver-checks`):

1. The `chore: release` commit is already on `main` and the
   `dynamo-protocols-v<version>` tag exists.
2. Rerun the workflow via **Actions → post-merge → Run workflow**. The
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
