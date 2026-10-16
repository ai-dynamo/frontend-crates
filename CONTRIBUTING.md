<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to frontend-crates

Thank you for your interest in contributing to `frontend-crates`!

This repository contains independently published Rust crates for protocols, rendering, tokenization, and parsing that can be adopted on their own to build OpenAI/Anthropic-compatible inference servers, plus a demo server wiring them together.

## Source of Truth

This repository is the source of truth for its published crates. [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) consumes released crate versions directly; source code is not mirrored or synced between the repositories.

What this means for contributors:

- **Changes to crate source code** under `protocols/`, `renderer/`, `tokenizers/`, or `parsers/` should be opened as PRs here.
- **Changes to Dynamo integration code** that consumes these crates should be opened against [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo).
- **Changes to repository scaffolding** — crate `Cargo.toml` metadata, `examples/dynamo-demo-server/`, `scripts/`, `.github/`, and docs in this repo — should be opened as PRs here.

If you're not sure where a change belongs, open an issue in either repository and we'll route it.

## How to Contribute

### Plan Large Changes Before Implementation

This workflow aligns with Dynamo's [contribution guide](https://github.com/ai-dynamo/dynamo/blob/05e984e968c9a30fea0d574359d728e9c86768bb/docs/fern/pages/community/contributing/contribution-flow.mdx#L12) and [Contribution Request format](https://github.com/ai-dynamo/dynamo/blob/05e984e968c9a30fea0d574359d728e9c86768bb/.github/ISSUE_TEMPLATE/contribution_request.yml).

For external contributions of **100 or more changed lines (additions plus deletions)**, new features, or substantial refactors, open a [Contribution Request](https://github.com/ai-dynamo/frontend-crates/issues/new?template=contribution_request.yml) before starting implementation. This includes new model parsers and changes that span multiple crates or modify public APIs. Wait for explicit maintainer approval of the proposed approach before implementing the change or submitting a PR.

Use the Contribution Request form with a `[CONTRIBUTION]:` title and complete all five required fields:

| Field | What to include |
| --- | --- |
| Type of Change | Bug fix, new feature, refactoring, or performance improvement. |
| Problem Statement | The problem or use case you want to address. For a new parser, include representative raw model output. |
| Proposed Solution | Your proposed approach and any public API changes. |
| Estimated PR Size | The expected number of changed lines. |
| Files/Components Affected | The crates, files, or components you expect to modify. |

If an existing issue already describes the work, add the same proposal fields there and wait for maintainer approval rather than opening a duplicate. Link the approved issue in the PR description with `Fixes #123` when the PR completes the issue, or `Relates to #123` when it addresses only part of the work.

Small, focused changes below 100 changed lines, such as typo fixes, documentation corrections, or narrow bug fixes, can go directly to a PR unless they introduce a feature or require agreement on the approach. Keep security disclosures private as described in [Reporting Issues](#reporting-issues).

### Implement and Submit

Standard GitHub fork-and-PR workflow:

1. Fork [this repository](https://github.com/ai-dynamo/frontend-crates).
2. Create a topic branch from `main`.
3. Make your changes; keep PRs focused (one logical change per PR).
4. Run the local checks for each affected crate:

   ```bash
   cargo fmt
   cargo clippy -- -D warnings
   cargo test
   ```

5. Add tests for new parser families, new tokenizer code paths, and any new public surface.
6. Make sure every new source file carries the SPDX header block (see [Source Headers](#source-headers) below).
7. Sign off your commits (see [Signing Your Work](#signing-your-work) below).
8. Open the PR using the [required PR description format](#required-pr-description-format) and link the approved issue when one is required above. CI must pass.

### Required PR Description Format

Use the [PR template](.github/pull_request_template.md) and include Overview, Details, Validation, and Where should the reviewer start? sections. Link the related issue when required by the contribution workflow above.

**Every bug-fix PR must include a Bug Fix Evidence section in the PR description.** Do not leave the reproduction only in an issue, test source, or commit message. Include the following numbered subsections:

1. **Minimal Sample Input:** The smallest concrete input that reproduces the bug, with the affected crate/version and a runnable command or test. For parser bugs, include the raw model text, tools/schema, request initialization or mode, and chunk boundaries when they affect the failure.
2. **Buggy Output (Before):** The actual incorrect output from the previous version for that input. Show the incorrect fields or events explicitly.
3. **Expected/Fixed Output (After):** The expected output and the observed output from the fixed version for the same input. Show the corrected fields or events explicitly; explain any remaining difference.
4. **Unified Regression Cases:** For parser bugs, add the reproduction and expected GOLDEN events as `UNIFIED.*` regression cases (test columns). List the exact case IDs, what each verifies, applicable model families, and intentional grammar/mode exclusions. Follow [case applicability](conformance/README.md#case-applicability) and the [parser-fix capture workflow](conformance/README.md#2-fix-a-dynamo-parser-and-refresh-its-expected-outputs). Capture the before and after versions against the same input, tools, initialization, chunk schedule, and GOLDEN so the regression column shows red before the fix and green after it.
5. **Visual Conformance Verification:** Regenerate `conformance/CONFORMANCE_v2.html` from the final candidate worktree using `conformance/utils/render_table_v2.sh`, open the report in a browser, and inspect the affected `UNIFIED.*` columns and their input/output popups. State the versions, model families, case IDs, and observed before/after results in the PR description. Confirm that the affected current Dynamo family has zero red and zero empty cells under the [Unified parser hard gate](conformance/README.md#unified-parser-hard-gate). A passing unit test or JSON check alone does not replace this visual verification. Keep generated HTML and JSON out of the commit.

For parser bug fixes, all five subsections are required before requesting review. For other bug fixes, provide the concrete input and before/after outputs, and explain why Unified cases and visual conformance verification do not apply. For changes with no bug fix, mark Bug Fix Evidence as not applicable and state why.

### Reporting Issues

- **Bugs / unexpected behavior**: open a [GitHub issue](https://github.com/ai-dynamo/frontend-crates/issues) with a minimal repro and the crate version (`cargo pkgid`).
- **Feature requests / new model parsers**: open a [GitHub issue](https://github.com/ai-dynamo/frontend-crates/issues) describing the use case. For new tool-calling or reasoning parsers, please include a representative sample of the model's raw output.
- **Security disclosures**: do **not** open a public issue. See [`SECURITY.md`](SECURITY.md).

### Community

- [CNCF Slack `#ai-dynamo`](https://communityinviter.com/apps/cloud-native/cncf)
- [Dynamo Discord](https://discord.gg/nvidia-dynamo)

## Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

- Any contribution which contains commits that are not Signed-Off will not be accepted.
- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO (https://developercertificate.org/):

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

## Source Headers

All new source files must carry the SPDX header block. Use the form appropriate for the file type.

**Rust / TypeScript / C-style comments:**

```
// SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
```

**Shell / Python / YAML / TOML:**

```
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

If a file is derived from third-party code (as `dynamo-protocols` is from `async-openai`), include the upstream attribution beneath the SPDX block — see [`ATTRIBUTIONS-Rust.md`](ATTRIBUTIONS-Rust.md) and the existing headers in `protocols/src/lib.rs` and `protocols/Cargo.toml` for the canonical form.

## License

By contributing, you agree that your contributions are licensed under the [Apache 2.0 license](LICENSE) of this project.
