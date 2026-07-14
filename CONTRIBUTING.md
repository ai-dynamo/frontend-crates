<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to frontend-crates

Thank you for your interest in contributing to `frontend-crates`!

This repository contains three independently published Rust crates — `dynamo-protocols`, `dynamo-tokenizers`, and `dynamo-parsers` — that can be adopted on their own to build OpenAI/Anthropic-compatible inference servers, plus a demo server wiring them together.

## Source of Truth and Sync Direction

The code under `protocols/` and `renderer/` currently mirrors `lib/{protocols,renderer}/` from [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo). The canonical sync direction is **dynamo → frontend-crates**, one-way and manual, driven by [`scripts/sync-from-dynamo.sh`](scripts/sync-from-dynamo.sh).

The parser crates under `parsers/` (`parsers/v1` = `dynamo-parsers`, `parsers/v2` = `dynamo-parsers-v2`, `parsers/v2-py` = the test-only binding) and the `tokenizers/` crate are **frontend-crates-owned** — the sync script no longer touches them. `tokenizers/` was detached when its tokenizer fixtures moved from the top-level `llm/tests/data` into `tokenizers/tests/data`. See [`docs/PARSERS-V2-MIGRATION-PLAN.md`](docs/PARSERS-V2-MIGRATION-PLAN.md).

What this means for contributors:

- **Changes to synced crate source code (`protocols/src/**`, `renderer/src/**`)** should be opened as PRs against [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) in the corresponding `lib/` directory. Once merged upstream, the sync script pulls them into this repository for the next published release.
- **Changes to parser and tokenizer source (`parsers/v1/src/**`, `parsers/v2/src/**`, `tokenizers/src/**`)** are made here directly — those crates are no longer synced from Dynamo.
- **Changes to repository scaffolding** — crate `Cargo.toml` metadata, `examples/dynamo-demo-server/`, `scripts/`, `.github/`, docs in this repo, the sync tooling itself — should be opened as PRs here.

If you're not sure where a change belongs, open the PR in either place and we'll route it.

## How to Contribute

Standard GitHub fork-and-PR workflow:

1. Fork the repository ([this one](https://github.com/ai-dynamo/frontend-crates), or [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) for crate source changes — see above).
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
8. Open the PR. CI must pass.

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
