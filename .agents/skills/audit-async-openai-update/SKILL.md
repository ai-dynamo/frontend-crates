---
name: audit-async-openai-update
description: Audit async-openai dependency upgrades in ai-dynamo/frontend-crates and Dynamo consumers. Use when bumping async-openai, reviewing an upgrade PR, checking whether locally owned or vendored OpenAI protocol types can be deleted or narrowed, detecting drift in mirrored structs and enums, or preparing validation evidence for an async-openai update.
---

# Audit async-openai Updates

Audit the dependency bump and the local compatibility layer together. Prefer upstream types, but preserve every proven wire-format relaxation and Dynamo-specific extension.

## Guardrails

- Work in an isolated clone or detached worktree. Do not experiment in an active PR worktree.
- Read `protocols/CLAUDE.md` before classifying ownership.
- Treat Serde attributes, enum tagging, untagged variant order, defaults, and omission behavior as API behavior.
- Distinguish missing fields from explicit `null`; `#[serde(default)]` usually handles only the former.
- Do not remove a public type without searching downstream Dynamo consumers. Prefer a re-export or type alias when it preserves compatibility.
- Keep dependency updates and unrelated cleanup in separate commits or PRs unless the user approves combining them.

## Workflow

### 1. Establish the versions

Record the old manifest requirement, old lock version, new requirement, new lock version, base commit, and Rust toolchain. For pre-1.0 crates, remember that `0.41` means `>=0.41.0, <0.42.0`.

Fetch both crate sources if necessary:

```bash
cargo info async-openai@OLD_VERSION
cargo info async-openai@NEW_VERSION
```

Locate them under `${CARGO_HOME:-$HOME/.cargo}/registry/src/*/async-openai-VERSION`. When Cargo runs in a container, mount a persistent Cargo home and pass the resulting source paths explicitly.

### 2. Generate the inventory

Run the bundled script from the frontend-crates root:

```bash
python .agents/skills/audit-async-openai-update/scripts/inventory.py \
  --repo . \
  --old-source /path/to/async-openai-OLD \
  --new-source /path/to/async-openai-NEW
```

Use the report as a candidate list, not as a verdict. Inspect:

- locally owned names newly introduced upstream;
- locally owned names whose upstream definition changed;
- longstanding local/upstream overlaps;
- ownership comments that cite upstream gaps or pinned versions;
- mirrored enums or structs that may have gained variants or fields.

Exclude fully custom protocols such as Anthropic unless the update touches their shared dependencies.

### 3. Classify every candidate

Use one classification:

- **Remove/re-export**: upstream accepts all required wire shapes and exposes the needed traits and builders.
- **Alias**: upstream is suitable, but a public Dynamo name must remain source-compatible.
- **Narrow**: upstream fixed part of the gap, while a smaller local wrapper or input-only type is still required.
- **Keep**: Dynamo needs a serving extension, permissive input, normalization, or known upstream bug workaround.
- **Drift fix**: an owned mirror must add upstream fields or enum variants.

For each decision, compare fields, field types, visibility, Serde attributes, defaults, tags, builders, derives, and request-versus-response use.

### 4. Prove the decision

For each removal or narrowing:

1. Add or identify a focused test for the original wire shape.
2. Confirm the old upstream type rejects it when the rationale depends on a historical gap.
3. Make the smallest substitution in the isolated worktree.
4. Search frontend-crates and Dynamo for constructors, matches, and explicit type annotations.
5. Preserve public names with aliases when practical.
6. Verify serialized JSON as well as successful deserialization.

Do not infer compatibility from compilation alone.

### 5. Run checks

Use the repository-pinned Rust toolchain:

```bash
cargo check --workspace --all-targets --locked
cargo test -p dynamo-protocols --all-targets --locked
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
git diff --check
```

If Git LFS fixtures are unavailable, report that explicitly and rerun the remaining workspace tests with only the fixture-dependent crate excluded.

### 6. Report

Produce a table with:

| Type or subtree | Local rationale | Upstream status | Action | Evidence | Downstream impact |
|---|---|---|---|---|---|

Separate cleanup enabled by the new version from pre-existing redundancy discovered during the audit. Include exact upstream old/new source links and relevant OpenAI API documentation in the PR body.
