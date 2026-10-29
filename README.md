# frontend-crates

Standalone Rust crates for building OpenAI/Anthropic-compatible inference servers, plus a small demo wiring them together.

| Crate | What it does |
| -- | -- |
| [`dynamo-protocols`](./protocols/)   | Request/response types for OpenAI Chat / Completions / Responses + Anthropic Messages. Built on `async-openai` v0.42 with inference-serving extensions. |
| [`dynamo-tokenizers`](./tokenizers/) | HuggingFace + tiktoken + FastTokenizer wrappers with fast incremental detokenization and prefix-caching for shared-prefix workloads. |
| [`dynamo-parsers`](./parsers/v1/)    | Reasoning + tool-calling parsers across 18+ model families (DeepSeek R1/V4, Qwen3, GPT-OSS, Kimi K2/K3, Gemma 4, Llama, Hermes, ...). The legacy batch parser and streaming jail. |
| [`dynamo-parsers-v2`](./parsers/v2/) | Unified ordered reasoning, text, and tool-call events, plus compatibility tool-only parsers. Published on a `0.x` API line. |
| [`dynamo-structural-tag`](./parsers/structural-tag/) | Shared structural-tag grammar and schema helpers used by both parser crates. |
| [`dynamo-renderer`](./renderer/)     | Chat-template / prompt rendering: OpenAI chat requests → model-ready prompts via HF `chat_template` (minijinja), plus native DeepSeek, Inkling, and Kimi K3 formatters. The *encode* side. |
| [`dynamo-multimodal`](./multimodal/) | Shared multimodal input processing and accounting, with a golden-tested Qwen-VL image pipeline. The optional `fetch` feature provides the protected `MediaFetcher`; trusted-source `fetch_bytes*` helpers remain stubs. |

Each library listed above is independently published to crates.io and can be adopted on its own. `dynamo-renderer` depends on `dynamo-protocols` and re-exports `dynamo-tokenizers`. `dynamo-parsers` depends on `dynamo-protocols` and `dynamo-structural-tag`; `dynamo-parsers-v2` depends on `dynamo-structural-tag` and owns its batch conversion helpers under `tool_calling/v1core`, without depending on the v1 parser crate. The repository itself is a Cargo workspace so shared dependency versions, CI checks, and the demo build stay consistent.

For new streaming integrations, start with the [Unified lifecycle example](parsers/v2/README.md#using-unified-parsing); existing batch callers can use [v1](parsers/v1/README.md#using-the-crate). Dynamo uses V2 for compatible deployments. Pin and test its `0.x` API; library registration alone does not add serving support. `parsers/v2-py` is an unpublished test binding. See the [migration plan](docs/PARSERS-V2-MIGRATION-PLAN.md) for proposed consolidation.

## Layout

```
frontend-crates/
├── protocols/              # dynamo-protocols
├── tokenizers/             # dynamo-tokenizers
├── parsers/
│   ├── v1/                 # dynamo-parsers        (stable batch parser)
│   ├── v2/                 # dynamo-parsers-v2     (incremental + Unified, 0.x)
│   ├── structural-tag/     # dynamo-structural-tag (shared grammar helpers)
│   └── v2-py/              # dynamo-parsers-v2-py  (test-only PyO3 binding, unpublished)
├── renderer/               # dynamo-renderer (deps protocols, tokenizers)
├── multimodal/             # dynamo-multimodal (Qwen-VL image pipeline)
├── examples/
│   └── dynamo-demo-server/ # axum server wiring them together
└── docs/                   # PARSERS-V2-MIGRATION-PLAN.md
```

## Building

This repo pins Rust with [`rust-toolchain.toml`](./rust-toolchain.toml). Use the workspace commands from the repository root:

```bash
cargo check --workspace --all-targets --locked
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
```

The root [`Cargo.lock`](./Cargo.lock) is committed for reproducible CI and demo-server builds. Library crates still publish normally; consumers resolve their own lockfiles.

Each crate can still be built or tested directly:

```bash
cargo build -p dynamo-protocols
cargo build -p dynamo-tokenizers
cargo build -p dynamo-parsers
cargo build -p dynamo-renderer
cargo build -p dynamo-multimodal
cargo build -p dynamo-demo-server --release
```

Repository hygiene checks:

```bash
cargo fmt --all -- --check
cargo machete
cargo deny --all-features check bans licenses
```
