# frontend-crates

Standalone Rust crates for building OpenAI/Anthropic-compatible inference servers, plus a small demo wiring them together.

| Crate | What it does |
| -- | -- |
| [`dynamo-protocols`](./protocols/)   | Request/response types for OpenAI Chat / Completions / Responses + Anthropic Messages. Built on `async-openai` v0.34 with inference-serving extensions. |
| [`dynamo-tokenizers`](./tokenizers/) | HuggingFace + tiktoken + FastTokenizer wrappers with fast incremental detokenization and prefix-caching for shared-prefix workloads. |
| [`dynamo-parsers`](./parsers/)       | Reasoning + tool-calling parsers across 18+ model families (DeepSeek R1/V4, Qwen3, GPT-OSS, Kimi K2, Gemma 4, Llama, Hermes, ...). Streaming-first. The *decode* side. |
| [`dynamo-renderer`](./renderer/)     | Chat-template / prompt rendering: OpenAI chat requests → model-ready prompt strings via HF `chat_template` (minijinja), plus native DeepSeek formatters. The *encode* side. |

Each crate is independently published to crates.io and can be adopted on its own. Only `dynamo-renderer` has internal deps — it depends on `dynamo-protocols` and re-exports `dynamo-tokenizers` for convenience; `dynamo-protocols`, `dynamo-tokenizers`, and `dynamo-parsers` are leaf crates with no internal deps. The repository itself is a Cargo workspace so shared dependency versions, CI checks, and the demo build stay consistent.

## Layout

```
frontend-crates/
├── protocols/              # dynamo-protocols
├── tokenizers/             # dynamo-tokenizers
├── parsers/                # dynamo-parsers
├── renderer/               # dynamo-renderer (deps protocols, tokenizers)
├── examples/
│   └── dynamo-demo-server/ # axum server wiring them together
└── conformance/            # parser conformance fixtures, checks, and renderers
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
cargo build -p dynamo-demo-server --release
```

Repository hygiene checks:

```bash
cargo fmt --all -- --check
cargo machete
cargo deny --all-features check bans licenses
```

## Source of Truth

This repository is the source of truth for these crates. Dynamo consumes the published crates instead of feeding a reverse sync back into this repository.
