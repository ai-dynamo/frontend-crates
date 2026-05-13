# frontend-crates

Three standalone Rust crates for building OpenAI/Anthropic-compatible inference servers, plus a small demo wiring them together.

| Crate | What it does |
| -- | -- |
| [`dynamo-protocols`](./protocols/)   | Request/response types for OpenAI Chat / Completions / Responses + Anthropic Messages. Built on `async-openai` v0.34 with inference-serving extensions. |
| [`dynamo-tokenizers`](./tokenizers/) | HuggingFace + tiktoken + FastTokenizer wrappers with fast incremental detokenization. |
| [`dynamo-parsers`](./parsers/)       | Reasoning + tool-calling parsers across 18+ model families (DeepSeek R1/V4, Qwen3, GPT-OSS, Kimi K2, Gemma 4, Llama, Hermes, ...). Streaming-first. |

Each crate is independently published to crates.io and can be adopted on its own. `dynamo-parsers` depends on `dynamo-protocols`; the other two have no internal deps. The repository itself is a Cargo workspace so shared dependency versions, CI checks, and the demo build stay consistent.

## Layout

```
frontend-crates/
├── protocols/              # dynamo-protocols
├── tokenizers/             # dynamo-tokenizers
├── parsers/                # dynamo-parsers (deps protocols)
├── examples/
│   └── dynamo-demo-server/ # axum server wiring all three together
└── scripts/
    └── sync-from-dynamo.sh # check / pull changes from ai-dynamo/dynamo
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
cargo build -p dynamo-demo-server --release
```

Repository hygiene checks:

```bash
cargo fmt --all -- --check
cargo machete
cargo deny --all-features check bans licenses
```

## Where the code lives

These crates currently mirror `lib/{protocols,tokenizers,parsers}/` from [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo). The sync is **one-way (dynamo → frontend-crates) and manual** for now — see [`scripts/sync-from-dynamo.sh`](./scripts/sync-from-dynamo.sh) to check for upstream changes.
