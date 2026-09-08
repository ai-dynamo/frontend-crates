# frontend-crates

Standalone Rust crates for building OpenAI/Anthropic-compatible inference servers, plus a small demo wiring them together.

| Crate | What it does |
| -- | -- |
| [`dynamo-protocols`](./protocols/)   | Request/response types for OpenAI Chat / Completions / Responses + Anthropic Messages. Built on `async-openai` v0.34 with inference-serving extensions. |
| [`dynamo-tokenizers`](./tokenizers/) | HuggingFace + tiktoken + FastTokenizer wrappers with fast incremental detokenization and prefix-caching for shared-prefix workloads. |
| [`dynamo-parsers`](./parsers/v1/)    | Reasoning + tool-calling parsers across 18+ model families (DeepSeek R1/V4, Qwen3, GPT-OSS, Kimi K2/K3, Gemma 4, Llama, Hermes, ...). The stable batch parser — the *decode* side, and the crate to depend on. |
| [`dynamo-renderer`](./renderer/)     | Chat-template / prompt rendering: OpenAI chat requests → model-ready prompts via HF `chat_template` (minijinja), plus native DeepSeek, Inkling, and Kimi K3 formatters. The *encode* side. |

Each crate is independently published to crates.io and can be adopted on its own. Only `dynamo-renderer` has internal deps — it depends on `dynamo-protocols` and re-exports `dynamo-tokenizers` for convenience; `dynamo-protocols`, `dynamo-tokenizers`, and `dynamo-parsers` are leaf crates with no internal deps. The repository itself is a Cargo workspace so shared dependency versions, CI checks, and the demo build stay consistent.

The three parser crates live under `parsers/`: `parsers/v1` is the stable, published `dynamo-parsers`. `parsers/v2` (`dynamo-parsers-v2`) is the **work-in-progress** pure-streaming parser on a `0.x` line — use `dynamo-parsers` (v1) for anything real. **v1 is interim**: once v2 reaches parity, v1 (batch + jail) is removed outright — v2 is the ultimate implementation, and new parser work goes there. `parsers/v2-py` is a test-only PyO3 binding for the conformance harness and is **not published**. See [`docs/PARSERS-V2-MIGRATION-PLAN.md`](./docs/PARSERS-V2-MIGRATION-PLAN.md).

## Layout

```
frontend-crates/
├── protocols/              # dynamo-protocols
├── tokenizers/             # dynamo-tokenizers
├── parsers/
│   ├── v1/                 # dynamo-parsers        (stable batch parser)
│   ├── v2/                 # dynamo-parsers-v2     (WIP streaming parser, 0.x)
│   └── v2-py/              # dynamo-parsers-v2-py  (test-only PyO3 binding, unpublished)
├── renderer/               # dynamo-renderer (deps protocols, tokenizers)
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
cargo build -p dynamo-demo-server --release
```

Repository hygiene checks:

```bash
cargo fmt --all -- --check
cargo machete
cargo deny --all-features check bans licenses
```
