# frontend-crates

Three standalone Rust crates for building OpenAI/Anthropic-compatible inference servers, plus a small demo wiring them together.

| Crate | What it does |
| -- | -- |
| [`dynamo-protocols`](./protocols/)   | Request/response types for OpenAI Chat / Completions / Responses + Anthropic Messages. Built on `async-openai` v0.34 with inference-serving extensions. |
| [`dynamo-tokenizers`](./tokenizers/) | HuggingFace + tiktoken + FastTokenizer wrappers with fast incremental detokenization. |
| [`dynamo-parsers`](./parsers/)       | Reasoning + tool-calling parsers across 18+ model families (DeepSeek R1/V4, Qwen3, GPT-OSS, Kimi K2, Gemma 4, Llama, Hermes, ...). Streaming-first. |

Each crate is independently published to crates.io and can be adopted on its own. `dynamo-parsers` depends on `dynamo-protocols`; the other two have no internal deps.

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

Each crate builds standalone:

```bash
cd protocols   && cargo build
cd tokenizers  && cargo build
cd parsers     && cargo build
```

The demo server pulls all three via path deps:

```bash
cd examples/dynamo-demo-server && cargo build --release
```

## Where the code lives

These crates currently mirror `lib/{protocols,tokenizers,parsers}/` from [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo). The sync is **one-way (dynamo → frontend-crates) and manual** for now — see [`scripts/sync-from-dynamo.sh`](./scripts/sync-from-dynamo.sh) to check for upstream changes.
