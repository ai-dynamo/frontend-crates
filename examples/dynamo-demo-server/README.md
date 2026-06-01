# dynamo-demo-server

A working OpenAI/Anthropic-compatible server in ~600 lines of Rust, wiring together the sibling crates (`dynamo-protocols`, `dynamo-tokenizers`, `dynamo-parsers`, `dynamo-renderer`). The backend is a dummy echo — swap `src/echo.rs` for your scheduler / forward pass and you have a real server.

## Build

```bash
cd examples/dynamo-demo-server
cargo build --release
```

This picks up the sibling crates via path deps (`../../protocols`, `../../tokenizers`, `../../parsers`, `../../renderer`) — no separate clone.

## Run

```bash
# Pass any HuggingFace repo id; tokenizer.json + tokenizer_config.json are
# fetched on startup. The chat_template in tokenizer_config.json powers /v1/render.
cargo run --release -- --model Qwen/Qwen2.5-0.5B-Instruct

# Or point at a local tokenizer.json (optionally a local chat-template config):
cargo run --release -- --tokenizer /path/to/tokenizer.json \
  --chat-template-config /path/to/tokenizer_config.json
```

Flags:

```
--model <MODEL>                  HuggingFace repo id (fetches tokenizer.json + tokenizer_config.json)
--tokenizer <PATH>               local tokenizer.json (alternative to --model)
--chat-template-config <PATH>    local tokenizer_config.json for /v1/render (else fetched via --model)
--host <HOST>                    default 0.0.0.0
--http-port <PORT>               default 3000
```

## Test

Hit every endpoint with the smoke script while the server is running:

```bash
scripts/smoke.sh                          # against localhost:3000
scripts/smoke.sh http://127.0.0.1:3000    # explicit base URL
```

Exits non-zero if any endpoint fails.

## Endpoints

| Endpoint                    | API                                  | Crates used                    |
| --------------------------- | ------------------------------------ | ------------------------------ |
| `POST /v1/chat/completions` | OpenAI Chat (streaming + tool calls) | protocols, parsers, tokenizers |
| `POST /v1/completions`      | OpenAI Completions                   | protocols, tokenizers          |
| `POST /v1/responses`        | OpenAI Responses                     | protocols, tokenizers          |
| `POST /v1/messages`         | Anthropic Messages (streaming)       | protocols, tokenizers          |
| `POST /v1/tokenize`         | encode → token ids                   | tokenizers                     |
| `POST /v1/detokenize`       | token ids → text                     | tokenizers                     |
| `POST /v1/tool-parse`       | tool-call parser (15+ formats)       | parsers                        |
| `POST /v1/reasoning-parse`  | reasoning parser                     | parsers                        |
| `POST /v1/render`           | chat request → rendered prompt       | renderer (protocols)           |
| `GET /health`               | —                                    | —                              |

## Layout

```
src/
  main.rs                 axum router + CLI + HF Hub tokenizer / config fetch
  engine.rs               AppState — holds the loaded Tokenizer + chat-template renderer
  echo.rs                 dummy backend (extracts text from request bodies)
  handlers/
    chat.rs               /v1/chat/completions  (streaming + tool parsing)
    completions.rs        /v1/completions
    responses.rs          /v1/responses
    anthropic.rs          /v1/messages          (Anthropic SSE format)
    tokenize.rs           /v1/tokenize, /v1/detokenize
    tool_parse.rs         /v1/tool-parse
    reasoning_parse.rs    /v1/reasoning-parse
    render.rs             /v1/render            (chat template → prompt)
```
