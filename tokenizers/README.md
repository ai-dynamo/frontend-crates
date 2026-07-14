# dynamo-tokenizers

Efficient, versatile tokenization for LLM inference. Wraps HuggingFace and TikToken tokenizers (plus a FastTokenizer hybrid mode) behind a small encode/decode/sequence API designed for streaming detokenization.

## Features

- **Multiple backends.** HuggingFace `tokenizers`, OpenAI `tiktoken`, and a FastTokenizer hybrid behind one trait.
- **Streaming-friendly.** `Sequence` tracks incremental token-id appends and emits text deltas without re-decoding the full prefix.
- **Prefix caching.** `CachedTokenizer` records prefix tokenizations at special-token boundaries; repeated prompts that share a system prefix re-encode only the trailing suffix, turning O(N) work into O(suffix_len).
- **Hash verification.** Detect tokenizer drift across model versions.

## Quick start

```rust
use dynamo_tokenizers::hf::HuggingFaceTokenizer;
use dynamo_tokenizers::traits::{Encoder, Decoder};

// tokenizer.json downloaded from any HuggingFace model repo
let tokenizer = HuggingFaceTokenizer::from_file("/path/to/tokenizer.json")
    .expect("load tokenizer");

let encoding = tokenizer.encode("Your sample text here")
    .expect("encode");
println!("{:?}", encoding);

let decoded = tokenizer.decode(&encoding.token_ids, false)
    .expect("decode");
assert_eq!(decoded, "Your sample text here");
```

## Streaming detokenization with `Sequence`

```rust
use dynamo_tokenizers::{Sequence, Tokenizer};
use std::sync::Arc;

let tokenizer = Tokenizer::from(Arc::new(tokenizer));
let mut sequence = Sequence::new(tokenizer.clone());

sequence.append_text("Your sample text here")
    .expect("append text");

// As each new token id is produced by the engine, append it
// and get back just the incremental text delta:
let delta = sequence.append_token_id(1337)
    .expect("append token_id");
```

## Prefix caching with `CachedTokenizer`

Multi-turn chat workloads re-tokenize a large shared prefix (system prompt +
prior turns) on every request. `CachedTokenizer` wraps a compatible tokenizer and
caches prefix tokenizations at special-token boundaries (e.g. `<|im_start|>`,
`<|im_end|>`, `<s>`, `</s>`). On a hit it merges the cached prefix tokens with a
fresh encode of the trailing suffix only.

Boundaries are taken **only** immediately after a registered special token —
those are atomic in BPE, so the merge is exact:
`encode(prefix) + encode(suffix) == encode(prefix + suffix)`. There is no
whitespace/punctuation fallback; the cache prefers a miss over a corrupt split.

```rust
use dynamo_tokenizers::{CachedTokenizer, HuggingFaceTokenizer};
use dynamo_tokenizers::traits::{Encoder, Tokenizer};
use std::sync::Arc;

let hf = HuggingFaceTokenizer::from_file("/path/to/tokenizer.json")
    .expect("load tokenizer");
let inner: Arc<dyn Tokenizer> = Arc::new(hf);

// The atomic special tokens the model uses as turn delimiters.
// An empty list disables caching: encode/encode_batch pass straight through.
let specials = vec!["<|im_start|>".to_string(), "<|im_end|>".to_string()];

let cached = CachedTokenizer::new(inner, specials, 256 * 1024 * 1024)
    .expect("tokenizer must support prefix caching"); // 256 MiB budget

let encoding = cached.encode("<|im_start|>system\nYou are helpful.<|im_end|>")
    .expect("encode");

let stats = cached.cache_stats();
println!("hits={} misses={} hit_rate={:.2}", stats.hits, stats.misses, stats.hit_rate);
```

TikToken models expose the exact special-token strings registered with their BPE, so
the same cache can be constructed without duplicating model metadata:

```rust
use dynamo_tokenizers::{CachedTokenizer, TikTokenTokenizer};
use dynamo_tokenizers::traits::Tokenizer;
use std::sync::Arc;

let tiktoken = TikTokenTokenizer::from_file_auto("/path/to/tiktoken.model")
    .expect("load tokenizer");
let specials = tiktoken.special_tokens().to_vec();
let inner: Arc<dyn Tokenizer> = Arc::new(tiktoken);
let cached = CachedTokenizer::new(inner, specials, 256 * 1024 * 1024)
    .expect("tokenizer must support prefix caching");
```

Entries are evicted by approximate LRU once `max_memory_bytes` is exceeded. The
cache lives as long as the `CachedTokenizer` instance. Use `.with_observer(...)`
to push request-level hit/miss events into your metrics. Use
`.with_token_observer(...)` to receive exact cached and uncached token counts
after each successful encode while L1 is active. Partial hits report both
categories, so consumers can increment `cached_tokens_total` and
`uncached_tokens_total` counters and derive a token-level reuse ratio.
