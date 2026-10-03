# dynamo-tokenizers

Efficient, versatile tokenization for LLM inference. Wraps Hugging Face,
TikToken, fastokens, and Baseten Tokenizer backends behind a small
encode/decode/sequence API designed for streaming detokenization.

## Features

- **Multiple backends.** Hugging Face `tokenizers`, OpenAI `tiktoken`,
  `fastokens`, and `basetenkenizer` behind one trait.
- **Streaming-friendly.** `Sequence` tracks incremental token-id appends and emits text deltas without re-decoding the full prefix.
- **Prefix caching.** `CachedTokenizer` records prefix tokenizations at special-token boundaries; repeated prompts that share a system prefix re-encode only the trailing suffix, turning O(N) work into O(suffix_len).
- **Hash verification.** Detect tokenizer drift across model versions.

## Segmented prompts

`BasetenTokenizer` supports segmented encoding for renderers such as Kimi K3's
XTML renderer, where trusted control tokens and untrusted message content must
remain distinct:

Many Kimi model repositories ship tiktoken assets rather than a directly
loadable `tokenizer.json`. Baseten publishes compatible `tokenizer.json`
artifacts for these models; for Kimi K3, use
[`baseten/kimi-k3-tokenizer`](https://huggingface.co/baseten/kimi-k3-tokenizer).
Download the file and pass its path to `BasetenTokenizer::from_file`.

```rust
use dynamo_tokenizers::{
    BasetenTokenizer, EncodeSegment,
    traits::Encoder,
};

let tokenizer = BasetenTokenizer::from_file("/path/to/tokenizer.json")?;
let segments = [
    EncodeSegment::new("<|open|>message role=\"user\"<|sep|>", true),
    EncodeSegment::new(user_message, false),
    EncodeSegment::new("<|close|>message<|sep|><|end_of_msg|>", true),
];
let encoding = tokenizer.encode_segments(&segments)?;
```

Segmented encoding preserves legacy tiktoken chunk boundaries for long-input
token-ID parity, as required by Kimi K3.

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

Checkpoints that ship only a `tiktoken.model` (Kimi K2/K3) can also encode
through `fastokens` without a converted `tokenizer.json`:
`FastTikTokenTokenizer::from_file_auto("/path/to/tiktoken.model")` loads the
same ranks, regex, and special tokens as `TikTokenTokenizer::from_file_auto`.

Entries are admitted and evicted by Moka's W-TinyLFU policy. The byte budget counts
token-ID payloads, excluding cache metadata and tokenizer objects. Eviction is
deferred, so this is not a strict process-memory limit. A private cache lives as
long as its `CachedTokenizer` instance. Use `.with_observer(...)`
to push request-level hit/miss events into your metrics. Use
`.with_token_observer(...)` to receive exact cached and uncached token counts
after each successful encode while L1 is active. Partial hits report both
categories, so consumers can increment `cached_tokens_total` and
`uncached_tokens_total` counters and derive a token-level reuse ratio.

### Share one capacity budget across tokenizers

Construct one `SharedTokenizerCache::new(max_memory_bytes)` and pass clones to
`CachedTokenizer::new_with_cache(inner, specials, shared_cache, namespace)`.
Clones share storage, admission, and eviction across all namespaces.

The namespace is a byte string chosen by the caller. Include the model identity,
tokenizer file checksums, actual backend, and any encoding options that affect
token IDs. Equal namespaces must describe identical tokenizer behavior. The
cache hashes the length-delimited namespace before the text prefix; keys remain
32-byte BLAKE3 digests. A namespace is not a reserved share of the capacity.

Entries survive a tokenizer wrapper being dropped while the shared storage remains
alive. Recreating a wrapper with the same namespace can reuse those entries.
Entries leave through normal eviction or when the shared storage is dropped.
The boundary list must contain actual atomic special tokens recognized by the inner
tokenizer. Arbitrary strings are not safe cache boundaries.

`cache_stats()` reports wrapper-local hits and misses, plus namespace-wide entry
count and token bytes. Namespace storage statistics scan the shared cache on demand;
they can change under concurrent writes. Private caches use Moka's totals directly.
Wrappers disabled by the cache eligibility checks report zero statistics.
`SharedTokenizerCache::stats()` reports the combined entry count and token bytes
after pending maintenance.
