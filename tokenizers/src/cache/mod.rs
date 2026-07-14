// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: Copyright (c) 2024 Simo Lin, Chang Su, Keyang Ru (llm-tokenizer authors)
//
// Portions adapted from sgl-project/llm-tokenizer v1.3.2 (Apache-2.0).
// Upstream: https://github.com/lightseekorg/smg
// Modifications: removed L0 layer, removed `add_special_tokens` plumbing (Dynamo's
// `Encoder::encode` has no such flag), dropped fingerprinting, retargeted onto
// `crate::traits::Tokenizer`.

//! Tokenizer caching layer (L1: prefix matching at special-token boundaries).
//!
//! Wraps a cache-compatible [`Tokenizer`] in a cache that records prefix
//! tokenizations at every special-token boundary. On a hit, the cached prefix
//! tokens are merged with a fresh encode of the trailing suffix only — turning
//! O(N) tokenization work into O(suffix_len) when prompts share a system prefix.
//!
//! # Correctness
//!
//! Boundaries are taken **only** at positions immediately following a registered
//! special token (e.g. `<|im_start|>`, `<|im_end|>`, `<s>`, `</s>`). Special tokens
//! are atomic in BPE (`special: true, normalized: false`), so splitting there
//! preserves the invariant `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`.
//! No fallback to whitespace or punctuation — better to miss than to corrupt.
//!
//! # Storage normalization
//!
//! When L1 is enabled, **every** `encode` returns [`Encoding::Sp`] (token-ids only) —
//! hits merge cached prefix ids with a fresh suffix encode, and misses assemble the ids
//! from the per-boundary segment encodes (see [`L1Cache::populate_and_encode`]) — even
//! when the inner tokenizer would have produced [`Encoding::Hf`] (rich offsets/attention/
//! etc). All current downstream consumers in Dynamo only call [`Encoding::token_ids`], so
//! this lossy normalization is safe; revisit if a caller starts reading offsets or
//! attention masks from encodings produced through the cache.
//!
//! # Configuration
//!
//! - `special_tokens: Vec<String>` — must be supplied at construction (the
//!   [`Tokenizer`] trait is intentionally minimal and does not expose them).
//!   An empty list disables L1: `encode`/`encode_batch` short-circuit straight
//!   to the inner tokenizer with no lookup, no miss-counter bump, and no
//!   insert attempt.
//! - `max_memory_bytes` — L1 byte budget; entries evicted via approximate LRU.
//!
//! # Provenance
//!
//! Adapted from `llm-tokenizer` v1.3.2 (`cache/l1.rs`, `cache/mod.rs`). L0 and
//! fingerprinting were dropped; L1 alone covers the headline multi-turn-chat
//! workload, and the in-memory cache lifetime is bound to a single tokenizer
//! instance so fingerprint-based invalidation is unnecessary.

mod l1;

use std::sync::Arc;

pub use l1::{CacheEventFn, L1Cache, L1CacheStats};

use crate::{
    Encoding, Result, TokenIdType,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// Token-level cache usage for one successful encode.
///
/// A partial cache hit reports both cached prefix tokens and uncached suffix tokens.
/// Their sum always equals the number of tokens returned by the encode operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheTokenUsage {
    /// Tokens returned from the cached prefix.
    pub cached_tokens: usize,
    /// Tokens freshly encoded from the uncached suffix.
    pub uncached_tokens: usize,
}

/// Optional observer for token-level cache usage.
pub type CacheTokenUsageFn = Arc<dyn Fn(CacheTokenUsage) + Send + Sync>;

/// Caching wrapper around an inner tokenizer.
///
/// Implements [`Encoder`], [`Decoder`], and [`Tokenizer`]; decode calls pass
/// through to the inner tokenizer (decoding is fast and rarely repeated).
pub struct CachedTokenizer {
    inner: Arc<dyn Tokenizer>,
    l1: L1Cache,
    /// Whether L1 is active. False when the special-token set is empty (e.g. the tiktoken
    /// wrapping path): `encode`/`encode_batch` then bypass the cache entirely. The special
    /// tokens themselves live in the `L1Cache` (its boundary automaton).
    l1_enabled: bool,
    /// When true, cache the newly-tokenized suffix on a partial hit so the next turn
    /// of a growing conversation hits deeper (see [`L1Cache::extend_after_match`]).
    extend_on_hit: bool,
    /// Called once after every successful encode while L1 is active.
    token_observer: Option<CacheTokenUsageFn>,
}

impl CachedTokenizer {
    /// Construct a cached tokenizer.
    ///
    /// `special_tokens` is the list of atomic special-token strings the inner
    /// tokenizer recognizes (typically extracted via the HuggingFace tokenizer's
    /// `get_added_tokens_decoder()` filtering by `special == true`). An empty list
    /// disables L1 — `encode`/`encode_batch` short-circuit to the inner tokenizer
    /// without touching the cache or its counters.
    ///
    /// `max_memory_bytes` is the L1 cache byte budget.
    ///
    /// # Errors
    ///
    /// Returns the inner tokenizer's compatibility error when it cannot be
    /// safely wrapped in the prefix cache.
    pub fn new(
        inner: Arc<dyn Tokenizer>,
        special_tokens: Vec<String>,
        max_memory_bytes: usize,
    ) -> Result<Self> {
        inner.validate_prefix_cache()?;

        let l1_enabled = !special_tokens.is_empty();
        Ok(Self {
            inner,
            l1: L1Cache::new(max_memory_bytes, special_tokens),
            l1_enabled,
            extend_on_hit: false,
            token_observer: None,
        })
    }

    /// Enable partial-hit extension. When on, a partial cache hit also caches the
    /// freshly-tokenized suffix at its deepest special-token boundary, so each turn of
    /// a growing multi-turn conversation hits deeper than the last and per-turn
    /// tokenization cost stops growing with conversation length. Default off.
    pub fn with_extend(mut self, enabled: bool) -> Self {
        self.extend_on_hit = enabled;
        self
    }

    /// Install hit/miss callbacks so each L1 lookup pushes an event into the
    /// supplied closures (e.g. `Prometheus::Counter::inc`). Replaces any
    /// previously-set observer.
    pub fn with_observer(mut self, on_hit: CacheEventFn, on_miss: CacheEventFn) -> Self {
        self.l1.set_observer(on_hit, on_miss);
        self
    }

    /// Install a callback that receives exact cached and uncached token counts after each
    /// successful encode while L1 is active. A partial hit reports both categories, which
    /// lets consumers maintain token-level cache totals and derive a reuse ratio. Replaces
    /// any previously-set token observer.
    ///
    /// This observer is not called when the special-token set is empty (and L1 is therefore
    /// disabled) or when encoding returns an error.
    pub fn with_token_observer(mut self, observer: CacheTokenUsageFn) -> Self {
        self.token_observer = Some(observer);
        self
    }

    fn observe_token_usage(&self, cached_tokens: usize, total_tokens: usize) {
        if let Some(observer) = &self.token_observer {
            let uncached_tokens = total_tokens
                .checked_sub(cached_tokens)
                .expect("cached token count cannot exceed total token count");
            observer(CacheTokenUsage {
                cached_tokens,
                uncached_tokens,
            });
        }
    }

    /// Snapshot of L1 cache statistics (cumulative hits/misses/entries/memory).
    pub fn cache_stats(&self) -> L1CacheStats {
        self.l1.stats()
    }

    /// Clear all cached entries and reset counters.
    pub fn clear_cache(&self) {
        self.l1.clear();
    }

    /// Access the underlying tokenizer (e.g. for downcasting to a concrete type).
    pub fn inner(&self) -> &Arc<dyn Tokenizer> {
        &self.inner
    }
}

impl Encoder for CachedTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        // No specials => no boundaries are ever produced. Skip the lookup, miss-counter
        // bump, and insert attempt entirely — otherwise the tiktoken wrapping path (which
        // deliberately passes an empty list) pays the cost on every call with no chance
        // of a hit.
        if !self.l1_enabled {
            return self.inner.encode(input);
        }

        if let Some((prefix_tokens, prefix_len, deepest_boundary)) =
            self.l1.longest_prefix_match(input)
        {
            let cached_tokens = prefix_tokens.len();
            let suffix = &input[prefix_len..];
            let encoding = if suffix.is_empty() {
                Encoding::Sp(prefix_tokens.to_vec())
            } else if self.extend_on_hit {
                // Cache the new suffix at its deepest boundary so the next turn hits
                // deeper, then return the full merged tokens. The deepest boundary was
                // already found by `longest_prefix_match`, so no rescan is needed here.
                Encoding::Sp(self.l1.extend_after_match(
                    input,
                    prefix_tokens,
                    prefix_len,
                    deepest_boundary,
                    self.inner.as_ref(),
                )?)
            } else {
                let suffix_enc = self.inner.encode(suffix)?;
                // Reserve exact capacity so appending the suffix doesn't grow-realloc and
                // re-copy the (large) cached prefix.
                let mut merged: Vec<TokenIdType> =
                    Vec::with_capacity(prefix_tokens.len() + suffix_enc.token_ids().len());
                merged.extend_from_slice(&prefix_tokens);
                merged.extend_from_slice(suffix_enc.token_ids());
                Encoding::Sp(merged)
            };
            self.observe_token_usage(cached_tokens, encoding.token_ids().len());
            return Ok(encoding);
        }

        // Miss path: tokenize once, caching the cumulative prefix at every boundary as we
        // go. The returned ids equal an uncached encode (special tokens are atomic), so we
        // avoid the redundant second tokenization a separate full-encode + insert would
        // cost. Returns Encoding::Sp — consistent with the hit path (see the storage-
        // normalization note in the module docs).
        let encoding = Encoding::Sp(self.l1.populate_and_encode(input, self.inner.as_ref())?);
        self.observe_token_usage(0, encoding.token_ids().len());
        Ok(encoding)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        // True passthrough when L1 is disabled — delegate to the inner's native
        // batch path (which may be rayon-parallel for HF) instead of falling
        // through per-item.
        if !self.l1_enabled {
            return self.inner.encode_batch(inputs);
        }

        // Per-item cache lookup — do NOT delegate to inner.encode_batch, which would
        // bypass the cache. Sequential iteration is fine; if rayon is added later it
        // belongs here, not inside `encode`.
        inputs.iter().map(|&i| self.encode(i)).collect()
    }
}

impl Decoder for CachedTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        // Decode is not cached — passthrough to inner.
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CachedTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HuggingFaceTokenizer;
    use std::sync::{Mutex, atomic::AtomicU64, atomic::Ordering};

    struct FailingTokenizer;

    impl Encoder for FailingTokenizer {
        fn encode(&self, _input: &str) -> Result<Encoding> {
            Err(anyhow::anyhow!("intentional encode failure"))
        }

        fn encode_batch(&self, _inputs: &[&str]) -> Result<Vec<Encoding>> {
            Err(anyhow::anyhow!("intentional encode failure"))
        }
    }

    impl Decoder for FailingTokenizer {
        fn decode(
            &self,
            _token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> Result<DecodeResult> {
            Err(anyhow::anyhow!("intentional decode failure"))
        }
    }

    impl Tokenizer for FailingTokenizer {
        fn validate_prefix_cache(&self) -> Result<()> {
            Ok(())
        }
    }

    const TINYLLAMA_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
    );

    fn inner() -> Arc<dyn Tokenizer> {
        Arc::new(HuggingFaceTokenizer::from_file(TINYLLAMA_PATH).expect("load TinyLlama"))
    }

    fn specials() -> Vec<String> {
        vec!["<s>".into(), "</s>".into()]
    }

    fn collect_token_usage(
        tokenizer: CachedTokenizer,
    ) -> (CachedTokenizer, Arc<Mutex<Vec<CacheTokenUsage>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let observed = events.clone();
        let tokenizer = tokenizer.with_token_observer(Arc::new(move |usage| {
            observed.lock().unwrap().push(usage);
        }));
        (tokenizer, events)
    }

    #[test]
    fn rejects_hf_tokenizer_that_adds_special_tokens() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(
            HuggingFaceTokenizer::from_file(TINYLLAMA_PATH)
                .expect("load TinyLlama")
                .with_options(crate::TokenizerOptions {
                    add_special_tokens: true,
                }),
        );

        let result = CachedTokenizer::new(tokenizer, specials(), 4096);
        let Err(error) = result else {
            panic!("add_special_tokens=true must be rejected");
        };
        assert_eq!(
            error.to_string(),
            "HuggingFace tokenizers configured with add_special_tokens=true must remain uncached"
        );
    }

    #[test]
    fn empty_specials_passes_through_correctly() {
        // L1 disabled by empty specials list — encode must produce correct ids
        // AND short-circuit to the inner tokenizer (no miss-counter bump, no
        // insert attempt). Otherwise the tiktoken integration would log a
        // miss per request with zero hits forever.
        let tok = inner();
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(tok.clone(), Vec::new(), 4096)
                .expect("TinyLlama must support prefix caching"),
        );
        let s = "<s>hello world</s>";
        let a = cached.encode(s).unwrap();
        let b = tok.encode(s).unwrap();
        assert_eq!(a.token_ids(), b.token_ids());
        let stats = cached.cache_stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.misses, 0, "empty specials must not increment misses");
        assert_eq!(stats.hits, 0);
        assert!(
            events.lock().unwrap().is_empty(),
            "empty specials must not emit token usage"
        );
    }

    #[test]
    fn token_observer_reports_full_miss_and_partial_hit_with_and_without_extension() {
        for extend_on_hit in [false, true] {
            let tok = inner();
            let hits = Arc::new(AtomicU64::new(0));
            let misses = Arc::new(AtomicU64::new(0));
            let hit_counter = hits.clone();
            let miss_counter = misses.clone();
            let cached = CachedTokenizer::new(tok, specials(), 64 * 1024)
                .expect("TinyLlama must support prefix caching")
                .with_extend(extend_on_hit)
                .with_observer(
                    Arc::new(move || {
                        hit_counter.fetch_add(1, Ordering::Relaxed);
                    }),
                    Arc::new(move || {
                        miss_counter.fetch_add(1, Ordering::Relaxed);
                    }),
                );
            let (cached, events) = collect_token_usage(cached);

            let shared = "<s>system\nYou are helpful.</s><s>user\n";
            let first = format!("{shared}First question?</s>");
            let second = format!("{shared}Second different prompt entirely.</s>");

            let first_encoding = cached.encode(&first).unwrap();
            let second_encoding = cached.encode(&second).unwrap();

            let events = events.lock().unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(
                events[0],
                CacheTokenUsage {
                    cached_tokens: 0,
                    uncached_tokens: first_encoding.token_ids().len(),
                }
            );
            assert!(events[1].cached_tokens > 0);
            assert!(events[1].uncached_tokens > 0);
            assert_eq!(
                events[1].cached_tokens + events[1].uncached_tokens,
                second_encoding.token_ids().len()
            );
            assert_eq!(hits.load(Ordering::Relaxed), 1);
            assert_eq!(misses.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn token_observer_does_not_report_failed_encodes() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(FailingTokenizer);
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(tokenizer, specials(), 4096)
                .expect("test tokenizer explicitly supports prefix caching"),
        );

        assert!(cached.encode("<s>this fails</s>").is_err());
        assert!(events.lock().unwrap().is_empty());
    }

    #[test]
    fn two_turn_chat_correctness_and_hit() {
        let tok = inner();
        let cached = CachedTokenizer::new(tok.clone(), specials(), 64 * 1024)
            .expect("TinyLlama must support prefix caching");

        let template = "<s>system\nYou are helpful.</s><s>user\n";
        let first = format!("{template}First question?</s>");
        let second = format!("{template}Second different prompt entirely.</s>");

        // Warm the cache.
        let _ = cached.encode(&first).unwrap();

        // Second request: shared prefix → L1 hit, suffix-only fresh encode.
        let cached_second = cached.encode(&second).unwrap();
        let plain_second = tok.encode(&second).unwrap();
        assert_eq!(
            cached_second.token_ids(),
            plain_second.token_ids(),
            "cached encode must equal plain encode for second turn"
        );

        let stats = cached.cache_stats();
        assert!(stats.hits >= 1, "expected L1 hit on second request");
    }

    #[test]
    fn decode_passes_through() {
        let tok = inner();
        let cached = CachedTokenizer::new(tok.clone(), specials(), 4096)
            .expect("TinyLlama must support prefix caching");
        let enc = cached.encode("<s>hello</s>").unwrap();
        let direct = tok.decode(enc.token_ids(), false).unwrap();
        let through = cached.decode(enc.token_ids(), false).unwrap();
        assert_eq!(direct, through);
    }

    #[test]
    fn encode_batch_uses_cache() {
        let tok = inner();
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(tok.clone(), specials(), 64 * 1024)
                .expect("TinyLlama must support prefix caching"),
        );
        let shared = "<s>system\nShared persona.</s><s>user\n";
        let inputs = [
            format!("{shared}q1</s>"),
            format!("{shared}q2</s>"),
            format!("{shared}q3</s>"),
        ];
        let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
        let outs = cached.encode_batch(&refs).unwrap();
        assert_eq!(outs.len(), 3);
        let events = events.lock().unwrap();
        assert_eq!(events.len(), outs.len());
        for (event, output) in events.iter().zip(&outs) {
            assert_eq!(
                event.cached_tokens + event.uncached_tokens,
                output.token_ids().len()
            );
        }
        assert_eq!(events[0].cached_tokens, 0);
        assert!(events[1..].iter().all(|event| event.cached_tokens > 0));
        // First call populates, second/third hit.
        assert!(cached.cache_stats().hits >= 2, "expected hits on q2 and q3");
    }
}
