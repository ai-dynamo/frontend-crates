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
//! Both the plain-text `encode` path and the segmented `encode_segments` path
//! (Kimi K3-style renderer output) are served from the same cache.
//!
//! # Correctness
//!
//! Boundaries are taken **only** at positions immediately following a registered
//! special token (e.g. `<|im_start|>`, `<|im_end|>`, `<s>`, `</s>`). Special tokens
//! are atomic in BPE (`special: true, normalized: false`), so splitting there
//! preserves the invariant `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`.
//! No fallback to whitespace or punctuation — better to miss than to corrupt.
//!
//! Atomicity alone is insufficient when registered special-token strings can overlap.
//! [`CachedTokenizer::new`] disables L1 for such sets because the boundary scanner could
//! otherwise split inside the token selected by the underlying tokenizer.
//!
//! Segmented inputs cut at segment ends instead of scanning for special tokens. That is
//! exact only when the inner [`Encoder::encode_segments`] encodes each segment
//! independently and concatenates the ids, so the path is enabled only for tokenizers
//! that opt in through [`Tokenizer::validate_segmented_prefix_cache`]; others keep the
//! uncached passthrough. Keys for this path frame each segment with its `allow_special`
//! flag and byte length under a separate blake3 derive-key context, so a segmented
//! prefix never aliases a plain-text prefix, a different trust layout, or a different
//! split of the same flattened text.
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
//!   insert attempt. A list whose members can overlap disables L1 identically.
//! - `encode_segments` is cached at segment boundaries with trust-aware keys (see
//!   above) when the inner tokenizer opts in; it never flattens segments, so the
//!   special-token trust boundaries reach the inner tokenizer unchanged. With L1
//!   disabled, or for a tokenizer that has not opted in, it passes straight through.
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

use l1::PrefixLookup;
pub use l1::{CacheEventFn, L1Cache, L1CacheStats};

use crate::{
    EncodeSegment, Encoding, Result, TokenIdType,
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
    l1_enabled: bool,
    /// The inner tokenizer declared composable segmented encoding, so
    /// `encode_segments` may split at segment boundaries.
    segments_cacheable: bool,
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
    /// without touching the cache or its counters. An overlapping token set also disables
    /// L1, with a warning, because its boundaries are ambiguous.
    ///
    /// `max_memory_bytes` is the L1 cache byte budget.
    ///
    /// # Errors
    ///
    /// Returns the inner tokenizer's compatibility error when it cannot be
    /// safely wrapped in the prefix cache.
    pub fn new(
        inner: Arc<dyn Tokenizer>,
        mut special_tokens: Vec<String>,
        max_memory_bytes: usize,
    ) -> Result<Self> {
        inner.validate_prefix_cache()?;
        special_tokens.retain(|token| !token.is_empty());

        // Overlapping matches can create a cache boundary inside a token selected by the
        // inner tokenizer. Preserve correctness by bypassing this optional optimization.
        let overlapping_specials = match l1::first_unsafe_overlap(&special_tokens) {
            Some((first, second)) => {
                tracing::warn!(
                    target: "tokenizer",
                    first_token = first,
                    second_token = second,
                    special_token_count = special_tokens.len(),
                    "special tokens can overlap; tokenizer prefix cache disabled"
                );
                true
            }
            None => false,
        };

        let l1_enabled = !special_tokens.is_empty() && !overlapping_specials;
        let cache_tokens = if l1_enabled {
            special_tokens
        } else {
            Vec::new()
        };
        let segments_cacheable = match inner.validate_segmented_prefix_cache() {
            Ok(()) => true,
            Err(reason) => {
                if l1_enabled {
                    tracing::info!(
                        target: "tokenizer",
                        %reason,
                        "segmented encodes bypass the tokenizer prefix cache"
                    );
                }
                false
            }
        };
        Ok(Self {
            inner,
            l1: L1Cache::new(max_memory_bytes, cache_tokens),
            l1_enabled,
            segments_cacheable,
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
        if !self.l1_enabled {
            return self.inner.encode(input);
        }

        let matched = match self.l1.lookup_prefix(input) {
            PrefixLookup::Hit(matched) => matched,
            PrefixLookup::Miss(prefix_hashes) => {
                let encoding = Encoding::Sp(self.l1.populate_and_encode_with_hashes(
                    input,
                    prefix_hashes.into_iter(),
                    self.inner.as_ref(),
                )?);
                self.observe_token_usage(0, encoding.token_ids().len());
                return Ok(encoding);
            }
        };

        let cached_tokens = matched.tokens.len();
        let encoding = if self.extend_on_hit {
            Encoding::Sp(self.l1.extend_after_match_with_hash(
                input,
                matched,
                self.inner.as_ref(),
            )?)
        } else {
            let suffix_enc = self.inner.encode(&input[matched.prefix_len..])?;
            // Reserve once to avoid copying the cached prefix during vector growth.
            let mut merged: Vec<TokenIdType> =
                Vec::with_capacity(matched.tokens.len() + suffix_enc.token_ids().len());
            merged.extend_from_slice(&matched.tokens);
            merged.extend_from_slice(suffix_enc.token_ids());
            Encoding::Sp(merged)
        };
        self.observe_token_usage(cached_tokens, encoding.token_ids().len());
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

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        if !self.l1_enabled {
            return self.inner.encode_segments(segments);
        }
        if !self.segments_cacheable {
            let encoding = self.inner.encode_segments(segments)?;
            self.observe_token_usage(0, encoding.token_ids().len());
            return Ok(encoding);
        }

        // Segments are never flattened: cut points are segment ends and the uncached
        // remainder reaches the inner tokenizer with its trust flags intact.
        let matched = match self.l1.lookup_prefix_segments(segments) {
            PrefixLookup::Hit(matched) => matched,
            PrefixLookup::Miss(prefix_hashes) => {
                let encoding = Encoding::Sp(self.l1.populate_and_encode_segments_with_hashes(
                    segments,
                    prefix_hashes.into_iter(),
                    self.inner.as_ref(),
                )?);
                self.observe_token_usage(0, encoding.token_ids().len());
                return Ok(encoding);
            }
        };

        let cached_tokens = matched.tokens.len();
        let encoding = if self.extend_on_hit {
            Encoding::Sp(self.l1.extend_after_match_segments_with_hash(
                segments,
                matched,
                self.inner.as_ref(),
            )?)
        } else {
            // On this path `prefix_len` counts complete leading segments.
            let suffix_enc = self
                .inner
                .encode_segments(&segments[matched.prefix_len..])?;
            let mut merged: Vec<TokenIdType> =
                Vec::with_capacity(matched.tokens.len() + suffix_enc.token_ids().len());
            merged.extend_from_slice(&matched.tokens);
            merged.extend_from_slice(suffix_enc.token_ids());
            Encoding::Sp(merged)
        };
        self.observe_token_usage(cached_tokens, encoding.token_ids().len());
        Ok(encoding)
    }
}

impl Decoder for CachedTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        // Decode is not cached — passthrough to inner.
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CachedTokenizer {
    fn vocab_size(&self) -> Option<usize> {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        self.inner.token_to_id(token)
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        self.inner.special_token_ids()
    }

    fn num_special_tokens_added(&self) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HuggingFaceTokenizer;
    use std::sync::{Mutex, atomic::AtomicU64, atomic::Ordering};
    use tokenizers::Tokenizer as HfTokenizer;

    struct FailingTokenizer;

    struct SegmentTokenizer;

    impl Encoder for SegmentTokenizer {
        fn encode(&self, input: &str) -> Result<Encoding> {
            Ok(Encoding::Sp(vec![input.len() as u32]))
        }

        fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
            inputs.iter().map(|input| self.encode(input)).collect()
        }

        fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
            let ids = segments
                .iter()
                .flat_map(|segment| [segment.allow_special as u32, segment.text.len() as u32])
                .collect();
            Ok(Encoding::Sp(ids))
        }
    }

    impl Decoder for SegmentTokenizer {
        fn decode(
            &self,
            _token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> Result<DecodeResult> {
            Ok(DecodeResult::Complete(String::new()))
        }
    }

    impl Tokenizer for SegmentTokenizer {
        fn validate_prefix_cache(&self) -> Result<()> {
            Ok(())
        }

        fn validate_segmented_prefix_cache(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Same output as `SegmentTokenizer`, but never opts into segmented prefix caching.
    struct OpaqueSegmentTokenizer;

    impl Encoder for OpaqueSegmentTokenizer {
        fn encode(&self, input: &str) -> Result<Encoding> {
            SegmentTokenizer.encode(input)
        }

        fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
            SegmentTokenizer.encode_batch(inputs)
        }

        fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
            SegmentTokenizer.encode_segments(segments)
        }
    }

    impl Decoder for OpaqueSegmentTokenizer {
        fn decode(
            &self,
            token_ids: &[TokenIdType],
            skip_special_tokens: bool,
        ) -> Result<DecodeResult> {
            SegmentTokenizer.decode(token_ids, skip_special_tokens)
        }
    }

    impl Tokenizer for OpaqueSegmentTokenizer {
        fn validate_prefix_cache(&self) -> Result<()> {
            Ok(())
        }
    }

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

        fn validate_segmented_prefix_cache(&self) -> Result<()> {
            Ok(())
        }

        fn vocab_size(&self) -> Option<usize> {
            None
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
        // Empty token strings carry no boundary information and must not make L1 active.
        let tok = inner();
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(tok.clone(), vec![String::new()], 4096)
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
    fn laguna_overlapping_specials_bypass_cache() {
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 2, "content": "〈|EOS|〉", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 14, "content": "〈|", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 15, "content": "|〉", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "〈|EOS|〉": 2, "〈|": 14, "|〉": 15, "tail": 16},
                "unk_token": "<unk>"
            }
        }"#;

        let hf = HfTokenizer::from_bytes(TOKENIZER_JSON).expect("load test tokenizer");
        let tok: Arc<dyn Tokenizer> = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf));
        let overlapping = vec!["〈|EOS|〉".into(), "〈|".into(), "|〉".into()];
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(tok.clone(), overlapping, 4096)
                .expect("HuggingFace tokenizer must support prefix caching"),
        );

        let expected = tok.encode("〈|EOS|〉").unwrap();
        assert_eq!(expected.token_ids(), &[2]);
        assert_eq!(
            cached.encode("〈|EOS|〉").unwrap().token_ids(),
            expected.token_ids()
        );
        let stats = cached.cache_stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(
            stats.misses, 0,
            "overlapping specials must not increment misses"
        );
        assert_eq!(stats.hits, 0);
        assert!(
            events.lock().unwrap().is_empty(),
            "overlapping specials must not emit token usage"
        );
    }

    #[test]
    fn segmented_encoding_without_specials_passes_through() {
        let inner: Arc<dyn Tokenizer> = Arc::new(SegmentTokenizer);
        let segments = [
            EncodeSegment::new("<ctl>", true),
            EncodeSegment::new("user content", false),
        ];
        let expected = inner.encode_segments(&segments).unwrap();
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(inner.clone(), Vec::new(), 4096)
                .expect("test tokenizer supports prefix caching"),
        );

        let actual = cached.encode_segments(&segments).unwrap();

        assert_eq!(actual.token_ids(), expected.token_ids());
        let stats = cached.cache_stats();
        assert_eq!((stats.entries, stats.hits, stats.misses), (0, 0, 0));
        assert!(
            events.lock().unwrap().is_empty(),
            "disabled L1 must not emit token usage"
        );
    }

    #[test]
    fn segmented_encoding_without_opt_in_passes_through_uncached() {
        // The inner tokenizer supports segments and the text-path cache, but has not
        // declared segment composability: ids come straight from it, nothing is cached,
        // and the observer still sees the fully uncached encode.
        let inner: Arc<dyn Tokenizer> = Arc::new(OpaqueSegmentTokenizer);
        let segments = [
            EncodeSegment::new("<ctl>", true),
            EncodeSegment::new("user content", false),
            EncodeSegment::new("<ctl>", true),
        ];
        let expected = inner.encode_segments(&segments).unwrap();
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(inner.clone(), vec!["<ctl>".to_string()], 4096)
                .expect("test tokenizer supports prefix caching"),
        );

        for _ in 0..2 {
            let actual = cached.encode_segments(&segments).unwrap();
            assert_eq!(actual.token_ids(), expected.token_ids());
        }
        let stats = cached.cache_stats();
        assert_eq!((stats.entries, stats.hits, stats.misses), (0, 0, 0));
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[
                CacheTokenUsage {
                    cached_tokens: 0,
                    uncached_tokens: 6,
                },
                CacheTokenUsage {
                    cached_tokens: 0,
                    uncached_tokens: 6,
                },
            ]
        );

        // The text path of the same tokenizer is unaffected by the missing opt-in.
        let _ = cached.encode("<ctl>user content<ctl>").unwrap();
        assert!(cached.cache_stats().entries > 0);
    }

    #[test]
    fn segmented_single_segment_has_no_boundary_and_passes_through_as_a_miss() {
        // One segment has no cut point: the lookup records a miss, nothing is inserted, and
        // the ids are the inner tokenizer's (mirrors a plain-text input with no specials).
        let inner: Arc<dyn Tokenizer> = Arc::new(SegmentTokenizer);
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(inner.clone(), vec!["<ctl>".to_string()], 4096)
                .expect("test tokenizer supports prefix caching"),
        );
        let only = [EncodeSegment::new("<ctl>lonely", true)];

        let actual = cached.encode_segments(&only).unwrap();
        assert_eq!(
            actual.token_ids(),
            inner.encode_segments(&only).unwrap().token_ids()
        );
        let stats = cached.cache_stats();
        assert_eq!((stats.entries, stats.hits, stats.misses), (0, 0, 1));
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[CacheTokenUsage {
                cached_tokens: 0,
                uncached_tokens: 2,
            }]
        );
    }

    #[test]
    fn segmented_encoding_populates_then_hits_at_segment_boundaries() {
        // SegmentTokenizer emits `[allow_special, len]` per segment, so a wrong split or
        // a stale prefix shows up as a token-id mismatch, not just a bad counter.
        let inner: Arc<dyn Tokenizer> = Arc::new(SegmentTokenizer);
        let turn1 = [
            EncodeSegment::new("<ctl>", true),
            EncodeSegment::new("user content", false),
            EncodeSegment::new("<ctl>", true),
        ];
        let mut turn2 = turn1.to_vec();
        turn2.extend([
            EncodeSegment::new("assistant reply", false),
            EncodeSegment::new("<ctl>", true),
        ]);
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(inner.clone(), vec!["<ctl>".to_string()], 4096)
                .expect("test tokenizer supports prefix caching"),
        );

        // Miss: one entry after every complete leading segment except the last.
        let first = cached.encode_segments(&turn1).unwrap();
        assert_eq!(
            first.token_ids(),
            inner.encode_segments(&turn1).unwrap().token_ids()
        );
        let stats = cached.cache_stats();
        assert_eq!(stats.entries, 2);
        assert_eq!((stats.hits, stats.misses), (0, 1));

        // Hit: the deepest cached boundary covers turn1's first two segments (4 ids);
        // only the remaining three segments are freshly encoded.
        let second = cached.encode_segments(&turn2).unwrap();
        assert_eq!(
            second.token_ids(),
            inner.encode_segments(&turn2).unwrap().token_ids()
        );
        let stats = cached.cache_stats();
        assert_eq!((stats.hits, stats.misses), (1, 1));
        assert_eq!(stats.entries, 2, "extend is off, so a hit must not insert");
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[
                CacheTokenUsage {
                    cached_tokens: 0,
                    uncached_tokens: 6,
                },
                CacheTokenUsage {
                    cached_tokens: 4,
                    uncached_tokens: 6,
                },
            ]
        );
    }

    #[test]
    fn segmented_keys_do_not_alias_across_trust_or_split_differences() {
        let inner: Arc<dyn Tokenizer> = Arc::new(SegmentTokenizer);
        let cached = CachedTokenizer::new(inner.clone(), vec!["<ctl>".to_string()], 4096)
            .expect("test tokenizer supports prefix caching");

        let trusted = [
            EncodeSegment::new("<ctl>", true),
            EncodeSegment::new("a", false),
            EncodeSegment::new("b", false),
        ];
        // Same flattened text as `trusted`, but the marker is untrusted content here...
        let untrusted = [
            EncodeSegment::new("<ctl>", false),
            EncodeSegment::new("a", false),
            EncodeSegment::new("b", false),
        ];
        // ...and here the split falls elsewhere.
        let resplit = [
            EncodeSegment::new("<ctl>a", true),
            EncodeSegment::new("b", false),
        ];

        let _ = cached.encode_segments(&trusted).unwrap();
        for candidate in [&untrusted[..], &resplit[..]] {
            let expected = inner.encode_segments(candidate).unwrap();
            let actual = cached.encode_segments(candidate).unwrap();
            assert_eq!(actual.token_ids(), expected.token_ids());
        }
        assert_eq!(
            cached.cache_stats().hits,
            0,
            "identical flattened text with a different trust layout or split must not hit"
        );
    }

    #[test]
    fn segmented_and_plain_text_keys_are_domain_separated() {
        // `encode` emits `[len]` per boundary run and `encode_segments` emits `[flag, len]`
        // per segment, so a key shared across the two paths would return the wrong ids.
        // The plain-text encode here only seeds plain-text keys for the same bytes; its
        // own output is not the subject (this mock is not prefix-stable under `encode`).
        let inner: Arc<dyn Tokenizer> = Arc::new(SegmentTokenizer);
        let cached = CachedTokenizer::new(inner.clone(), vec!["<ctl>".to_string()], 4096)
            .expect("test tokenizer supports prefix caching");
        let segments = [
            EncodeSegment::new("<ctl>", true),
            EncodeSegment::new("user content", false),
            EncodeSegment::new("<ctl>", true),
        ];
        let flattened: String = segments.iter().map(|segment| segment.text).collect();

        let _ = cached.encode(&flattened).unwrap();
        assert!(
            cached.cache_stats().entries > 0,
            "plain-text miss must populate"
        );

        let segmented = cached.encode_segments(&segments).unwrap();
        assert_eq!(
            segmented.token_ids(),
            inner.encode_segments(&segments).unwrap().token_ids()
        );
        assert_eq!(
            cached.cache_stats().hits,
            0,
            "segmented lookup must not hit plain-text entries for the same bytes"
        );
    }

    #[test]
    fn segmented_extend_on_hit_caches_only_the_deepest_boundary() {
        let inner: Arc<dyn Tokenizer> = Arc::new(SegmentTokenizer);
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(inner.clone(), vec!["<ctl>".to_string()], 64 * 1024)
                .expect("test tokenizer supports prefix caching")
                .with_extend(true),
        );
        let mut convo = vec![
            EncodeSegment::new("<ctl>", true),
            EncodeSegment::new("system prompt", false),
            EncodeSegment::new("<ctl>", true),
        ];

        let _ = cached.encode_segments(&convo).unwrap();
        let mut entries = cached.cache_stats().entries;
        assert_eq!(entries, 2);

        for turn in 1..=4 {
            convo.extend([
                EncodeSegment::new("user text", false),
                EncodeSegment::new("<ctl>", true),
                EncodeSegment::new("assistant text", false),
                EncodeSegment::new("<ctl>", true),
            ]);
            let actual = cached.encode_segments(&convo).unwrap();
            assert_eq!(
                actual.token_ids(),
                inner.encode_segments(&convo).unwrap().token_ids(),
                "turn {turn}"
            );
            let stats = cached.cache_stats();
            assert_eq!(
                stats.entries,
                entries + 1,
                "turn {turn}: a partial hit with extend on adds exactly one entry"
            );
            entries = stats.entries;
        }

        // Turn 0 is a full miss (3 segments, 6 ids). Every later turn reuses everything up
        // to the previous turn's deepest boundary and encodes the four new segments plus
        // the final one (10 ids), so the uncached work stays flat as the history grows.
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 5);
        assert_eq!(
            events[0],
            CacheTokenUsage {
                cached_tokens: 0,
                uncached_tokens: 6,
            }
        );
        for (turn, usage) in events.iter().enumerate().skip(1) {
            assert_eq!(usage.uncached_tokens, 10, "turn {turn}");
            assert_eq!(usage.cached_tokens, 4 + 8 * (turn - 1), "turn {turn}");
        }
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
    fn token_observer_does_not_report_failed_segmented_encodes() {
        // FailingTokenizer keeps the trait default `encode_segments`, which errors, so the
        // segmented miss path fails inside the inner encode: no event, no entry.
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(FailingTokenizer);
        let (cached, events) = collect_token_usage(
            CachedTokenizer::new(tokenizer, specials(), 4096)
                .expect("test tokenizer explicitly supports prefix caching"),
        );
        let segments = [
            EncodeSegment::new("<s>", true),
            EncodeSegment::new("this fails", false),
            EncodeSegment::new("</s>", true),
        ];

        assert!(cached.encode_segments(&segments).is_err());
        assert!(events.lock().unwrap().is_empty());
        assert_eq!(cached.cache_stats().entries, 0);
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

    #[test]
    fn vocab_introspection_forwards_to_inner() {
        let tok = inner();
        let cached = CachedTokenizer::new(tok.clone(), specials(), 4096)
            .expect("TinyLlama must support prefix caching");
        assert_eq!(cached.vocab_size(), tok.vocab_size());
        assert_eq!(
            cached.token_to_id("<s>").unwrap(),
            tok.token_to_id("<s>").unwrap()
        );
        assert_eq!(
            cached.special_token_ids().unwrap(),
            tok.special_token_ids().unwrap()
        );
    }

    #[test]
    fn special_token_accounting_matches_cached_encoder_behavior() {
        let cached = CachedTokenizer::new(inner(), specials(), 4096)
            .expect("TinyLlama must support prefix caching")
            .with_options(crate::TokenizerOptions {
                add_special_tokens: true,
            });
        let cached_ids = cached.encode("hello").unwrap();
        let hf_ids = HuggingFaceTokenizer::from_file(TINYLLAMA_PATH)
            .expect("load TinyLlama")
            .with_options(crate::TokenizerOptions {
                add_special_tokens: true,
            })
            .encode("hello")
            .unwrap();

        assert_eq!(cached.num_special_tokens_added().unwrap(), 0);
        assert_eq!(hf_ids.token_ids().len(), cached_ids.token_ids().len() + 1);
        assert_eq!(&hf_ids.token_ids()[1..], cached_ids.token_ids());
    }

    #[test]
    fn unoverridden_introspection_methods_use_defaults() {
        let tokenizer = SegmentTokenizer;
        assert_eq!(tokenizer.vocab_size(), None);
        assert!(tokenizer.token_to_id("anything").is_err());
        assert!(tokenizer.special_token_ids().is_err());
        assert!(tokenizer.num_special_tokens_added().is_err());
    }
}
