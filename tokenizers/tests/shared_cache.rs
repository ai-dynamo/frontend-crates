// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_tokenizers::{
    CachedTokenizer, DecodeResult, Encoding, HuggingFaceTokenizer, Result, SharedTokenizerCache,
    cache::L1Cache,
    traits::{Decoder, Encoder, Tokenizer},
};

// Two tokenizers can assign different IDs to exactly the same text.
struct ByteTokenizer(u32);

impl Encoder for ByteTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        Ok(Encoding::Sp(
            input.bytes().map(|b| u32::from(b) + self.0).collect(),
        ))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for ByteTokenizer {
    fn decode(&self, _: &[u32], _: bool) -> Result<DecodeResult> {
        anyhow::bail!("decode is unused in this test")
    }
}

impl Tokenizer for ByteTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        Ok(())
    }
}

fn specials() -> Vec<String> {
    vec!["<s>".into(), "</s>".into()]
}

fn cached(cache: &SharedTokenizerCache, namespace: &[u8], offset: u32) -> CachedTokenizer {
    CachedTokenizer::new_with_cache(
        Arc::new(ByteTokenizer(offset)),
        specials(),
        cache.clone(),
        namespace,
    )
    .unwrap()
    .with_extend(true)
}

#[test]
fn shared_entries_survive_reload_with_isolated_namespaces_and_statistics() {
    let storage = SharedTokenizerCache::new(1024 * 1024);
    let input = "<s>hello</s>tail";
    let a = cached(&storage, b"model-a", 0);
    a.encode(input).unwrap();
    assert_eq!(a.cache_stats().misses, 1);
    drop(a);

    let a = cached(&storage, b"model-a", 0);
    let b = cached(&storage, b"model-b", 256);
    for (tokenizer, offset) in [(&a, 0), (&b, 256)] {
        assert_eq!(
            tokenizer.encode(input).unwrap().token_ids(),
            ByteTokenizer(offset).encode(input).unwrap().token_ids()
        );
    }
    assert_eq!(a.cache_stats().hits, 1, "unchanged reload reuses entries");
    assert_eq!(
        b.cache_stats().misses,
        1,
        "another model cannot reuse those IDs"
    );
    let a_stats = a.cache_stats();
    let b_stats = b.cache_stats();
    let total = storage.stats();
    assert_eq!(total.entries, a_stats.entries + b_stats.entries);
    assert_eq!(
        total.memory_bytes,
        a_stats.memory_bytes + b_stats.memory_bytes
    );

    let a_l1 = L1Cache::new_with_cache(storage.clone(), specials(), b"model-a");
    assert!(!a_l1.is_empty());
    assert_eq!(a_l1.len(), a_stats.entries);
    let unused = L1Cache::new_with_cache(storage.clone(), specials(), b"unused");
    assert!(unused.is_empty());
    assert_eq!(unused.len(), 0);

    // Disabled wrappers must not report another wrapper's namespace usage.
    for boundaries in [vec![], vec!["<s>".into(), "<s>x".into()]] {
        let disabled = CachedTokenizer::new_with_cache(
            Arc::new(ByteTokenizer(0)),
            boundaries,
            storage.clone(),
            b"model-a",
        )
        .unwrap();
        disabled.encode(input).unwrap();
        let stats = disabled.cache_stats();
        assert_eq!((stats.entries, stats.memory_bytes), (0, 0));
        assert_eq!((stats.hits, stats.misses, stats.hit_rate), (0, 0, 0.0));
    }
    assert_eq!(storage.stats(), total);

    let private =
        CachedTokenizer::new(Arc::new(ByteTokenizer(0)), specials(), 1024 * 1024).unwrap();
    private.encode(input).unwrap();
    let stats = private.cache_stats();
    assert_eq!(stats.entries, a_stats.entries);
    assert_eq!(stats.memory_bytes, a_stats.memory_bytes);
}

#[test]
fn shared_misses_and_partial_hits_match_uncached_encoding() {
    let raw: Arc<dyn Tokenizer> = Arc::new(
        HuggingFaceTokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
        ))
        .unwrap(),
    );
    let turns = [
        "<s>system\nHello 世界</s><s>user\nOne",
        "<s>system\nHello 世界</s><s>user\nOne</s><s>assistant\nTwo</s>tail",
        "<s>system\nHello 世界</s><s>user\nOne</s><s>assistant\nTwo</s>tail</s><s>user\nThree",
    ];
    for extend in [false, true] {
        let storage = SharedTokenizerCache::new(1024 * 1024);
        let tokenizer =
            CachedTokenizer::new_with_cache(raw.clone(), specials(), storage, b"parity")
                .unwrap()
                .with_extend(extend);
        for input in turns {
            assert_eq!(
                tokenizer.encode(input).unwrap().token_ids(),
                raw.encode(input).unwrap().token_ids()
            );
        }
        assert_eq!(tokenizer.cache_stats().misses, 1);
        assert_eq!(tokenizer.cache_stats().hits, 2);
    }

    let storage = SharedTokenizerCache::new(1024 * 1024);
    let l1 = L1Cache::new_with_cache(storage, specials(), b"public-l1");
    l1.insert_at_boundaries(turns[0], raw.as_ref()).unwrap();
    let (tokens, offset, deepest) = l1.longest_prefix_match(turns[1]).unwrap();
    assert_eq!(
        l1.extend_after_match(turns[1], tokens, offset, deepest, raw.as_ref())
            .unwrap(),
        raw.encode(turns[1]).unwrap().token_ids()
    );
    let (_, offset, _) = l1.longest_prefix_match(turns[1]).unwrap();
    assert_eq!(offset, deepest);
}

#[test]
fn concurrent_namespaces_use_one_capacity_including_zero_capacity() {
    for capacity in [0, 4096] {
        let storage = SharedTokenizerCache::new(capacity);
        assert_eq!(storage.max_memory_bytes(), capacity);
        std::thread::scope(|scope| {
            for model in 0..4u32 {
                let tokenizer = cached(&storage, &model.to_le_bytes(), model * 256);
                scope.spawn(move || {
                    for index in 0..100 {
                        let input = format!("<s>{index} {} </s>tail", "prompt ".repeat(16));
                        assert_eq!(
                            tokenizer.encode(&input).unwrap().token_ids(),
                            ByteTokenizer(model * 256)
                                .encode(&input)
                                .unwrap()
                                .token_ids()
                        );
                    }
                });
            }
        });
        let stats = storage.stats();
        assert!(stats.memory_bytes <= capacity);
        if capacity == 0 {
            assert_eq!(stats.entries, 0);
        } else {
            assert!(stats.entries > 0);
        }
    }
}
