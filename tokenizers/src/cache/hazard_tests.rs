// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer setups where the text after a special token encodes
//! differently on its own than inside the whole prompt. A prefix cache that
//! splits there gives other IDs than an uncached encode, so the cache must
//! turn itself off. The setups that are safe must keep the cache.

use std::sync::Arc;

use serde_json::{Value, json};
use tokenizers::Tokenizer as HfTokenizer;

use super::CachedTokenizer;
use crate::HuggingFaceTokenizer;
use crate::traits::{Encoder, Tokenizer};

/// An added token as `tokenizer.json` lists it, with every flag off.
fn added(id: u32, content: &str, special: bool) -> Value {
    json!({
        "id": id, "content": content, "special": special, "single_word": false,
        "lstrip": false, "rstrip": false, "normalized": false
    })
}

/// `added` with `flag` set.
fn flagged(id: u32, content: &str, special: bool, flag: &str) -> Value {
    let mut token = added(id, content, special);
    token[flag] = json!(true);
    token
}

/// A BPE `tokenizer.json` with no merges, so each character of `vocab` is
/// one token.
fn tokenizer_json(
    added_tokens: Value,
    normalizer: Value,
    pre_tokenizer: Value,
    vocab: Value,
) -> Value {
    json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added_tokens, "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer, "post_processor": null, "decoder": null,
        "model": {"type": "BPE", "vocab": vocab, "merges": []}
    })
}

fn hf(json: &Value) -> Arc<dyn Tokenizer> {
    let hf = HfTokenizer::from_bytes(json.to_string()).expect("load test tokenizer");
    Arc::new(HuggingFaceTokenizer::from_tokenizer(hf))
}

/// The special tokens of `json`, as dynamo's model card takes them.
fn specials(json: &Value) -> Vec<String> {
    json["added_tokens"]
        .as_array()
        .expect("added tokens")
        .iter()
        .filter(|token| token["special"] == json!(true))
        .map(|token| token["content"].as_str().expect("content").to_owned())
        .collect()
}

fn ids(tokenizer: &dyn Encoder, input: &str) -> Vec<u32> {
    tokenizer
        .encode(input)
        .expect("encode")
        .token_ids()
        .to_vec()
}

/// `input` encoded uncached, then twice through the cache: a miss, then a
/// hit. Also whether the cache stored anything.
fn three_ways(
    tokenizer: Arc<dyn Tokenizer>,
    specials: Vec<String>,
    input: &str,
) -> (Vec<u32>, [Vec<u32>; 2], bool) {
    let uncached = ids(tokenizer.as_ref(), input);
    let cached = CachedTokenizer::new(tokenizer, specials, 1 << 20).expect("cacheable");
    let cold = ids(&cached, input);
    let warm = ids(&cached, input);
    let stats = cached.cache_stats();
    (uncached, [cold, warm], stats.misses + stats.hits > 0)
}

/// `rstrip` on a special token: it takes the spaces after it, which go back
/// to the next part when the cache splits after it.
#[test]
fn a_special_token_that_strips_spaces_after_it_turns_the_cache_off() {
    let json = tokenizer_json(
        json!([flagged(3, "<sep>", true, "rstrip")]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1, " ": 2}),
    );
    let (uncached, cached, used) = three_ways(hf(&json), specials(&json), "a<sep> b");
    assert_eq!(
        uncached,
        [0, 3, 1],
        "control: the special token takes the space"
    );
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(!used, "the cache is off");
}

/// `lstrip` on a special token is safe: the spaces it takes come before
/// it, in the same part as the token.
#[test]
fn a_special_token_that_strips_spaces_before_it_keeps_the_cache() {
    let json = tokenizer_json(
        json!([flagged(3, "<sep>", true, "lstrip"), added(4, "<end>", true)]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1, " ": 2}),
    );
    for input in ["a <sep>b", "a<end> <sep>b"] {
        let (uncached, cached, used) = three_ways(hf(&json), specials(&json), input);
        assert_eq!(cached, [uncached.clone(), uncached], "{input}");
        assert!(used, "{input}: the cache is on");
    }
}

/// A special token matched in normalized text, with a normalizer that
/// rewrites text around it: the whole prompt normalizes across the place
/// where the cache splits.
#[test]
fn a_normalized_special_token_with_a_normalizer_turns_the_cache_off() {
    let json = tokenizer_json(
        json!([flagged(2, "<sep>", true, "normalized")]),
        json!({"type": "Replace", "pattern": {"String": "<sep>b"}, "content": "a"}),
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let (uncached, cached, used) = three_ways(hf(&json), specials(&json), "<sep>b");
    assert_eq!(
        uncached,
        [0],
        "control: the normalizer rewrites the whole input"
    );
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(!used, "the cache is off");
}

/// With no normalizer, normalized matching is plain matching: safe.
#[test]
fn a_normalized_special_token_without_a_normalizer_keeps_the_cache() {
    let json = tokenizer_json(
        json!([flagged(2, "<sep>", true, "normalized")]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let (uncached, cached, used) = three_ways(hf(&json), specials(&json), "a<sep>b");
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(used, "the cache is on");
}

/// `tokenizer.json` matches the token in normalized text, and
/// `tokenizer_config.json` says it is not normalized. The merge changes the
/// flag the tokenizer reports, but the tokenizer still matches the token in
/// normalized text.
#[test]
fn a_normalized_special_token_the_config_calls_plain_turns_the_cache_off() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = tokenizer_json(
        json!([flagged(2, "<sep>", true, "normalized")]),
        json!({"type": "Replace", "pattern": {"String": "<sep>b"}, "content": "a"}),
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let config = json!({"added_tokens_decoder": {"2": added(2, "<sep>", true)}});
    std::fs::write(dir.path().join("tokenizer.json"), json.to_string()).expect("write");
    std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).expect("write");
    let path = dir.path().join("tokenizer.json");
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(HuggingFaceTokenizer::from_file(path.to_str().expect("UTF-8")).expect("load"));
    let (uncached, cached, used) = three_ways(tokenizer, specials(&json), "<sep>b");
    assert_eq!(
        uncached,
        [0],
        "control: the token is still matched after normalization"
    );
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(!used, "the cache is off");
}

/// A plain added token that holds a special token: the tokenizer matches
/// the longer token, and the cache splits inside it.
#[test]
fn a_special_token_inside_a_plain_added_token_turns_the_cache_off() {
    let json = tokenizer_json(
        json!([added(3, "<sep>", true), added(4, "a<sep>b", false)]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let (uncached, cached, used) = three_ways(hf(&json), specials(&json), "a<sep>b");
    assert_eq!(
        uncached.len(),
        1,
        "control: the tokenizer takes the longer token"
    );
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(!used, "the cache is off");
}

/// An added token that matches only as a whole word: the character before
/// the cache's split decides whether it matches.
#[test]
fn a_whole_word_added_token_turns_the_cache_off() {
    let json = tokenizer_json(
        json!([
            added(2, "SEP", true),
            flagged(3, "ab", false, "single_word")
        ]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let (uncached, cached, used) = three_ways(hf(&json), specials(&json), "SEPab");
    assert_eq!(
        uncached,
        [2, 0, 1],
        "control: `ab` is not a whole word here"
    );
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(!used, "the cache is off");
}

/// A Metaspace pre-tokenizer that marks only the first word of the text:
/// the cache encodes the part after a split as a new text, and marks it.
#[test]
fn a_metaspace_that_marks_only_the_first_word_turns_the_cache_off() {
    let json = tokenizer_json(
        json!([added(3, "<sep>", true)]),
        Value::Null,
        json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "first", "split": true}),
        json!({"a": 0, "\u{2581}": 1, "b": 2}),
    );
    let (uncached, cached, used) = three_ways(hf(&json), specials(&json), "a<sep>b");
    assert_eq!(
        uncached,
        [1, 0, 3, 2],
        "control: only the first word is marked"
    );
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(!used, "the cache is off");
}

/// A Metaspace pre-tokenizer that marks every word is safe.
#[test]
fn a_metaspace_that_marks_every_word_keeps_the_cache() {
    let json = tokenizer_json(
        json!([added(3, "<sep>", true)]),
        Value::Null,
        json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true}),
        json!({"a": 0, "\u{2581}": 1, "b": 2}),
    );
    let (uncached, cached, used) = three_ways(hf(&json), specials(&json), "a<sep>b");
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(used, "the cache is on");
}

/// The Baseten backend reads `tokenizer.json` itself, and gets the same
/// check: here a special token with `rstrip`.
#[test]
fn the_baseten_backend_turns_the_cache_off_too() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = tokenizer_json(
        json!([flagged(3, "<sep>", true, "rstrip")]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1, " ": 2}),
    );
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, json.to_string()).expect("write");
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(crate::BasetenTokenizer::from_file(path.to_str().expect("UTF-8")).expect("load"));
    let (uncached, cached, used) = three_ways(tokenizer, specials(&json), "a<sep> b");
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(!used, "the cache is off");
}

/// The fastokens backend reads the check from its Hugging Face side, which
/// loads the same `tokenizer.json`.
#[test]
fn the_fastokens_backend_turns_the_cache_off_too() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = tokenizer_json(
        json!([flagged(3, "<sep>", true, "rstrip")]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1, " ": 2}),
    );
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, json.to_string()).expect("write");
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(crate::FastTokenizer::from_file(path.to_str().expect("UTF-8")).expect("load"));
    let (uncached, cached, used) = three_ways(tokenizer, specials(&json), "a<sep> b");
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(!used, "the cache is off");
}

#[test]
fn a_normalized_special_inside_a_normalized_plain_token_turns_the_cache_off() {
    let json = tokenizer_json(
        json!([
            flagged(2, "<sep>", true, "normalized"),
            flagged(3, "a<sep>b", false, "normalized")
        ]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let (uncached, cached, _) = three_ways(hf(&json), specials(&json), "a<sep>b");
    assert_eq!(uncached, [3], "control: the longer normalized token wins");
    assert_eq!(cached, [uncached.clone(), uncached]);
}

#[test]
fn promoting_a_plain_token_preserves_the_uncached_ids() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = tokenizer_json(
        json!([flagged(2, "<sep>", false, "normalized")]),
        json!({"type": "Replace", "pattern": {"String": "<sep>b"}, "content": "a"}),
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let config = json!({"added_tokens_decoder": {"2": added(2, "<sep>", true)}});
    std::fs::write(dir.path().join("tokenizer.json"), json.to_string()).expect("write");
    std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).expect("write");
    let path = dir.path().join("tokenizer.json");
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().expect("UTF-8")).expect("load");
    assert_eq!(ids(&tokenizer, "<sep>b"), [2, 1]);
}

#[test]
fn lstrip_cannot_consume_a_normalized_whitespace_special_across_a_split() {
    let json = tokenizer_json(
        json!([
            flagged(2, "\n", true, "normalized"),
            flagged(3, "<sep>", true, "lstrip")
        ]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let (uncached, cached, _) = three_ways(hf(&json), specials(&json), "a\n<sep>b");
    assert_eq!(uncached, [0, 3, 1], "control: lstrip consumes the newline");
    assert_eq!(cached, [uncached.clone(), uncached]);
}

#[test]
fn a_raw_special_preempts_a_normalized_plain_token_in_hf() {
    let json = tokenizer_json(
        json!([
            added(2, "<sep>", true),
            flagged(3, "a<sep>b", false, "normalized")
        ]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let (uncached, cached, used) = three_ways(hf(&json), specials(&json), "a<sep>b");
    assert_eq!(uncached, [0, 2, 1]);
    assert_eq!(cached, [uncached.clone(), uncached]);
    assert!(used, "HF keeps its safe cache");
}

#[test]
fn a_normalized_plain_token_cannot_shadow_a_fast_backend_split() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = tokenizer_json(
        json!([
            added(2, "<sep>", true),
            flagged(3, "a<sep>b", false, "normalized")
        ]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1}),
    );
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, json.to_string()).expect("write");
    let path = path.to_str().expect("UTF-8");

    let backends: [Arc<dyn Tokenizer>; 2] = [
        Arc::new(crate::FastTokenizer::from_file(path).expect("load Fastokens")),
        Arc::new(crate::BasetenTokenizer::from_file(path).expect("load Baseten")),
    ];
    let mut results = Vec::new();
    for backend in backends {
        let (uncached, cached, _) = three_ways(backend, specials(&json), "a<sep>b");
        assert_eq!(uncached, [3], "control: the longer added token wins");
        results.push(cached);
    }
    assert_eq!(
        results,
        vec![[vec![3], vec![3]], [vec![3], vec![3]]],
        "Fastokens and Baseten must match their own uncached IDs"
    );
}

#[test]
fn fastokens_uses_the_encoders_strip_flags_when_the_config_disagrees() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = tokenizer_json(
        json!([flagged(3, "<sep>", true, "rstrip")]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1, " ": 2}),
    );
    let config = json!({"added_tokens_decoder": {"3": added(3, "<sep>", true)}});
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, json.to_string()).expect("write tokenizer");
    std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string())
        .expect("write config");
    let path = path.to_str().expect("UTF-8");
    let hf = HuggingFaceTokenizer::from_file(path).expect("load HF");
    assert_eq!(ids(&hf, "a<sep> b"), [0, 3, 2, 1]);
    assert!(hf.prefix_cache_hazard().is_none());
    let fast: Arc<dyn Tokenizer> =
        Arc::new(crate::FastTokenizer::from_file(path).expect("load fastokens"));
    let (uncached, cached, _) = three_ways(fast, specials(&json), "a<sep> b");
    assert_eq!(
        uncached,
        [0, 3, 1],
        "fastokens keeps tokenizer.json's rstrip"
    );
    assert_eq!(cached, [uncached.clone(), uncached]);
}

#[test]
fn fastokens_checks_plain_added_tokens_loaded_only_from_the_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = tokenizer_json(
        json!([added(3, "<sep>", true)]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1, " ": 2}),
    );
    let config = json!({"added_tokens_decoder": {"4": added(4, "a<sep>b", false)}});
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, json.to_string()).expect("write tokenizer");
    std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string())
        .expect("write config");
    let path = path.to_str().expect("UTF-8");
    let hf = HuggingFaceTokenizer::from_file(path).expect("load HF");
    assert_eq!(ids(&hf, "a<sep>b"), [0, 3, 1]);
    assert!(hf.prefix_cache_hazard().is_none());
    let fast: Arc<dyn Tokenizer> =
        Arc::new(crate::FastTokenizer::from_file(path).expect("load fastokens"));
    let (uncached, cached, _) = three_ways(fast, specials(&json), "a<sep>b");
    assert_eq!(uncached, [4], "fastokens loads the config's plain token");
    assert_eq!(cached, [uncached.clone(), uncached]);
}

#[test]
fn fastokens_checks_strip_flags_of_a_plain_token_the_config_makes_special() {
    let dir = tempfile::tempdir().expect("tempdir");
    let json = tokenizer_json(
        json!([flagged(3, "<sep>", false, "rstrip")]),
        Value::Null,
        Value::Null,
        json!({"a": 0, "b": 1, " ": 2}),
    );
    let config = json!({"added_tokens_decoder": {"3": added(3, "<sep>", true)}});
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, json.to_string()).expect("write tokenizer");
    std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string())
        .expect("write config");
    let path = path.to_str().expect("UTF-8");
    let hf = HuggingFaceTokenizer::from_file(path).expect("load HF");
    assert_eq!(ids(&hf, "a<sep> b"), [0, 3, 2, 1]);
    assert!(hf.prefix_cache_hazard().is_none());
    let fast: Arc<dyn Tokenizer> =
        Arc::new(crate::FastTokenizer::from_file(path).expect("load fastokens"));
    assert!(fast.special_token_ids().expect("special ids").contains(&3));
    let (uncached, cached, _) = three_ways(fast, vec!["<sep>".to_owned()], "a<sep> b");
    assert_eq!(
        uncached,
        [0, 3, 1],
        "fastokens keeps the plain token's rstrip"
    );
    assert_eq!(cached, [uncached.clone(), uncached]);
}
