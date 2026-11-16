// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Throwaway generator. It freezes what the legacy (`tokenizers` 0.23.2) Hugging Face backend
//! returns, so the port to `tokenizers` 1.x can be checked without keeping the legacy crate.
//!
//! ```bash
//! HF_EXPECT_COMMIT=$(git rev-parse --short HEAD) \
//!   cargo test -p dynamo-tokenizers --test gen_hf_expectations -- --ignored --nocapture
//! ```

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use dynamo_tokenizers::{
    DecodeStream, EncodeSegment, FastTokenizer, HuggingFaceTokenizer, TokenizerOptions,
    hf::merge_special_tokens_from_config,
    traits::Tokenizer,
};
use serde_json::{Value, json};
use tokenizers::models::bpe::{BPE, Vocab};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
const OUT: &str = "hf-expectations";

const TEXTS: &[&str] = &[
    "",
    " ",
    "hello world",
    "hello worlds",
    "Hello, world!",
    "  leading and trailing  ",
    "café 北京 😀",
    "e\u{301}\n\t",
    "naïve façade — “quotes” …",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890 3.14159 -42",
    "fn main() {\n    println!(\"hi\");\n}\n",
    "line1\nline2\r\nline3",
    "tabs\tand\u{a0}nbsp",
    "مرحبا بالعالم",
    "こんにちは世界",
    "👩‍👩‍👧‍👦 family 🇺🇸",
    "\u{fffd} replacement",
    "<special>hello<special>",
    "a<|x|>b",
    "UPPER lower MiXeD",
    "hello 😀 北京!",
    "He said hello, world. The lot.",
];

/// Texts whose every id prefix is decoded and streamed: they split multi-byte characters.
const PREFIX_TEXTS: &[&str] = &["hello 😀 北京!", "café 北京 😀"];

fn byte_level_tokenizer() -> tokenizers::Tokenizer {
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let mut vocab = Vocab::from_iter(
        alphabet
            .into_iter()
            .enumerate()
            .map(|(id, c)| (c.to_string(), id as u32)),
    );
    let merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o")];
    for (left, right) in merges {
        vocab.insert(format!("{left}{right}"), vocab.len() as u32);
    }
    let bpe = BPE::builder()
        .vocab_and_merges(
            vocab,
            merges.map(|(a, b)| (a.to_owned(), b.to_owned())).to_vec(),
        )
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, true),
    ));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
}

fn legacy_from(config: &Value) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(config).unwrap()).unwrap()
}

fn with_special(mut tokenizer: tokenizers::Tokenizer) -> tokenizers::Tokenizer {
    tokenizer
        .add_special_tokens([tokenizers::AddedToken::from("<special>", true)])
        .unwrap();
    tokenizer
}

fn out_dir(name: &str) -> PathBuf {
    let dir = Path::new(DATA).join(OUT).join(name);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A configuration as legacy saves it: the shape Dynamo handed the 1.x reader before the port.
fn save_legacy(name: &str, tokenizer: &tokenizers::Tokenizer) {
    tokenizer
        .save(out_dir(name).join("tokenizer.json"), true)
        .unwrap();
}

/// A configuration as authored, never passed through legacy serialization.
fn save_raw(name: &str, config: &Value, tokenizer_config: Option<&Value>) {
    let dir = out_dir(name);
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(config).unwrap() + "\n",
    )
    .unwrap();
    if let Some(tokenizer_config) = tokenizer_config {
        fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_string_pretty(tokenizer_config).unwrap() + "\n",
        )
        .unwrap();
    }
}

fn added(id: u32, content: &str, special: bool) -> Value {
    json!({"id": id, "content": content, "special": special, "single_word": false,
        "lstrip": false, "rstrip": false, "normalized": false})
}

fn write_synthetic_configs() -> Vec<String> {
    let names = std::cell::RefCell::new(Vec::new());
    let legacy = |name: &str, tokenizer: tokenizers::Tokenizer| {
        save_legacy(name, &tokenizer);
        names.borrow_mut().push(name.to_string());
    };

    legacy("byte-level-bpe", with_special(byte_level_tokenizer()));

    let mut config: Value =
        serde_json::from_str(&byte_level_tokenizer().to_string(false).unwrap()).unwrap();
    let byte_level = config["decoder"].clone();
    config["decoder"] = json!({"type": "Sequence", "decoders": [byte_level.clone()]});
    legacy(
        "byte-level-bpe-sequence-decoder",
        with_special(legacy_from(&config)),
    );
    config["decoder"] = json!({"type": "Sequence", "decoders": [byte_level,
        {"type": "Replace", "pattern": {"String": "h"}, "content": "x"}]});
    legacy("byte-level-bpe-replace-decoder", legacy_from(&config));

    let mut normalized_added = byte_level_tokenizer();
    normalized_added
        .with_normalizer(Some(tokenizers::normalizers::Lowercase))
        .unwrap();
    normalized_added
        .add_special_tokens([tokenizers::AddedToken::from("CONTROL", true).normalized(true)])
        .unwrap();
    legacy("byte-level-bpe-normalized-added-token", normalized_added);

    for (name, model, pre_tokenizer, decoder) in [
        (
            "wordpiece",
            json!({"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "##s": 3, "café": 4, "北": 5, "京": 6, "😀": 7}}),
            json!({"type": "Whitespace"}),
            json!({"type": "WordPiece", "prefix": "##", "cleanup": true}),
        ),
        (
            "wordlevel",
            json!({"type": "WordLevel", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "café": 3, "北京": 4, "😀": 5}}),
            json!({"type": "Whitespace"}),
            json!(null),
        ),
        (
            "unigram",
            json!({"type": "Unigram", "unk_id": 0, "byte_fallback": false,
                "vocab": [["<unk>", 0.0], ["hello", -1.0], ["world", -1.0], [" ", -1.0],
                    ["café", -1.0], ["北", -1.0], ["京", -1.0], ["😀", -1.0]]}),
            json!(null),
            json!({"type": "Fuse"}),
        ),
    ] {
        legacy(
            name,
            with_special(legacy_from(&json!({
                "version": "1.0", "model": model, "pre_tokenizer": pre_tokenizer,
                "decoder": decoder, "added_tokens": [], "normalizer": null,
                "post_processor": null, "padding": null, "truncation": null,
            }))),
        );
    }

    // In memory `<new>` is 101; saved and reloaded it is numbered from the vocabulary length.
    let mut sparse = legacy_from(&json!({
        "version": "1.0", "truncation": null, "padding": null, "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "added_tokens": [added(100, "<s>", true)],
        "model": {"type": "WordLevel", "unk_token": "[UNK]",
            "vocab": {"[UNK]": 0, "hello": 1, "<s>": 100}},
    }));
    sparse
        .add_special_tokens([tokenizers::AddedToken::from("<new>", true)])
        .unwrap();
    assert_eq!(sparse.token_to_id("<new>"), Some(101));
    legacy("sparse-added-id", sparse);

    let raw = |name: &str, config: Value, tokenizer_config: Option<Value>| {
        save_raw(name, &config, tokenizer_config.as_ref());
        names.borrow_mut().push(name.to_string());
    };

    // Every byte token, as SentencePiece-derived vocabularies have.
    let mut byte_vocab = serde_json::Map::new();
    for byte in 0..=255u32 {
        byte_vocab.insert(format!("<0x{byte:02X}>"), json!(byte));
    }
    for (id, token) in [" hello", "!", "<eos>"].into_iter().enumerate() {
        byte_vocab.insert(token.to_string(), json!(256 + id));
    }
    raw(
        "byte-fallback-bpe",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(258, "<eos>", true)],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
            "model": {"type": "BPE", "vocab": byte_vocab, "merges": [], "byte_fallback": true}
        }),
        None,
    );

    // Only four of the 256 byte tokens.
    raw(
        "byte-fallback-bpe-partial-alphabet",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(6, "<eos>", true)],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
            "model": {"type": "BPE", "vocab": {"<0x61>": 0, "<0xF5>": 1, "<0xC3>": 2, "<0xA9>": 3,
                " hello": 4, "!": 5, "<eos>": 6}, "merges": [], "byte_fallback": true}
        }),
        None,
    );

    raw(
        "merge-gate-wordlevel",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(0, "<unk>", true)],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<|special_kept|>": 3, "<|special_dropped|>": 4}}
        }),
        Some(json!({"added_tokens_decoder": {
            "3": {"content": "<|special_kept|>", "special": true, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false},
            "4": {"content": "<|special_dropped|>", "special": false, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false}
        }})),
    );

    // The config names a special token that tokenizer.json lists as an ordinary added token,
    // and one that is in neither the vocabulary nor the added tokens.
    raw(
        "merge-promote-wordlevel",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(0, "<unk>", true), added(3, "<|marker|>", false)],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<|marker|>": 3}}
        }),
        Some(json!({"added_tokens_decoder": {
            "3": {"content": "<|marker|>", "special": true, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false},
            "9": {"content": "<|brand_new|>", "special": true, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false}
        }})),
    );

    let bos_template = json!({
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 0}}],
        "special_tokens": {"<bos>": {"id": "<bos>", "ids": [3], "tokens": ["<bos>"]}}
    });
    raw(
        "template-bos-wordlevel",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(0, "<unk>", true), added(3, "<bos>", true)],
            "normalizer": null, "pre_tokenizer": null, "post_processor": bos_template, "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<bos>": 3}}
        }),
        None,
    );

    let char_vocab = |mut vocab: Value, chars: &str| {
        for ch in chars.chars() {
            let id = vocab.as_object().unwrap().len();
            if vocab.get(ch.to_string()).is_none() {
                vocab[ch.to_string()] = json!(id);
            }
        }
        vocab
    };
    raw(
        "segmented-call-bos-bpe",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "normalizer": null, "pre_tokenizer": null, "decoder": {"type": "Fuse"},
            "added_tokens": [added(1, "<|call|>", true), added(2, "<bos>", true)],
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}}},
            "model": {"type": "BPE", "merges": [], "unk_token": "<unk>",
                "vocab": char_vocab(json!({"<unk>": 0, "<|call|>": 1, "<bos>": 2}), "<|cal>")}
        }),
        None,
    );
    raw(
        "segmented-think-bpe",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": {"type": "Fuse"},
            "added_tokens": [added(1, "<think>", false)],
            "model": {"type": "BPE", "merges": [], "unk_token": "<unk>",
                "vocab": char_vocab(json!({"<unk>": 0, "<think>": 1}), "<think>")}
        }),
        None,
    );

    // `H`+`e` merges to id 18, declared here as the added token `He`.
    let mut control_he: Value =
        serde_json::from_str(&fs::read_to_string(format!("{DATA}/minimal-bpe/tokenizer.json")).unwrap())
            .unwrap();
    control_he["added_tokens"] = json!([added(18, "He", true)]);
    raw("minimal-bpe-control-he", control_he, None);

    raw(
        "overlapping-specials-wordlevel",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(0, "<unk>", true), added(2, "〈|EOS|〉", true),
                added(14, "〈|", true), added(15, "|〉", true)],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "〈|EOS|〉": 2, "〈|": 14, "|〉": 15, "tail": 16}}
        }),
        None,
    );

    // Added tokens that strip neighbouring whitespace or match only whole words.
    let flagged = |id: u32, content: &str, special: bool, flag: &str| {
        let mut token = added(id, content, special);
        token[flag] = json!(true);
        token
    };
    raw(
        "added-token-flags-wordlevel",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(0, "[UNK]", true), flagged(3, "<l>", true, "lstrip"),
                flagged(4, "<r>", true, "rstrip"), flagged(5, "<w>", false, "single_word")],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "<l>": 3, "<r>": 4, "<w>": 5, "x": 6, "y": 7}}
        }),
        None,
    );

    // A BERT-style embedding tokenizer: normalizer, `[CLS] $A [SEP]` template, WordPiece decoder.
    raw(
        "bert-wordpiece",
        json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(0, "[UNK]", true), added(1, "[CLS]", true), added(2, "[SEP]", true)],
            "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                "strip_accents": null, "lowercase": true},
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
                "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
                "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
                    "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]}}},
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
            "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "world": 4, "cafe": 5, "##s": 6,
                    "北": 7, "京": 8, ",": 9, "!": 10, ".": 11, "upper": 12, "lower": 13, "mixed": 14,
                    "the": 15, "quick": 16, "dog": 17}}
        }),
        None,
    );

    // Truncation and padding blocks, which serving never applies.
    let mut truncated = legacy_from(&json!({
        "version": "1.0", "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null, "decoder": null, "added_tokens": [added(0, "[UNK]", true), added(3, "[PAD]", true)],
        "padding": null, "truncation": null,
        "model": {"type": "WordLevel", "unk_token": "[UNK]",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "[PAD]": 3}},
    }));
    truncated
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 2,
            ..Default::default()
        }))
        .unwrap();
    truncated.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(4),
        pad_id: 3,
        pad_token: "[PAD]".into(),
        ..Default::default()
    }));
    legacy("truncation-padding-wordlevel", truncated);

    names.into_inner()
}

fn result_ids(result: dynamo_tokenizers::Result<dynamo_tokenizers::Encoding>) -> Value {
    match result {
        Ok(encoding) => json!({"ids": encoding.token_ids()}),
        Err(error) => json!({"error": error.to_string()}),
    }
}

fn decode_pair(tokenizer: &dyn Tokenizer, ids: &[u32]) -> Value {
    let decode = |skip| match tokenizer.decode(ids, skip) {
        Ok(text) => json!(String::from(text)),
        Err(error) => json!({"error": error.to_string()}),
    };
    json!([decode(false), decode(true)])
}

fn stream(tokenizer: &Arc<dyn Tokenizer>, prompt: &[u32], ids: &[u32], skip: bool) -> Value {
    let mut stream = DecodeStream::new(tokenizer.clone(), prompt, skip);
    let chunk = |result: dynamo_tokenizers::Result<Option<String>>| match result {
        Ok(text) => json!(text),
        Err(error) => json!({"error": error.to_string()}),
    };
    let steps: Vec<Value> = ids.iter().map(|id| chunk(stream.step(*id))).collect();
    json!({"steps": steps, "finish": chunk(stream.finish())})
}

fn stream_pair(tokenizer: &Arc<dyn Tokenizer>, ids: &[u32]) -> Value {
    json!([
        stream(tokenizer, &[], ids, false),
        stream(tokenizer, &[], ids, true)
    ])
}

fn sequence_record(tokenizer: &Arc<dyn Tokenizer>, ids: &[u32]) -> Value {
    json!({
        "ids": ids,
        "decode": decode_pair(tokenizer.as_ref(), ids),
        "unstable": [tokenizer.has_unstable_suffix(ids, false), tokenizer.has_unstable_suffix(ids, true)],
        "stream": stream_pair(tokenizer, ids),
    })
}

/// [`HuggingFaceTokenizer::from_file`], without the file's truncation and padding blocks.
/// Serving never applied them: the model card removed both before wrapping the tokenizer.
fn load(
    path: &Path,
    options: TokenizerOptions,
) -> dynamo_tokenizers::Result<HuggingFaceTokenizer> {
    let mut legacy = tokenizers::Tokenizer::from_file(path)
        .map_err(|err| dynamo_tokenizers::Error::msg(format!("Error loading tokenizer: {err}")))?;
    if let Some(parent) = path.parent() {
        merge_special_tokens_from_config(&mut legacy, parent);
    }
    legacy.with_truncation(None).unwrap();
    legacy.with_padding(None);
    Ok(HuggingFaceTokenizer::from_tokenizer(legacy).with_options(options))
}

fn record(model_dir: &str) -> Value {
    let dir = Path::new(DATA).join(model_dir);
    let path = dir.join("tokenizer.json");
    let path_str = path.to_str().unwrap();

    let plain = match load(&path, TokenizerOptions::default()) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            return json!({"model_dir": model_dir, "load_error": error.to_string()});
        }
    };
    let with_special_tokens = load(
        &path,
        TokenizerOptions {
            add_special_tokens: true,
        },
    )
    .unwrap();
    let validate_prefix_cache = plain.validate_prefix_cache().is_ok();
    let validate_segment_prefix_cache = plain.validate_segment_prefix_cache().is_ok();
    let num_special_tokens_added = plain.num_special_tokens_added().unwrap();
    let plain: Arc<dyn Tokenizer> = Arc::new(plain);
    let with_special_tokens: Arc<dyn Tokenizer> = Arc::new(with_special_tokens);
    let fast: Option<Arc<dyn Tokenizer>> = FastTokenizer::from_file(path_str)
        .ok()
        .map(|fast| Arc::new(fast) as Arc<dyn Tokenizer>);

    // The vocabulary as the legacy tokenizer reports it after the same special-token merge.
    let mut legacy = tokenizers::Tokenizer::from_file(&path).unwrap();
    merge_special_tokens_from_config(&mut legacy, &dir);
    let mut added_tokens: Vec<_> = legacy.get_added_tokens_decoder().into_iter().collect();
    added_tokens.sort_by_key(|(id, _)| *id);
    let mut special_tokens: Vec<String> = added_tokens
        .iter()
        .filter(|(_, token)| token.special)
        .map(|(_, token)| token.content.clone())
        .collect();
    special_tokens.sort();
    special_tokens.dedup();
    let token_id_bound = legacy
        .get_vocab(true)
        .into_values()
        .max()
        .map(|id| id as usize + 1);

    let mut texts: Vec<String> = TEXTS.iter().map(|text| text.to_string()).collect();
    let first_added: Vec<&str> = added_tokens
        .iter()
        .take(6)
        .map(|(_, token)| token.content.as_str())
        .collect();
    if model_dir.ends_with("added-token-flags-wordlevel") {
        texts.extend(
            ["hello   <l>   world", "hello<r>   world", "x<w>y <w> x", "<w>", "hello <l><r> world"]
                .map(String::from),
        );
    }
    if !first_added.is_empty() {
        texts.push(first_added.join(""));
        texts.push(first_added.join(" hello "));
        texts.push(format!("{0}hello world{0}", first_added[0]));
        texts.push(format!(" {} ", first_added[0]));
    }

    let text_records: Vec<Value> = texts
        .iter()
        .map(|text| {
            let mut record = json!({"text": text});
            match plain.encode(text) {
                Ok(encoding) => {
                    let ids = encoding.token_ids();
                    record["ids"] = json!(ids);
                    record["decode"] = decode_pair(plain.as_ref(), ids);
                    record["stream"] = stream_pair(&plain, ids);
                }
                Err(error) => record["error"] = json!(error.to_string()),
            }
            match with_special_tokens.encode(text) {
                Ok(encoding) => {
                    let ids = encoding.token_ids();
                    record["special_ids"] = json!(ids);
                    record["special_decode"] = decode_pair(plain.as_ref(), ids);
                }
                Err(error) => record["special_error"] = json!(error.to_string()),
            }
            if let Some(fast) = &fast {
                record["fast"] = match fast.encode(text) {
                    Ok(encoding) => {
                        let ids = encoding.token_ids();
                        json!({
                            "ids": ids,
                            "decode": decode_pair(fast.as_ref(), ids),
                            "stream": stream_pair(fast, ids),
                        })
                    }
                    Err(error) => json!({"error": error.to_string()}),
                };
            }
            record
        })
        .collect();

    let prefix_records: Vec<Value> = PREFIX_TEXTS
        .iter()
        .filter_map(|text| {
            let encoding = plain.encode(text).ok()?;
            let ids = encoding.token_ids();
            let prefixes: Vec<Value> = (1..=ids.len())
                .map(|end| {
                    let decoded = plain.decode(&ids[..end], false).unwrap();
                    json!({
                        "decode": decoded.as_str(),
                        "partial": decoded.is_partial(),
                        "unstable": plain.has_unstable_suffix(&ids[..end], false),
                    })
                })
                .collect();
            let streams: Vec<Value> = (0..ids.len())
                .map(|prompt| stream(&plain, &ids[..prompt], &ids[prompt..], false))
                .collect();
            Some(json!({"text": text, "ids": ids, "prefixes": prefixes, "streams": streams}))
        })
        .collect();

    let added_records: Vec<Value> = added_tokens
        .iter()
        .map(|(id, token)| {
            json!({
                "id": id,
                "content": token.content,
                "special": token.special,
                "encode": result_ids(plain.encode(&token.content)),
                "decode": decode_pair(plain.as_ref(), &[*id]),
            })
        })
        .collect();

    let marker = first_added.first().copied().unwrap_or("<special>");
    let joined = first_added.join("");
    let segment_lists: Vec<Vec<(String, bool)>> = vec![
        vec![],
        vec![(format!("{marker}hello"), true), (format!(" world{marker}"), true)],
        vec![(format!("{marker}hello"), true), (format!(" world{marker}"), false)],
        vec![("hello world".into(), true)],
        vec![("hello world".into(), false)],
        vec![("He said hello".into(), false), (" world".into(), true)],
        vec![(joined.clone(), true)],
        vec![(joined, false)],
        vec![("café 北京 😀".into(), false), ("".into(), true), ("!".into(), false)],
    ];
    let segment_records: Vec<Value> = segment_lists
        .iter()
        .map(|segments| {
            let borrowed: Vec<EncodeSegment<'_>> = segments
                .iter()
                .map(|(text, allow_special)| EncodeSegment::new(text, *allow_special))
                .collect();
            json!({
                "segments": segments,
                "plain": result_ids(plain.encode_segments(&borrowed)),
                "special": result_ids(with_special_tokens.encode_segments(&borrowed)),
                "fast": fast.as_ref().map(|fast| result_ids(fast.encode_segments(&borrowed))),
            })
        })
        .collect();

    let bound = token_id_bound.unwrap_or(0) as u32;
    let head: Vec<u32> = (0..bound.min(12)).collect();
    let mut sequences: Vec<Vec<u32>> = vec![head.clone(), head.into_iter().rev().collect()];
    // Invalid tails, complete and incomplete multibyte characters, and a special between bytes.
    if model_dir.ends_with("byte-fallback-bpe-partial-alphabet") {
        sequences.extend([
            vec![0, 1, 4],
            vec![2, 3, 1, 4],
            vec![2, 3, 5],
            vec![0],
            vec![2, 3],
            vec![2],
            vec![0, 1],
            vec![0, 6, 1, 4],
        ]);
    }
    if model_dir.ends_with("byte-fallback-bpe") {
        let (a, invalid, lead, tail, hello, bang, eos) = (0x61, 0xF5, 0xC3, 0xA9, 256, 257, 258);
        sequences.extend([
            vec![a, invalid, hello],
            vec![lead, tail, invalid, hello],
            vec![lead, tail, bang],
            vec![a],
            vec![lead, tail],
            vec![lead],
            vec![a, invalid],
            vec![a, eos, invalid, hello],
            vec![hello, bang, a, invalid, hello],
        ]);
    }
    if model_dir.ends_with("TinyLlama_v1.1") {
        let id = |piece: &str| plain.token_to_id(piece).unwrap().unwrap();
        sequences.extend([
            vec![id("<0x61>"), id("<0xF5>")],
            vec![id("<0x61>"), id("</s>"), id("<0xF5>")],
            vec![id("<0xC3>"), id("<0xA9>"), id("</s>")],
        ]);
    }
    let sequence_records: Vec<Value> = sequences
        .iter()
        .map(|ids| sequence_record(&plain, ids))
        .collect();

    let mut probes: Vec<String> = ["hello", "world", "<special>", "<missing>", "He", " ", ""]
        .iter()
        .map(|probe| probe.to_string())
        .collect();
    probes.extend(added_tokens.iter().map(|(_, token)| token.content.clone()));
    probes.sort();
    probes.dedup();
    let token_to_id: serde_json::Map<String, Value> = probes
        .into_iter()
        .map(|probe| {
            let id = plain.token_to_id(&probe).unwrap();
            (probe, json!(id))
        })
        .collect();

    json!({
        "model_dir": model_dir,
        "vocab_size": plain.vocab_size(),
        "token_id_bound": token_id_bound,
        "special_token_ids": plain.special_token_ids().unwrap(),
        "special_tokens": special_tokens,
        "num_special_tokens_added": num_special_tokens_added,
        "validate_prefix_cache": validate_prefix_cache,
        "validate_segment_prefix_cache": validate_segment_prefix_cache,
        "fast_loads": fast.is_some(),
        "token_to_id": token_to_id,
        "added_tokens": added_records,
        "texts": text_records,
        "prefixes": prefix_records,
        "segments": segment_records,
        "sequences": sequence_records,
    })
}

/// One key per line, and one element per line for the record arrays, so a diff stays readable.
fn render(value: &Value) -> String {
    const RECORD_ARRAYS: &[&str] = &["added_tokens", "texts", "prefixes", "segments", "sequences"];
    let object = value.as_object().unwrap();
    let mut out = String::from("{\n");
    for (index, (key, value)) in object.iter().enumerate() {
        out.push_str(&format!("{}: ", serde_json::to_string(key).unwrap()));
        match value.as_array() {
            Some(items) if RECORD_ARRAYS.contains(&key.as_str()) && !items.is_empty() => {
                out.push_str("[\n");
                for (item_index, item) in items.iter().enumerate() {
                    out.push_str(&serde_json::to_string(item).unwrap());
                    out.push_str(if item_index + 1 == items.len() { "\n" } else { ",\n" });
                }
                out.push(']');
            }
            _ => out.push_str(&serde_json::to_string(value).unwrap()),
        }
        out.push_str(if index + 1 == object.len() { "\n" } else { ",\n" });
    }
    out.push_str("}\n");
    out
}

#[test]
#[ignore = "throwaway generator; run explicitly"]
fn generate() {
    let commit = std::env::var("HF_EXPECT_COMMIT").expect("set HF_EXPECT_COMMIT");
    let mut cases: Vec<(String, String)> = [
        "minimal-bpe",
        "sample-models/mock-deepseek-r1",
        "sample-models/mock-llama-3.1-8b-instruct",
        "sample-models/mock-special-tokens-only",
        "sample-models/mock-tokenizer-json-fallback",
        "sample-models/TinyLlama_v1.1",
    ]
    .iter()
    .map(|dir| (dir.rsplit('/').next().unwrap().to_string(), dir.to_string()))
    .collect();
    cases.extend(
        write_synthetic_configs()
            .into_iter()
            .map(|name| (name.clone(), format!("{OUT}/{name}"))),
    );

    for (name, model_dir) in &cases {
        let mut expected = record(model_dir);
        expected["provenance"] = json!({
            "dynamo_tokenizers": env!("CARGO_PKG_VERSION"),
            "tokenizers": "0.23.2",
            "commit": commit,
        });
        fs::write(out_dir(name).join("expected.json"), render(&expected)).unwrap();
        println!(
            "{name}: {}",
            expected
                .get("load_error")
                .map_or("recorded".to_string(), |error| format!("load error: {error}"))
        );
    }
}
