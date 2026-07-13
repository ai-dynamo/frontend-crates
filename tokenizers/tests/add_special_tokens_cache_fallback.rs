// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    fs,
    io::Write,
    sync::{Arc, Mutex},
};

use dynamo_tokenizers::{
    Encoding, HuggingFaceTokenizer, Tokenizer, TokenizerCacheConfig, TokenizerConfig,
    TokenizerOptions,
    traits::{Encoder, Tokenizer as _},
};
use tempfile::TempDir;
use tracing_subscriber::fmt::MakeWriter;

const FALLBACK_WARNING: &str = "add_special_tokens=true is incompatible with tokenizer caching; using uncached HuggingFace tokenizer";

const TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [
        {"id": 0, "content": "<unk>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 1, "content": "<bos>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 2, "content": "<eos>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 3, "content": "<|im_start|>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 4, "content": "<|im_end|>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
    ],
    "normalizer": null,
    "pre_tokenizer": {"type": "WhitespaceSplit"},
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<eos>", "type_id": 0}}
        ],
        "pair": [
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": "<eos>", "type_id": 1}}
        ],
        "special_tokens": {
            "<bos>": {"id": "<bos>", "ids": [1], "tokens": ["<bos>"]},
            "<eos>": {"id": "<eos>", "ids": [2], "tokens": ["<eos>"]}
        }
    },
    "decoder": null,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<|im_start|>": 3,
            "<|im_end|>": 4,
            "system": 5,
            "user": 6,
            "hello": 7,
            "world": 8,
            "again": 9
        },
        "unk_token": "<unk>"
    }
}"#;

#[derive(Clone, Default)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);

struct LockedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for LockedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for SharedWriter {
    type Writer = LockedWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LockedWriter(self.0.clone())
    }
}

impl SharedWriter {
    fn contents(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

fn tokenizer_fixture() -> (TempDir, String) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("tokenizer.json");
    fs::write(&path, TOKENIZER_JSON).unwrap();
    (dir, path.to_string_lossy().into_owned())
}

fn ids(encoding: &Encoding) -> &[u32] {
    encoding.token_ids()
}

#[test]
fn cache_request_with_add_special_tokens_falls_back_to_hf() {
    let (_dir, path) = tokenizer_fixture();
    let options = TokenizerOptions {
        add_special_tokens: true,
    };
    let direct = HuggingFaceTokenizer::from_file(&path)
        .unwrap()
        .with_options(options);
    let logs = SharedWriter::default();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(logs.clone())
        .finish();

    tracing::subscriber::with_default(subscriber, || {
        let selected = Tokenizer::from_file_with_config(
            &path,
            TokenizerConfig::new(options)
                .with_cache(TokenizerCacheConfig::new(1024 * 1024).with_extend(true)),
        )
        .unwrap();

        let warning_after_construction = logs.contents();
        assert_eq!(
            warning_after_construction.matches(FALLBACK_WARNING).count(),
            1
        );

        let prompts = [
            "<|im_start|> system <|im_end|> <|im_start|> user hello world <|im_end|>",
            "<|im_start|> user hello again <|im_end|>",
        ];

        for prompt in prompts {
            let expected = direct.encode(prompt).unwrap();
            for _ in 0..3 {
                let actual = selected.encode(prompt).unwrap();
                assert!(matches!(actual, Encoding::Hf(_)));
                assert_eq!(ids(&actual), ids(&expected));
            }
        }

        let expected_batch = direct.encode_batch(&prompts).unwrap();
        let actual_batch = selected.encode_batch(&prompts).unwrap();
        assert_eq!(actual_batch.len(), expected_batch.len());
        for (actual, expected) in actual_batch.iter().zip(&expected_batch) {
            assert!(matches!(actual, Encoding::Hf(_)));
            assert_eq!(ids(actual), ids(expected));
        }

        let first = ids(&actual_batch[0]);
        assert_eq!(first.first(), Some(&1));
        assert_eq!(first.last(), Some(&2));
        assert_eq!(first.iter().filter(|&&id| id == 1).count(), 1);
        assert_eq!(first.iter().filter(|&&id| id == 2).count(), 1);

        assert_eq!(logs.contents(), warning_after_construction);
    });
}
