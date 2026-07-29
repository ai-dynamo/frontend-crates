// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten Tokenizer backend for high-performance BPE encoding and decoding.
//!
//! Some Kimi repositories ship tiktoken assets without a directly loadable
//! `tokenizer.json`. Baseten publishes compatible tokenizer artifacts,
//! including [`baseten/kimi-k3-tokenizer`](https://huggingface.co/baseten/kimi-k3-tokenizer).

use std::path::Path;

use super::{
    EncodeSegment, Encoding, Error, Result, TokenIdType, TokenizerOptions,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// Tokenizer backed by the `basetenkenizer` crate.
pub struct BasetenTokenizer {
    tokenizer: basetenkenizer::Tokenizer,
    options: TokenizerOptions,
}

impl BasetenTokenizer {
    /// Load a tokenizer from a Hugging Face `tokenizer.json` file.
    pub fn from_file(path: &str) -> Result<Self> {
        let path = Path::new(path);
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::msg(format!("Error reading Baseten tokenizer: {e}")))?;
        let mut json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::msg(format!("Error parsing Baseten tokenizer: {e}")))?;
        if let Some(parent) = path.parent() {
            merge_special_tokens_from_config(&mut json, parent);
        }
        let tokenizer = basetenkenizer::Tokenizer::from_json(json)
            .map_err(|e| Error::msg(format!("Error loading Baseten tokenizer: {e}")))?;
        Ok(Self {
            tokenizer,
            options: TokenizerOptions::default(),
        })
    }
}

fn merge_special_tokens_from_config(json: &mut serde_json::Value, model_dir: &Path) {
    let config_path = model_dir.join("tokenizer_config.json");
    let Ok(raw) = std::fs::read_to_string(&config_path) else {
        return;
    };
    let config: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(error) => {
            tracing::debug!(
                target: "tokenizer",
                path = %config_path.display(),
                error = %error,
                "tokenizer_config.json parse failed; skipping special-token merge"
            );
            return;
        }
    };
    let Some(decoder) = config
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
    else {
        return;
    };

    if json.get("added_tokens").is_none() {
        json["added_tokens"] = serde_json::json!([]);
    }
    let Some(added_tokens) = json
        .get_mut("added_tokens")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };

    for (id, spec) in decoder {
        let Some(id) = id.parse::<u32>().ok() else {
            continue;
        };
        let Some(spec) = spec.as_object() else {
            continue;
        };
        if spec.get("special").and_then(serde_json::Value::as_bool) != Some(true) {
            continue;
        }
        let Some(content) = spec
            .get("content")
            .and_then(serde_json::Value::as_str)
            .filter(|content| !content.is_empty())
        else {
            continue;
        };

        if let Some(existing) = added_tokens
            .iter_mut()
            .find(|token| token.get("content").and_then(serde_json::Value::as_str) == Some(content))
        {
            existing["special"] = serde_json::Value::Bool(true);
            continue;
        }

        let mut token = serde_json::Map::from_iter([
            ("id".to_string(), serde_json::json!(id)),
            ("content".to_string(), serde_json::json!(content)),
            ("special".to_string(), serde_json::Value::Bool(true)),
        ]);
        for field in ["single_word", "lstrip", "rstrip", "normalized"] {
            token.insert(
                field.to_string(),
                serde_json::Value::Bool(
                    spec.get(field)
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                ),
            );
        }
        added_tokens.push(serde_json::Value::Object(token));
    }
}

impl Encoder for BasetenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let ids = self
            .tokenizer
            .encode_with_special_tokens(input, self.options.add_special_tokens)
            .map_err(|e| Error::msg(format!("Baseten tokenizer encode error: {e}")))?;
        Ok(Encoding::Sp(ids))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.tokenizer
            .encode_batch(inputs, self.options.add_special_tokens)
            .map(|ids| ids.into_iter().map(Encoding::Sp).collect())
            .map_err(|e| Error::msg(format!("Baseten tokenizer batch encode error: {e}")))
    }

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        let segments = segments
            .iter()
            .map(|segment| (segment.text, segment.allow_special));
        let ids = self
            .tokenizer
            .encode_segments_tiktoken_safe(segments, self.options.add_special_tokens)
            .map_err(|e| Error::msg(format!("Baseten tokenizer segment encode error: {e}")))?;
        Ok(Encoding::Sp(ids))
    }
}

impl Decoder for BasetenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map(DecodeResult::from)
            .map_err(|e| Error::msg(format!("Baseten tokenizer decode error: {e}")))
    }
}

impl Tokenizer for BasetenTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        if self.options.add_special_tokens {
            return Err(Error::msg(
                "Baseten tokenizers configured with add_special_tokens=true must remain uncached",
            ));
        }
        Ok(())
    }

    fn with_options(mut self, options: TokenizerOptions) -> Self {
        self.options = options;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        HuggingFaceTokenizer, Tokenizer as TokenizerWrapper, traits::Tokenizer as TokenizerTrait,
    };

    const TOKENIZER_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/minimal-bpe/tokenizer.json"
    );

    #[test]
    fn encode_matches_hugging_face() {
        let baseten = BasetenTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let hf = HuggingFaceTokenizer::from_file(TOKENIZER_PATH).unwrap();

        for text in ["Hello, world!", "Hello", " world", "He llo"] {
            let baseten_ids = baseten.encode(text).unwrap();
            let hf_ids = hf.encode(text).unwrap();
            assert_eq!(
                baseten_ids.token_ids(),
                hf_ids.token_ids(),
                "Baseten and Hugging Face must produce identical token IDs for '{text}'"
            );
        }
    }

    #[test]
    fn batch_encode_matches_sequential_encode() {
        let tokenizer = BasetenTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let inputs = ["Hello", " world", "Hello, world!"];
        let batch = tokenizer.encode_batch(&inputs).unwrap();

        for (encoding, input) in batch.iter().zip(inputs) {
            let sequential = tokenizer.encode(input).unwrap();
            assert_eq!(encoding.token_ids(), sequential.token_ids());
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tokenizer = BasetenTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let encoding = tokenizer.encode("Hello, world!").unwrap();
        let decoded = tokenizer.decode(encoding.token_ids(), true).unwrap();

        assert_eq!(decoded.as_str(), "Hello, world!");
    }

    #[test]
    fn works_with_decode_stream() {
        let tokenizer = Arc::new(BasetenTokenizer::from_file(TOKENIZER_PATH).unwrap());
        let wrapper = TokenizerWrapper::from(tokenizer);
        let prompt_ids = wrapper.encode("Hello").unwrap().token_ids().to_vec();
        let continuation_ids = wrapper.encode(", world!").unwrap().token_ids().to_vec();
        let mut stream = wrapper.decode_stream(&prompt_ids, true);
        let mut accumulated = String::new();

        for id in &continuation_ids {
            if let Some(chunk) = stream.step(*id).unwrap() {
                accumulated.push_str(&chunk);
            }
        }

        let mut all_ids = prompt_ids.clone();
        all_ids.extend_from_slice(&continuation_ids);
        let full_text: String = wrapper.decode(&all_ids, true).unwrap().into();
        let prompt_text: String = wrapper.decode(&prompt_ids, true).unwrap().into();
        assert_eq!(accumulated, full_text[prompt_text.len()..]);
    }

    #[test]
    fn prefix_cache_rejects_special_token_post_processing() {
        let plain = BasetenTokenizer::from_file(TOKENIZER_PATH).unwrap();
        assert!(plain.validate_prefix_cache().is_ok());

        let with_special_tokens = BasetenTokenizer::from_file(TOKENIZER_PATH)
            .unwrap()
            .with_options(TokenizerOptions {
                add_special_tokens: true,
            });
        assert!(with_special_tokens.validate_prefix_cache().is_err());
    }

    #[test]
    fn segments_preserve_special_token_trust_boundary() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tokenizer.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(TOKENIZER_PATH).unwrap()).unwrap();
        let vocab = json["model"]["vocab"].as_object_mut().unwrap();
        vocab.insert("<".to_string(), serde_json::json!(23));
        vocab.insert(">".to_string(), serde_json::json!(24));
        vocab.insert("c".to_string(), serde_json::json!(25));
        json["added_tokens"] = serde_json::json!([{
            "id": 26,
            "content": "<ctl>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }]);
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();

        let tokenizer: Arc<dyn TokenizerTrait> =
            Arc::new(BasetenTokenizer::from_file(path.to_str().unwrap()).unwrap());
        let segments = [
            EncodeSegment::new("Hello", true),
            EncodeSegment::new("<ctl>", false),
            EncodeSegment::new(" world!", true),
        ];

        let segmented = tokenizer.encode_segments(&segments).unwrap();
        let flattened = tokenizer.encode("Hello<ctl> world!").unwrap();

        assert_ne!(
            segmented.token_ids(),
            flattened.token_ids(),
            "untrusted control-token-looking text must not become an added token"
        );
        assert!(flattened.token_ids().contains(&26));
        assert!(!segmented.token_ids().contains(&26));
    }

    #[test]
    fn segments_honor_add_special_tokens_option() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tokenizer.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(TOKENIZER_PATH).unwrap()).unwrap();
        json["model"]["vocab"]["<bos>"] = serde_json::json!(23);
        json["added_tokens"] = serde_json::json!([{
            "id": 23,
            "content": "<bos>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }]);
        json["post_processor"] = serde_json::json!({
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}}
            ],
            "pair": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}}
            ],
            "special_tokens": {
                "<bos>": {"id": "<bos>", "ids": [23], "tokens": ["<bos>"]}
            }
        });
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let segments = [EncodeSegment::new("Hello", false)];

        let plain = BasetenTokenizer::from_file(path.to_str().unwrap()).unwrap();
        let plain_ids = plain
            .encode_segments(&segments)
            .unwrap()
            .token_ids()
            .to_vec();

        let with_bos = BasetenTokenizer::from_file(path.to_str().unwrap())
            .unwrap()
            .with_options(TokenizerOptions {
                add_special_tokens: true,
            });
        assert_eq!(
            with_bos.encode_segments(&segments).unwrap().token_ids(),
            [&[23], plain_ids.as_slice()].concat()
        );
    }

    #[test]
    fn merges_config_only_special_tokens() {
        let temp = tempfile::tempdir().unwrap();
        let tokenizer_path = temp.path().join("tokenizer.json");
        std::fs::copy(TOKENIZER_PATH, &tokenizer_path).unwrap();
        std::fs::write(
            temp.path().join("tokenizer_config.json"),
            serde_json::json!({
                "added_tokens_decoder": {
                    "23": {
                        "content": "<ctl>",
                        "special": true,
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": false
                    }
                }
            })
            .to_string(),
        )
        .unwrap();

        let tokenizer = BasetenTokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
        let encoding = tokenizer.encode("<ctl>").unwrap();
        assert_eq!(encoding.token_ids(), &[23]);
        assert_eq!(tokenizer.decode(&[23], false).unwrap().as_str(), "<ctl>");
        assert_eq!(tokenizer.decode(&[23], true).unwrap().as_str(), "");
    }
}
