// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental Gigatoken backend.
//!
//! Encoding uses Gigatoken's persistent worker pool. Decoding stays on the
//! HuggingFace implementation so this backend preserves the crate's
//! incremental-decoding behavior.

use std::path::Path;

use anyhow::{Context as _, ensure};
use gigatoken::{Tokenizer as GigatokenEncoder, WorkerPool, encode_docs_ragged};

use super::{
    Encoding, Result, TokenIdType,
    hf::HuggingFaceTokenizer,
    tiktoken::TikTokenTokenizer,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// Hybrid tokenizer: Gigatoken BPE encoding with HuggingFace decoding.
///
/// Gigatoken's public Rust batch API owns mutable per-worker pretoken caches
/// behind [`WorkerPool`], allowing this wrapper to satisfy Dynamo's shared
/// `&self` encoder interface without serializing callers behind one mutex.
pub struct GigatokenTokenizer {
    encoder: GigatokenEncoder,
    workers: WorkerPool,
    decoder: Box<dyn Decoder>,
}

impl GigatokenTokenizer {
    /// Load a Hugging Face byte-level BPE `tokenizer.json`.
    pub fn from_file(path: &str) -> Result<Self> {
        let encoder = gigatoken::load_tokenizer::hf::load_hf_bpe(Path::new(path))
            .map_err(|error| anyhow::anyhow!("Error loading Gigatoken tokenizer: {error:#}"))?;
        let decoder = HuggingFaceTokenizer::from_file(path)?;
        Ok(Self {
            encoder,
            workers: WorkerPool::new(),
            decoder: Box::new(decoder),
        })
    }

    /// Load a rank-per-line TikToken model and its sibling tokenizer metadata.
    ///
    /// Repositories using this format carry their split regex only in remote
    /// Python code, so callers must name Gigatoken's matching pretokenizer
    /// scheme (for example, `kimi`).
    pub fn from_tiktoken_model(path: &str, pretokenizer: &str) -> Result<Self> {
        let model_path = Path::new(path);
        let model_dir = model_path
            .parent()
            .context("Cannot determine parent directory of TikToken model")?;
        let config_path = model_dir.join("tokenizer_config.json");
        let pretokenizer = gigatoken::pretokenize::PretokenizerType::from_name(pretokenizer)
            .with_context(|| format!("Unknown Gigatoken pretokenizer scheme {pretokenizer:?}"))?;
        let encoder = gigatoken::load_tokenizer::tiktoken::load_tiktoken_model(
            model_path,
            &config_path,
            pretokenizer,
        )
        .map_err(|error| anyhow::anyhow!("Error loading Gigatoken TikToken model: {error:#}"))?;
        let decoder = TikTokenTokenizer::from_file_auto(path)?;
        Ok(Self {
            encoder,
            workers: WorkerPool::new(),
            decoder: Box::new(decoder),
        })
    }

    fn encode_inputs(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        let documents: Vec<&[u8]> = inputs.iter().map(|input| input.as_bytes()).collect();
        let (token_ids, lengths) = encode_docs_ragged(&self.workers, &self.encoder, &documents);

        ensure!(
            lengths.len() == inputs.len(),
            "Gigatoken returned {} rows for {} inputs",
            lengths.len(),
            inputs.len()
        );

        let mut offset = 0usize;
        let mut encodings = Vec::with_capacity(lengths.len());
        for length in lengths {
            let length = usize::try_from(length)
                .context("Gigatoken returned a negative token count for an input")?;
            let end = offset
                .checked_add(length)
                .context("Gigatoken token count overflow")?;
            let ids = token_ids
                .get(offset..end)
                .context("Gigatoken returned inconsistent token row lengths")?;
            encodings.push(Encoding::Sp(ids.to_vec()));
            offset = end;
        }

        ensure!(
            offset == token_ids.len(),
            "Gigatoken returned {} unassigned token IDs",
            token_ids.len().saturating_sub(offset)
        );
        Ok(encodings)
    }
}

impl Encoder for GigatokenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        self.encode_inputs(&[input])?
            .pop()
            .context("Gigatoken returned no encoding for one input")
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.encode_inputs(inputs)
    }
}

impl Decoder for GigatokenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.decoder.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for GigatokenTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HuggingFaceTokenizer;

    #[test]
    #[ignore = "requires QWEN_TOKENIZER=/path/to/Qwen3/tokenizer.json"]
    fn gigatoken_matches_huggingface_for_qwen3() {
        let tokenizer_path =
            std::env::var("QWEN_TOKENIZER").expect("QWEN_TOKENIZER must point to tokenizer.json");
        let gigatoken = GigatokenTokenizer::from_file(&tokenizer_path).unwrap();
        let hf = HuggingFaceTokenizer::from_file(&tokenizer_path).unwrap();
        let inputs = [
            "Hello, world!",
            "你好，世界！",
            "<|im_start|>user\nExplain this Rust function.<|im_end|>\n",
            "fn route(request: &Request) -> Result<Response> { todo!() }",
        ];

        for input in inputs {
            assert_eq!(
                gigatoken.encode(input).unwrap().token_ids(),
                hf.encode(input).unwrap().token_ids(),
                "single encoding mismatch for {input:?}"
            );
        }

        let actual = gigatoken.encode_batch(&inputs).unwrap();
        let expected = hf.encode_batch(&inputs).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.token_ids(), expected.token_ids());
        }

        let ids = gigatoken.encode("Hello, world!").unwrap();
        let decoded = gigatoken.decode(ids.token_ids(), false).unwrap();
        assert!(!decoded.as_str().is_empty());
    }

    #[test]
    #[ignore = "requires KIMI_TOKENIZER=/path/to/Kimi-K2.6/tiktoken.model"]
    fn gigatoken_matches_tiktoken_for_kimi_k26() {
        let tokenizer_path =
            std::env::var("KIMI_TOKENIZER").expect("KIMI_TOKENIZER must point to tiktoken.model");
        let gigatoken = GigatokenTokenizer::from_tiktoken_model(&tokenizer_path, "kimi").unwrap();
        let tiktoken = TikTokenTokenizer::from_file_auto(&tokenizer_path).unwrap();
        let inputs = [
            "Hello, world!",
            "你好，世界！",
            "[BOS]<|im_user|>Explain this Rust function.<|im_end|>",
            "fn route(request: &Request) -> Result<Response> { todo!() }",
        ];

        for input in inputs {
            assert_eq!(
                gigatoken.encode(input).unwrap().token_ids(),
                tiktoken.encode(input).unwrap().token_ids(),
                "single encoding mismatch for {input:?}"
            );
        }

        let actual = gigatoken.encode_batch(&inputs).unwrap();
        let expected = tiktoken.encode_batch(&inputs).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.token_ids(), expected.token_ids());
        }
    }
}
