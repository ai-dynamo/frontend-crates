// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fastokens backend using the `fastokens` crate for high-performance BPE encoding.
//!
//! This module preserves the existing hybrid behavior: `fastokens` handles encoding and
//! `HuggingFaceTokenizer` handles decoding. Both are loaded from the same `tokenizer.json`
//! file.
//!
//! [`FastTikTokenTokenizer`] instead loads a bare `tiktoken.model` and decodes its ranks
//! directly.

use std::collections::HashSet;
use std::path::Path;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::{
    EncodeSegment, Encoding, Error, Result, TokenIdType,
    hf::HuggingFaceTokenizer,
    tiktoken,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

fn fast_encode(encoder: &fastokens::Tokenizer, input: &str) -> Result<Encoding> {
    let ids = encoder
        .encode(input)
        .map_err(|e| Error::msg(format!("Fastokens encode error: {e}")))?;
    Ok(Encoding::Sp(ids))
}

fn fast_encode_segments(
    encoder: &fastokens::Tokenizer,
    segments: &[EncodeSegment<'_>],
) -> Result<Encoding> {
    let segments: Vec<fastokens::EncodeSegment<'_>> = segments
        .iter()
        .map(|segment| fastokens::EncodeSegment {
            text: segment.text,
            allow_special: segment.allow_special,
        })
        .collect();
    let ids = encoder
        .encode_segments(&segments)
        .map_err(|e| Error::msg(format!("Fastokens segmented encode error: {e}")))?;
    Ok(Encoding::Sp(ids))
}

/// Hybrid tokenizer: fast BPE encoding via `fastokens`, decoding via HuggingFace.
///
/// Both backends are loaded from the same `tokenizer.json` file.
pub struct FastTokenizer {
    fast_encoder: fastokens::Tokenizer,
    hf_decoder: HuggingFaceTokenizer,
}

impl FastTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let fast_encoder = fastokens::Tokenizer::from_file(Path::new(path))
            .map_err(|e| Error::msg(format!("Error loading fastokens tokenizer: {e}")))?;
        let hf_decoder = HuggingFaceTokenizer::from_file(path)?;
        Ok(Self {
            fast_encoder,
            hf_decoder,
        })
    }
}

impl Encoder for FastTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        fast_encode(&self.fast_encoder, input)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.par_iter().map(|input| self.encode(input)).collect()
    }

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        fast_encode_segments(&self.fast_encoder, segments)
    }
}

impl Decoder for FastTokenizer {
    fn has_unstable_suffix(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> bool {
        self.hf_decoder
            .has_unstable_suffix(token_ids, skip_special_tokens)
    }

    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.hf_decoder.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for FastTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        Ok(())
    }

    // `fast_encoder` and `hf_decoder` are loaded from the same tokenizer.json,
    // so the HF side's vocabulary introspection applies to both.
    fn vocab_size(&self) -> Option<usize> {
        self.hf_decoder.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        self.hf_decoder.token_to_id(token)
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        self.hf_decoder.special_token_ids()
    }

    fn num_special_tokens_added(&self) -> Result<usize> {
        Ok(0)
    }
}

/// `fastokens` over a bare `tiktoken.model`, for checkpoints such as Kimi K2/K3 that ship no
/// `tokenizer.json`. Loads the same ranks, regex, and special tokens as
/// [`TikTokenTokenizer::from_file_auto`](crate::TikTokenTokenizer::from_file_auto).
///
/// Plain-text prefix caching ([`CachedTokenizer`](crate::CachedTokenizer)) is rejected for
/// this backend: `fastokens` pre-tokenizes text that contains a special token with its regex
/// engine but plain text with a hand-written scanner, and the two disagree on the Unicode
/// case folding of `(?i:'s)` (`'ſ`, U+017F). Splitting a prompt after a special token
/// therefore changes ids, which is exactly the invariant the cache relies on. Segmented
/// encodes are unaffected: every segment is encoded on its own with or without a cache.
pub struct FastTikTokenTokenizer {
    inner: fastokens::Tokenizer,
    /// Decoding joins raw bytes itself because `fastokens`' decoder returns a lossy `String`,
    /// which cannot tell a vocabulary token ending in `EF BF BD` from a truncated sequence.
    id_to_bytes: FxHashMap<u32, Vec<u8>>,
    special_token_ids: HashSet<u32>,
    special_tokens: Vec<String>,
}

impl FastTikTokenTokenizer {
    /// Load a tiktoken model file, reading the BPE pattern from `config.json` and the special
    /// tokens from `tokenizer_config.json` in the same directory.
    pub fn from_file_auto(path: &str) -> Result<Self> {
        let directory = Path::new(path)
            .parent()
            .ok_or_else(|| Error::msg("Cannot determine parent directory of tiktoken file"))?;
        let pattern = tiktoken::detect_bpe_pattern(directory)?;
        let encoder = tiktoken::parse_tiktoken_file(path)?;
        let num_base_tokens = encoder.values().max().map_or(0, |&m| m + 1) as usize;
        let special_tokens = tiktoken::load_special_tokens(directory, num_base_tokens)?;

        let ranks: Vec<(Vec<u8>, u32)> = encoder.into_iter().collect();
        let mut id_to_bytes: FxHashMap<u32, Vec<u8>> = ranks
            .iter()
            .map(|(bytes, rank)| (*rank, bytes.clone()))
            .collect();
        id_to_bytes.extend(
            special_tokens
                .iter()
                .map(|(content, &id)| (id, content.as_bytes().to_vec())),
        );
        let special_token_ids = special_tokens.values().copied().collect();
        let special_token_strings = tiktoken::sorted_special_token_strings(&special_tokens);

        let config =
            fastokens::tiktoken::TiktokenConfig::new(pattern, special_tokens.into_iter().collect());
        let inner = fastokens::Tokenizer::from_tiktoken_ranks(&ranks, config).map_err(|e| {
            Error::msg(format!(
                "Error loading fastokens tiktoken tokenizer from {path}: {e}"
            ))
        })?;
        Ok(Self {
            inner,
            id_to_bytes,
            special_token_ids,
            special_tokens: special_token_strings,
        })
    }

    /// Atomic special-token strings registered with the encoder, sorted; the boundary set
    /// for [`CachedTokenizer`](crate::CachedTokenizer).
    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }
}

impl Encoder for FastTikTokenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        fast_encode(&self.inner, input)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.par_iter().map(|input| self.encode(input)).collect()
    }

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        fast_encode_segments(&self.inner, segments)
    }
}

impl Decoder for FastTikTokenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        let mut bytes = Vec::new();
        for id in token_ids {
            if skip_special_tokens && self.special_token_ids.contains(id) {
                continue;
            }
            if let Some(token) = self.id_to_bytes.get(id) {
                bytes.extend_from_slice(token);
            }
        }
        match String::from_utf8(bytes) {
            Ok(text) => Ok(DecodeResult::Complete(text)),
            Err(e) => Ok(DecodeResult::from_decoded(
                String::from_utf8_lossy(e.as_bytes()).into_owned(),
            )),
        }
    }
}

impl Tokenizer for FastTikTokenTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        Err(Error::msg(
            "fastokens over tiktoken.model does not satisfy the prefix-cache invariant: its \
             scanner (plain text) and regex path (text containing special tokens) disagree on \
             the case folding of (?i:'s), so encode(prefix) + encode(suffix) can differ from \
             encode(prefix + suffix) across a special-token boundary",
        ))
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        Ok(self.inner.token_to_id(token))
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        let mut ids: Vec<TokenIdType> = self.special_token_ids.iter().copied().collect();
        ids.sort_unstable();
        Ok(ids)
    }

    fn num_special_tokens_added(&self) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HuggingFaceTokenizer, TokenizerOptions};

    // Minimal synthetic BPE tokenizer with no normalizer or post-processor --
    // compatible with fastokens. Vocab covers: H,T,a,d,e,h,i,l,o,r,s,t,w + punctuation.
    const TOKENIZER_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/minimal-bpe/tokenizer.json"
    );
    const SEGMENTED_TOKENIZER_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
    );

    #[test]
    fn byte_fallback_stream_matches_full_decode() {
        let tokenizer: crate::Tokenizer =
            std::sync::Arc::new(FastTokenizer::from_file(SEGMENTED_TOKENIZER_PATH).unwrap()).into();
        for (pieces, skip) in [
            (vec!["<0x61>", "<0xF5>"], false),
            (vec!["<0x61>", "</s>", "<0xF5>"], false),
            (vec!["<0x61>", "</s>", "<0xF5>"], true),
        ] {
            let ids: Vec<_> = pieces
                .iter()
                .map(|piece| tokenizer.token_to_id(piece).unwrap().unwrap())
                .collect();
            let expected: String = tokenizer.decode(&ids, skip).unwrap().into();
            let mut stream = tokenizer.decode_stream(&[], skip);
            let mut actual = String::new();
            for id in ids {
                actual.push_str(&stream.step(id).unwrap().unwrap_or_default());
            }
            actual.push_str(&stream.finish().unwrap().unwrap_or_default());
            assert_eq!(actual, expected, "{pieces:?}, skip={skip}");
        }
    }

    #[test]
    fn test_fast_encode_decode_roundtrip() {
        let tokenizer = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        // Encode then decode: verifies both paths execute without error.
        // With a null decoder, HF inserts spaces between tokens so exact equality
        // is not expected here -- we just verify the operations succeed and produce
        // non-empty results.
        let text = "Hello, world!";
        let encoding = tokenizer.encode(text).unwrap();
        assert!(!encoding.token_ids().is_empty());
        let decoded: String = tokenizer.decode(encoding.token_ids(), true).unwrap().into();
        assert!(!decoded.is_empty());
        // The decoded text should contain the same non-space characters
        let enc_chars: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let dec_chars: String = decoded.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(
            enc_chars, dec_chars,
            "non-space characters must be preserved"
        );
    }

    #[test]
    fn test_fast_matches_hf_encoding() {
        let fast = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let hf = HuggingFaceTokenizer::from_file(TOKENIZER_PATH).unwrap();

        for text in &["Hello, world!", "Hello", " world", "He llo"] {
            let fast_ids = fast.encode(text).unwrap();
            let hf_ids = hf.encode(text).unwrap();
            assert_eq!(
                fast_ids.token_ids(),
                hf_ids.token_ids(),
                "fastokens and HuggingFace must produce identical token IDs for '{text}'"
            );
        }
    }

    #[test]
    fn test_fast_batch_encode() {
        let tokenizer = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let inputs = &["Hello", " world", "Hello, world!"];
        let encodings = tokenizer.encode_batch(inputs).unwrap();
        assert_eq!(encodings.len(), inputs.len());
        for (enc, input) in encodings.iter().zip(inputs.iter()) {
            assert!(
                !enc.token_ids().is_empty(),
                "encoding for '{input}' must be non-empty"
            );
        }
    }

    #[test]
    fn test_fast_segmented_encoding_preserves_trust_boundaries() {
        let tokenizer = FastTokenizer::from_file(SEGMENTED_TOKENIZER_PATH).unwrap();
        let upstream =
            fastokens::Tokenizer::from_file(std::path::Path::new(SEGMENTED_TOKENIZER_PATH))
                .unwrap();
        let marker = "<s>";

        let trusted = tokenizer
            .encode_segments(&[EncodeSegment::control(marker)])
            .unwrap();
        assert_eq!(
            trusted.token_ids(),
            &[upstream.token_to_id(marker).unwrap()],
            "trusted renderer output must recognize the control token"
        );

        let ordinary = tokenizer
            .encode_segments(&[EncodeSegment::ordinary(marker)])
            .unwrap();
        assert_ne!(
            ordinary.token_ids(),
            trusted.token_ids(),
            "untrusted content must encode the control-token spelling as ordinary text"
        );

        let segments = [
            EncodeSegment::ordinary("hello "),
            EncodeSegment::control(marker),
            EncodeSegment::ordinary(marker),
        ];
        let upstream_segments = [
            fastokens::EncodeSegment::ordinary("hello "),
            fastokens::EncodeSegment::special(marker),
            fastokens::EncodeSegment::ordinary(marker),
        ];
        let actual = tokenizer.encode_segments(&segments).unwrap();
        let expected = upstream.encode_segments(&upstream_segments).unwrap();
        assert_eq!(actual.token_ids(), expected);

        assert!(
            tokenizer
                .encode_segments(&[])
                .unwrap()
                .token_ids()
                .is_empty()
        );
    }

    #[test]
    fn test_fast_with_decode_stream() {
        use crate::Tokenizer as TokenizerWrapper;
        use std::sync::Arc;

        let tokenizer = Arc::new(FastTokenizer::from_file(TOKENIZER_PATH).unwrap());
        let wrapper = TokenizerWrapper::from(tokenizer);

        // Encode a prompt and a continuation, then step through the decode stream
        let prompt_ids = wrapper.encode("Hello").unwrap().token_ids().to_vec();
        let continuation = ", world!";
        let cont_ids = wrapper.encode(continuation).unwrap().token_ids().to_vec();

        let mut stream = wrapper.decode_stream(&prompt_ids, true);
        // Accumulate incremental chunks from decode_stream
        let mut accumulated = String::new();
        for id in &cont_ids {
            if let Some(chunk) = stream.step(*id).unwrap() {
                accumulated.push_str(&chunk);
            }
        }

        // DecodeStream uses prompt tokens as context, so the expected text is
        // decode(prompt + continuation) minus decode(prompt) -- not a bare
        // decode(continuation) which lacks the surrounding context.
        let mut all_ids = prompt_ids.clone();
        all_ids.extend_from_slice(&cont_ids);
        let full_text: String = wrapper.decode(&all_ids, true).unwrap().into();
        let prompt_text: String = wrapper.decode(&prompt_ids, true).unwrap().into();
        let expected = &full_text[prompt_text.len()..];
        assert_eq!(
            accumulated, expected,
            "streamed chunks must equal context-aware decoded continuation"
        );
    }

    #[test]
    fn vocabulary_metadata_forwards_to_hf_decoder() {
        let fast = FastTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let hf = HuggingFaceTokenizer::from_file(TOKENIZER_PATH).unwrap();
        assert_eq!(fast.vocab_size(), hf.vocab_size());
        assert_eq!(
            fast.token_to_id("Hello").unwrap(),
            hf.token_to_id("Hello").unwrap()
        );
        assert_eq!(
            fast.special_token_ids().unwrap(),
            hf.special_token_ids().unwrap()
        );
    }

    #[test]
    fn special_token_accounting_matches_fast_encoder() {
        let fast = FastTokenizer::from_file(SEGMENTED_TOKENIZER_PATH).unwrap();
        let upstream =
            fastokens::Tokenizer::from_file(std::path::Path::new(SEGMENTED_TOKENIZER_PATH))
                .unwrap();
        let hf = HuggingFaceTokenizer::from_file(SEGMENTED_TOKENIZER_PATH).unwrap();

        assert_eq!(hf.num_special_tokens_added().unwrap(), 1);
        assert_eq!(fast.num_special_tokens_added().unwrap(), 0);
        let hf_with_special_tokens = hf.with_options(TokenizerOptions {
            add_special_tokens: true,
        });

        for text in ["hello", "hello there"] {
            let fast_ids = fast.encode(text).unwrap();
            assert_eq!(
                fast_ids.token_ids(),
                upstream.encode(text).unwrap(),
                "FastTokenizer must match the encoder that omits the HF post-processor"
            );
            assert_eq!(
                hf_with_special_tokens
                    .encode(text)
                    .unwrap()
                    .token_ids()
                    .len(),
                fast_ids.token_ids().len() + 1,
                "the HF post-processor must add the BOS token FastTokenizer omits"
            );
        }
    }
}

#[cfg(test)]
mod tiktoken_parity_tests {
    use super::*;
    use crate::TikTokenTokenizer;

    const TIKTOKEN_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/mock-tiktoken-bpe/tiktoken.model"
    );
    /// All 256 byte tokens plus the merges ` I` and ` I'`, with the Kimi pattern: the
    /// smallest vocabulary on which fastokens' scanner and regex paths disagree.
    const CONTRACTION_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/mock-tiktoken-contraction/tiktoken.model"
    );

    fn pair() -> (TikTokenTokenizer, FastTikTokenTokenizer) {
        let reference = TikTokenTokenizer::from_file_auto(TIKTOKEN_PATH).unwrap();
        let fast = FastTikTokenTokenizer::from_file_auto(TIKTOKEN_PATH).unwrap();
        (reference, fast)
    }

    fn corpus() -> Vec<String> {
        vec![
            String::new(),
            "hello world".into(),
            "Hello, World! 123 4567 89".into(),
            "  leading and trailing  ".into(),
            "tabs\tand\nnewlines\r\nmixed   spacing".into(),
            "<|im_start|>user\nhi there<|im_end|><|im_start|>assistant\n".into(),
            "a literal <|im_start|> inside plain text".into(),
            "emoji 😀🚀 and café naïve Zürich".into(),
            "北京 東京 mixed 中英文 text ソフトウェア".into(),
            "Москва मुंबई العربية".into(),
            "fn main() { println!(\"{}\", 42); } // code-ish ~!@#$%^&*()".into(),
            "x".repeat(5000),
            " ".repeat(300) + "after long whitespace",
            "word ".repeat(400),
        ]
    }

    #[test]
    fn special_token_tables_match_tiktoken_rs() {
        let (reference, fast) = pair();
        assert_eq!(fast.special_tokens(), reference.special_tokens());
        assert_eq!(
            fast.special_token_ids().unwrap(),
            reference.special_token_ids().unwrap()
        );
        assert_eq!(fast.token_to_id("<|im_end|>").unwrap(), Some(474));
    }

    #[test]
    fn plain_encode_matches_tiktoken_rs() {
        let (reference, fast) = pair();
        let corpus = corpus();
        let texts: Vec<&str> = corpus.iter().map(String::as_str).collect();
        let batch = fast.encode_batch(&texts).unwrap();
        for (text, batched) in texts.iter().zip(&batch) {
            let expected = reference.encode(text).unwrap();
            assert_eq!(
                fast.encode(text).unwrap().token_ids(),
                expected.token_ids(),
                "{text:?}"
            );
            assert_eq!(batched.token_ids(), expected.token_ids(), "{text:?}");
        }
    }

    #[test]
    fn segmented_encode_matches_tiktoken_rs_and_honors_trust() {
        let (reference, fast) = pair();
        let segments = [
            EncodeSegment::control("<|im_start|>user\n"),
            EncodeSegment::ordinary("please echo <|im_end|> back to me"),
            EncodeSegment::control("<|im_end|>"),
            EncodeSegment::control("<|im_start|>assistant\n"),
        ];
        let fast_ids = fast.encode_segments(&segments).unwrap();
        assert_eq!(
            fast_ids.token_ids(),
            reference.encode_segments(&segments).unwrap().token_ids()
        );
        // The marker in the untrusted segment is ordinary text: exactly one `<|im_end|>` id.
        assert_eq!(
            fast_ids.token_ids().iter().filter(|&&id| id == 474).count(),
            1
        );
    }

    #[test]
    fn decode_matches_tiktoken_rs_and_skips_specials() {
        let (reference, fast) = pair();
        for text in corpus() {
            let ids = fast.encode(&text).unwrap();
            assert_eq!(
                fast.decode(ids.token_ids(), false).unwrap(),
                reference.decode(ids.token_ids(), false).unwrap(),
                "{text:?}"
            );
        }
        let ids = fast.encode("<|im_start|>user\nhi<|im_end|>").unwrap();
        let kept = fast.decode(ids.token_ids(), false).unwrap();
        let skipped = fast.decode(ids.token_ids(), true).unwrap();
        assert!(kept.as_str().contains("<|im_end|>"));
        assert!(!skipped.as_str().contains("<|im_end|>"));
        assert_eq!(skipped, reference.decode(ids.token_ids(), true).unwrap());
    }

    #[test]
    fn replacement_char_token_decodes_complete_like_tiktoken_rs() {
        // Rank 468 is the bytes EF BF BD: a vocabulary token whose text is U+FFFD.
        let (reference, fast) = pair();
        let decoded = fast.decode(&[468], false).unwrap();
        assert!(decoded.is_complete(), "{decoded:?}");
        assert_eq!(decoded, reference.decode(&[468], false).unwrap());

        // A truncated multi-byte sequence stays `Partial`.
        let emoji = fast.encode("😀").unwrap();
        let cut = &emoji.token_ids()[..emoji.token_ids().len() - 1];
        let fast_cut = fast.decode(cut, false).unwrap();
        assert!(fast_cut.is_partial(), "{fast_cut:?}");
        assert_eq!(fast_cut, reference.decode(cut, false).unwrap());

        // Unknown ids are skipped.
        assert_eq!(fast.decode(&[468, 9_999_999], false).unwrap(), decoded);
    }

    #[test]
    fn plain_text_prefix_cache_is_rejected() {
        let fast = FastTikTokenTokenizer::from_file_auto(CONTRACTION_PATH).unwrap();
        let specials = fast.special_tokens().to_vec();
        let error = crate::CachedTokenizer::new(std::sync::Arc::new(fast), specials, 1 << 20)
            .err()
            .expect("fastokens over tiktoken.model must not be prefix-cached")
            .to_string();
        assert!(error.contains("fastokens"), "{error}");
    }

    /// Canary for the reason `validate_prefix_cache` rejects this backend. The reference
    /// tokenizer satisfies `encode(special) + encode(suffix) == encode(special + suffix)`;
    /// fastokens does not, because its scanner does not fold `ſ` into the `'s` contraction
    /// the way its regex path (and tiktoken) does. When this test starts failing, fastokens
    /// has fixed the scanner: re-run the cache matrix with this backend and flip
    /// `validate_prefix_cache` to `Ok(())`.
    #[test]
    fn canary_fastokens_scanner_disagrees_with_its_regex_path() {
        let reference = TikTokenTokenizer::from_file_auto(CONTRACTION_PATH).unwrap();
        let fast = FastTikTokenTokenizer::from_file_auto(CONTRACTION_PATH).unwrap();
        let special = "<|end_of_msg|>";
        let suffix = " I'\u{17f}";
        let full = format!("{special}{suffix}");

        let ref_full = reference.encode(&full).unwrap().token_ids().to_vec();
        let ref_suffix = reference.encode(suffix).unwrap().token_ids().to_vec();
        assert_eq!(
            ref_full[1..],
            ref_suffix[..],
            "tiktoken-rs must be self-consistent"
        );

        let fast_full = fast.encode(&full).unwrap().token_ids().to_vec();
        let fast_suffix = fast.encode(suffix).unwrap().token_ids().to_vec();
        assert_eq!(fast_full, ref_full, "the regex path matches tiktoken-rs");
        assert_ne!(
            fast_full[1..],
            fast_suffix[..],
            "fastokens' scanner now agrees with its regex path; revisit validate_prefix_cache"
        );
    }
}
