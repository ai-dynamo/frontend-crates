// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mistral Tekken tokenizer (`tekken.json`).
//!
//! Tekken is a tiktoken-style byte-level BPE. The file stores the BPE ranks,
//! the split regex, and the special tokens. Special tokens own ids
//! `0..num_special_tokens`; an ordinary BPE token with rank `r` has id
//! `r + num_special_tokens`. This mirrors mistral-common's `Tekkenizer`.
//!
//! Plain `encode` never recognizes special-token text, matching
//! mistral-common, which encodes message text as ordinary BPE and inserts
//! structural tokens by id. Chat prompts must therefore arrive as segments
//! from a renderer that marks its control tokens ([`EncodeSegment::control`]);
//! a text-only Jinja prompt would encode `<s>[INST]` as plain text.

use std::path::Path;

use anyhow::Context as _;
use base64::Engine as _;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use tiktoken_rs::CoreBPE;

use super::{
    EncodeSegment, Encoding, Error, Result, TokenIdType, TokenizerOptions,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

/// Special tokens kept by `decode(skip_special_tokens = true)`.
///
/// Tool-call and reasoning parsers need these markers in the decoded text.
/// Upstream vLLM's `MistralTokenizer.convert_ids_to_tokens` keeps the same set
/// when skipping special tokens.
const PARSER_MARKERS: [&str; 4] = ["[TOOL_CALLS]", "[ARGS]", "[THINK]", "[/THINK]"];

/// Released tekken files reserve 1,000 special tokens. The bound keeps a
/// malformed file from allocating billions of `<SPECIAL_n>` fillers.
const MAX_SPECIAL_TOKENS: usize = 1 << 16;

/// Byte-level BPE needs every single byte as a token.
const BYTE_TOKENS: usize = 256;

#[derive(Deserialize)]
struct TekkenFile {
    config: TekkenConfig,
    vocab: Vec<TekkenTokenInfo>,
    special_tokens: Option<Vec<TekkenSpecialToken>>,
    model_settings_builder: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct TekkenConfig {
    pattern: String,
    default_vocab_size: usize,
    default_num_special_tokens: usize,
    version: String,
}

#[derive(Deserialize)]
struct TekkenTokenInfo {
    rank: usize,
    token_bytes: String,
}

#[derive(Deserialize)]
struct TekkenSpecialToken {
    rank: usize,
    token_str: String,
}

pub struct TekkenTokenizer {
    bpe: CoreBPE,
    num_special_tokens: u32,
    vocab_size: u32,
    special_token_ids: FxHashMap<String, TokenIdType>,
    kept_on_skip: Vec<TokenIdType>,
    version: String,
    model_settings_builder: Option<serde_json::Value>,
}

impl TekkenTokenizer {
    /// Load a Mistral `tekken.json` tokenizer file.
    pub fn from_file(path: &str) -> Result<Self> {
        let file: TekkenFile = {
            let contents = std::fs::read(path)
                .with_context(|| format!("Failed to read tekken file '{path}'"))?;
            serde_json::from_slice(&contents)
                .with_context(|| format!("Failed to parse tekken file '{path}'"))?
        };
        Self::from_parts(file)
    }

    fn from_parts(file: TekkenFile) -> Result<Self> {
        let TekkenConfig {
            pattern,
            default_vocab_size,
            default_num_special_tokens,
            version,
        } = file.config;
        let vocab_size = u32::try_from(default_vocab_size)
            .context("tekken default_vocab_size does not fit a token id")?;
        let num_special_tokens = u32::try_from(default_num_special_tokens)
            .context("tekken default_num_special_tokens does not fit a token id")?;
        if default_num_special_tokens > MAX_SPECIAL_TOKENS {
            anyhow::bail!(
                "tekken reserves {default_num_special_tokens} special tokens, more than the supported {MAX_SPECIAL_TOKENS}"
            );
        }

        let Some(specials) = file.special_tokens else {
            anyhow::bail!(
                "tekken {version} file has no special_tokens; only files that list them are supported"
            );
        };
        if specials.len() > default_num_special_tokens {
            anyhow::bail!(
                "tekken file lists {} special tokens but reserves only {default_num_special_tokens}",
                specials.len()
            );
        }
        let Some(inner_vocab_size) = default_vocab_size.checked_sub(default_num_special_tokens)
        else {
            anyhow::bail!("tekken default_vocab_size is smaller than its special tokens");
        };
        // CoreBPE panics when it meets a byte that has no token.
        if inner_vocab_size < BYTE_TOKENS {
            anyhow::bail!(
                "tekken keeps {inner_vocab_size} ordinary tokens, fewer than the {BYTE_TOKENS} single bytes"
            );
        }
        if file.vocab.len() < inner_vocab_size {
            anyhow::bail!(
                "tekken vocab has {} entries, fewer than the {inner_vocab_size} the config requires",
                file.vocab.len()
            );
        }

        // Ranks the file leaves unlisted are filled the way mistral-common does.
        let mut special_tokens: Vec<String> = (0..default_num_special_tokens)
            .map(|id| format!("<SPECIAL_{id}>"))
            .collect();
        for (index, token) in specials.into_iter().enumerate() {
            if token.rank != index {
                anyhow::bail!(
                    "tekken special token '{}' has rank {}, expected {index}",
                    token.token_str,
                    token.rank
                );
            }
            // An empty special token would make CoreBPE's special-token regex
            // match the empty string and loop forever.
            if token.token_str.is_empty() {
                anyhow::bail!("tekken special token {index} is empty");
            }
            special_tokens[index] = token.token_str;
        }

        let mut special_token_ids = FxHashMap::default();
        for (id, token) in (0..num_special_tokens).zip(special_tokens) {
            if let Some(previous) = special_token_ids.insert(token, id) {
                anyhow::bail!("tekken special token ids {previous} and {id} share one string");
            }
        }

        let engine = base64::engine::general_purpose::STANDARD;
        let mut ordinary_token_ids = FxHashMap::default();
        for (rank, token) in (0..).zip(file.vocab.into_iter().take(inner_vocab_size)) {
            if token.rank != rank as usize {
                anyhow::bail!("tekken vocab entry {rank} has rank {}", token.rank);
            }
            let bytes = engine
                .decode(&token.token_bytes)
                .with_context(|| format!("Invalid base64 in tekken vocab entry {rank}"))?;
            // mistral-common requires ranks 0..256 to be the single bytes in order.
            if (rank as usize) < BYTE_TOKENS && bytes != [rank as u8] {
                anyhow::bail!("tekken vocab entry {rank} is not the single byte {rank}");
            }
            // BPE merge priority compares ranks only by order, so shifting every
            // rank by the special-token count makes CoreBPE emit tekken ids.
            if ordinary_token_ids
                .insert(bytes, num_special_tokens + rank)
                .is_some()
            {
                anyhow::bail!("tekken vocab entry {rank} duplicates an earlier token");
            }
        }

        let bpe = CoreBPE::new(ordinary_token_ids, special_token_ids.clone(), &pattern)
            .map_err(|err| Error::msg(format!("Error creating tekken BPE: {err}")))?;

        let kept_on_skip = PARSER_MARKERS
            .iter()
            .filter_map(|marker| special_token_ids.get(*marker).copied())
            .collect();

        Ok(Self {
            bpe,
            num_special_tokens,
            vocab_size,
            special_token_ids,
            kept_on_skip,
            version,
            model_settings_builder: file.model_settings_builder,
        })
    }

    /// Whether `path` names a Mistral tekken tokenizer file (`tekken*.json`,
    /// such as `tekken.json` or mistral-common's versioned `tekken_240911.json`).
    pub fn is_tekken_file(path: &Path) -> bool {
        path.file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| name.starts_with("tekken") && name.ends_with(".json"))
    }

    /// The tokenizer format version, such as `v15`.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// The numeric part of [`Self::version`], such as `15` for `v15`.
    pub fn version_number(&self) -> Option<u32> {
        self.version.strip_prefix('v')?.parse().ok()
    }

    /// The raw `model_settings_builder` section (v15+), which sets the
    /// allowed values and defaults of `[MODEL_SETTINGS]` prompt fields.
    pub fn model_settings_builder(&self) -> Option<&serde_json::Value> {
        self.model_settings_builder.as_ref()
    }

    /// Reject ids outside the vocabulary before decoding.
    ///
    /// `CoreBPE::_decode_native_and_split` panics on an unknown id. Every id
    /// below `vocab_size` is known: special ids `0..N` are filled densely and
    /// ordinary ranks are contiguous and unique, as `from_parts` checks.
    fn validate_ids(&self, token_ids: &[TokenIdType]) -> Result<()> {
        match token_ids.iter().find(|&&id| id >= self.vocab_size) {
            Some(id) => anyhow::bail!(
                "token id {id} is outside the tekken vocabulary of {}",
                self.vocab_size
            ),
            None => Ok(()),
        }
    }
}

impl Encoder for TekkenTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        Ok(Encoding::Sp(self.bpe.encode_ordinary(input)))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.par_iter().map(|input| self.encode(input)).collect()
    }

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        let mut token_ids = Vec::new();
        for segment in segments {
            if !segment.allow_special {
                token_ids.extend(self.bpe.encode_ordinary(segment.text));
            } else if let Some(&id) = self.special_token_ids.get(segment.text) {
                // Renderers emit one control segment per marker; a direct lookup
                // avoids CoreBPE rebuilding its special-token set on every call.
                token_ids.push(id);
            } else {
                token_ids.extend(self.bpe.encode_with_special_tokens(segment.text));
            }
        }
        Ok(Encoding::Sp(token_ids))
    }
}

impl Decoder for TekkenTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.validate_ids(token_ids)?;
        let ids: Vec<u32> = if skip_special_tokens {
            token_ids
                .iter()
                .filter(|&&id| id >= self.num_special_tokens || self.kept_on_skip.contains(&id))
                .copied()
                .collect()
        } else {
            token_ids.to_vec()
        };

        // Same strategy as the tiktoken backend: strict UTF-8 first, then a
        // lossy fallback whose trailing U+FFFD marks an incomplete byte run.
        let bytes: Vec<u8> = self.bpe._decode_native_and_split(ids).flatten().collect();
        match String::from_utf8(bytes) {
            Ok(text) => Ok(DecodeResult::Complete(text)),
            Err(e) => {
                let text = String::from_utf8_lossy(e.as_bytes()).into_owned();
                Ok(DecodeResult::from_decoded(text))
            }
        }
    }
}

impl Tokenizer for TekkenTokenizer {
    // `validate_prefix_cache` keeps its default error: the prefix cache splits
    // text at special-token strings, but `encode` treats them as ordinary text.

    /// `add_special_tokens` is not applied: `encode` never adds `<s>`, which
    /// the Mistral renderer emits itself, as mistral-common does.
    fn with_options(self, _options: TokenizerOptions) -> Self {
        self
    }

    fn vocab_size(&self) -> Option<usize> {
        Some(self.vocab_size as usize)
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        if let Some(&id) = self.special_token_ids.get(token) {
            return Ok(Some(id));
        }
        // A string that BPE encodes to exactly one id is that token.
        Ok(match self.bpe.encode_ordinary(token).as_slice() {
            &[id] => Some(id),
            _ => None,
        })
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        Ok((0..self.num_special_tokens).collect())
    }

    // Tekken's `encode` never adds BOS/EOS; renderers emit `<s>` explicitly.
    fn num_special_tokens_added(&self) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecodeStream;
    use std::sync::Arc;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/mock-tekken/tekken.json"
    );
    const EXPECTED_IDS: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/mock-tekken/expected_ids.json"
    );

    #[derive(Deserialize)]
    struct ExpectedIds {
        text: String,
        ids: Vec<TokenIdType>,
    }

    fn id_of(tokenizer: &TekkenTokenizer, token: &str) -> Option<TokenIdType> {
        tokenizer.token_to_id(token).unwrap()
    }

    fn fixture() -> TekkenTokenizer {
        TekkenTokenizer::from_file(FIXTURE).unwrap()
    }

    #[test]
    fn encode_matches_mistral_common() {
        let tokenizer = fixture();
        let expected: Vec<ExpectedIds> =
            serde_json::from_str(&std::fs::read_to_string(EXPECTED_IDS).unwrap()).unwrap();
        assert!(!expected.is_empty());
        for case in expected {
            let ids = tokenizer.encode(&case.text).unwrap();
            assert_eq!(ids.token_ids(), case.ids.as_slice(), "text {:?}", case.text);
            let decoded: String = tokenizer.decode(&case.ids, false).unwrap().into();
            assert_eq!(decoded, case.text);
        }
    }

    #[test]
    fn metadata_comes_from_the_file() {
        let tokenizer = fixture();
        assert_eq!(tokenizer.version(), "v15");
        assert_eq!(tokenizer.version_number(), Some(15));
        assert_eq!(tokenizer.vocab_size(), Some(1040));
        assert_eq!(id_of(&tokenizer, "<s>"), Some(1));
        assert_eq!(id_of(&tokenizer, "</s>"), Some(2));
        assert_eq!(id_of(&tokenizer, "[TOOL_CALLS]"), Some(9));
        assert_eq!(id_of(&tokenizer, "[THINK]"), Some(34));
        // Ranks the file leaves unnamed become numbered fillers.
        assert_eq!(id_of(&tokenizer, "<SPECIAL_39>"), Some(39));
        assert_eq!(tokenizer.special_token_ids().unwrap().len(), 40);
        assert_eq!(tokenizer.token_to_id("[ARGS]").unwrap(), Some(32));
        assert_eq!(
            tokenizer.token_to_id("a").unwrap(),
            Some(40 + u32::from(b'a'))
        );
        assert_eq!(tokenizer.token_to_id("not-a-token-xyz").unwrap(), None);
        let builder = tokenizer.model_settings_builder().unwrap();
        assert_eq!(builder["reasoning_effort"]["default"], "none");
    }

    #[test]
    fn special_token_text_is_ordinary_unless_marked_control() {
        let tokenizer = fixture();
        let inst = id_of(&tokenizer, "[INST]").unwrap();

        let plain = tokenizer.encode("[INST]").unwrap();
        assert!(!plain.token_ids().contains(&inst));

        let segments = tokenizer
            .encode_segments(&[
                EncodeSegment::control("<s>[INST]"),
                EncodeSegment::ordinary("hi [INST]"),
                EncodeSegment::control("[/INST]"),
            ])
            .unwrap();
        let ids = segments.token_ids();
        assert_eq!(ids[..2], [1, inst]);
        assert_eq!(*ids.last().unwrap(), 4);
        assert_eq!(ids.iter().filter(|&&id| id == inst).count(), 1);
    }

    #[test]
    fn skipping_special_tokens_keeps_parser_markers() {
        let tokenizer = fixture();
        let id = |token: &str| id_of(&tokenizer, token).unwrap();
        let mut ids = vec![id("<s>"), id("[THINK]")];
        ids.extend(tokenizer.encode("plan").unwrap().token_ids());
        ids.extend([id("[/THINK]"), id("[TOOL_CALLS]")]);
        ids.extend(tokenizer.encode("f").unwrap().token_ids());
        ids.push(id("[ARGS]"));
        ids.extend(tokenizer.encode("{}").unwrap().token_ids());
        ids.push(id("</s>"));

        let skipped: String = tokenizer.decode(&ids, true).unwrap().into();
        assert_eq!(skipped, "[THINK]plan[/THINK][TOOL_CALLS]f[ARGS]{}");
        let kept: String = tokenizer.decode(&ids, false).unwrap().into();
        assert_eq!(kept, "<s>[THINK]plan[/THINK][TOOL_CALLS]f[ARGS]{}</s>");
    }

    #[test]
    fn decode_rejects_ids_outside_the_vocabulary() {
        let tokenizer = fixture();
        assert!(tokenizer.decode(&[1040], false).is_err());
    }

    #[test]
    fn decode_stream_reassembles_byte_fallback_characters() {
        let tokenizer = fixture();
        let ids = tokenizer.encode("北京 😀").unwrap().token_ids().to_vec();
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(tokenizer);
        let mut stream = DecodeStream::new(tokenizer, &[], false);
        let mut text = String::new();
        for id in ids {
            if let Some(chunk) = stream.step(id).unwrap() {
                assert!(!chunk.contains('\u{FFFD}'), "chunk {chunk:?}");
                text.push_str(&chunk);
            }
        }
        if let Some(chunk) = stream.finish().unwrap() {
            text.push_str(&chunk);
        }
        assert_eq!(text, "北京 😀");
    }

    #[test]
    fn file_dispatch_selects_tekken() {
        let tokenizer = crate::create_tokenizer_from_file(FIXTURE).unwrap();
        assert_eq!(tokenizer.vocab_size(), Some(1040));
        let ids = tokenizer.encode("Hello, world!").unwrap();
        assert_eq!(
            ids.token_ids(),
            fixture().encode("Hello, world!").unwrap().token_ids()
        );
    }

    #[test]
    fn control_segments_match_special_token_matching() {
        let tokenizer = fixture();
        let fast = tokenizer
            .encode_segments(&[EncodeSegment::control("[INST]")])
            .unwrap();
        let mixed = tokenizer
            .encode_segments(&[EncodeSegment::control("[INST][/INST]")])
            .unwrap();
        assert_eq!(fast.token_ids(), [3]);
        assert_eq!(mixed.token_ids(), [3, 4]);
    }

    /// Load the fixture after `edit`, and return the load error.
    fn load_error(edit: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut file: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap();
        edit(&mut file);
        let file: TekkenFile = serde_json::from_value(file).unwrap();
        TekkenTokenizer::from_parts(file).err().unwrap().to_string()
    }

    #[test]
    fn rejects_empty_special_tokens() {
        let error = load_error(|file| file["special_tokens"][5]["token_str"] = "".into());
        assert!(error.contains("empty"), "{error}");
    }

    #[test]
    fn rejects_vocab_without_the_byte_alphabet() {
        let error = load_error(|file| {
            let bytes = base64::engine::general_purpose::STANDARD.encode([0xff, 0xfe, 0xfd]);
            file["vocab"][65]["token_bytes"] = bytes.into();
        });
        assert!(error.contains("single byte 65"), "{error}");

        let error = load_error(|file| file["config"]["default_vocab_size"] = (40 + 255).into());
        assert!(error.contains("fewer than the 256"), "{error}");
    }

    #[test]
    fn rejects_oversized_special_token_reservation() {
        let error = load_error(|file| {
            file["config"]["default_vocab_size"] = u32::MAX.into();
            file["config"]["default_num_special_tokens"] = u32::MAX.into();
            file["special_tokens"] = serde_json::json!([]);
            file["vocab"] = serde_json::json!([]);
        });
        assert!(error.contains("special tokens, more than"), "{error}");
    }

    #[test]
    fn prefix_cache_is_refused() {
        assert!(fixture().validate_prefix_cache().is_err());
    }

    #[test]
    fn recognizes_tekken_file_names() {
        assert!(TekkenTokenizer::is_tekken_file(Path::new("/m/tekken.json")));
        assert!(TekkenTokenizer::is_tekken_file(Path::new(
            "tekken_240911.json"
        )));
        assert!(!TekkenTokenizer::is_tekken_file(Path::new(
            "/m/tokenizer.json"
        )));
        assert!(!TekkenTokenizer::is_tekken_file(Path::new(
            "/m/tekken.model"
        )));
    }
}
