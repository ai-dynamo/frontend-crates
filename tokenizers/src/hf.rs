// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hugging Face `tokenizer.json` backend, on the `tokenizers` 1.x pipeline.
//!
//! The 1.x [`PipelineTokenizer`] is read-only: it encodes and decodes what a `tokenizer.json`
//! describes and cannot be changed after it is built. Everything this backend adjusts is
//! therefore done to the JSON before the pipeline is read from it.

use std::{collections::HashSet, path::Path};

use anyhow::{Context as _, ensure};
use serde_json::Value;
use tokenizers::{
    DecoderRuntime,
    normalizers::metaspace::PrependBehavior,
    pipeline::{
        EncodeOptions, Encoding as PipelineEncoding, Inputs, Override, PipelineModel,
        PipelineNormalizer, PipelineToken, PipelineTokenizer,
    },
};

use super::{
    EncodeSegment, Encoding, Error, Result, TokenIdType, TokenizerOptions,
    traits::{DecodeResult, Decoder, Encoder, PrefixSplitRules, Tokenizer},
};

/// The library that reads, encodes and decodes, as errors name it.
const RUNTIME: &str = "Hugging Face tokenizers 1.x";

pub struct HuggingFaceTokenizer {
    pipeline: PipelineTokenizer,
    /// The same configuration with no added vocabulary, which encodes the ordinary segments of
    /// [`Encoder::encode_segments`]. `None` on a tokenizer loaded only to decode.
    ordinary: Option<PipelineTokenizer>,
    /// `<0xNN>` ids under a ByteFallback decoder. A trailing run of them can still decode
    /// differently once the next byte arrives.
    byte_fallback_ids: HashSet<u32>,
    special_ids: HashSet<u32>,
    /// Every added-token id, `special: true` or not. An ordinary segment of
    /// [`Encoder::encode_segments`] must never produce one of these; see
    /// that method.
    added_ids: HashSet<u32>,
    special_tokens: Vec<String>,
    token_id_bound: Option<usize>,
    /// Construction-time options; see [`TokenizerOptions`]. Defaults preserve
    /// the historical behavior (e.g. `add_special_tokens: false`). Set via
    /// [`Tokenizer::with_options`].
    options: TokenizerOptions,
    /// Where the configuration came from, for errors.
    source: String,
}

impl HuggingFaceTokenizer {
    /// Every added-token id of this vocabulary, `special: true` or not.
    pub(crate) fn added_ids(&self) -> &HashSet<u32> {
        &self.added_ids
    }

    /// Load from `tokenizer.json`, merging in special tokens declared only in
    /// a sibling `tokenizer_config.json`'s `added_tokens_decoder`. Without
    /// this, some releases (Qwen2-VL-2B's `<|image_pad|>`) BPE-shatter and
    /// silently break MM-aware routing. The merge is idempotent.
    ///
    /// The file's `truncation` and `padding` blocks are never applied, as Python
    /// `transformers` ignores them unless a call asks for them.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::load(path.as_ref(), true)
    }

    /// [`Self::from_file`] for a tokenizer that only decodes and answers vocabulary lookups, as
    /// [`FastTokenizer`](crate::FastTokenizer) uses it. It skips the second pipeline that
    /// segmented encoding needs.
    pub(crate) fn decoder_from_file(path: &Path) -> Result<Self> {
        Self::load(path, false)
    }

    /// Build from a parsed `tokenizer.json`. No `tokenizer_config.json` is consulted; apply
    /// [`merge_special_tokens_from_config`] first if the model has one.
    pub fn from_json(config: Value) -> Result<Self> {
        Self::build(config, "tokenizer configuration".to_string(), true)
    }

    fn load(path: &Path, segments: bool) -> Result<Self> {
        let source = path.display().to_string();
        let contents = std::fs::read_to_string(path)
            .map_err(|err| Error::msg(format!("Error loading tokenizer {source}: {err}")))?;
        let mut config: Value = serde_json::from_str(&contents).map_err(|err| {
            crate::log_json_err(&source, &contents, &err);
            Error::msg(format!("Error loading tokenizer {source}: {err}"))
        })?;
        if let Some(parent) = path.parent() {
            merge_special_tokens_from_config(&mut config, parent);
        }
        Self::build(config, source, segments)
    }

    fn build(mut config: Value, source: String, segments: bool) -> Result<Self> {
        let added_contents: Vec<String> = config
            .get("added_tokens")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|token| token.get("content").and_then(Value::as_str))
            .filter(|content| !content.is_empty())
            .map(str::to_owned)
            .collect();

        tokenizers::canonicalize_value(&mut config)
            .with_context(|| format!("failed to convert {source} for {RUNTIME}"))?;
        let pipeline = read_pipeline(&config, &source)?;

        // Every added token must keep its spelling. The pipeline keys a normalized added token
        // by its normalized form: the original spelling has no id, and a special one is skipped
        // on decode where its text used to be kept. Reject that rather than change the ids sent
        // to the model or the text returned.
        let added_vocabulary = pipeline.get_added_vocabulary();
        for content in &added_contents {
            let round_trips = pipeline.token_to_id(content).is_some_and(|id| {
                added_vocabulary
                    .id_to_token_bytes(id)
                    .is_some_and(|(bytes, _)| bytes == content.as_bytes())
            });
            ensure!(
                round_trips,
                "unsupported added token {content:?} in {source}: {RUNTIME} changes its id or decoded form"
            );
        }

        // The pipeline decodes byte-level BPE straight from bytes, bypassing the decoder.
        // Only accept configurations for which that shortcut preserves the decoded text.
        if matches!(pipeline.get_model(), PipelineModel::BPE(bpe) if bpe.is_byte_level()) {
            let supported = match pipeline.get_decoder() {
                Some(DecoderRuntime::ByteLevel(_)) => true,
                Some(DecoderRuntime::Sequence(decoders)) => {
                    matches!(decoders.as_slice(), [DecoderRuntime::ByteLevel(_)])
                }
                _ => false,
            };
            ensure!(
                supported,
                "unsupported byte-level BPE decoder in {source}: {RUNTIME} requires a direct ByteLevel decoder or a sequence containing only that decoder"
            );
        }

        if pipeline.get_truncation().is_some() {
            tracing::warn!(
                target: "tokenizer",
                source = %source,
                "tokenizer.json declares a truncation config; ignoring it so prompts are not silently clipped"
            );
        }
        if pipeline.get_padding().is_some() {
            tracing::warn!(
                target: "tokenizer",
                source = %source,
                "tokenizer.json declares a padding config; ignoring it for online tokenization"
            );
        }

        // ByteFallback decodes a contiguous run atomically: one invalid byte
        // replaces the entire run, including previously valid ASCII/UTF-8.
        // Discover the at-most-256 byte IDs once, not on the request hot path.
        fn has_byte_fallback(decoder: &DecoderRuntime) -> bool {
            match decoder {
                DecoderRuntime::ByteFallback(_) => true,
                DecoderRuntime::Sequence(decoders) => decoders.iter().any(has_byte_fallback),
                _ => false,
            }
        }
        let has_byte_fallback = pipeline.get_decoder().is_some_and(has_byte_fallback);
        let mut byte_fallback_ids = HashSet::new();
        let mut token_id_bound = None;
        for (token, id) in pipeline.get_vocab(true) {
            token_id_bound = token_id_bound.max(Some(id as usize + 1));
            if has_byte_fallback
                && token.len() == 6
                && token.starts_with("<0x")
                && token.ends_with('>')
                && u8::from_str_radix(&token[3..5], 16).is_ok()
            {
                byte_fallback_ids.insert(id);
            }
        }

        let added_tokens = added_vocabulary.get_added_tokens_decoder();
        let special_ids = added_tokens
            .iter()
            .filter_map(|(id, token)| token.special.then_some(*id))
            .collect();
        let mut special_tokens: Vec<String> = added_tokens
            .values()
            .filter(|token| token.special)
            .map(|token| token.content.clone())
            .collect();
        special_tokens.sort();
        special_tokens.dedup();
        let added_ids = added_tokens.into_keys().collect();

        let ordinary = if segments {
            config["added_tokens"] = Value::Array(Vec::new());
            Some(read_pipeline(&config, &source).with_context(|| {
                format!("failed to build the pipeline for ordinary segments of {source}")
            })?)
        } else {
            None
        };

        Ok(HuggingFaceTokenizer {
            pipeline,
            ordinary,
            byte_fallback_ids,
            special_ids,
            added_ids,
            special_tokens,
            token_id_bound,
            options: TokenizerOptions::default(),
            source,
        })
    }

    /// The contents of the added tokens marked `special`, sorted: the strings a prefix cache
    /// must treat as atomic. See [`CachedTokenizer`](crate::CachedTokenizer).
    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }

    /// One more than the largest token id, added tokens included. Ids can have gaps, so
    /// [`Tokenizer::vocab_size`] is not this bound.
    pub fn token_id_bound(&self) -> Option<usize> {
        self.token_id_bound
    }

    /// Truncation and padding stay off: they belong to a batching layer, and the prefix cache
    /// would otherwise apply them to each separately encoded piece.
    fn encode_options(&self, add_special_tokens: bool) -> EncodeOptions {
        EncodeOptions {
            add_special_tokens,
            encode_special_tokens: false,
            padding: Override::Off,
            truncation: Override::Off,
        }
    }

    fn encode_with(
        &self,
        pipeline: &PipelineTokenizer,
        inputs: impl Into<Inputs>,
        expected: usize,
        options: &EncodeOptions,
    ) -> Result<Vec<PipelineEncoding>> {
        let batch = pipeline
            .encode(inputs, options)
            .wait()
            .map_err(|err| Error::msg(format!("Error tokenizing input: {err}")))
            .with_context(|| self.source.clone())?;
        ensure!(
            batch.len() == expected,
            "{RUNTIME} returned {} encodings for {expected} inputs for {}",
            batch.len(),
            self.source
        );
        Ok(batch)
    }

    /// Encode one input on the calling thread, appending its ids to `out`.
    ///
    /// [`PipelineTokenizer::encode`] is not used for a single input: it sends any input of
    /// 16 KiB or more to a process-wide rayon pool sized to the machine. A frontend already
    /// runs one request per thread, and waking pool threads costs several times the CPU of
    /// the encode it splits, because a rayon worker sweeps every other worker's queue before
    /// it sleeps again. `encode_into` is the pipeline's serial entry point and returns the
    /// same ids.
    fn encode_into(
        &self,
        pipeline: &PipelineTokenizer,
        input: &str,
        options: &EncodeOptions,
        out: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        pipeline
            .encode_into(input, options, out)
            .map_err(|err| Error::msg(format!("Error tokenizing input: {err}")))
            .with_context(|| self.source.clone())
    }
}

fn read_pipeline(config: &Value, source: &str) -> Result<PipelineTokenizer> {
    let json =
        serde_json::to_string(config).with_context(|| format!("failed to serialize {source}"))?;
    tokenizers::from_json(&json)
        .map_err(|err| Error::msg(format!("Error loading tokenizer: {err}")))
        .with_context(|| format!("failed to construct a {RUNTIME} tokenizer from {source}"))
}

/// Promote `tokenizer_config.json`'s `special: true` `added_tokens_decoder`
/// entries into a parsed `tokenizer.json`. Missing-file / parse errors are swallowed since
/// the file is optional. See [`HuggingFaceTokenizer::from_file`].
///
/// A token that `tokenizer.json` already lists takes the config's flags; any other is appended
/// to `added_tokens`. The tokenizer reuses the vocabulary id of a token the model already has
/// and numbers the rest after the vocabulary, so the config's id is informational.
///
/// `pub` so a caller that reads `tokenizer.json` itself applies the same promotion as every
/// backend here; otherwise the tokenizer and a prefix cache built around it would disagree on
/// which strings are atomic specials.
pub fn merge_special_tokens_from_config(config: &mut Value, model_dir: &Path) {
    const FLAGS: [&str; 4] = ["single_word", "lstrip", "rstrip", "normalized"];

    let cfg_path = model_dir.join("tokenizer_config.json");
    let Ok(raw) = std::fs::read_to_string(&cfg_path) else {
        return;
    };
    let cfg: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(
                target: "tokenizer",
                path = %cfg_path.display(),
                error = %e,
                "tokenizer_config.json parse failed; skipping special-token merge"
            );
            return;
        }
    };
    let Some(decoder) = cfg.get("added_tokens_decoder").and_then(Value::as_object) else {
        return;
    };
    let Some(root) = config.as_object_mut() else {
        return;
    };
    let Some(added_tokens) = root
        .entry("added_tokens")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
    else {
        return;
    };

    let mut promoted: Vec<&str> = Vec::new();
    for (id, spec) in decoder {
        let Ok(id) = id.parse::<u32>() else {
            continue;
        };
        let Some(spec) = spec.as_object() else {
            continue;
        };
        if spec.get("special").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        let Some(content) = spec
            .get("content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        else {
            continue;
        };
        let mut token = serde_json::Map::from_iter([
            ("content".to_string(), Value::from(content)),
            ("special".to_string(), Value::Bool(true)),
        ]);
        for flag in FLAGS {
            let value = spec.get(flag).and_then(Value::as_bool).unwrap_or(false);
            token.insert(flag.to_string(), Value::Bool(value));
        }

        let existing = added_tokens
            .iter_mut()
            .find(|token| token.get("content").and_then(Value::as_str) == Some(content));
        match existing.and_then(Value::as_object_mut) {
            Some(existing) => {
                let unchanged = token
                    .iter()
                    .all(|(key, value)| existing.get(key).unwrap_or(&Value::Bool(false)) == value);
                if unchanged {
                    continue;
                }
                existing.extend(token);
            }
            None => {
                token.insert("id".to_string(), Value::from(id));
                added_tokens.push(Value::Object(token));
            }
        }
        promoted.push(content);
    }

    if !promoted.is_empty() {
        // Warn (not debug) when the merge actually promotes anything —
        // intentionally loud so accidental promotion of a previously-
        // non-special token shows up immediately in worker logs. Lists
        // the literal token strings so debugging doesn't require a
        // second pass through `added_tokens_decoder`.
        tracing::warn!(
            target: "tokenizer",
            path = %cfg_path.display(),
            added = promoted.len(),
            promoted = ?promoted,
            "merged additional special tokens from tokenizer_config.json"
        );
    }
}

impl Encoder for HuggingFaceTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let options = self.encode_options(self.options.add_special_tokens);
        let mut ids = Vec::with_capacity(input.len() / 4);
        self.encode_into(&self.pipeline, input, &options, &mut ids)?;
        Ok(Encoding::Sp(
            ids.into_iter().map(PipelineToken::id).collect(),
        ))
    }

    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        let Some(ordinary) = &self.ordinary else {
            return Err(Error::msg(
                "this Hugging Face tokenizer was loaded to decode only",
            ));
        };
        // Post-processing applies once to the joined sequence, rather than independently at
        // every trust boundary.
        let options = self.encode_options(false);
        let mut ids = Vec::new();
        for segment in segments {
            // No added vocabulary at all for an ordinary segment: `encode_special_tokens`
            // skips only `special: true` entries, and chat markers such as Qwen3's
            // `<think>` are `special: false`.
            let pipeline = if segment.allow_special {
                &self.pipeline
            } else {
                ordinary
            };
            let segment_start = ids.len();
            self.encode_into(pipeline, segment.text, &options, &mut ids)?;
            if !segment.allow_special {
                crate::reject_control_ids_in_ordinary_segment(
                    PipelineToken::cast_slice(&ids[segment_start..]),
                    &self.added_ids,
                )?;
            }
        }
        if self.options.add_special_tokens {
            let processed = self
                .pipeline
                .get_post_processor()
                .single
                .post_process::<true>(ids, None);
            ids = processed.ids().to_vec();
        }
        Ok(Encoding::Sp(PipelineToken::cast_slice(&ids).to_vec()))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        let options = self.encode_options(self.options.add_special_tokens);
        Ok(self
            .encode_with(&self.pipeline, inputs, inputs.len(), &options)?
            .iter()
            .map(|encoding| Encoding::Sp(PipelineToken::cast_slice(encoding.ids()).to_vec()))
            .collect())
    }
}

impl Decoder for HuggingFaceTokenizer {
    fn has_unstable_suffix(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> bool {
        token_ids
            .iter()
            .rev()
            .find(|id| !skip_special_tokens || !self.special_ids.contains(id))
            .is_some_and(|id| self.byte_fallback_ids.contains(id))
    }

    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.pipeline
            .decode(token_ids, skip_special_tokens)
            .map(DecodeResult::from_decoded)
            .map_err(|err| Error::msg(format!("Error de-tokenizing input: {err}")))
            .with_context(|| self.source.clone())
    }
}

/// The Metaspace marker when it is prepended only to text at offset 0 of the input. The 1.x
/// pipeline runs Metaspace as a normalizer, and a sequence arrives already flattened.
fn first_word_marker(normalizer: &PipelineNormalizer) -> Option<char> {
    match normalizer {
        PipelineNormalizer::Metaspace(metaspace)
            if metaspace.prepend() == PrependBehavior::First =>
        {
            Some(metaspace.delimiter())
        }
        _ => None,
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn validate_prefix_cache(&self) -> Result<()> {
        if self.options.add_special_tokens {
            return Err(Error::msg(
                "HuggingFace tokenizers configured with add_special_tokens=true must remain uncached",
            ));
        }
        Ok(())
    }

    fn prefix_split_rules(&self) -> PrefixSplitRules {
        let mut rules = PrefixSplitRules::default();
        let added_tokens = self
            .pipeline
            .get_added_vocabulary()
            .get_added_tokens_decoder();
        for token in added_tokens.into_values() {
            if token.rstrip {
                rules.rstrip.push(token.content.clone());
            }
            if token.single_word {
                rules.single_word.push(token.content);
            }
        }
        rules.first_word_marker = self
            .pipeline
            .get_normalizers()
            .iter()
            .find_map(first_word_marker);
        rules
    }

    /// Without added special tokens, a segment that allows special tokens goes through the
    /// pipeline `encode` runs.
    fn validate_segment_prefix_cache(&self) -> Result<()> {
        self.validate_prefix_cache()
    }

    /// HF honors [`TokenizerOptions::add_special_tokens`] (BOS/EOS per the
    /// tokenizer's post-processor template).
    fn with_options(mut self, options: TokenizerOptions) -> Self {
        self.options = options;
        self
    }

    fn vocab_size(&self) -> Option<usize> {
        Some(self.pipeline.get_vocab_size(true))
    }

    fn token_to_id(&self, token: &str) -> Result<Option<TokenIdType>> {
        // The pipeline resolves the empty string to an added token's id.
        if token.is_empty() {
            return Ok(None);
        }
        Ok(self.pipeline.token_to_id(token))
    }

    fn special_token_ids(&self) -> Result<Vec<TokenIdType>> {
        let mut ids: Vec<TokenIdType> = self.special_ids.iter().copied().collect();
        ids.sort_unstable();
        Ok(ids)
    }

    fn num_special_tokens_added(&self) -> Result<usize> {
        Ok(self.pipeline.get_post_processor().single.n_special())
    }
}

#[cfg(test)]
mod byte_fallback_stream_tests {
    use super::*;

    // `<0xNN>` is id NN: the reader requires every byte token of a byte-fallback vocabulary.
    const A: u32 = 0x61;
    const INVALID: u32 = 0xF5;
    const LEAD: u32 = 0xC3;
    const TAIL: u32 = 0xA9;
    const HELLO: u32 = 256;
    const BANG: u32 = 257;
    const EOS: u32 = 258;

    fn tokenizer() -> super::super::Tokenizer {
        let mut vocab = serde_json::Map::new();
        for byte in 0..=255u32 {
            vocab.insert(format!("<0x{byte:02X}>"), byte.into());
        }
        vocab.insert(" hello".into(), HELLO.into());
        vocab.insert("!".into(), BANG.into());
        vocab.insert("<eos>".into(), EOS.into());
        let config = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [{"id": EOS, "content": "<eos>", "special": true,
                "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "ByteFallback"}, {"type": "Fuse"}]},
            "model": {"type": "BPE", "vocab": vocab, "merges": [], "byte_fallback": true}
        });
        std::sync::Arc::new(HuggingFaceTokenizer::from_json(config).unwrap()).into()
    }

    #[test]
    fn byte_runs_match_full_decode() {
        let tokenizer = tokenizer();
        // Invalid tails rewrite the whole byte run, even previously valid ASCII
        // or a complete multibyte character. Valid, incomplete and literal
        // replacement output must all survive a terminal flush unchanged.
        for (ids, skip) in [
            (vec![A, INVALID, HELLO], false),
            (vec![LEAD, TAIL, INVALID, HELLO], false),
            (vec![LEAD, TAIL, BANG], false),
            (vec![A], false),
            (vec![LEAD, TAIL], false),
            (vec![LEAD], false),
            (vec![A, INVALID], false),
            (vec![A, EOS, INVALID, HELLO], false),
            (vec![A, EOS, INVALID, HELLO], true),
        ] {
            let expected: String = tokenizer.decode(&ids, skip).unwrap().into();
            let mut stream = tokenizer.decode_stream(&[], skip);
            let mut actual = String::new();
            for id in &ids {
                actual.push_str(&stream.step(*id).unwrap().unwrap_or_default());
            }
            actual.push_str(&stream.finish().unwrap().unwrap_or_default());
            assert_eq!(actual, expected, "ids={ids:?}, skip={skip}");
            assert_eq!(stream.finish().unwrap(), None);
        }
    }

    #[test]
    fn only_byte_suffix_is_delayed_and_prompt_is_not_emitted() {
        let tokenizer = tokenizer();
        let mut stream = tokenizer.decode_stream(&[HELLO], true);
        assert_eq!(stream.step(BANG).unwrap().as_deref(), Some("!"));
        assert_eq!(stream.step(A).unwrap(), None);
        assert_eq!(stream.step(INVALID).unwrap(), None);
        assert_eq!(stream.step(HELLO).unwrap().as_deref(), Some("�� hello"));
        assert_eq!(stream.finish().unwrap(), None);
    }
}

#[cfg(test)]
mod tests {
    //! The existing tool-calling / reasoning parser tests inject already-
    //! decoded text and never run `tokenizer.decode`, so they cannot
    //! catch the class of bug where a non-special marker gets accidentally
    //! promoted to "special" and silently disappears under
    //! `skip_special_tokens=True`. One unit test pins both halves of the
    //! gate (promote special:true / skip special:false) end-to-end through
    //! the actual encode -> decode round trip — the layer above which the
    //! parser tests cannot reach.
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn merge_gate_round_trips_through_decode() {
        // Minimal WordLevel `tokenizer.json`. `<|special_kept|>` and
        // `<|special_dropped|>` are deliberately NOT pre-declared as
        // special in `added_tokens` here — that's exactly the shape
        // `merge_special_tokens_from_config` is supposed to fix from
        // tokenizer_config.json.
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<|special_kept|>": 3, "<|special_dropped|>": 4},
                "unk_token": "<unk>"
            }
        }"#;

        // `<|special_kept|>` is `special: true` → must be promoted, and
        // therefore stripped under skip_special_tokens=true.
        // `<|special_dropped|>` is `special: false` → must be skipped,
        // and therefore survive skip_special_tokens=true. The latter is
        // the Ryan/Keiven concern: tool-call / reasoning markers are
        // universally declared `special: false` precisely so the parser
        // still sees them; a regression here would silently break
        // parsing without any of the parser unit tests turning red.
        const TOKENIZER_CONFIG_JSON: &str = r#"{
            "added_tokens_decoder": {
                "3": {"content": "<|special_kept|>",    "special": true,  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                "4": {"content": "<|special_dropped|>", "special": false, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
            }
        }"#;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        fs::write(
            dir.path().join("tokenizer_config.json"),
            TOKENIZER_CONFIG_JSON,
        )
        .unwrap();

        let tokenizer =
            HuggingFaceTokenizer::from_file(dir.path().join("tokenizer.json").to_str().unwrap())
                .unwrap();

        // Registry assertion: only the special:true entry was promoted.
        assert_eq!(
            tokenizer.special_tokens(),
            ["<unk>".to_string(), "<|special_kept|>".to_string()],
            "<|special_kept|> promoted; <|special_dropped|> stayed non-special"
        );

        // Decode round-trip assertion (the layer parser tests cannot
        // reach): the registry mutation actually changes downstream
        // decode behavior in the expected direction.
        let enc_kept = tokenizer.encode("<|special_kept|>").unwrap();
        assert_eq!(enc_kept.token_ids(), [3]);
        let decoded_strip: String = tokenizer.decode(enc_kept.token_ids(), true).unwrap().into();
        assert!(
            !decoded_strip.contains("<|special_kept|>"),
            "promoted special:true token must be stripped under skip_special_tokens=true; got {decoded_strip:?}"
        );

        let enc_drop = tokenizer.encode("<|special_dropped|>").unwrap();
        let decoded_keep: String = tokenizer.decode(enc_drop.token_ids(), true).unwrap().into();
        assert!(
            decoded_keep.contains("<|special_dropped|>"),
            "non-promoted special:false token must survive skip_special_tokens=true; got {decoded_keep:?}"
        );
    }

    #[test]
    fn add_special_tokens_flag_controls_encode() {
        // WordLevel tokenizer with a TemplateProcessing post-processor that
        // prepends <bos>. The post-processor only fires when encode is
        // called with add_special_tokens=true, which is exactly the knob
        // `TokenizerOptions`/`with_options` exposes.
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 3, "content": "<bos>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": {
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
                    "<bos>": {"id": "<bos>", "ids": [3], "tokens": ["<bos>"]}
                }
            },
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<bos>": 3},
                "unk_token": "<unk>"
            }
        }"#;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        let path = dir.path().join("tokenizer.json");
        let path = path.to_str().unwrap();

        let ids = |enc: &Encoding| enc.token_ids().to_vec();

        // Default: unchanged historical behavior — no BOS.
        let plain = HuggingFaceTokenizer::from_file(path).unwrap();
        assert_eq!(ids(&plain.encode("hello").unwrap()), vec![1]);

        let with_bos =
            HuggingFaceTokenizer::from_file(path)
                .unwrap()
                .with_options(TokenizerOptions {
                    add_special_tokens: true,
                });
        assert_eq!(ids(&with_bos.encode("hello").unwrap()), vec![3, 1]);
        let batch = with_bos.encode_batch(&["hello", "world"]).unwrap();
        assert_eq!(ids(&batch[0]), vec![3, 1]);
        assert_eq!(ids(&batch[1]), vec![3, 2]);

        // Same knob through the top-level wrapper: from_file keeps the
        // historical default, from_file_with_options threads the flag to
        // construction before the tokenizer goes behind the shared Arc.
        use crate::Tokenizer as TokenizerWrapper;
        let wrapper_plain = TokenizerWrapper::from_file(path).unwrap();
        assert_eq!(ids(&wrapper_plain.encode("hello").unwrap()), vec![1]);

        let wrapper_bos = TokenizerWrapper::from_file_with_options(
            path,
            TokenizerOptions {
                add_special_tokens: true,
            },
        )
        .unwrap();
        assert_eq!(ids(&wrapper_bos.encode("hello").unwrap()), vec![3, 1]);
    }

    #[test]
    fn truncation_and_padding_blocks_are_never_applied() {
        // A truncation block clips every prompt and a padding block pads it; Python
        // `transformers` applies neither unless a call asks. The blocks stay in the file and
        // are ignored, so the tokenizer remains safe to wrap in the prefix cache.
        let tokenizer = HuggingFaceTokenizer::from_json(serde_json::json!({
            "version": "1.0",
            "truncation": {"direction": "Right", "max_length": 2, "strategy": "LongestFirst", "stride": 0},
            "padding": {"strategy": {"Fixed": 4}, "direction": "Right", "pad_to_multiple_of": null,
                "pad_id": 3, "pad_type_id": 0, "pad_token": "[PAD]"},
            "added_tokens": [{"id": 3, "content": "[PAD]", "special": true, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false}],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "[PAD]": 3}}
        }))
        .unwrap();
        assert!(tokenizer.validate_prefix_cache().is_ok());
        assert_eq!(tokenizer.encode("hello").unwrap().token_ids(), [1]);
        assert_eq!(
            tokenizer.encode("hello world hello").unwrap().token_ids(),
            [1, 2, 1]
        );
        assert_eq!(
            tokenizer
                .encode_segments(&[
                    crate::EncodeSegment::control("hello world"),
                    crate::EncodeSegment::ordinary("hello"),
                ])
                .unwrap()
                .token_ids(),
            [1, 2, 1]
        );
    }

    #[test]
    fn vocab_introspection_accessors() {
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2},
                "unk_token": "<unk>"
            }
        }"#;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        let path = dir.path().join("tokenizer.json");

        let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(tokenizer.vocab_size(), Some(3));
        assert_eq!(tokenizer.token_to_id("hello").unwrap(), Some(1));
        assert_eq!(tokenizer.special_token_ids().unwrap(), vec![0]);
        assert_eq!(tokenizer.num_special_tokens_added().unwrap(), 0);
    }

    #[test]
    fn num_special_tokens_added_reflects_post_processor_additions() {
        const TOKENIZER_JSON: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                {"id": 3, "content": "<bos>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": {
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
                    "<bos>": {"id": "<bos>", "ids": [3], "tokens": ["<bos>"]}
                }
            },
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<bos>": 3},
                "unk_token": "<unk>"
            }
        }"#;

        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("tokenizer.json"), TOKENIZER_JSON).unwrap();
        let path = dir.path().join("tokenizer.json");

        let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(tokenizer.num_special_tokens_added().unwrap(), 1);
    }
}

#[cfg(test)]
mod segmented_tests {
    use super::*;
    use crate::EncodeSegment;

    #[test]
    fn segmented_hf_preserves_literal_controls_and_adds_bos_once() {
        let mut vocab = serde_json::json!({"<unk>":0,"<|call|>":1,"<bos>":2});
        for ch in "<|cal>".chars() {
            let id = vocab.as_object().unwrap().len();
            vocab[ch.to_string()] = serde_json::json!(id);
        }
        let config = serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null,
            "normalizer":null, "pre_tokenizer":null, "decoder":{"type":"Fuse"},
            "added_tokens":[{"id":1,"content":"<|call|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},
                {"id":2,"content":"<bos>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],
            "post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
                "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<bos>":{"id":"<bos>","ids":[2],"tokens":["<bos>"]}}},
            "model":{"type":"BPE","vocab":vocab,"merges":[],"unk_token":"<unk>"}
        });
        for add_special_tokens in [false, true] {
            let tokenizer = HuggingFaceTokenizer::from_json(config.clone())
                .unwrap()
                .with_options(TokenizerOptions { add_special_tokens });
            let encoding = tokenizer
                .encode_segments(&[
                    EncodeSegment::control("<|call|>"),
                    EncodeSegment::ordinary("<|call|>"),
                ])
                .unwrap();
            let ids = encoding.token_ids();
            assert_eq!(ids.iter().filter(|id| **id == 1).count(), 1);
            assert_eq!(
                ids.iter().filter(|id| **id == 2).count(),
                usize::from(add_special_tokens)
            );
            let decoded: String = tokenizer.decode(ids, true).unwrap().into();
            assert_eq!(decoded, "<|call|>");
            assert_eq!(
                tokenizer
                    .encode("<|call|>")
                    .unwrap()
                    .token_ids()
                    .iter()
                    .filter(|id| **id == 1)
                    .count(),
                1
            );
        }
    }
    /// A single input is encoded on the calling thread; a batch goes through the pipeline's
    /// thread pool once it is large enough. Both must give the same ids for an input that the
    /// pool would split at its special token.
    #[test]
    fn single_input_encode_matches_the_pipelines_batch_path() {
        let mut vocab = serde_json::json!({"<unk>":0,"<|call|>":1,"<bos>":2});
        for ch in "<|cal>".chars() {
            let id = vocab.as_object().unwrap().len();
            vocab[ch.to_string()] = serde_json::json!(id);
        }
        let config = serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null,
            "normalizer":null, "pre_tokenizer":null, "decoder":{"type":"Fuse"},
            "added_tokens":[{"id":1,"content":"<|call|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},
                {"id":2,"content":"<bos>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],
            "post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
                "pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<bos>":{"id":"<bos>","ids":[2],"tokens":["<bos>"]}}},
            "model":{"type":"BPE","vocab":vocab,"merges":[],"unk_token":"<unk>"}
        });
        // Two stretches of text, each over the pool's 8 KiB task size, around one special token.
        let input = format!("{}<|call|>{}", "call".repeat(3_000), "lac".repeat(3_000));
        assert!(input.len() > 16 * 1024);
        for add_special_tokens in [false, true] {
            let tokenizer = HuggingFaceTokenizer::from_json(config.clone())
                .unwrap()
                .with_options(TokenizerOptions { add_special_tokens });
            let single = tokenizer.encode(&input).unwrap();
            let batch = tokenizer.encode_batch(&[input.as_str()]).unwrap();
            assert_eq!(single.token_ids(), batch[0].token_ids());
            let segmented = tokenizer
                .encode_segments(&[EncodeSegment::control(&input)])
                .unwrap();
            assert_eq!(single.token_ids(), segmented.token_ids());

            let ids = single.token_ids();
            assert_eq!(ids.len(), 21_001 + usize::from(add_special_tokens));
            assert_eq!(ids.iter().filter(|id| **id == 1).count(), 1);
            assert_eq!(
                ids.iter().filter(|id| **id == 2).count(),
                usize::from(add_special_tokens)
            );
        }
    }

    #[test]
    fn segmented_hf_encodes_non_special_added_tokens_as_text() {
        // HF's `encode_special_tokens` skips only `special: true` entries;
        // chat markers such as Qwen3's `<think>` are registered `special: false`.
        let mut vocab = serde_json::json!({"<unk>":0,"<think>":1});
        for ch in "<think>".chars() {
            let id = vocab.as_object().unwrap().len();
            vocab[ch.to_string()] = serde_json::json!(id);
        }
        let config = serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null,
            "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":{"type":"Fuse"},
            "added_tokens":[{"id":1,"content":"<think>","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],
            "model":{"type":"BPE","vocab":vocab,"merges":[],"unk_token":"<unk>"}
        });
        let tokenizer = HuggingFaceTokenizer::from_json(config).unwrap();
        let ids = |segment| {
            tokenizer
                .encode_segments(&[segment])
                .unwrap()
                .token_ids()
                .to_vec()
        };
        assert_eq!(ids(EncodeSegment::control("<think>")), [1]);
        let ordinary = ids(EncodeSegment::ordinary("<think>"));
        assert_eq!(ordinary.len(), "<think>".len());
        assert!(!ordinary.contains(&1));
    }

    #[test]
    fn segmented_hf_rejects_ordinary_control_ids() {
        // `H`+`e` merges to id 18, declared here as the added token `He`:
        // ordinary text must not come back as that control id.
        let mut json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minimal-bpe/tokenizer.json"
            ))
            .unwrap(),
        )
        .unwrap();
        json["added_tokens"] = serde_json::json!([{
            "id": 18,
            "content": "He",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }]);
        let tokenizer = HuggingFaceTokenizer::from_json(json).unwrap();
        let control = tokenizer
            .encode_segments(&[EncodeSegment::control("He")])
            .unwrap();
        assert_eq!(control.token_ids(), &[18]);
        let error = tokenizer
            .encode_segments(&[EncodeSegment::ordinary("He")])
            .unwrap_err();
        assert!(
            error.to_string().contains("control-token id 18"),
            "unexpected error: {error}"
        );
    }
}
