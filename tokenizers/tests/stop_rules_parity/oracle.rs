// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Dynamo frontend's code at dynamo `37f2f0a712`, copied as the oracle:
//! the end-of-sequence chain of `HFConfig::from_json_file`
//! (`lib/llm/src/model_card.rs:2176-2410`), and the stop-token part of
//! `Decoder::step_with_token_text` and `Decoder::finish`
//! (`lib/llm/src/backend.rs:895-1166`).
//!
//! Each change from the stock code is marked `ORACLE CHANGE`.

#![allow(clippy::all)]

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use dynamo_tokenizers::{DecodeStream, TokenIdType, file_json_field};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// lib/llm/src/model_card.rs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HFConfig {
    architectures: Vec<String>,
    model_type: String,
    text_config: Option<HFTextConfig>,
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFTextConfig {
    bos_token_id: Option<TokenIdType>,
    eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    final_eos_token_ids: Vec<TokenIdType>,
    max_position_embeddings: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    vocab_size: Option<usize>,
}

impl HFConfig {
    /// ORACLE CHANGE: returns the final EOS IDs, not the config, and reads
    /// `config.json` with `serde_json` where the stock code uses `json_five`.
    pub fn eos_token_ids(file: &Path) -> Result<Vec<TokenIdType>> {
        let file_path = file;
        let contents = std::fs::read_to_string(file_path)?;
        let mut config: Self = serde_json::from_str(&contents)?;
        if config.text_config.is_none() {
            let text_config: HFTextConfig = serde_json::from_str(&contents)?;
            config.text_config = Some(text_config);
        }

        let Some(text_config) = config.text_config.as_mut() else {
            anyhow::bail!(
                "Missing text config fields (model_type, eos_token_ids, etc) in config.json"
            );
        };

        let model_dir = file_path.parent().unwrap_or_else(|| Path::new(""));
        let gencfg_path = model_dir.join("generation_config.json");

        let mut final_eos_token_ids: Vec<TokenIdType> = {
            file_json_field::<serde_json::Value>(&gencfg_path, "eos_token_id")
                .ok()
                .and_then(|v| {
                    if v.is_number() {
                        v.as_number()
                            .and_then(|n| n.as_u64())
                            .map(|n| vec![n as TokenIdType])
                    } else if v.is_array() {
                        let arr = v.as_array().unwrap();
                        Some(
                            arr.iter()
                                .filter_map(|inner_v| {
                                    inner_v
                                        .as_number()
                                        .and_then(|n| n.as_u64())
                                        .map(|n| n as TokenIdType)
                                })
                                .collect(),
                        )
                    } else {
                        None
                    }
                })
        }
        .or_else(|| {
            config
                .eos_token_id
                .as_ref()
                .or(text_config.eos_token_id.as_ref())
                .and_then(|v| {
                    if v.is_number() {
                        v.as_number()
                            .and_then(|n| n.as_u64())
                            .map(|n| vec![n as TokenIdType])
                    } else {
                        serde_json::from_value(v.clone())
                            .map(Some)
                            .unwrap_or_else(|_err| None)
                    }
                })
        })
        .or_else(|| {
            resolve_token_id_from_tokenizer_files(model_dir, "eos_token").map(|id| vec![id])
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "missing eos_token_id in generation_config.json, config.json, \
                 tokenizer_config.json, and special_tokens_map.json — cannot load"
            )
        })?;

        if let Ok(tokenizer_eos_id) = resolve_token_id_from_tokenizer_config(
            &model_dir.join("tokenizer_config.json"),
            "eos_token",
        ) && !final_eos_token_ids.contains(&tokenizer_eos_id)
        {
            final_eos_token_ids.push(tokenizer_eos_id);
        }

        Ok(final_eos_token_ids)
    }
}

fn resolve_token_id_from_tokenizer_files(model_dir: &Path, token_key: &str) -> Option<TokenIdType> {
    if let Ok(id) =
        resolve_token_id_from_tokenizer_config(&model_dir.join("tokenizer_config.json"), token_key)
    {
        return Some(id);
    }
    resolve_token_id_from_special_tokens_map(model_dir, token_key).ok()
}

fn resolve_token_id_from_tokenizer_config(
    path: &Path,
    token_key: &str,
) -> anyhow::Result<TokenIdType> {
    let config = read_json(path)
        .with_context(|| format!("Failed to read or parse tokenizer_config.json: {:?}", path))?;
    let token_str = extract_token_string(config.get(token_key), token_key)?;
    if let Some(added_tokens) = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
        && let Ok(id) = lookup_id_in_added_tokens_decoder(added_tokens, &token_str)
    {
        return Ok(id);
    }
    let model_dir = path.parent().unwrap_or_else(|| Path::new(""));
    lookup_id_in_tokenizer_json(model_dir, &token_str).ok_or_else(|| {
        anyhow::anyhow!(
            "{token_key} '{token_str}' from tokenizer_config.json not found in \
             added_tokens_decoder or tokenizer.json added_tokens"
        )
    })
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn lookup_id_in_tokenizer_config(model_dir: &Path, token_str: &str) -> Option<TokenIdType> {
    let cfg = read_json(&model_dir.join("tokenizer_config.json"))?;
    let added = cfg.get("added_tokens_decoder")?.as_object()?;
    lookup_id_in_added_tokens_decoder(added, token_str).ok()
}

fn lookup_id_in_tokenizer_json(model_dir: &Path, token_str: &str) -> Option<TokenIdType> {
    let tok = read_json(&model_dir.join("tokenizer.json"))?;
    tok.get("added_tokens")?
        .as_array()?
        .iter()
        .filter(|e| e.get("content").and_then(|v| v.as_str()) == Some(token_str))
        .find_map(|e| e.get("id").and_then(|v| v.as_u64()))
        .map(|id| id as TokenIdType)
}

fn resolve_token_id_from_special_tokens_map(
    model_dir: &Path,
    token_key: &str,
) -> anyhow::Result<TokenIdType> {
    let stm = read_json(&model_dir.join("special_tokens_map.json"))
        .context("Failed to read or parse special_tokens_map.json")?;
    let token_str = extract_token_string(stm.get(token_key), token_key)?;

    lookup_id_in_tokenizer_config(model_dir, &token_str)
        .or_else(|| lookup_id_in_tokenizer_json(model_dir, &token_str))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{token_key} '{token_str}' from special_tokens_map.json not found in \
                 tokenizer_config.json added_tokens_decoder or tokenizer.json added_tokens"
            )
        })
}

fn extract_token_string(
    field: Option<&serde_json::Value>,
    token_key: &str,
) -> anyhow::Result<String> {
    match field {
        Some(serde_json::Value::String(s)) => Ok(s.clone()),
        Some(serde_json::Value::Object(obj)) => obj
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("{} is an object without 'content' field", token_key)),
        _ => anyhow::bail!("{} not found or not a string", token_key),
    }
}

fn lookup_id_in_added_tokens_decoder(
    added_tokens: &serde_json::Map<String, serde_json::Value>,
    token_str: &str,
) -> anyhow::Result<TokenIdType> {
    for (id_str, token_info) in added_tokens {
        let content = token_info
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if content == token_str {
            let token_id: TokenIdType = id_str.parse().with_context(|| {
                format!(
                    "Failed to parse token ID '{}' from added_tokens_decoder",
                    id_str
                )
            })?;
            return Ok(token_id);
        }
    }
    anyhow::bail!("token '{}' not found in added_tokens_decoder", token_str)
}

// ---------------------------------------------------------------------------
// lib/llm/src/backend.rs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopTrigger {
    HiddenStopTokenDetected(TokenIdType),
    UserStopTokenDetected(TokenIdType),
    VisibleStopTokenDetected(TokenIdType),
}

#[derive(Debug, Default)]
pub struct StepResult {
    pub released_text: Option<String>,
    pub stop_trigger: Option<StopTrigger>,
}

impl StepResult {
    fn ok(released_text: Option<String>) -> Self {
        Self {
            released_text,
            stop_trigger: None,
        }
    }
}

/// ORACLE CHANGE: the stop-token fields only. No tokenizer (so no per-token
/// text), no tracker, and no stop strings: with none, the stock
/// `process_decoded_text` returns its input and `flush_jailed` nothing.
pub struct Decoder {
    decode_stream: DecodeStream,
    pending_token_ids: Vec<TokenIdType>,
    hidden_stop_ids: HashSet<TokenIdType>,
    visible_stop_ids: HashSet<TokenIdType>,
    user_stop_ids: HashSet<TokenIdType>,
    no_stop_trim: bool,
    min_tokens: u32,
    generated_tokens: u32,
}

impl Decoder {
    /// ORACLE CHANGE: takes the three ID lists of `StopConditions`
    /// (`stop_token_ids`, `stop_token_ids_hidden`, `stop_token_ids_visible`).
    pub fn new(
        decode_stream: DecodeStream,
        stop_token_ids: Vec<TokenIdType>,
        stop_token_ids_hidden: Vec<TokenIdType>,
        stop_token_ids_visible: Vec<TokenIdType>,
        min_tokens: u32,
        no_stop_trim: bool,
    ) -> Self {
        let user_stop_ids: HashSet<TokenIdType> = stop_token_ids.iter().copied().collect();
        let system_stop_ids: HashSet<TokenIdType> = stop_token_ids_hidden.iter().copied().collect();
        let visible_stop_ids: HashSet<TokenIdType> =
            stop_token_ids_visible.iter().copied().collect();
        let hidden_stop_ids = user_stop_ids.union(&system_stop_ids).copied().collect();
        Self {
            decode_stream,
            pending_token_ids: Vec::new(),
            hidden_stop_ids,
            visible_stop_ids,
            user_stop_ids,
            no_stop_trim,
            min_tokens,
            generated_tokens: 0,
        }
    }

    pub fn step(&mut self, token_id: TokenIdType) -> Result<StepResult> {
        let below_min_tokens = self.generated_tokens < self.min_tokens;
        self.generated_tokens += 1;

        if !below_min_tokens
            && self.hidden_stop_ids.contains(&token_id)
            && !self.visible_stop_ids.contains(&token_id)
            && !self.no_stop_trim
        {
            let mut result = self.finish()?;
            result.stop_trigger.get_or_insert_with(|| {
                if self.user_stop_ids.contains(&token_id) {
                    StopTrigger::UserStopTokenDetected(token_id)
                } else {
                    StopTrigger::HiddenStopTokenDetected(token_id)
                }
            });
            return Ok(result);
        }

        self.pending_token_ids.push(token_id);
        let mut decoded_text = self.decode_stream.step(token_id)?;
        if decoded_text.is_some() {
            self.pending_token_ids.clear();
        }

        if below_min_tokens {
            return Ok(StepResult::ok(decoded_text));
        }

        if self.visible_stop_ids.contains(&token_id) || self.hidden_stop_ids.contains(&token_id) {
            let tail = self.decode_stream.finish()?;
            self.pending_token_ids.clear();
            if let Some(tail) = tail {
                decoded_text.get_or_insert_default().push_str(&tail);
            }
        }

        let mut decoded = StepResult::ok(decoded_text);

        let trigger = if self.visible_stop_ids.contains(&token_id) {
            StopTrigger::VisibleStopTokenDetected(token_id)
        } else if self.user_stop_ids.contains(&token_id) {
            StopTrigger::UserStopTokenDetected(token_id)
        } else if self.hidden_stop_ids.contains(&token_id) {
            StopTrigger::HiddenStopTokenDetected(token_id)
        } else {
            return Ok(decoded);
        };
        decoded.stop_trigger = Some(trigger);
        Ok(decoded)
    }

    pub fn finish(&mut self) -> Result<StepResult> {
        let tail = self.decode_stream.finish()?;
        self.pending_token_ids.clear();
        Ok(StepResult::ok(tail))
    }
}
