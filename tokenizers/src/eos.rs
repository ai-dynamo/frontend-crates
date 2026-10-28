// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A model's end-of-sequence token IDs, read from its files as the Dynamo
//! frontend's model card reads them.
//!
//! The first file that names them wins: `generation_config.json`, then
//! `config.json` (at the top, or in `text_config`), then the `eos_token` of
//! `tokenizer_config.json`, then that of `special_tokens_map.json`. The
//! `eos_token` of `tokenizer_config.json` is then added when it differs,
//! because some models end a reply with it although their config names
//! another (Qwen3.5 names `<|endoftext|>` and ends with `<|im_end|>`).
//!
//! A token named by its text resolves to its ID through
//! `tokenizer_config.json`'s `added_tokens_decoder`, then `tokenizer.json`'s
//! `added_tokens`.

use std::path::Path;

use anyhow::Context as _;
use serde_json::Value;

use crate::TokenIdType;

/// The end-of-sequence token IDs of the model whose files are in `dir`.
///
/// `config.json` is read as JSON. The Dynamo frontend reads it as JSON5, and
/// refuses the model when it does not parse; here a file that does not
/// parse names no IDs, and the chain goes on.
///
/// # Errors
///
/// When no file names an end-of-sequence token, or the one named has no ID.
pub fn eos_token_ids(dir: &Path) -> anyhow::Result<Vec<TokenIdType>> {
    let mut ids = generation_config_ids(dir)
        .or_else(|| config_ids(dir))
        .or_else(|| tokenizer_files_id(dir, "eos_token").map(|id| vec![id]))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "missing eos_token_id in generation_config.json, config.json, \
                 tokenizer_config.json, and special_tokens_map.json"
            )
        })?;
    if let Ok(id) = tokenizer_config_id(&dir.join("tokenizer_config.json"), "eos_token")
        && !ids.contains(&id)
    {
        ids.push(id);
    }
    Ok(ids)
}

/// `generation_config.json`'s `eos_token_id`: one ID, or a list whose
/// entries that are IDs count.
fn generation_config_ids(dir: &Path) -> Option<Vec<TokenIdType>> {
    let value = read_json(&dir.join("generation_config.json"))?
        .get("eos_token_id")?
        .clone();
    match value {
        Value::Number(number) => number.as_u64().map(|id| vec![id as TokenIdType]),
        Value::Array(values) => Some(
            values
                .iter()
                .filter_map(|value| value.as_u64().map(|id| id as TokenIdType))
                .collect(),
        ),
        _ => None,
    }
}

/// `config.json`'s `eos_token_id`, at the top or in `text_config`: one ID,
/// or a list that is all IDs.
fn config_ids(dir: &Path) -> Option<Vec<TokenIdType>> {
    let config = read_json(&dir.join("config.json"))?;
    let value = config
        .get("eos_token_id")
        .filter(|value| !value.is_null())
        .or_else(|| {
            config
                .get("text_config")?
                .get("eos_token_id")
                .filter(|value| !value.is_null())
        })?;
    if let Some(id) = value.as_u64() {
        return Some(vec![id as TokenIdType]);
    }
    match serde_json::from_value(value.clone()) {
        Ok(ids) => Some(ids),
        Err(error) => {
            tracing::error!(
                ?value,
                "eos_token_id is not a number or an array, cannot deserialize: {error}"
            );
            None
        }
    }
}

/// The ID of `token_key` from `tokenizer_config.json`, or else from
/// `special_tokens_map.json`.
fn tokenizer_files_id(dir: &Path, token_key: &str) -> Option<TokenIdType> {
    tokenizer_config_id(&dir.join("tokenizer_config.json"), token_key)
        .ok()
        .or_else(|| special_tokens_map_id(dir, token_key).ok())
}

/// The ID of `token_key` in the `tokenizer_config.json` at `path`: from its
/// `added_tokens_decoder`, or else from `tokenizer.json`'s `added_tokens`.
fn tokenizer_config_id(path: &Path, token_key: &str) -> anyhow::Result<TokenIdType> {
    let config = read_json(path)
        .with_context(|| format!("Failed to read or parse tokenizer_config.json: {path:?}"))?;
    let token = token_string(config.get(token_key), token_key)?;
    if let Some(added) = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        && let Ok(id) = added_tokens_decoder_id(added, &token)
    {
        return Ok(id);
    }
    let dir = path.parent().unwrap_or_else(|| Path::new(""));
    tokenizer_json_id(dir, &token).ok_or_else(|| {
        anyhow::anyhow!(
            "{token_key} '{token}' from tokenizer_config.json not found in \
             added_tokens_decoder or tokenizer.json added_tokens"
        )
    })
}

/// The ID of `token_key` in `special_tokens_map.json`, which names only its
/// text: from `tokenizer_config.json`, or else `tokenizer.json`.
fn special_tokens_map_id(dir: &Path, token_key: &str) -> anyhow::Result<TokenIdType> {
    let map = read_json(&dir.join("special_tokens_map.json"))
        .context("Failed to read or parse special_tokens_map.json")?;
    let token = token_string(map.get(token_key), token_key)?;
    read_json(&dir.join("tokenizer_config.json"))
        .and_then(|config| {
            let added = config.get("added_tokens_decoder")?.as_object()?;
            added_tokens_decoder_id(added, &token).ok()
        })
        .or_else(|| tokenizer_json_id(dir, &token))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{token_key} '{token}' from special_tokens_map.json not found in \
                 tokenizer_config.json added_tokens_decoder or tokenizer.json added_tokens"
            )
        })
}

/// A token's text, from a field that is `"<text>"` or `{"content": "<text>"}`.
fn token_string(field: Option<&Value>, token_key: &str) -> anyhow::Result<String> {
    match field {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Object(object)) => object
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| anyhow::anyhow!("{token_key} is an object without 'content' field")),
        _ => anyhow::bail!("{token_key} not found or not a string"),
    }
}

/// The ID of the first entry of `added` whose content is `token`. An entry
/// whose ID does not parse fails the lookup.
fn added_tokens_decoder_id(
    added: &serde_json::Map<String, Value>,
    token: &str,
) -> anyhow::Result<TokenIdType> {
    for (id, entry) in added {
        if entry.get("content").and_then(Value::as_str).unwrap_or("") == token {
            return id.parse().with_context(|| {
                format!("Failed to parse token ID '{id}' from added_tokens_decoder")
            });
        }
    }
    anyhow::bail!("token '{token}' not found in added_tokens_decoder")
}

/// The ID of `token` in `tokenizer.json`'s `added_tokens`.
fn tokenizer_json_id(dir: &Path, token: &str) -> Option<TokenIdType> {
    read_json(&dir.join("tokenizer.json"))?
        .get("added_tokens")?
        .as_array()?
        .iter()
        .filter(|entry| entry.get("content").and_then(Value::as_str) == Some(token))
        .find_map(|entry| entry.get("id").and_then(Value::as_u64))
        .map(|id| id as TokenIdType)
}

/// A file's JSON, or `None` when it is missing or does not parse.
fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}
