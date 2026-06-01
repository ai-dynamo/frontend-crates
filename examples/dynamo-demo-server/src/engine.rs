// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared server state -- the integration point for the Dynamo crates.
//!
//! Holds an optional tokenizer loaded at startup. Handlers use it for accurate
//! token usage counts and the `/v1/tokenize` + `/v1/detokenize` endpoints.
//! Optionally also holds a [`PromptFormatter`] built from a model's
//! `tokenizer_config.json`, used by the `/v1/render` endpoint to showcase
//! dynamo-renderer's chat-template rendering.

use std::sync::Arc;

use dynamo_renderer::{ChatTemplate, ContextMixins, PromptFormatter};
use dynamo_tokenizers::Tokenizer;

#[derive(Clone)]
pub struct AppState {
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// Chat-template renderer, present when a `tokenizer_config.json` carrying a
    /// `chat_template` was loaded at startup. `PromptFormatter` is itself an
    /// `Arc`-backed handle, so cloning `AppState` per request is cheap.
    pub formatter: Option<PromptFormatter>,
}

impl AppState {
    pub fn new(tokenizer_path: Option<&str>, chat_template_config_path: Option<&str>) -> Self {
        let tokenizer = tokenizer_path.and_then(|path| match Tokenizer::from_file(path) {
            Ok(tk) => {
                tracing::info!(path, "loaded tokenizer");
                Some(Arc::new(tk))
            }
            Err(e) => {
                tracing::warn!(path, error = %e, "failed to load tokenizer; running without");
                None
            }
        });
        let formatter = chat_template_config_path.and_then(load_formatter);
        Self {
            tokenizer,
            formatter,
        }
    }

    /// Count tokens in a string. Falls back to len/4 heuristic when no tokenizer is loaded.
    pub fn count_tokens(&self, text: &str) -> u32 {
        match &self.tokenizer {
            Some(tk) => tk
                .encode(text)
                .map(|enc| enc.token_ids().len() as u32)
                .unwrap_or_else(|_| (text.len() as u32) / 4),
            None => (text.len() as u32) / 4,
        }
    }

    pub fn encode(&self, text: &str) -> Option<Vec<u32>> {
        self.tokenizer
            .as_ref()
            .and_then(|tk| tk.encode(text).ok())
            .map(|enc| enc.token_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Option<String> {
        self.tokenizer
            .as_ref()
            .and_then(|tk| tk.decode(ids, skip_special).ok())
            .map(String::from)
    }
}

/// Parse a `tokenizer_config.json` and build a chat-template [`PromptFormatter`]
/// from its `chat_template`. Returns `None` (with a warning) if the file can't
/// be read, parsed, or carries no `chat_template` -- the server then runs
/// without `/v1/render` support rather than failing to start.
fn load_formatter(path: &str) -> Option<PromptFormatter> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(path, error = %e, "failed to read tokenizer_config.json; /v1/render disabled");
            return None;
        }
    };
    let config: ChatTemplate = match serde_json::from_str(&raw) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(path, error = %e, "failed to parse tokenizer_config.json; /v1/render disabled");
            return None;
        }
    };
    if config.chat_template.is_none() {
        tracing::warn!(
            path,
            "tokenizer_config.json has no chat_template; /v1/render disabled"
        );
        return None;
    }
    match PromptFormatter::from_parts(config, ContextMixins::default(), false) {
        Ok(f) => {
            tracing::info!(path, "loaded chat-template renderer");
            Some(f)
        }
        Err(e) => {
            tracing::warn!(path, error = %e, "failed to build renderer; /v1/render disabled");
            None
        }
    }
}
