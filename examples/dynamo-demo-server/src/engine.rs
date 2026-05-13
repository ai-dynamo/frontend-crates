//! Shared server state -- the integration point for the three Dynamo crates.
//!
//! Holds an optional tokenizer loaded at startup. Handlers use it for accurate
//! token usage counts and the `/v1/tokenize` + `/v1/detokenize` endpoints.

use std::sync::Arc;

use dynamo_tokenizers::Tokenizer;

#[derive(Clone)]
pub struct AppState {
    pub tokenizer: Option<Arc<Tokenizer>>,
}

impl AppState {
    pub fn new(tokenizer_path: Option<&str>) -> Self {
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
        Self { tokenizer }
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
