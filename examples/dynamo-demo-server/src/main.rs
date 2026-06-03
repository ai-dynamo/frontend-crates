// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod echo;
mod engine;
mod handlers;

use std::path::PathBuf;

use axum::{
    Router,
    routing::{get, post},
};
use clap::Parser;

use crate::engine::AppState;

/// dynamo-demo-server -- minimal OpenAI/Anthropic-compatible server showcasing
/// dynamo-protocols + dynamo-parsers + dynamo-renderer + dynamo-tokenizers.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// HuggingFace repo id to fetch tokenizer.json + tokenizer_config.json from
    /// (e.g. "Qwen/Qwen2.5-7B-Instruct"). Mutually exclusive with --tokenizer.
    #[arg(long, env = "MODEL")]
    model: Option<String>,

    /// HF revision (branch/tag/sha). Defaults to "main".
    #[arg(long, env = "MODEL_REVISION", default_value = "main")]
    revision: String,

    /// Path to a local tokenizer.json (or .model / .tiktoken). Mutually exclusive with --model.
    #[arg(long, env = "TOKENIZER_PATH")]
    tokenizer: Option<PathBuf>,

    /// Path to a local tokenizer_config.json carrying a `chat_template`, used by
    /// the `/v1/render` endpoint. When --model is used this is fetched from HF
    /// automatically; pass this to point at a local file instead.
    #[arg(long, env = "CHAT_TEMPLATE_CONFIG")]
    chat_template_config: Option<PathBuf>,

    /// Host to bind.
    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    host: String,

    /// HTTP port to listen on.
    #[arg(long, env = "HTTP_PORT", default_value_t = 3000)]
    http_port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    if cli.model.is_some() && cli.tokenizer.is_some() {
        anyhow::bail!("--model and --tokenizer are mutually exclusive");
    }

    let tokenizer_path = match (&cli.model, &cli.tokenizer) {
        (Some(repo), _) => Some(fetch_hf_tokenizer(repo, &cli.revision).await?),
        (_, Some(path)) => Some(path.clone()),
        _ => None,
    };

    // Resolve a tokenizer_config.json for chat-template rendering: an explicit
    // --chat-template-config wins; otherwise fetch it alongside the tokenizer
    // when --model is given. A missing/optional file is fine -- /v1/render just
    // stays disabled.
    let chat_template_config_path = match (&cli.chat_template_config, &cli.model) {
        (Some(path), _) => Some(path.clone()),
        (None, Some(repo)) => fetch_hf_tokenizer_config(repo, &cli.revision).await,
        _ => None,
    };

    let state = AppState::new(
        tokenizer_path.as_deref().and_then(|p| p.to_str()),
        chat_template_config_path
            .as_deref()
            .and_then(|p| p.to_str()),
    );
    let has_tk = state.tokenizer.is_some();
    let has_renderer = state.formatter.is_some();

    let app = Router::new()
        .route("/v1/chat/completions", post(handlers::chat::handler))
        .route("/v1/completions", post(handlers::completions::handler))
        .route("/v1/responses", post(handlers::responses::handler))
        .route("/v1/messages", post(handlers::anthropic::handler))
        .route("/v1/tokenize", post(handlers::tokenize::encode))
        .route("/v1/detokenize", post(handlers::tokenize::decode))
        .route("/v1/tool-parse", post(handlers::tool_parse::handler))
        .route(
            "/v1/reasoning-parse",
            post(handlers::reasoning_parse::handler),
        )
        .route("/v1/render", post(handlers::render::handler))
        .route("/health", get(|| async { "ok" }))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.http_port);
    tracing::info!(
        addr = %addr,
        tokenizer = has_tk,
        renderer = has_renderer,
        "dynamo-demo-server: protocols + parsers + renderer + tokenizers"
    );
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Download `tokenizer.json` from HF Hub and return the cached path.
async fn fetch_hf_tokenizer(repo_id: &str, revision: &str) -> anyhow::Result<PathBuf> {
    use hf_hub::{Repo, RepoType, api::tokio::ApiBuilder};

    tracing::info!(
        repo = repo_id,
        revision,
        "fetching tokenizer.json from HF Hub"
    );
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let path = repo.get("tokenizer.json").await?;
    tracing::info!(path = %path.display(), "tokenizer cached");
    Ok(path)
}

/// Download `tokenizer_config.json` from HF Hub and return the cached path.
/// Returns `None` (with a warning) if the repo doesn't ship one -- rendering is
/// optional, so this never aborts startup.
async fn fetch_hf_tokenizer_config(repo_id: &str, revision: &str) -> Option<PathBuf> {
    use hf_hub::{Repo, RepoType, api::tokio::ApiBuilder};

    tracing::info!(
        repo = repo_id,
        revision,
        "fetching tokenizer_config.json from HF Hub"
    );
    let api = ApiBuilder::new().with_progress(true).build().ok()?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    match repo.get("tokenizer_config.json").await {
        Ok(path) => {
            tracing::info!(path = %path.display(), "tokenizer_config cached");
            Some(path)
        }
        Err(e) => {
            tracing::warn!(repo = repo_id, error = %e, "no tokenizer_config.json; /v1/render disabled");
            None
        }
    }
}
