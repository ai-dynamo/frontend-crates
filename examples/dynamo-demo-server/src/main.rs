mod echo;
mod engine;
mod handlers;

use std::path::PathBuf;

use axum::{Router, routing::{get, post}};
use clap::Parser;

use crate::engine::AppState;

/// dynamo-demo-server -- minimal OpenAI/Anthropic-compatible server showcasing
/// dynamo-protocols + dynamo-parsers + dynamo-tokenizers.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// HuggingFace repo id to fetch tokenizer.json from (e.g. "Qwen/Qwen2.5-7B-Instruct").
    /// Mutually exclusive with --tokenizer.
    #[arg(long, env = "MODEL")]
    model: Option<String>,

    /// HF revision (branch/tag/sha). Defaults to "main".
    #[arg(long, env = "MODEL_REVISION", default_value = "main")]
    revision: String,

    /// Path to a local tokenizer.json (or .model / .tiktoken). Mutually exclusive with --model.
    #[arg(long, env = "TOKENIZER_PATH")]
    tokenizer: Option<PathBuf>,

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
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
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

    let state = AppState::new(tokenizer_path.as_deref().and_then(|p| p.to_str()));
    let has_tk = state.tokenizer.is_some();

    let app = Router::new()
        .route("/v1/chat/completions", post(handlers::chat::handler))
        .route("/v1/completions", post(handlers::completions::handler))
        .route("/v1/responses", post(handlers::responses::handler))
        .route("/v1/messages", post(handlers::anthropic::handler))
        .route("/v1/tokenize", post(handlers::tokenize::encode))
        .route("/v1/detokenize", post(handlers::tokenize::decode))
        .route("/v1/tool-parse", post(handlers::tool_parse::handler))
        .route("/v1/reasoning-parse", post(handlers::reasoning_parse::handler))
        .route("/health", get(|| async { "ok" }))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.http_port);
    tracing::info!(
        addr = %addr,
        tokenizer = has_tk,
        "dynamo-demo-server: protocols + parsers + tokenizers"
    );
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Download `tokenizer.json` from HF Hub and return the cached path.
async fn fetch_hf_tokenizer(repo_id: &str, revision: &str) -> anyhow::Result<PathBuf> {
    use hf_hub::{Repo, RepoType, api::tokio::ApiBuilder};

    tracing::info!(repo = repo_id, revision, "fetching tokenizer.json from HF Hub");
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
