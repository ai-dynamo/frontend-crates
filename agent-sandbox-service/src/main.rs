// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agent_rt_sandbox::{
    DurableSandboxSupervisor, DurableSandboxSupervisorConfig, KubeAgentSandboxControlPlane,
    KubeAgentSandboxControlPlaneConfig, KubernetesSandboxConfig, KubernetesSandboxProvider,
    PostgresExecutionStore, SandboxdClient, SandboxdClientConfig,
};
use agent_rt_sandbox_service::config::SandboxCatalogFile;
use agent_rt_sandbox_service::{TrustedProxyAuth, router};
use clap::Parser;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[arg(long, env = "AGENT_SANDBOX_LISTEN", default_value = "0.0.0.0:8090")]
    listen: SocketAddr,

    #[arg(long, env = "AGENT_SANDBOX_DATABASE_URL")]
    database_url: String,

    #[arg(long, env = "AGENT_SANDBOX_DATABASE_POOL_SIZE", default_value_t = 16)]
    database_pool_size: usize,

    #[arg(long, env = "AGENT_SANDBOX_CATALOG")]
    catalog: PathBuf,

    #[arg(
        long,
        env = "AGENT_SANDBOX_TRUSTED_PROXY_TOKEN",
        hide_env_values = true
    )]
    trusted_proxy_token: String,

    #[arg(long, env = "POD_NAME")]
    owner_id: String,

    #[arg(
        long,
        env = "AGENT_SANDBOX_MAX_REQUEST_BYTES",
        default_value_t = 2 * 1024 * 1024
    )]
    max_request_bytes: usize,

    #[arg(long, env = "AGENT_SANDBOX_LEASE_MILLIS", default_value_t = 30_000)]
    lease_millis: u64,

    #[arg(long, env = "AGENT_SANDBOX_RENEW_MILLIS", default_value_t = 10_000)]
    renew_millis: u64,

    #[arg(
        long,
        env = "AGENT_SANDBOX_CANCELLATION_GRACE_MILLIS",
        default_value_t = 5_000
    )]
    cancellation_grace_millis: u64,

    #[arg(
        long,
        env = "AGENT_SANDBOX_CLAIM_READY_TIMEOUT_MILLIS",
        default_value_t = 60_000
    )]
    claim_ready_timeout_millis: u64,

    #[arg(long, env = "AGENT_SANDBOX_CLAIM_POLL_MILLIS", default_value_t = 200)]
    claim_poll_millis: u64,

    #[arg(long, env = "AGENT_SANDBOX_SANDBOXD_GRPC_PORT", default_value_t = 9090)]
    sandboxd_grpc_port: u16,

    #[arg(long, env = "AGENT_SANDBOX_SANDBOXD_REST_PORT", default_value_t = 8080)]
    sandboxd_rest_port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("failed to install the rustls ring provider"))?;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    anyhow::ensure!(cli.database_pool_size > 0, "database pool must be nonzero");
    anyhow::ensure!(cli.max_request_bytes > 0, "request limit must be nonzero");

    let catalog_json = std::fs::read_to_string(&cli.catalog)?;
    let catalog = SandboxCatalogFile::from_json(&catalog_json)?;
    let provider_config = KubernetesSandboxConfig::try_from(catalog)?;
    let auth = TrustedProxyAuth::new(&cli.trusted_proxy_token)?;

    let store = Arc::new(
        PostgresExecutionStore::connect_no_tls(&cli.database_url, cli.database_pool_size).await?,
    );
    let data_plane = Arc::new(SandboxdClient::new(SandboxdClientConfig {
        grpc_port: cli.sandboxd_grpc_port,
        rest_port: cli.sandboxd_rest_port,
        ..SandboxdClientConfig::default()
    })?);
    let supervisor = DurableSandboxSupervisor::new(
        data_plane,
        store,
        DurableSandboxSupervisorConfig {
            owner_id: cli.owner_id,
            lease_duration: Duration::from_millis(cli.lease_millis),
            renew_interval: Duration::from_millis(cli.renew_millis),
            cancellation_grace: Duration::from_millis(cli.cancellation_grace_millis),
        },
    )?;
    let kubernetes = KubeAgentSandboxControlPlane::new(
        kube::Client::try_default().await?,
        KubeAgentSandboxControlPlaneConfig {
            ready_timeout: Duration::from_millis(cli.claim_ready_timeout_millis),
            poll_interval: Duration::from_millis(cli.claim_poll_millis),
        },
    )?;
    let provider = Arc::new(KubernetesSandboxProvider::new(
        kubernetes,
        supervisor,
        provider_config,
    ));
    let app = router(provider, auth, cli.max_request_bytes);

    let listener = tokio::net::TcpListener::bind(cli.listen).await?;
    tracing::info!(address = %cli.listen, "sandbox executor service listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "failed to install Ctrl-C handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => tracing::error!(%error, "failed to install SIGTERM handler"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
