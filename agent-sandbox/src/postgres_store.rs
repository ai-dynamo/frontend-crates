// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-replica execution storage using PostgreSQL row locks and database time.

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod, Runtime};
use thiserror::Error;
use tokio_postgres::NoTls;

const MIGRATION: &str = include_str!("../migrations/postgres/0001_sandbox_executions.sql");

#[derive(Debug, Error)]
pub enum PostgresExecutionStoreError {
    #[error("PostgreSQL failed: {0}")]
    Database(#[from] tokio_postgres::Error),
    #[error("PostgreSQL pool failed: {0}")]
    Pool(#[from] deadpool_postgres::PoolError),
    #[error("PostgreSQL pool configuration failed: {0}")]
    PoolBuild(#[from] deadpool_postgres::BuildError),
    #[error("persisted sandbox JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("persisted sandbox execution violates an invariant: {0}")]
    Invariant(&'static str),
}

#[derive(Clone)]
pub struct PostgresExecutionStore {
    pool: Pool,
}

impl PostgresExecutionStore {
    /// Builds a pool without transport TLS. Production deployments that need
    /// TLS should construct their connector and use [`Self::from_pool`].
    pub async fn connect_no_tls(
        database_url: &str,
        max_pool_size: usize,
    ) -> Result<Self, PostgresExecutionStoreError> {
        let postgres_config = database_url.parse::<tokio_postgres::Config>()?;
        let manager = Manager::from_config(
            postgres_config,
            NoTls,
            ManagerConfig {
                recycling_method: RecyclingMethod::Fast,
            },
        );
        let pool = Pool::builder(manager)
            .runtime(Runtime::Tokio1)
            .max_size(max_pool_size)
            .build()?;
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    pub fn from_pool(pool: Pool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub async fn migrate(&self) -> Result<(), PostgresExecutionStoreError> {
        let mut client = self.pool.get().await?;
        let transaction = client.transaction().await?;
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended('agent-rt-sandbox-migrations', 0))",
                &[],
            )
            .await?;
        transaction.batch_execute(MIGRATION).await?;
        transaction.commit().await?;
        Ok(())
    }
}
