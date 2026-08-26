// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable storage adapters for `dynamo-agent-rt`.

#![cfg_attr(docsrs, feature(doc_cfg))]

mod error;

#[cfg(feature = "duckdb")]
mod duckdb;

pub use error::StoreInvariantError;

#[cfg(feature = "duckdb")]
pub use duckdb::{DuckDbStore, DuckDbStoreError};
