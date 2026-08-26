// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deployment-owned implementations of `dynamo-agent-rt` tool contracts.

mod web_search;

pub use web_search::{
    BraveWebSearchError, BraveWebSearchExecutor, BraveWebSearchFailurePolicy,
    BraveWebSearchProfile, WebSearchArguments, WebSearchArgumentsError, WebSearchConfigError,
    WebSearchFreshness, WebSearchOutput, WebSearchResult,
};
