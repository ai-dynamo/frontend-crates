// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve one media source to raw bytes — shared by routers and engines so
//! both resolve sources identically (the Python parity anchor is
//! `transformers.image_utils.load_image`, which engine fetchers like SGLang's
//! `get_image_bytes` mirror).
//!
//! Source precedence: `http(s)://` (bounded download,
//! [`FetchOptions::timeout`] and the proxy env vars honored the way
//! `requests` honors them, including `NO_PROXY` host/subdomain/CIDR/port
//! matching), `file://` / absolute path, `data:` URL, else bare base64.
//! Every read is charged against a byte budget *as it streams*, so an
//! oversized source stops mid-download instead of going fully resident first.
//!
//! This resolves one source, synchronously — scheduling many fetches
//! concurrently (and any async runtime) stays the consumer's concern.

/// Cap on any single resolved payload — HTTP, file, or base64 — so no source
/// form can exhaust memory.
pub const MAX_FETCH_BYTES: u64 = 64 << 20;

/// A byte allowance shared by every source of one request, charged as they
/// stream, so concurrent fetches stop at their combined size rather than each
/// stopping at [`MAX_FETCH_BYTES`].
#[derive(Debug)]
pub struct ByteBudget(#[allow(dead_code)] std::sync::atomic::AtomicU64);

impl ByteBudget {
    pub fn new(total: u64) -> Self {
        Self(std::sync::atomic::AtomicU64::new(total))
    }
}

/// Knobs of the network stage; [`Default`] matches Python engines' defaults.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FetchOptions {
    /// Per-source HTTP GET timeout (default 3 s).
    pub timeout: std::time::Duration,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            timeout: std::time::Duration::from_secs(3),
        }
    }
}

/// Resolve one string-typed media source into raw encoded bytes.
pub fn fetch_bytes(src: &str) -> Result<Vec<u8>, String> {
    fetch_bytes_budgeted(src, &ByteBudget::new(MAX_FETCH_BYTES))
}

/// [`fetch_bytes`] against a caller-owned allowance, for resolving several
/// sources under one whole-request bound. [`MAX_FETCH_BYTES`] still caps each.
pub fn fetch_bytes_budgeted(src: &str, budget: &ByteBudget) -> Result<Vec<u8>, String> {
    fetch_bytes_budgeted_with(src, budget, &FetchOptions::default())
}

/// [`fetch_bytes_budgeted`] with explicit [`FetchOptions`].
pub fn fetch_bytes_budgeted_with(
    src: &str,
    budget: &ByteBudget,
    opts: &FetchOptions,
) -> Result<Vec<u8>, String> {
    let _ = (src, budget, opts);
    todo!(
        "source-precedence dispatch, chunked budget-charged reads, requests-parity proxy handling"
    )
}
