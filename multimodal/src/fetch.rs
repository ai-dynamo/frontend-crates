// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Planned compatibility helper for resolving trusted media sources to raw
//! bytes. Parity anchor: `transformers.image_utils.load_image`.
//!
//! This module is a stub: its fetch functions always return
//! [`MmError::Unsupported`](crate::MmError::Unsupported). The behavior described
//! below is the intended contract for a future implementation.
//!
//! This module is not an API-request security boundary: source syntax includes
//! local files, and this interface does not define a host allowlist or private
//! network and redirect policy. Frontends accepting untrusted URLs should use
//! their protected fetcher and pass the resolved bytes to this crate.
//!
//! Source precedence: `http(s)://`, `file://` / absolute path, `data:` URL,
//! else bare base64. HTTP downloads will honor [`FetchOptions::timeout`] and
//! the proxy env vars with `requests` semantics (including `NO_PROXY`
//! matching). Reads will be charged against a byte budget as they stream, so
//! an oversized source stops mid-download instead of going fully resident first.
//!
//! Resolution will be synchronous and per-source; concurrency and async
//! scheduling stay the consumer's concern.

/// Intended cap on any single resolved payload — HTTP, file, or base64.
pub const MAX_FETCH_BYTES: u64 = 64 << 20;

/// Planned byte allowance shared by every source of one request. Charging is
/// not implemented; the fetch functions currently return an error.
#[derive(Debug)]
pub struct ByteBudget(#[allow(dead_code)] std::sync::atomic::AtomicU64);

impl ByteBudget {
    pub fn new(total: u64) -> Self {
        Self(std::sync::atomic::AtomicU64::new(total))
    }
}

/// Planned knobs of the network stage; [`Default`] matches Python engines' defaults.
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

/// Resolve one trusted, string-typed media source into raw encoded bytes.
///
/// Do not call this directly on untrusted request URLs; see the module-level
/// security note.
///
/// # Errors
///
/// Always returns [`MmError::Unsupported`](crate::MmError::Unsupported); not
/// implemented yet.
pub fn fetch_bytes(src: &str) -> crate::Result<Vec<u8>> {
    fetch_bytes_budgeted(src, &ByteBudget::new(MAX_FETCH_BYTES))
}

/// [`fetch_bytes`] against a caller-owned allowance, for resolving several
/// sources under one whole-request bound. [`MAX_FETCH_BYTES`] still caps each.
///
/// # Errors
///
/// Always returns [`MmError::Unsupported`](crate::MmError::Unsupported); not
/// implemented yet.
pub fn fetch_bytes_budgeted(src: &str, budget: &ByteBudget) -> crate::Result<Vec<u8>> {
    fetch_bytes_budgeted_with(src, budget, &FetchOptions::default())
}

/// [`fetch_bytes_budgeted`] with explicit [`FetchOptions`].
///
/// # Errors
///
/// Always returns [`MmError::Unsupported`](crate::MmError::Unsupported); not
/// implemented yet.
pub fn fetch_bytes_budgeted_with(
    src: &str,
    budget: &ByteBudget,
    opts: &FetchOptions,
) -> crate::Result<Vec<u8>> {
    let _ = (src, budget, opts);
    Err(crate::MmError::unsupported(
        "fetch is not implemented yet; resolve media with your own fetcher",
    ))
}
