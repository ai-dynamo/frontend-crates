// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional compatibility helper for resolving trusted media sources to raw
//! bytes. Parity anchor: `transformers.image_utils.load_image`.
//!
//! This module is not an API-request security boundary: source syntax includes
//! local files, and this interface does not define a host allowlist or private
//! network and redirect policy. Frontends accepting untrusted URLs should use
//! their protected fetcher and pass the resolved bytes to this crate.
//!
//! Source precedence: `http(s)://`, `file://` / absolute path, `data:` URL,
//! else bare base64. HTTP downloads honor [`FetchOptions::timeout`] and the
//! proxy env vars with `requests` semantics (including `NO_PROXY` matching).
//! Reads are charged against a byte budget as they stream, so an oversized
//! source stops mid-download instead of going fully resident first.
//!
//! Resolution is synchronous and per-source; concurrency and async scheduling
//! stay the consumer's concern.

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

/// Resolve one trusted, string-typed media source into raw encoded bytes.
///
/// Do not call this directly on untrusted request URLs; see the module-level
/// security note.
pub fn fetch_bytes(src: &str) -> crate::Result<Vec<u8>> {
    fetch_bytes_budgeted(src, &ByteBudget::new(MAX_FETCH_BYTES))
}

/// [`fetch_bytes`] against a caller-owned allowance, for resolving several
/// sources under one whole-request bound. [`MAX_FETCH_BYTES`] still caps each.
pub fn fetch_bytes_budgeted(src: &str, budget: &ByteBudget) -> crate::Result<Vec<u8>> {
    fetch_bytes_budgeted_with(src, budget, &FetchOptions::default())
}

/// [`fetch_bytes_budgeted`] with explicit [`FetchOptions`].
pub fn fetch_bytes_budgeted_with(
    src: &str,
    budget: &ByteBudget,
    opts: &FetchOptions,
) -> crate::Result<Vec<u8>> {
    let _ = (src, budget, opts);
    todo!(
        "source-precedence dispatch, chunked budget-charged reads, requests-parity proxy handling"
    )
}
