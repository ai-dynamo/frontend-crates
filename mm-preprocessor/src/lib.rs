// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family multimodal preprocessing for LLM inference serving — a
//! Rust replacement for the image pipelines behind HF `AutoProcessor`.
//!
//! Model families implement [`processor::MmFamilyProcessor`] (decoded media →
//! named tensors, prompt geometry as data, position encodings, pixel-free
//! token accounting), selected through [`registry::ProcessorSpec`] — resolved
//! from the HF config files ([`registry::spec_from_model_dir`]) or handed
//! over pre-resolved. The crate also carries what routers and engines must
//! agree on: media source resolution (the `fetch` module, feature `fetch`)
//! and content-hash identity ([`content_hash_u64`]). Request orchestration —
//! concurrency, caps, failure policy, packing — stays in the consumer's
//! driver, as on the Python path; the README maps the boundary.
//!
//! Bit-exactness is the contract: the resize kernels ([`image::resize`]) and
//! each family's normalize/patchify reproduce the mirrored HF processor's
//! arithmetic exactly, so an engine can swap this crate in without output
//! drift.
//!
//! Errors are `Result<T, String>`: every `Err` is a human-readable
//! preprocessing failure for the consumer to surface, not a recoverable
//! taxonomy.
//!
//! The crate reads no environment variables and owns no threads until a
//! consumer arms the rayon pool; see [`execution`].

pub mod execution;
#[cfg(feature = "fetch")]
pub mod fetch;
pub mod image;
pub mod models;
pub mod processor;
pub mod registry;
pub mod token_layout;

/// Media identity hash: blake3 truncated to its first 8 bytes, big-endian.
/// One definition, so router cache-affinity keys and engine prefix-cache/dedup
/// keys agree.
pub fn content_hash_u64(data: &[u8]) -> u64 {
    let _ = data;
    todo!("blake3, first 8 bytes BE")
}
