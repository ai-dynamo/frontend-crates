// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family multimodal preprocessing for LLM inference serving — a
//! Rust replacement for the image pipelines behind HF `AutoProcessor`.
//!
//! Scope mirrors what HF ships: model families implement
//! [`processor::MmFamilyProcessor`] (decoded media → named tensors; prompt
//! geometry as data; position encodings; pixel-free token accounting),
//! selected and configured through [`registry::ProcessorSpec`] — resolved
//! from the HF config files ([`registry::spec_from_model_dir`], the
//! `AutoProcessor.from_pretrained` equivalent) or handed over pre-resolved.
//! The crate also carries the consumer-agnostic utilities a router and an
//! engine must agree on: media source resolution (the `fetch` module, feature
//! `fetch`) and content-hash identity ([`content_hash_u64`]). Request orchestration
//! (concurrency, caps, failure policy, packing) stays in the consumer's
//! driver, exactly as on the Python path; see the crate README for the boundary
//! and per-consumer sketches.
//!
//! Bit-exactness is the contract: the resize kernels ([`image::resize`]) and
//! each family's normalize/patchify reproduce the mirrored HF processor's
//! arithmetic exactly, so a serving engine can swap this crate in for the
//! Python path without output drift.
//!
//! Errors are `Result<T, String>` throughout: every `Err` is a
//! human-readable preprocessing failure for the consumer to surface, not a
//! recoverable taxonomy.
//!
//! The crate reads no environment variables and owns no threads until asked:
//! kernels run inline on the caller by default, fanning out on a crate-owned
//! rayon pool only after a consumer arms it (`execution::init_pool`). The
//! default-on `parallel` feature merely links rayon; opting out
//! (`default-features = false`) forces the inline path at compile time.

pub mod execution;
#[cfg(feature = "fetch")]
pub mod fetch;
pub mod image;
pub mod models;
pub mod processor;
pub mod registry;
pub mod token_layout;

/// Content hash for media identity: blake3 truncated to its first 8 bytes,
/// big-endian. One shared definition, so a router's cache-affinity keys and
/// an engine's prefix-cache/dedup keys agree on the same bytes.
pub fn content_hash_u64(data: &[u8]) -> u64 {
    let _ = data;
    todo!("blake3, first 8 bytes BE")
}
