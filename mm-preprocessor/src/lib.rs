// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family multimodal preprocessing for LLM inference serving — a
//! Rust replacement for the image pipelines behind HF `AutoProcessor`.
//!
//! Model families implement [`pipeline::MmFamilyProcessor`] (turning decoded
//! media into named tensors and describing prompt geometry as data); the
//! model-independent [`driver::process`] owns request control flow: fetch →
//! hash → decode → per-item preprocess → token layout → positions. A family
//! is selected and configured through [`registry::PipelineSpec`], either as
//! the typed value or from a JSON spec of resolved processor parameters
//! ([`registry::pipeline_from_spec`]).
//!
//! Bit-exactness is the contract: the resize kernels ([`image::resize`]) and
//! each family's normalize/patchify reproduce the mirrored HF processor's
//! arithmetic exactly, so a serving engine can swap this crate in for the
//! Python path without output drift.
//!
//! Errors are `Result<T, String>` throughout: every `Err` is a
//! request-rejection message a serving engine surfaces to its client, not a
//! recoverable taxonomy.
//!
//! The crate reads no environment variables and, without the `parallel`
//! feature, owns no threads; pool sizing (`par::init_pool`, feature
//! `parallel`) and fetch timeouts (`fetch::FetchOptions`, feature `fetch`)
//! are explicit configuration.

pub mod driver;
#[cfg(feature = "fetch")]
pub mod fetch;
pub mod image;
pub mod par;
pub mod pipeline;
pub mod qwen_vl;
pub mod registry;
pub mod token_layout;

/// Content hash for cache/dedup identity: blake3 truncated to its first 8
/// bytes, big-endian.
pub fn content_hash_u64(data: &[u8]) -> u64 {
    let digest = blake3::hash(data);
    u64::from_be_bytes(digest.as_bytes()[..8].try_into().unwrap())
}
