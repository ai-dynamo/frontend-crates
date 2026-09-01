// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family multimodal preprocessing for LLM inference serving — a
//! Rust replacement for the image pipelines behind HF `AutoProcessor`.
//!
//! Scope mirrors what HF ships, nothing more: model families implement
//! [`processor::MmFamilyProcessor`] (decoded media → named tensors; prompt
//! geometry as data; position encodings), selected and configured through
//! [`registry::ProcessorSpec`] — the `AutoProcessor.from_pretrained`
//! equivalent. Request orchestration (source fetching, content hashing,
//! per-request caps, failure policy) is the serving engine's driver, exactly
//! as it is on the Python path where the engine drives the HF processor; see
//! `DESIGN.md` for the boundary and an engine-side driver sketch.
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
//! feature, owns no threads; pool sizing (`execution::init_pool`, feature
//! `parallel`) is explicit configuration.
//!
//! This is the skeleton stage: signatures are final, bodies land with the
//! implementation PRs noted on each `todo!`.

pub mod execution;
pub mod image;
pub mod models;
pub mod processor;
pub mod registry;
pub mod token_layout;
