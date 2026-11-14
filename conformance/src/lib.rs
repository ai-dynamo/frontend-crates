// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conformance fixtures, normalized storage and the authored Unified runner.

#[doc(hidden)]
#[path = "../tests/common/arguments.rs"]
pub mod arguments;
pub mod normalization;

mod runner;
mod yaml_normalization;

pub use runner::{CaseFailure, CaseUnsupported, RunOptions, RunReport, run_unified_family};
