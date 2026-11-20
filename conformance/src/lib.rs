// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conformance fixtures and normalized storage shared by native readers.

pub mod normalization;
pub mod renderer;
#[cfg(feature = "renderer-runner")]
pub mod renderer_runner;
