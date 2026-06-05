// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conformance harness for the vendored Dynamo parity fixtures.
//!
//! No library code lives here — the work is in the integration tests under
//! `tests/`, which load `conformance/parity/**` and run it through
//! `dynamo-parsers`. This empty lib just makes the package a well-formed
//! workspace member that `cargo test --workspace` picks up.
