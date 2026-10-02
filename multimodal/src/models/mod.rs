// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One module per model family. A family implements
//! [`crate::processor::MmFamilyProcessor`] and registers a spec variant in
//! [`crate::registry::ProcessorSpec`]; everything else (request flow, caps,
//! failure semantics) comes from the driver for free.

pub mod qwen_vl;
