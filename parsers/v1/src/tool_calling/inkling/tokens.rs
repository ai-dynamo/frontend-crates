// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inkling special tokens shared by the tool-call and reasoning parsers, kept in
//! one place so the two can't drift.

pub(crate) const MESSAGE_MODEL: &str = "<|message_model|>";
pub(crate) const INVOKE: &str = "<|content_invoke_tool_json|>";
pub(crate) const END_MESSAGE: &str = "<|end_message|>";
pub(crate) const END_SAMPLING: &str = "<|content_model_end_sampling|>";
