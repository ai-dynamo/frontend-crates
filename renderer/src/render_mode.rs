// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a chat template evaluates the messages it is given.
//!
//! [`RenderMode::Adaptive`] is the historical behaviour: message shapes are
//! adapted where the load-time probe found the template rejects them, so a
//! conversation renders instead of raising.
//!
//! [`RenderMode::Verbatim`] applies no probe-driven message-shape adaptation, so
//! the template evaluates the conversation the request carried rather than one
//! reshaped to suit it. Every adaptation changes the bytes a counting caller
//! would see, which is why a counting caller needs this mode.
//!
//! Verbatim is scoped to that adaptation and nothing else. It is **not** a claim
//! of byte compatibility with another renderer: the template source is still
//! syntax-normalized and adapts a few known model templates, a typed request's
//! custom tool calls are still projected into function-call shape before
//! rendering, and the Jinja environment still supplies this crate's own filters.
//! Comparing rendered bytes against a different library remains a separate
//! question this mode does not answer.
//!
//! This is a property of the formatter, not of one render: it decides which
//! contract the instance implements and is fixed at construction. The instant a
//! render observes is a separate, per-request concern carried by
//! [`crate::OAIChatLikeRequest::render_instant`].

/// Which contract a formatter implements when it evaluates a template.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderMode {
    /// Adapt message shapes so templates that would reject them still render.
    #[default]
    Adaptive,

    /// Apply no probe-driven message-shape adaptation.
    ///
    /// System turns are not demoted into user turns, consecutive user turns are
    /// not coalesced, content arrays are not normalized, and reasoning/tool-call
    /// arguments are passed through as given. A template that rejects the
    /// supplied shape returns an error rather than rendering a rewritten
    /// conversation.
    ///
    /// This covers message adaptation only; see the module docs for what it does
    /// not promise.
    Verbatim,
}

impl RenderMode {
    /// True when message shapes may be adapted for this mode.
    pub fn adapts_messages(self) -> bool {
        matches!(self, Self::Adaptive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_adaptive() {
        assert_eq!(RenderMode::default(), RenderMode::Adaptive);
        assert!(RenderMode::default().adapts_messages());
    }

    #[test]
    fn verbatim_never_adapts() {
        assert!(!RenderMode::Verbatim.adapts_messages());
    }
}
