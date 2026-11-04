// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a chat template evaluates the messages it is given.
//!
//! [`RenderMode::Adaptive`] is the historical behaviour: message shapes are
//! adapted where the load-time probe found the template rejects them, so a
//! conversation renders instead of raising.
//!
//! [`RenderMode::Verbatim`] sends the messages through unmodified. A caller that
//! must reproduce another renderer byte-for-byte — an input tokenizer counting
//! what `transformers` would have counted, for example — needs the template to
//! see exactly the conversation the request carried, because every adaptation
//! changes the bytes it would count.
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

    /// Render the supplied messages exactly as given.
    ///
    /// No message-shape adaptation is applied: system turns are not demoted
    /// into user turns, consecutive user turns are not coalesced, content
    /// arrays are not normalized, and reasoning/tool-call arguments are passed
    /// through as given. A template that rejects the supplied shape returns an
    /// error rather than rendering a rewritten conversation.
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
