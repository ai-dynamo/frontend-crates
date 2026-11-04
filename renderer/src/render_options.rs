// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-render options that change *how* a chat template is evaluated.
//!
//! [`RenderOptions::default`] preserves the historical behaviour: message shapes
//! are adapted where the load-time probe found the template rejects them, and
//! `strftime_now`/`datetime` read the wall clock.
//!
//! Callers that must reproduce another renderer byte-for-byte — an input
//! tokenizer counting what `transformers` would have counted, for example —
//! need two things this type carries:
//!
//! * [`strict`](RenderOptions::strict): pass `messages` through unmodified, so
//!   a template that accepts a shape renders it verbatim instead of being
//!   reshaped into one it accepts.
//! * [`now`](RenderOptions::now): pin every clock-reading template function to
//!   one instant, so the same request renders identically on every attempt and
//!   a deferred render cannot disagree with the instant captured for it.

use std::time::SystemTime;

/// Options controlling how a chat template is evaluated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderOptions {
    /// Render `messages` exactly as supplied.
    ///
    /// When set, no message-shape adaptation is applied: system turns are not
    /// demoted into user turns, consecutive user turns are not coalesced,
    /// content arrays are not normalized, and reasoning/tool-call arguments are
    /// passed through as given. A template that rejects the supplied shape
    /// returns an error rather than rendering a rewritten conversation.
    pub strict: bool,

    /// Instant observed by every clock-reading template function in this render.
    ///
    /// `None` reads the wall clock, which is the historical behaviour. `Some`
    /// pins `strftime_now` to the given instant for the whole render; the
    /// binding is per render, so concurrent renders cannot observe each
    /// other's instant.
    pub now: Option<SystemTime>,
}

impl RenderOptions {
    /// Options that render `messages` verbatim from the given instant.
    pub fn strict_at(now: SystemTime) -> Self {
        Self {
            strict: true,
            now: Some(now),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn default_keeps_historical_behaviour() {
        let options = RenderOptions::default();
        assert!(!options.strict);
        assert!(options.now.is_none());
    }

    #[test]
    fn strict_at_pins_both() {
        let options = RenderOptions::strict_at(SystemTime::UNIX_EPOCH + Duration::from_secs(7));
        assert!(options.strict);
        assert_eq!(
            options.now,
            Some(SystemTime::UNIX_EPOCH + Duration::from_secs(7))
        );
    }
}
