// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) const HARMONY_MARKERS: &[&str] = &[
    "<|start|>",
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<|call|>",
    "<|end|>",
    "<|return|>",
];

#[derive(Debug, Clone, Default)]
pub(crate) struct HarmonyPayload {
    body_pending: bool,
    depth: usize,
    in_string: bool,
    escape: bool,
}

impl HarmonyPayload {
    pub(crate) fn begin_body(&mut self) {
        *self = Self {
            body_pending: true,
            ..Self::default()
        };
    }

    pub(crate) fn in_string(&self) -> bool {
        self.in_string
    }

    pub(crate) fn in_json(&self) -> bool {
        self.depth > 0 || self.in_string
    }

    // Token bytes can split UTF-8 characters; JSON delimiters are ASCII.
    pub(crate) fn push_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if self.body_pending {
                if byte.is_ascii_whitespace() {
                    continue;
                }
                self.body_pending = false;
                match byte {
                    b'{' | b'[' => self.depth = 1,
                    b'"' => self.in_string = true,
                    _ => {}
                }
            } else if self.escape {
                self.escape = false;
            } else if self.in_string {
                match byte {
                    b'\\' => self.escape = true,
                    b'"' => self.in_string = false,
                    _ => {}
                }
            } else if self.depth > 0 {
                match byte {
                    b'"' => self.in_string = true,
                    b'{' | b'[' => self.depth += 1,
                    b'}' | b']' => self.depth -= 1,
                    _ => {}
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HarmonyVisible {
    line_start: bool,
    payload: Option<HarmonyPayload>,
}

impl Default for HarmonyVisible {
    fn default() -> Self {
        Self {
            line_start: true,
            payload: None,
        }
    }
}

impl HarmonyVisible {
    #[cfg(test)]
    pub(crate) fn in_json(&self) -> bool {
        self.payload.as_ref().is_some_and(HarmonyPayload::in_json)
    }

    pub(crate) fn in_string(&self) -> bool {
        self.payload.as_ref().is_some_and(HarmonyPayload::in_string)
    }

    // The reasoning projection separates visible messages with a newline. Keep
    // that boundary across chunks without treating new chunks as new messages.
    pub(crate) fn push_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if let Some(payload) = &mut self.payload {
                payload.push_bytes(&[byte]);
                if !payload.in_json() {
                    self.payload = None;
                }
                continue;
            }
            if self.line_start && byte.is_ascii_whitespace() {
                continue;
            }
            if matches!(byte, b'{' | b'[') || (self.line_start && byte == b'"') {
                let mut payload = HarmonyPayload::default();
                payload.begin_body();
                payload.push_bytes(&[byte]);
                self.payload = Some(payload);
            }
            self.line_start = byte == b'\n';
        }
    }

    pub(crate) fn protected_prefix_len(&mut self, text: &str) -> usize {
        let mut end = 0;
        for (index, &byte) in text.as_bytes().iter().enumerate() {
            if !self.in_string() && byte == b'<' {
                let remaining = &text[index..];
                if HARMONY_MARKERS
                    .iter()
                    .any(|marker| remaining.starts_with(marker) || marker.starts_with(remaining))
                {
                    break;
                }
            }
            self.push_bytes(&[byte]);
            end = index + 1;
        }
        end
    }
}
