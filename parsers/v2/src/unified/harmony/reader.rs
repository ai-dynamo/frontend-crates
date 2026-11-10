// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Delta-only storage for the released Harmony reader's strict token protocol.
//! Body decoding follows OpenAI's Apache-2.0 `openai-harmony` 0.0.8
//! `src/encoding.rs::StreamableParser`; header grammar stays owned by that reader.

use std::collections::HashSet;

use openai_harmony::chat::Role;
use openai_harmony::{HarmonyEncoding, StreamableParser};

#[derive(Clone)]
enum State {
    ExpectStart,
    Header(Vec<u32>),
    Body {
        role: Option<Role>,
        recipient: Option<String>,
        channel: Option<String>,
    },
}

pub(super) struct DeltaReader {
    encoding: HarmonyEncoding,
    start: u32,
    message: u32,
    stop_tokens: HashSet<u32>,
    state: State,
    undecoded_tokens: Vec<u32>,
    undecoded_bytes: Vec<u8>,
    last_delta: Option<String>,
    completed: usize,
    skipped_pending: usize,
}

impl DeltaReader {
    pub(super) fn new(encoding: HarmonyEncoding) -> anyhow::Result<Self> {
        Ok(Self {
            start: super::token_id(super::START)?,
            message: super::token_id(super::MESSAGE)?,
            stop_tokens: encoding.stop_tokens()?,
            encoding,
            state: State::ExpectStart,
            undecoded_tokens: Vec::new(),
            undecoded_bytes: Vec::new(),
            last_delta: None,
            completed: 0,
            skipped_pending: 0,
        })
    }

    pub(super) fn fork_continuation(&self) -> Self {
        Self {
            encoding: self.encoding.clone(),
            start: self.start,
            message: self.message,
            stop_tokens: self.stop_tokens.clone(),
            state: self.state.clone(),
            undecoded_tokens: self.undecoded_tokens.clone(),
            undecoded_bytes: self.undecoded_bytes.clone(),
            last_delta: None,
            completed: self.completed,
            skipped_pending: 0,
        }
    }

    pub(super) fn process(&mut self, id: u32) -> anyhow::Result<()> {
        match std::mem::replace(&mut self.state, State::ExpectStart) {
            State::ExpectStart => {
                anyhow::ensure!(
                    id == self.start,
                    "Unexpected token {} while expecting start token {}",
                    id,
                    self.start
                );
                self.state = State::Header(Vec::new());
            }
            State::Header(mut tokens) => {
                if id == self.message {
                    // The released reader owns role, channel and recipient grammar.
                    let mut parser = StreamableParser::new(self.encoding.clone(), None)?;
                    parser.process(self.start)?;
                    for token in tokens {
                        parser.process(token)?;
                    }
                    parser.process(id)?;
                    self.state = State::Body {
                        role: parser.current_role(),
                        recipient: parser.current_recipient(),
                        channel: parser.current_channel(),
                    };
                } else {
                    tokens.push(id);
                    self.state = State::Header(tokens);
                }
            }
            body @ State::Body { .. } => {
                if self.stop_tokens.contains(&id) {
                    self.complete_message();
                } else {
                    self.state = body;
                    self.decode_body(id);
                }
            }
        }
        Ok(())
    }

    pub(super) fn process_eos(&mut self) -> anyhow::Result<()> {
        match &self.state {
            State::ExpectStart => {}
            State::Header(_) => {
                anyhow::bail!("Unexpected EOS while waiting for message header to complete");
            }
            State::Body { .. } => self.complete_message(),
        }
        Ok(())
    }

    fn complete_message(&mut self) {
        self.skipped_pending += self.discard_pending();
        self.last_delta = None;
        self.completed += 1;
        self.state = State::ExpectStart;
    }

    fn decode_body(&mut self, id: u32) {
        self.undecoded_tokens.push(id);
        let Ok(bytes) = self
            .encoding
            .tokenizer()
            .decode_bytes(&self.undecoded_tokens)
        else {
            self.last_delta = None;
            return;
        };
        self.undecoded_bytes.extend(bytes);
        match String::from_utf8(self.undecoded_bytes.clone()) {
            Ok(text) => {
                self.last_delta = Some(text);
                self.undecoded_bytes.clear();
            }
            Err(error) => {
                let utf8_error = error.utf8_error();
                let bytes = error.into_bytes();
                let valid_len = utf8_error.valid_up_to();
                let mut delta = String::new();
                if valid_len > 0 {
                    // `valid_up_to` is the decoder's validated UTF-8 boundary.
                    delta.push_str(
                        std::str::from_utf8(&bytes[..valid_len]).expect("validated UTF-8"),
                    );
                    self.undecoded_bytes.drain(..valid_len);
                }
                match utf8_error.error_len() {
                    Some(length) => {
                        delta.push('\u{fffd}');
                        self.undecoded_bytes.drain(..length);
                    }
                    None => self.last_delta = None,
                }
                if !delta.is_empty() {
                    self.last_delta = Some(delta);
                }
            }
        }
        self.undecoded_tokens.clear();
    }

    pub(super) fn discard_pending(&mut self) -> usize {
        let skipped = self.undecoded_tokens.len() + self.undecoded_bytes.len();
        self.undecoded_tokens.clear();
        self.undecoded_bytes.clear();
        skipped
    }

    pub(super) fn take_skipped_pending(&mut self) -> usize {
        std::mem::take(&mut self.skipped_pending)
    }

    pub(super) fn message_count(&self) -> usize {
        self.completed
    }

    pub(super) fn last_content_delta(&self) -> Option<&str> {
        self.last_delta.as_deref()
    }

    pub(super) fn current_role(&self) -> Option<Role> {
        match &self.state {
            State::Body { role, .. } => role.clone(),
            _ => None,
        }
    }

    pub(super) fn current_recipient(&self) -> Option<&str> {
        match &self.state {
            State::Body { recipient, .. } => recipient.as_deref(),
            _ => None,
        }
    }

    pub(super) fn current_channel(&self) -> Option<&str> {
        match &self.state {
            State::Body { channel, .. } => channel.as_deref(),
            _ => None,
        }
    }

    pub(super) fn header_tokens(&self) -> Option<&[u32]> {
        match &self.state {
            State::Header(tokens) => Some(tokens),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoding() -> HarmonyEncoding {
        super::super::harmony_encoding().unwrap().clone()
    }

    fn parity(encoding: &HarmonyEncoding, ids: &[u32]) {
        let mut actual = DeltaReader::new(encoding.clone()).unwrap();
        let mut expected = StreamableParser::new(encoding.clone(), None).unwrap();
        for id in ids {
            let left = actual.process(*id);
            let right = expected.process(*id).map(|_| ());
            assert_eq!(
                left.as_ref().map_err(ToString::to_string),
                right.as_ref().map_err(ToString::to_string),
                "token={id}"
            );
            if left.is_err() {
                break;
            }
            assert_eq!(
                actual.last_content_delta(),
                expected.last_content_delta().unwrap().as_deref()
            );
            assert_eq!(actual.current_role(), expected.current_role());
            assert_eq!(
                actual.current_recipient(),
                expected.current_recipient().as_deref()
            );
            assert_eq!(
                actual.current_channel(),
                expected.current_channel().as_deref()
            );
            assert_eq!(actual.message_count(), expected.messages().len());
        }
        assert_eq!(
            actual.process_eos().map_err(|error| error.to_string()),
            expected
                .process_eos()
                .map(|_| ())
                .map_err(|error| error.to_string())
        );
    }

    #[test]
    fn released_header_and_delta_contract_matches_roles_and_metadata() {
        let encoding = encoding();
        for role in [
            "assistant",
            "user",
            "system",
            "developer",
            "tool",
            "python",
            "unknown",
        ] {
            for metadata in [
                "",
                " to=functions.lookup",
                "<|channel|>analysis",
                "<|channel|>final",
                "<|channel|>commentary to=functions.lookup",
                " to=python<|channel|>analysis",
                "<|constrain|>json",
                "<|channel|>",
            ] {
                for stop in ["<|end|>", "<|return|>", "<|call|>"] {
                    let text = format!("<|start|>{role}{metadata}<|message|>漢🙂x{stop}");
                    parity(
                        &encoding,
                        &encoding.tokenizer().encode_with_special_tokens(&text),
                    );
                }
            }
        }
    }

    #[test]
    fn decoder_and_stop_behavior_matches_each_byte_and_pending_eof() {
        let encoding = encoding();
        let header = encoding
            .tokenizer()
            .encode_with_special_tokens("<|start|>assistant<|message|>");
        let byte_token = |byte| {
            (0..256u32)
                .find(|id| encoding.tokenizer().decode_bytes([*id]).unwrap() == [byte])
                .unwrap()
        };
        for bytes in [
            vec![0xff, 0xfe, b'x'],
            vec![0xf0, 0x90],
            vec![0xe6, 0xbc, 0xa2, b'x'],
            vec![0xf0, 0x9f, 0x99, 0x82],
            vec![b'a', 0xff, 0xfe, 0xe6],
        ] {
            let mut ids = header.clone();
            ids.extend(bytes.iter().map(|byte| byte_token(*byte)));
            parity(&encoding, &ids);
            for stop in encoding.stop_tokens().unwrap() {
                let mut stopped = ids.clone();
                stopped.push(stop);
                parity(&encoding, &stopped);
            }
        }
        let mut unknown = header;
        unknown.extend([u32::MAX, byte_token(b'x')]);
        parity(&encoding, &unknown);
    }

    #[test]
    fn strict_headers_preserve_stops_and_eof_recovery_tokens() {
        let encoding = encoding();
        let mut ids = encoding
            .tokenizer()
            .encode_with_special_tokens("<|start|>assistant");
        for stop in encoding.stop_tokens().unwrap() {
            ids.push(stop);
        }
        let mut reader = DeltaReader::new(encoding.clone()).unwrap();
        for id in &ids {
            reader.process(*id).unwrap();
        }
        assert_eq!(reader.header_tokens(), Some(&ids[1..]));
        assert!(reader.process_eos().is_err());
        assert_eq!(reader.header_tokens(), Some(&ids[1..]));
        parity(&encoding, &ids);
    }

    #[test]
    fn fork_keeps_live_state_without_body_history_or_reemitting_delta() {
        let encoding = encoding();
        let text = format!(
            "<|start|>assistant<|channel|>analysis<|message|>{}",
            "漢🙂".repeat(1024)
        );
        let ids = encoding.tokenizer().encode_with_special_tokens(&text);
        let mut reader = DeltaReader::new(encoding.clone()).unwrap();
        for id in ids {
            reader.process(id).unwrap();
        }
        let mut fork = reader.fork_continuation();
        assert!(fork.last_content_delta().is_none());
        assert_eq!(fork.current_channel(), Some("analysis"));
        assert!(fork.header_tokens().is_none());
        for id in encoding
            .tokenizer()
            .encode_with_special_tokens("next<|end|>")
        {
            reader.process(id).unwrap();
            fork.process(id).unwrap();
            assert_eq!(reader.last_content_delta(), fork.last_content_delta());
            assert_eq!(reader.message_count(), fork.message_count());
        }
        let mut incomplete = DeltaReader::new(encoding.clone()).unwrap();
        for id in encoding
            .tokenizer()
            .encode_with_special_tokens("<|start|>assistant<|channel|>final")
        {
            incomplete.process(id).unwrap();
        }
        assert_eq!(
            incomplete.fork_continuation().header_tokens(),
            incomplete.header_tokens()
        );

        let byte_token = |byte| {
            (0..256u32)
                .find(|id| encoding.tokenizer().decode_bytes([*id]).unwrap() == [byte])
                .unwrap()
        };
        let mut saw_pending_bytes = false;
        let mut saw_pending_tokens = false;
        for payload in [
            "漢🙂"
                .as_bytes()
                .iter()
                .map(|byte| byte_token(*byte))
                .collect(),
            vec![u32::MAX, byte_token(b'x')],
        ] {
            let mut input = encoding.tokenizer().encode_with_special_tokens(
                "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>",
            );
            input.extend(payload);
            input.extend(
                encoding.tokenizer().encode_with_special_tokens(
                    "<|call|><|start|>assistant<|message|>done<|return|>",
                ),
            );
            for split in 0..=input.len() {
                let mut source = DeltaReader::new(encoding.clone()).unwrap();
                for id in &input[..split] {
                    source.process(*id).unwrap();
                }
                saw_pending_bytes |= !source.undecoded_bytes.is_empty();
                saw_pending_tokens |= !source.undecoded_tokens.is_empty();
                let mut fork = source.fork_continuation();
                assert!(fork.last_content_delta().is_none());
                for id in &input[split..] {
                    source.process(*id).unwrap();
                    fork.process(*id).unwrap();
                    assert_eq!(source.last_content_delta(), fork.last_content_delta());
                    assert_eq!(source.current_role(), fork.current_role());
                    assert_eq!(source.current_channel(), fork.current_channel());
                    assert_eq!(source.current_recipient(), fork.current_recipient());
                    assert_eq!(source.message_count(), fork.message_count());
                }
            }
        }
        assert!(saw_pending_bytes);
        assert!(saw_pending_tokens);
    }

    #[test]
    fn stop_discards_completion_only_bytes_and_reports_them_once() {
        let encoding = encoding();
        let mut reader = DeltaReader::new(encoding.clone()).unwrap();
        for id in encoding
            .tokenizer()
            .encode_with_special_tokens("<|start|>assistant<|message|>")
        {
            reader.process(id).unwrap();
        }
        let incomplete = (0..256u32)
            .find(|id| encoding.tokenizer().decode_bytes([*id]).unwrap() == [0xf0])
            .unwrap();
        reader.process(incomplete).unwrap();
        assert!(reader.last_content_delta().is_none());
        let stop = *encoding.stop_tokens().unwrap().iter().next().unwrap();
        reader.process(stop).unwrap();
        assert!(reader.last_content_delta().is_none());
        assert_eq!(reader.take_skipped_pending(), 1);
        assert_eq!(reader.take_skipped_pending(), 0);
        assert_eq!(reader.discard_pending(), 0);
    }
}
