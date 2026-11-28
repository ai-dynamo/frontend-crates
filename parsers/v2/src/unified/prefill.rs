// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::UnifiedParserStartingState;

/// Rendered generation-prompt grammar, independent of selector spelling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptStyle {
    #[default]
    NoPrefill,
    Think,
    Marker,
    NamedChannels,
    PartialThought,
    RecipientChannels,
    Harmony,
    WholeTurnChannels,
}

/// How a rendered closer overrides a request's thinking toggle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptClosure {
    #[default]
    Unspecified,
    ClosedWhenPresent,
    OpenUnlessClosed,
}

/// Parser initialization derived from the actual rendered prompt.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ParserPromptState {
    pub reasoning: bool,
    /// A complete opener, suitable for split parsers that cannot replay a partial prefix.
    pub explicit_reasoning: bool,
    pub starting_state: Option<UnifiedParserStartingState>,
    pub prefix: Option<String>,
    pub reasoning_closed: Option<bool>,
    pub channel: Option<UnifiedParserStartingState>,
}

impl super::UnifiedParserDescriptor {
    /// Control tokens used to keep native syntax distinct from quoted JSON payloads.
    pub fn control_markers(&self) -> Vec<String> {
        let mut markers: Vec<String> = self
            .reasoning_markers
            .into_iter()
            .flat_map(|(start, end)| [start.to_owned(), end.to_owned()])
            .collect();
        match self.prompt_style {
            PromptStyle::NamedChannels => {
                for channel in ["think", "response", "tools", "call"] {
                    markers.push(format!("<|open|>{channel}"));
                    markers.push(format!("<|close|>{channel}"));
                }
                markers.push("<|sep|>".into());
            }
            PromptStyle::WholeTurnChannels => markers.extend(
                [
                    "<|start|>",
                    "<|channel|>",
                    "<|message|>",
                    "<|constrain|>",
                    "<|end|>",
                    "<|return|>",
                    "<|call|>",
                ]
                .map(str::to_owned),
            ),
            _ if self.batch_markers == Some(("<|start|>", "<|eom|>")) => markers
                .extend(["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"].map(str::to_owned)),
            _ => {}
        }
        markers.extend(self.native_call_markers().map(str::to_owned));
        markers
    }

    /// Whether complete output contains native reasoning-channel evidence outside quotes.
    pub fn has_reasoning_boundary(&self, content: &str) -> bool {
        if let Some((start, end)) = self.reasoning_markers {
            return [start, end]
                .iter()
                .any(|marker| first_unquoted_marker_position(content, marker).is_some());
        }
        match self.batch_prefill {
            super::BatchPrefill::NamedChannels => ["<|open|>", "<|close|>"]
                .iter()
                .any(|marker| kimi_channel_marker_position(content, marker, "think").is_some()),
            super::BatchPrefill::HarmonyChannels => BEAM_OPENERS
                .iter()
                .chain(BEAM_CLOSERS.iter())
                .any(|marker| content.contains(marker)),
            _ => self.batch_markers.is_some_and(|(start, end)| {
                [start, end]
                    .iter()
                    .any(|marker| first_unquoted_marker_position(content, marker).is_some())
            }),
        }
    }

    /// A native tool envelope is stronger evidence than a reconstructed guided-JSON request.
    pub fn has_native_call(&self, content: &str) -> bool {
        if self.batch_prefill == super::BatchPrefill::NamedChannels {
            ["call", "tools"]
                .iter()
                .any(|channel| kimi_channel_marker_position(content, "<|open|>", channel).is_some())
        } else {
            self.native_call_markers()
                .any(|marker| first_unquoted_marker_position(content, marker).is_some())
        }
    }

    /// Inspect the rendered prompt without constructing mutable parser state.
    pub fn resolve_prompt(
        &self,
        prompt: Option<&str>,
        header_search_end: Option<usize>,
    ) -> ParserPromptState {
        let Some(raw) = prompt else {
            return ParserPromptState::default();
        };
        let trimmed = raw.trim_end();
        let assistant = (self.prompt_style == PromptStyle::Harmony)
            .then(|| gpt_oss_assistant_prefill(raw, header_search_end))
            .flatten();
        let mut state = ParserPromptState::default();
        state.reasoning = match self.prompt_style {
            PromptStyle::NoPrefill => false,
            PromptStyle::Think => trimmed.ends_with("<think>"),
            PromptStyle::Marker => self
                .reasoning_markers
                .is_some_and(|(start, _)| trimmed.ends_with(start)),
            PromptStyle::NamedChannels => {
                let mut suffix = trimmed.chars().rev().filter(|c| !c.is_whitespace());
                "<|open|>think<|sep|>"
                    .chars()
                    .rev()
                    .all(|c| suffix.next() == Some(c))
            }
            PromptStyle::PartialThought => {
                trimmed.ends_with("<|channel>thought") || trimmed.ends_with("<|channel>")
            }
            PromptStyle::RecipientChannels => trimmed.ends_with("assistant to=self<|message|>"),
            PromptStyle::Harmony => assistant.is_some_and(gpt_oss_prefill_has_reasoning),
            PromptStyle::WholeTurnChannels => {
                beam_prompt_open_channel(trimmed) == Some(UnifiedParserStartingState::Reasoning)
            }
        };
        state.explicit_reasoning = state.reasoning;
        state.prefix = match self.prompt_style {
            PromptStyle::PartialThought => {
                const HEADER: &str = "<|channel>thought\n";
                ("<|channel>".len()..HEADER.len())
                    .rev()
                    .map(|end| &HEADER[..end])
                    .find(|prefix| raw.ends_with(prefix))
                    .map(str::to_owned)
            }
            PromptStyle::Harmony => assistant.map(compact_gpt_oss_prefill).map(str::to_owned),
            _ => None,
        };
        state.reasoning |=
            self.prompt_style == PromptStyle::PartialThought && state.prefix.is_some();
        state.starting_state = (self.prompt_style == PromptStyle::Harmony
            && state.prefix.is_none()
            && raw.contains("<|start|>"))
        .then_some(UnifiedParserStartingState::Response);
        state.channel = (self.prompt_style == PromptStyle::WholeTurnChannels)
            .then(|| beam_prompt_open_channel(trimmed))
            .flatten();
        state.reasoning_closed = match self.prompt_closure {
            PromptClosure::Unspecified => None,
            PromptClosure::ClosedWhenPresent if state.reasoning => Some(false),
            PromptClosure::ClosedWhenPresent => self
                .reasoning_markers
                .and_then(|(_, end)| trimmed.ends_with(end).then_some(true)),
            PromptClosure::OpenUnlessClosed => self
                .reasoning_markers
                .map(|(_, end)| trimmed.ends_with(end)),
        };
        state
    }
}

pub const BEAM_OPENERS: [&str; 2] = ["<|channel|>", "<|start|>"];
pub const BEAM_CLOSERS: [&str; 3] = ["<|end|>", "<|return|>", "<|call|>"];

/// Find a control marker outside complete prose quotes.
pub fn first_unquoted_marker_position(content: &str, marker: &str) -> Option<usize> {
    first_unquoted_marker_position_after(content, marker, 0)
}

/// Infer whether an XTML prompt opened the named reasoning channel.
pub fn named_channel_prefill(content: &str) -> UnifiedParserStartingState {
    marker_prefill_from_positions(
        kimi_channel_marker_position(content, "<|open|>", "think"),
        kimi_channel_marker_position(content, "<|close|>", "think"),
    )
}

pub fn first_unquoted_marker_position_after(
    content: &str,
    marker: &str,
    minimum: usize,
) -> Option<usize> {
    first_unquoted_position(content, |offset| {
        offset >= minimum && content[offset..].starts_with(marker)
    })
}

fn first_unquoted_position(
    content: &str,
    mut matches_at: impl FnMut(usize) -> bool,
) -> Option<usize> {
    let chars: Vec<(usize, char)> = content.char_indices().collect();
    // Whether an unescaped `"`, `'`, or `` ` `` appears anywhere at or after each
    // position, precomputed once in O(n) so the scan below doesn't rescan the
    // remainder of the string for every candidate quote character (which would be
    // O(n^2) on adversarial text with many non-closing quote-like characters).
    let closes_later = later_unescaped_quote_closes(&chars);

    let mut quote = None;
    let mut escaped = false;
    for (position, &(offset, character)) in chars.iter().enumerate() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                let previous = content[..offset].chars().next_back();
                let next = content[offset + character.len_utf8()..].chars().next();
                if active_quote == '\'' && is_internal_apostrophe(previous, next) {
                    continue;
                }
                quote = None;
            }
            continue;
        }
        let previous = content[..offset].chars().next_back();
        let next = content[offset + character.len_utf8()..].chars().next();
        let apostrophe = character == '\''
            && (previous.is_some_and(char::is_alphanumeric)
                || next.is_some_and(char::is_alphanumeric) && !closes_later[position + 1][1]);
        // A quote that never closes before end-of-string was never really quoting
        // anything; treating it as an open span would blind the rest of the scan to
        // a real marker that follows (e.g. a stray `"` before a genuine `</think>`).
        if let Some(slot) = quote_slot(character)
            && !apostrophe
            && closes_later[position + 1][slot]
        {
            quote = Some(character);
            continue;
        }
        if matches_at(offset) {
            return Some(offset);
        }
    }
    None
}

fn quote_slot(character: char) -> Option<usize> {
    match character {
        '"' => Some(0),
        '\'' => Some(1),
        '`' => Some(2),
        _ => None,
    }
}

pub fn is_internal_apostrophe(previous: Option<char>, next: Option<char>) -> bool {
    previous.is_some_and(char::is_alphanumeric) && next.is_some_and(char::is_alphanumeric)
}

/// For each position in `chars`, whether an unescaped `"`, `'`, or `` ` `` appears
/// anywhere later in the string — `result[i][quote_slot(c)]` answers "does an
/// unescaped `c` exist in `chars[i..]`". Built in one right-to-left O(n) pass: the
/// answer at `i` depends only on `chars[i]` and the answer at `i + 1` (or `i + 2` for
/// a backslash, which escapes exactly the next character), so this is an exact
/// backward re-expression of the left-to-right escaped-flag scan that used to be
/// re-run from scratch for every candidate quote character.
fn later_unescaped_quote_closes(chars: &[(usize, char)]) -> Vec<[bool; 3]> {
    let n = chars.len();
    let mut closes = vec![[false; 3]; n + 1];
    let mut i = n;
    while i > 0 {
        i -= 1;
        let character = chars[i].1;
        closes[i] = if character == '\\' {
            closes.get(i + 2).copied().unwrap_or([false; 3])
        } else {
            let mut next = closes[i + 1];
            let internal_apostrophe = character == '\''
                && is_internal_apostrophe(
                    i.checked_sub(1)
                        .and_then(|previous| chars.get(previous).map(|(_, c)| *c)),
                    chars.get(i + 1).map(|(_, c)| *c),
                );
            if let Some(slot) = quote_slot(character)
                && !internal_apostrophe
            {
                next[slot] = true;
            }
            next
        };
    }
    closes
}

pub fn kimi_channel_marker_position(content: &str, prefix: &str, channel: &str) -> Option<usize> {
    const SEPARATOR: &str = "<|sep|>";
    // Candidate offsets advance, so each separator is visited at most once.
    let mut separators = content.match_indices(SEPARATOR).peekable();
    first_unquoted_position(content, |offset| {
        let Some(header) = content[offset..].strip_prefix(prefix) else {
            return false;
        };
        let after = offset + prefix.len();
        let end = if prefix == "<|close|>" {
            header.find(['<', '[', '{']).unwrap_or(header.len())
        } else {
            while separators.peek().is_some_and(|(at, _)| *at < after) {
                separators.next();
            }
            let Some((separator, _)) = separators.peek() else {
                return false;
            };
            separator - after
        };
        header[..end]
            .trim_start()
            .strip_prefix(channel)
            .is_some_and(|tail| tail.chars().next().is_none_or(char::is_whitespace))
    })
}

pub fn infer_marker_prefill(
    content: &str,
    opener: &str,
    closer: &str,
) -> UnifiedParserStartingState {
    marker_prefill_from_positions(
        first_unquoted_marker_position(content, opener),
        first_unquoted_marker_position(content, closer),
    )
}

pub(super) fn marker_prefill_from_positions(
    open: Option<usize>,
    close: Option<usize>,
) -> UnifiedParserStartingState {
    match (open, close) {
        (None, Some(_)) => UnifiedParserStartingState::Reasoning,
        (Some(open), Some(close)) if close < open => UnifiedParserStartingState::Reasoning,
        _ => UnifiedParserStartingState::None,
    }
}

pub fn beam_prefill(content: &str) -> UnifiedParserStartingState {
    let opener = BEAM_OPENERS
        .iter()
        .filter_map(|marker| content.find(marker))
        .min();
    let closer = BEAM_CLOSERS
        .iter()
        .filter_map(|marker| content.find(marker).map(|at| (at, *marker)))
        .min_by_key(|(at, _)| *at);
    match (opener, closer) {
        (Some(open), Some((close, _))) if open < close => UnifiedParserStartingState::None,
        (Some(_), None) => UnifiedParserStartingState::None,
        (_, Some((_, "<|end|>"))) => UnifiedParserStartingState::Reasoning,
        (_, Some(_)) | (None, None) => UnifiedParserStartingState::Response,
    }
}

// Harmony suppresses generated body text if initialization restores a non-assistant role.
pub fn gpt_oss_assistant_prefill(prompt: &str, header_search_end: Option<usize>) -> Option<&str> {
    const START: &str = "<|start|>";
    let start = prompt[..header_search_end.unwrap_or(prompt.len())].rfind(START)?;
    let prefix = &prompt[start..];
    let role = prefix[START.len()..]
        .split(|c: char| c.is_whitespace() || c == '<')
        .next()?;
    (role == "assistant").then_some(prefix)
}

pub fn gpt_oss_prefill_has_reasoning(message: &str) -> bool {
    message
        .split_once("<|message|>")
        .is_some_and(|(header, _)| {
            header
                .split_once("<|channel|>")
                .is_some_and(|(_, channel)| channel.split_whitespace().next() == Some("analysis"))
        })
}

/// Plain response/reasoning text does not affect parser continuation state.
pub fn compact_gpt_oss_prefill(prefix: &str) -> &str {
    const MESSAGE: &str = "<|message|>";
    let Some((header, body)) = prefix.split_once(MESSAGE) else {
        return prefix;
    };
    if header.contains("to=") || body.contains("<|") {
        // Recipient-owned bodies may contain a partially consumed JSON argument.
        // TODO: Validate argument prefixes incrementally so released bytes no longer
        // need to remain in the shared per-request continuation body.
        return prefix;
    }
    &prefix[..header.len() + MESSAGE.len()]
}

/// Which Beam channel the rendered prompt leaves open after its last `<|start|>assistant`
/// header: `None` for the bare header slot (the model opens its own channel), `Reasoning`
/// for an open analysis message, `Response` for an open final message (assistant prefill).
pub fn beam_prompt_open_channel(prompt: &str) -> Option<super::UnifiedParserStartingState> {
    use super::UnifiedParserStartingState as State;
    let tail = &prompt[prompt.rfind("<|start|>assistant")? + "<|start|>assistant".len()..];
    let open = |opener: &str, closers: &[&str]| {
        tail.rfind(opener).is_some_and(|at| {
            let rest = &tail[at + opener.len()..];
            !closers.iter().any(|closer| rest.contains(closer))
        })
    };
    if open("<|channel|>final<|message|>", &["<|return|>", "<|end|>"]) {
        Some(State::Response)
    } else if open("<|channel|>analysis<|message|>", &["<|end|>"]) {
        Some(State::Reasoning)
    } else {
        Some(State::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kimi_channel_markers_keep_quote_and_header_boundaries() {
        let quoted = r#""<|open|>call<|sep|>""#;
        let wrong = "<|open|>callback<|sep|>";
        let call = r#"<|open|> call tool="lookup"<|sep|>"#;
        let prefix = format!("{quoted} {wrong} é ");
        let suffix = wrong.repeat(1_000);
        let input = format!("{prefix}{call}{}", call.repeat(1_000));
        assert_eq!(
            kimi_channel_marker_position(&input, "<|open|>", "call"),
            Some(prefix.len())
        );
        assert_eq!(
            kimi_channel_marker_position(&format!("{quoted}{suffix}"), "<|open|>", "call"),
            None
        );
        assert_eq!(
            named_channel_prefill(r#""<|open|>think<|sep|>" plan<|close|> think <|sep|>"#),
            UnifiedParserStartingState::Reasoning,
        );
        assert_eq!(
            named_channel_prefill("<|open|> think <|sep|>plan<|close|>think<|sep|>"),
            UnifiedParserStartingState::None,
        );
    }
}
