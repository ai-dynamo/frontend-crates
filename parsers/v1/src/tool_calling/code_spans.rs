// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Search only prose, stopping before a tool body (whose backticks are data).
pub(crate) enum MarkerSearch {
    Found(usize, usize),
    Pending(usize),
    None,
}

fn line_start(text: &str, pos: usize) -> usize {
    text[..pos].rfind('\n').map_or(0, |p| p + 1)
}

fn fence_start(text: &str, pos: usize, initial_indent: usize) -> bool {
    let start = line_start(text, pos);
    let prefix = &text[start..pos];
    let prior = if start == 0 { initial_indent } else { 0 };
    prior + prefix.len() <= 3 && prefix.bytes().all(|b| b == b' ')
}

pub(crate) fn find_marker(text: &str, patterns: &[&str], final_input: bool) -> MarkerSearch {
    find_marker_with_indent(text, patterns, final_input, 0)
}

pub(crate) fn advance_line_indent(mut indent: usize, text: &str) -> usize {
    for byte in text.bytes() {
        indent = match byte {
            b'\n' => 0,
            b' ' => (indent + 1).min(4),
            _ => 4,
        };
    }
    indent
}

pub(crate) fn find_marker_with_indent(
    text: &str,
    patterns: &[&str],
    final_input: bool,
    initial_indent: usize,
) -> MarkerSearch {
    let bytes = text.as_bytes();
    let mut cursor = 0;
    while cursor < text.len() {
        if let Some(index) = patterns
            .iter()
            .position(|p| !p.is_empty() && text[cursor..].starts_with(p))
        {
            return MarkerSearch::Found(cursor, index);
        }
        let delimiter = bytes[cursor];
        if delimiter != b'`' && delimiter != b'~' {
            cursor += text[cursor..].chars().next().unwrap().len_utf8();
            continue;
        }
        let run = bytes[cursor..]
            .iter()
            .take_while(|&&b| b == delimiter)
            .count();
        if bytes[..cursor]
            .iter()
            .rev()
            .take_while(|&&b| b == b'\\')
            .count()
            % 2
            == 1
        {
            cursor += run;
            continue;
        }
        let fenced = run >= 3 && fence_start(text, cursor, initial_indent);
        if cursor + run == text.len()
            && !final_input
            && (delimiter == b'`' || fence_start(text, cursor, initial_indent))
        {
            return MarkerSearch::Pending(line_start(text, cursor));
        }
        if delimiter == b'~' && !fenced {
            cursor += run;
            continue;
        }
        let mut search = cursor + run;
        let mut end = None;
        while search < text.len() {
            let Some(relative) = bytes[search..].iter().position(|&b| b == delimiter) else {
                break;
            };
            search += relative;
            let length = bytes[search..]
                .iter()
                .take_while(|&&b| b == delimiter)
                .count();
            if search + length == text.len() && !final_input {
                return MarkerSearch::Pending(line_start(text, cursor));
            }
            if fenced {
                if length >= run && fence_start(text, search, initial_indent) {
                    let tail = &text[search + length..];
                    let line = tail.split('\n').next().unwrap();
                    if line.trim_matches([' ', '\t', '\r']).is_empty() {
                        if !tail.contains('\n') && !final_input {
                            return MarkerSearch::Pending(line_start(text, cursor));
                        }
                        end = Some(search + length);
                        break;
                    }
                }
            } else if length == run {
                end = Some(search + length);
                break;
            }
            search += length;
        }
        if let Some(end) = end {
            cursor = end;
        } else if !final_input {
            return MarkerSearch::Pending(line_start(text, cursor));
        } else if fenced {
            return MarkerSearch::None;
        } else {
            // Unlike a fence, an unmatched inline opener is ordinary text.
            cursor += run;
        }
    }
    MarkerSearch::None
}

pub(crate) fn find_unquoted(text: &str, marker: &str) -> Option<usize> {
    match find_marker(text, &[marker], true) {
        MarkerSearch::Found(pos, _) => Some(pos),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaped_delimiters_do_not_hide_calls() {
        assert_eq!(find_unquoted(r"\`<tool_call>\`", "<tool_call>"), Some(2));
        assert_eq!(find_unquoted("`<tool_call>`", "<tool_call>"), None);
        assert_eq!(find_unquoted("``a ` <tool_call> ``", "<tool_call>"), None);
    }
}
