// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer setups that the prefix cache cannot split safely.
//!
//! The cache splits a prompt right after each special token, and encodes the
//! parts one by one. That gives the IDs of a whole encode only when the text
//! after a special token encodes the same on its own as inside the prompt.
//! Overlapping special tokens break that, and the cache already checks for
//! them ([`super::l1::first_unsafe_overlap`]). The setups here break it too.

use serde_json::Value;

use super::l1::tokens_can_overlap;

/// How an encoder matches added tokens.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Matching {
    /// As `tokenizers` does: first the tokens that are not normalized, in the
    /// raw text, then the normalized ones, in what is left.
    TwoPasses,
    /// All added tokens in one leftmost-longest pass over the raw text,
    /// whatever their `normalized` flag, as fastokens and Baseten do.
    OnePass,
}

/// The match flags of one added token, as `tokenizer.json` lists them.
pub(crate) struct AddedTokenFlags<'a> {
    pub(crate) content: &'a str,
    pub(crate) special: bool,
    pub(crate) lstrip: bool,
    pub(crate) rstrip: bool,
    pub(crate) single_word: bool,
    pub(crate) normalized: bool,
}

/// Why the cache cannot split this tokenizer's prompts safely, if it cannot.
///
/// `matching` says whether the encoder matches added tokens in two passes
/// ([`Matching::TwoPasses`]) or one. The setups:
///
/// - a special token with `rstrip`: it takes the spaces after it, which the
///   part after the split keeps;
/// - a special token with `normalized`, when the tokenizer has a normalizer:
///   the normalizer can rewrite text across the split;
/// - a special token that overlaps a plain added token matched in the same
///   pass or an earlier one: the tokenizer can match the plain token in its
///   place;
/// - an added token with `single_word`: the character before the split
///   decides whether it matches;
/// - a first-pass token with `lstrip` or `rstrip`, and a normalized special
///   token that starts or ends with whitespace: the first pass can take that
///   whitespace, and the special token then does not match where the cache
///   splits;
/// - a Metaspace pre-tokenizer with `prepend_scheme: first`: it marks the
///   start of the part after the split, which inside the prompt is not a
///   start.
///
/// Otherwise `lstrip` is safe: the spaces before a special token are in the
/// same part as the token, and a match never takes text an earlier match in
/// its pass took.
pub(crate) fn prefix_cache_hazard<'a>(
    tokens: &[AddedTokenFlags<'a>],
    matching: Matching,
    has_normalizer: bool,
    pre_tokenizer: Option<&Value>,
) -> Option<String> {
    let second_pass =
        |token: &AddedTokenFlags<'_>| matching == Matching::TwoPasses && token.normalized;
    let strips_in_first_pass = tokens
        .iter()
        .find(|token| !second_pass(token) && (token.lstrip || token.rstrip));
    for token in tokens {
        if token.single_word {
            return Some(format!(
                "the added token `{}` matches only as a whole word",
                token.content
            ));
        }
        if !token.special {
            continue;
        }
        if token.rstrip {
            return Some(format!(
                "the special token `{}` strips the spaces after it",
                token.content
            ));
        }
        if token.normalized && has_normalizer {
            return Some(format!(
                "the special token `{}` matches in normalized text",
                token.content
            ));
        }
        // A plain token matched in a later pass than the special token cannot
        // take its place: the special token is split out first.
        let shadow = tokens.iter().find(|other| {
            !other.special
                && (second_pass(token) || !second_pass(other))
                && tokens_can_overlap(token.content, other.content)
        });
        if let Some(other) = shadow {
            return Some(format!(
                "the special token `{}` overlaps the added token `{}`",
                token.content, other.content
            ));
        }
        let edge_is_space = token.content.starts_with(char::is_whitespace)
            || token.content.ends_with(char::is_whitespace);
        if let Some(stripper) = strips_in_first_pass.filter(|_| second_pass(token) && edge_is_space)
        {
            return Some(format!(
                "the added token `{}` can strip the whitespace of the special token `{}`",
                stripper.content, token.content
            ));
        }
    }
    pre_tokenizer
        .is_some_and(marks_only_the_first_word)
        .then(|| "the Metaspace pre-tokenizer marks only the first word".to_owned())
}

fn marks_only_the_first_word(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            (object.get("type").and_then(Value::as_str) == Some("Metaspace")
                && object.get("prepend_scheme").and_then(Value::as_str) == Some("first"))
                || object.values().any(marks_only_the_first_word)
        }
        Value::Array(values) => values.iter().any(marks_only_the_first_word),
        _ => false,
    }
}

/// [`prefix_cache_hazard`] of a `tokenizer.json`, for a backend that reads it
/// as JSON and matches added tokens in one pass.
pub(crate) fn prefix_cache_hazard_in_json(json: &Value) -> Option<String> {
    let flag = |token: &Value, name| token.get(name).and_then(Value::as_bool).unwrap_or(false);
    let tokens: Vec<_> = json
        .get("added_tokens")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|token| {
            Some(AddedTokenFlags {
                content: token.get("content")?.as_str()?,
                special: flag(token, "special"),
                lstrip: flag(token, "lstrip"),
                rstrip: flag(token, "rstrip"),
                single_word: flag(token, "single_word"),
                normalized: flag(token, "normalized"),
            })
        })
        .collect();
    let has_normalizer = json
        .get("normalizer")
        .is_some_and(|normalizer| !normalizer.is_null());
    prefix_cache_hazard(
        &tokens,
        Matching::OnePass,
        has_normalizer,
        json.get("pre_tokenizer"),
    )
}
