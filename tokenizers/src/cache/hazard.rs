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

/// The match flags of one added token, as `tokenizer.json` lists them.
pub(crate) struct AddedTokenFlags<'a> {
    pub(crate) content: &'a str,
    pub(crate) special: bool,
    pub(crate) rstrip: bool,
    pub(crate) single_word: bool,
    pub(crate) normalized: bool,
}

/// Why the cache cannot split this tokenizer's prompts safely, if it cannot:
///
/// - a special token with `rstrip`: it takes the spaces after it, which the
///   part after the split keeps;
/// - a special token with `normalized`, when the tokenizer has a normalizer:
///   the normalizer can rewrite text across the split;
/// - a special token that overlaps an added token that is not special and
///   not normalized: the tokenizer can match the other token in its place;
/// - an added token with `single_word`: the character before the split
///   decides whether it matches;
/// - a Metaspace pre-tokenizer with `prepend_scheme: first`: it marks the
///   start of the part after the split, which inside the prompt is not a
///   start.
///
/// `lstrip` is safe: the spaces before a special token are in the same
/// part as the token.
pub(crate) fn prefix_cache_hazard<'a>(
    tokens: &[AddedTokenFlags<'a>],
    has_normalizer: bool,
    pre_tokenizer: Option<&Value>,
) -> Option<String> {
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
        let shadow = tokens.iter().find(|other| {
            !other.special && !other.normalized && tokens_can_overlap(token.content, other.content)
        });
        if let Some(other) = shadow {
            return Some(format!(
                "the special token `{}` overlaps the added token `{}`",
                token.content, other.content
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

/// [`prefix_cache_hazard`] of a `tokenizer.json`, for backends that read it
/// as JSON.
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
                rstrip: flag(token, "rstrip"),
                single_word: flag(token, "single_word"),
                normalized: flag(token, "normalized"),
            })
        })
        .collect();
    let has_normalizer = json
        .get("normalizer")
        .is_some_and(|normalizer| !normalizer.is_null());
    prefix_cache_hazard(&tokens, has_normalizer, json.get("pre_tokenizer"))
}
