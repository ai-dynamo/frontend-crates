// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical local `$ref` targets for grammar backends.
//!
//! JSON Schema spells a `$ref` as a URI fragment: percent-encoded, then
//! JSON-Pointer-escaped (`~1` is `/`, `~0` is `~`). XGrammar's schema converter
//! (0.2.8, `json_schema_converter.cc`) resolves a local reference by literal
//! segment lookup instead: `#/$defs/registry~1Model~0v1%20name` looks for a key
//! spelled exactly `registry~1Model~0v1%20name`, finds nothing, and the whole
//! structural tag fails to compile — the request is rejected. A literal space or
//! `~` in a key resolves; a `/` cannot be expressed at all.
//!
//! [`canonicalize_local_refs`] rewrites the schema into the spelling the
//! converter resolves: every local `$ref` is decoded to literal segments, and a
//! `$defs` / `definitions` entry whose name is not a plain identifier is renamed
//! to one (references follow). Definition names never reach the generated
//! arguments, so the rename is invisible to the model and the caller. A schema
//! that is already plain comes back unchanged.

use std::collections::{BTreeMap, HashSet};

use serde_json::{Map, Value};

/// Keys that hold reusable definitions at the schema root.
const DEFINITION_KEYS: [&str; 2] = ["$defs", "definitions"];

/// Nesting beyond this is left as written (a grammar backend bounds it too).
const MAX_DEPTH: usize = 64;

/// Total nodes visited before the walk stops rewriting.
const MAX_NODES: usize = 1 << 16;

/// Rewrite local `$ref` targets into the literal spelling grammar backends
/// resolve, renaming definition keys that cannot be referenced literally.
///
/// Non-local references (`http://…`, `other.json#/…`), references with an
/// invalid percent escape or a stray `~`, and references whose decoded segment
/// still contains a `/` outside a renamed definition are left exactly as written.
pub fn canonicalize_local_refs(mut schema: Value) -> Value {
    let Some(root) = schema.as_object() else {
        return schema;
    };
    let renames = plan_definition_renames(root);
    let mut budget = MAX_NODES;
    rewrite_refs(&mut schema, &renames, 0, &mut budget);
    if let Some(root) = schema.as_object_mut() {
        apply_definition_renames(root, &renames);
    }
    schema
}

/// A definition name a grammar backend resolves verbatim and a URI fragment
/// carries without escaping.
fn is_plain_identifier(name: &str) -> bool {
    !name.is_empty() && name.chars().all(is_plain_char)
}

fn is_plain_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
}

/// `(definition container, original name) -> plain replacement`.
type Renames = BTreeMap<(&'static str, String), String>;

fn plan_definition_renames(root: &Map<String, Value>) -> Renames {
    let mut renames = Renames::new();
    for container in DEFINITION_KEYS {
        let Some(definitions) = root.get(container).and_then(Value::as_object) else {
            continue;
        };
        let mut taken: HashSet<String> = definitions.keys().cloned().collect();
        for name in definitions.keys().filter(|name| !is_plain_identifier(name)) {
            let base: String = name
                .chars()
                .map(|c| if is_plain_char(c) { c } else { '_' })
                .collect();
            let mut candidate = base.clone();
            let mut suffix = 2;
            while taken.contains(&candidate) {
                candidate = format!("{base}_{suffix}");
                suffix += 1;
            }
            taken.insert(candidate.clone());
            renames.insert((container, name.clone()), candidate);
        }
    }
    renames
}

fn apply_definition_renames(root: &mut Map<String, Value>, renames: &Renames) {
    for container in DEFINITION_KEYS {
        if !renames.keys().any(|(key, _)| *key == container) {
            continue;
        }
        let Some(Value::Object(definitions)) = root.get_mut(container) else {
            continue;
        };
        let entries = std::mem::take(definitions);
        for (name, definition) in entries {
            let name = renames
                .get(&(container, name.clone()))
                .cloned()
                .unwrap_or(name);
            definitions.insert(name, definition);
        }
    }
}

fn rewrite_refs(node: &mut Value, renames: &Renames, depth: usize, budget: &mut usize) {
    if depth > MAX_DEPTH || *budget == 0 {
        return;
    }
    *budget -= 1;
    match node {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get_mut("$ref")
                && let Some(canonical) = canonical_local_ref(reference, renames)
            {
                *reference = canonical;
            }
            for child in map.values_mut() {
                rewrite_refs(child, renames, depth + 1, budget);
            }
        }
        Value::Array(items) => {
            for item in items {
                rewrite_refs(item, renames, depth + 1, budget);
            }
        }
        _ => {}
    }
}

/// The literal-segment spelling of a local reference, or `None` when the
/// reference is not local, cannot be decoded, or would not change.
fn canonical_local_ref(reference: &str, renames: &Renames) -> Option<String> {
    let fragment = reference.strip_prefix('#')?;
    // URI percent-decoding precedes JSON Pointer's `~0`/`~1` decoding.
    let pointer = percent_decode(fragment)?;
    let path = pointer.strip_prefix('/')?;
    let mut segments = path
        .split('/')
        .map(unescape_pointer_segment)
        .collect::<Option<Vec<_>>>()?;
    if let [container, name, ..] = segments.as_mut_slice()
        && let Some(container) = DEFINITION_KEYS.iter().find(|key| *key == container)
        && let Some(renamed) = renames.get(&(container, name.clone()))
    {
        *name = renamed.clone();
    }
    // A decoded `/` outside a renamed definition cannot be spelled literally;
    // keep the author's text rather than emit a pointer that means something else.
    if segments.iter().any(|segment| segment.contains('/')) {
        return None;
    }
    let canonical = format!("#/{}", segments.join("/"));
    (canonical != reference).then_some(canonical)
}

/// Decode `%XX` escapes; `None` on a malformed escape or invalid UTF-8.
fn percent_decode(text: &str) -> Option<String> {
    if !text.contains('%') {
        return Some(text.to_string());
    }
    let mut bytes = text.bytes();
    let mut decoded = Vec::with_capacity(text.len());
    while let Some(byte) = bytes.next() {
        decoded.push(if byte == b'%' {
            let high = char::from(bytes.next()?).to_digit(16)?;
            let low = char::from(bytes.next()?).to_digit(16)?;
            ((high << 4) | low) as u8
        } else {
            byte
        });
    }
    String::from_utf8(decoded).ok()
}

/// RFC 6901: `~1` is `/`, `~0` is `~`; any other `~` is malformed.
fn unescape_pointer_segment(segment: &str) -> Option<String> {
    let mut out = String::with_capacity(segment.len());
    let mut chars = segment.chars();
    while let Some(c) = chars.next() {
        if c != '~' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('0') => out.push('~'),
            Some('1') => out.push('/'),
            _ => return None,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn issue_schema() -> Value {
        json!({
            "type": "object",
            "properties": {"model_name": {"$ref": "#/$defs/registry~1Model~0v1%20name"}},
            "$defs": {"registry/Model~v1 name": {"type": "string"}},
            "required": ["model_name"]
        })
    }

    #[test]
    fn escaped_definition_is_renamed_and_its_reference_follows() {
        // ai-dynamo/dynamo#17661: the spelling XGrammar 0.2.8 compiles.
        assert_eq!(
            canonicalize_local_refs(issue_schema()),
            json!({
                "type": "object",
                "properties": {"model_name": {"$ref": "#/$defs/registry_Model_v1_name"}},
                "$defs": {"registry_Model_v1_name": {"type": "string"}},
                "required": ["model_name"]
            })
        );
    }

    #[test]
    fn plain_schema_is_unchanged() {
        let schema = json!({
            "type": "object",
            "properties": {"city": {"$ref": "#/$defs/City"}, "n": {"type": "integer"}},
            "$defs": {"City": {"type": "string"}, "my-def.v1": {"type": "boolean"}},
            "required": ["city"],
            "additionalProperties": false
        });
        assert_eq!(canonicalize_local_refs(schema.clone()), schema);
        for literal in [json!(true), json!(null), json!([1, 2]), json!("x")] {
            assert_eq!(canonicalize_local_refs(literal.clone()), literal);
        }
    }

    #[test]
    fn definitions_container_is_handled_like_defs() {
        let schema = json!({
            "type": "object",
            "properties": {"c": {"$ref": "#/definitions/Caf%C3%A9%20Menu"}},
            "definitions": {"Café Menu": {"type": "string"}}
        });
        assert_eq!(
            canonicalize_local_refs(schema),
            json!({
                "type": "object",
                "properties": {"c": {"$ref": "#/definitions/Caf__Menu"}},
                "definitions": {"Caf__Menu": {"type": "string"}}
            })
        );
    }

    #[test]
    fn percent_encoded_container_and_escapes_decode_to_the_renamed_key() {
        // `%24defs` / `%7E1` spell `$defs` / `~1`; the key itself is renamed because
        // it holds a `/`.
        let schema = json!({
            "$defs": {"A/B~C": {"type": "integer"}},
            "properties": {"v": {"$ref": "#/%24defs/A%7E1B%7E0C"}}
        });
        assert_eq!(
            canonicalize_local_refs(schema),
            json!({
                "$defs": {"A_B_C": {"type": "integer"}},
                "properties": {"v": {"$ref": "#/$defs/A_B_C"}}
            })
        );
    }

    #[test]
    fn invalid_escapes_and_non_local_references_are_left_alone() {
        let schema = json!({
            "$defs": {"Bad%ZZ": {"type": "string"}, "Ok": {"type": "string"}},
            "properties": {
                "a": {"$ref": "#/$defs/Bad%ZZ"},
                "b": {"$ref": "#/$defs/Ok~"},
                "c": {"$ref": "https://example.com/schema.json#/$defs/Remote%20X"},
                "d": {"$ref": "other.json#/x"},
                "e": {"$ref": "#"},
                "f": {"$ref": 7}
            }
        });
        let canonical = canonicalize_local_refs(schema.clone());
        assert_eq!(canonical["properties"], schema["properties"]);
        // The unreferencable key is still renamed; nothing points at it any more.
        assert_eq!(
            canonical["$defs"],
            json!({"Bad_ZZ": {"type": "string"}, "Ok": {"type": "string"}})
        );
    }

    #[test]
    fn references_nested_inside_definitions_and_arrays_are_rewritten() {
        let schema = json!({
            "type": "object",
            "properties": {"c": {"$ref": "#/$defs/a~1b"}},
            "$defs": {
                "a/b": {
                    "type": "object",
                    "properties": {"d": {"anyOf": [{"$ref": "#/$defs/x%20y"}, {"type": "null"}]}}
                },
                "x y": {"type": "integer"}
            }
        });
        assert_eq!(
            canonicalize_local_refs(schema),
            json!({
                "type": "object",
                "properties": {"c": {"$ref": "#/$defs/a_b"}},
                "$defs": {
                    "a_b": {
                        "type": "object",
                        "properties": {"d": {"anyOf": [{"$ref": "#/$defs/x_y"}, {"type": "null"}]}}
                    },
                    "x_y": {"type": "integer"}
                }
            })
        );
    }

    #[test]
    fn renamed_keys_do_not_collide_with_existing_or_each_other() {
        let schema = json!({
            "$defs": {"a b": {"type": "string"}, "a_b": {"type": "integer"}, "a/b": {"type": "boolean"}},
            "properties": {
                "x": {"$ref": "#/$defs/a%20b"},
                "y": {"$ref": "#/$defs/a_b"},
                "z": {"$ref": "#/$defs/a~1b"}
            }
        });
        let canonical = canonicalize_local_refs(schema);
        let defs = canonical["$defs"].as_object().unwrap();
        assert_eq!(defs.len(), 3);
        assert_eq!(defs["a_b"], json!({"type": "integer"}));
        let target = |name: &str| {
            canonical["properties"][name]["$ref"]
                .as_str()
                .unwrap()
                .to_string()
        };
        let (x, z) = (target("x"), target("z"));
        assert_eq!(target("y"), "#/$defs/a_b");
        assert_ne!(x, z);
        assert_eq!(defs[&x["#/$defs/".len()..]], json!({"type": "string"}));
        assert_eq!(defs[&z["#/$defs/".len()..]], json!({"type": "boolean"}));
    }

    #[test]
    fn a_decoded_slash_outside_a_definition_keeps_the_authored_text() {
        let schema = json!({
            "properties": {"a/b": {"type": "integer"}, "c": {"$ref": "#/properties/a~1b"}}
        });
        assert_eq!(canonicalize_local_refs(schema.clone()), schema);
    }

    #[test]
    fn deep_nesting_terminates() {
        let mut schema = json!({"$ref": "#/$defs/x%20y"});
        for _ in 0..(MAX_DEPTH + 40) {
            schema = json!({"items": schema});
        }
        let root = json!({"$defs": {"x y": {"type": "integer"}}, "properties": {"deep": schema}});
        let canonical = canonicalize_local_refs(root);
        assert_eq!(canonical["$defs"], json!({"x_y": {"type": "integer"}}));
    }
}
