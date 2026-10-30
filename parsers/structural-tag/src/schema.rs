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
//! `$defs` / `definitions` entry (at the root or in any subschema) whose name is
//! not a plain identifier is renamed to one, with its references following.
//! Definition names never reach the generated arguments, so the rename is
//! invisible to the model and the caller.
//!
//! The transform is all-or-nothing: the schema is analysed in full before
//! anything is rewritten, a definition is renamed only when every reference
//! that resolves to it today can be rewritten, and a schema too large or too
//! deep to analyse completely is returned exactly as written. Only
//! schema-valued keyword positions are read: an object inside `const`, `enum`,
//! `default` or `examples` is argument data and is never touched, and a
//! property named `$ref` or `$defs` is a name, not a keyword.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

/// Keys that hold reusable definitions, at the root or inside a subschema.
const DEFINITION_KEYS: [&str; 2] = ["$defs", "definitions"];

/// Deeper schemas are returned as written (a grammar backend bounds them too).
const MAX_DEPTH: usize = 64;

/// Larger schemas (by node count) are returned as written.
const MAX_NODES: usize = 1 << 16;

/// Rewrite local `$ref` targets into the literal spelling grammar backends
/// resolve, renaming definition keys that cannot be referenced literally.
///
/// Non-local references (`http://…`, `other.json#/…`), references with an
/// invalid percent escape or a stray `~`, and references whose decoded segment
/// still contains a `/` outside a renamed definition are left exactly as written,
/// and a definition such a reference resolves to by literal lookup keeps its
/// name. A schema the bounded analysis cannot finish is returned unchanged.
pub fn canonicalize_local_refs(mut schema: Value) -> Value {
    if !schema.is_object() {
        return schema;
    }
    let Some(plan) = Plan::analyze(&schema) else {
        return schema;
    };
    plan.apply(&mut schema);
    schema
}

/// A pointer path from the root, one decoded key or array index per segment.
type Path = Vec<String>;

/// Every `$ref` site and definition container in the schema.
#[derive(Default)]
struct Survey {
    /// `(path to the definitions map, its keys)`.
    containers: Vec<(Path, Vec<String>)>,
    /// `(raw text, decoded local segments)` for every string `$ref`.
    refs: Vec<(String, Option<Path>)>,
}

impl Survey {
    /// `None` when the schema exceeds the depth or node budget.
    fn collect(schema: &Value) -> Option<Self> {
        let mut survey = Self::default();
        let mut budget = MAX_NODES;
        survey.walk(schema, &mut Vec::new(), &mut budget)?;
        Some(survey)
    }

    /// Visit one schema. Only schema-valued keyword positions are entered;
    /// instance data (`const`, `enum`, `default`, `examples`, unknown keywords)
    /// is never read as a schema.
    fn walk(&mut self, schema: &Value, path: &mut Path, budget: &mut usize) -> Option<()> {
        if path.len() > MAX_DEPTH {
            return None;
        }
        *budget = budget.checked_sub(1)?;
        let Value::Object(map) = schema else {
            return Some(());
        };
        if let Some(Value::String(reference)) = map.get("$ref") {
            self.refs
                .push((reference.clone(), decode_local_ref(reference)));
        }
        for (key, child) in map {
            match keyword_kind(key, child) {
                KeywordKind::Instance => {}
                KeywordKind::Schema => {
                    path.push(key.clone());
                    self.walk(child, path, budget)?;
                    path.pop();
                }
                KeywordKind::SchemaList => {
                    let Some(items) = child.as_array() else {
                        continue;
                    };
                    path.push(key.clone());
                    for (index, item) in items.iter().enumerate() {
                        path.push(index.to_string());
                        self.walk(item, path, budget)?;
                        path.pop();
                    }
                    path.pop();
                }
                KeywordKind::SchemaMap => {
                    let Some(entries) = child.as_object() else {
                        continue;
                    };
                    path.push(key.clone());
                    if DEFINITION_KEYS.contains(&key.as_str()) {
                        self.containers
                            .push((path.clone(), entries.keys().cloned().collect()));
                    }
                    for (name, entry) in entries {
                        path.push(name.clone());
                        self.walk(entry, path, budget)?;
                        path.pop();
                    }
                    path.pop();
                }
            }
        }
        Some(())
    }
}

/// How a keyword's value relates to the schema that holds it.
#[derive(Clone, Copy)]
enum KeywordKind {
    /// Instance data or an annotation: never a schema (`const`, `enum`,
    /// `default`, `examples`, `$ref` itself, unknown keywords).
    Instance,
    /// One subschema.
    Schema,
    /// An array of subschemas.
    SchemaList,
    /// Names mapped to subschemas; the names are data, not keywords.
    SchemaMap,
}

/// Classify a keyword per JSON Schema (draft 2020-12 plus the draft-04/07
/// spellings tool schemas still use). Unknown keywords are annotations.
fn keyword_kind(key: &str, value: &Value) -> KeywordKind {
    match key {
        "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas" => {
            KeywordKind::SchemaMap
        }
        // Draft-07 `dependencies` mixes schemas with property-name arrays; only
        // its object entries are schemas, which `walk` already requires.
        "dependencies" => KeywordKind::SchemaMap,
        "allOf" | "anyOf" | "oneOf" | "prefixItems" => KeywordKind::SchemaList,
        // Draft-04/07 tuple form of `items` is a list; otherwise one schema.
        "items" => {
            if value.is_array() {
                KeywordKind::SchemaList
            } else {
                KeywordKind::Schema
            }
        }
        "additionalProperties"
        | "additionalItems"
        | "unevaluatedProperties"
        | "unevaluatedItems"
        | "contains"
        | "propertyNames"
        | "not"
        | "if"
        | "then"
        | "else"
        | "contentSchema" => KeywordKind::Schema,
        _ => KeywordKind::Instance,
    }
}

/// Definition renames, keyed by container path and original name.
type Renames = BTreeMap<(Path, String), String>;

/// What to change, decided before anything is touched.
struct Plan {
    renames: Renames,
}

impl Plan {
    fn analyze(schema: &Value) -> Option<Self> {
        let survey = Survey::collect(schema)?;

        // Start from every definition that cannot be referenced literally, then
        // withdraw a rename while some reference that resolves to that definition
        // today cannot be rewritten (an invalid escape, or a decoded `/` outside a
        // renamed key). Each round only removes candidates, so this terminates.
        let mut candidates: BTreeSet<(Path, String)> = survey
            .containers
            .iter()
            .flat_map(|(path, keys)| {
                keys.iter()
                    .filter(|key| !is_plain_identifier(key))
                    .map(|key| (path.clone(), key.clone()))
            })
            .collect();
        loop {
            let pinned: Vec<(Path, String)> = survey
                .refs
                .iter()
                .filter(|(raw, decoded)| {
                    !rewritable(decoded.as_deref(), &survey.containers, &candidates) && raw != "#"
                })
                .filter_map(|(raw, _)| literal_target(raw, &survey.containers))
                .filter(|target| candidates.contains(target))
                .collect();
            if pinned.is_empty() {
                break;
            }
            for target in pinned {
                candidates.remove(&target);
            }
        }

        let mut renames = Renames::new();
        for (path, keys) in &survey.containers {
            let mut taken: BTreeSet<String> = keys.iter().cloned().collect();
            for key in keys {
                if !candidates.contains(&(path.clone(), key.clone())) {
                    continue;
                }
                let base: String = key
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
                renames.insert((path.clone(), key.clone()), candidate);
            }
        }
        Some(Self { renames })
    }

    fn apply(&self, schema: &mut Value) {
        let containers: Vec<Path> = self
            .renames
            .keys()
            .map(|(path, _)| path.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        self.rewrite_refs(schema, &containers);
        for path in &containers {
            let Some(Value::Object(definitions)) = node_at_mut(schema, path) else {
                continue;
            };
            let entries = std::mem::take(definitions);
            for (name, definition) in entries {
                let name = self
                    .renames
                    .get(&(path.clone(), name.clone()))
                    .cloned()
                    .unwrap_or(name);
                definitions.insert(name, definition);
            }
        }
    }

    /// Mirror of [`Survey::walk`]: the same keyword positions, mutably.
    fn rewrite_refs(&self, schema: &mut Value, containers: &[Path]) {
        let Value::Object(map) = schema else {
            return;
        };
        if let Some(Value::String(reference)) = map.get_mut("$ref")
            && let Some(canonical) = self.canonical_ref(reference, containers)
        {
            *reference = canonical;
        }
        for (key, child) in map.iter_mut() {
            match keyword_kind(key, child) {
                KeywordKind::Instance => {}
                KeywordKind::Schema => self.rewrite_refs(child, containers),
                KeywordKind::SchemaList => {
                    if let Some(items) = child.as_array_mut() {
                        for item in items {
                            self.rewrite_refs(item, containers);
                        }
                    }
                }
                KeywordKind::SchemaMap => {
                    if let Some(entries) = child.as_object_mut() {
                        for entry in entries.values_mut() {
                            self.rewrite_refs(entry, containers);
                        }
                    }
                }
            }
        }
    }

    /// The literal-segment spelling of a local reference, or `None` when it is
    /// not local, cannot be decoded, would still hold a `/`, or would not change.
    fn canonical_ref(&self, reference: &str, containers: &[Path]) -> Option<String> {
        let mut segments = decode_local_ref(reference)?;
        for path in containers {
            if segments.len() > path.len()
                && segments[..path.len()] == path[..]
                && let Some(renamed) = self
                    .renames
                    .get(&(path.clone(), segments[path.len()].clone()))
            {
                segments[path.len()] = renamed.clone();
            }
        }
        if segments.iter().any(|segment| segment.contains('/')) {
            return None;
        }
        let canonical = format!("#/{}", segments.join("/"));
        (canonical != reference).then_some(canonical)
    }
}

/// Whether a reference will be rewritten once `candidates` are renamed: it
/// decodes, and every decoded `/` sits in a key position that is being renamed.
fn rewritable(
    decoded: Option<&[String]>,
    containers: &[(Path, Vec<String>)],
    candidates: &BTreeSet<(Path, String)>,
) -> bool {
    let Some(segments) = decoded else {
        return false;
    };
    segments.iter().enumerate().all(|(index, segment)| {
        !segment.contains('/')
            || containers.iter().any(|(path, _)| {
                path.len() == index
                    && segments[..index] == path[..]
                    && candidates.contains(&(path.clone(), segment.clone()))
            })
    })
}

/// The definition a reference left as written resolves to by literal segment
/// lookup, if any.
fn literal_target(raw: &str, containers: &[(Path, Vec<String>)]) -> Option<(Path, String)> {
    let segments: Vec<&str> = raw.strip_prefix("#/")?.split('/').collect();
    containers.iter().find_map(|(path, keys)| {
        (segments.len() > path.len()
            && segments[..path.len()]
                .iter()
                .zip(path)
                .all(|(segment, step)| *segment == step.as_str())
            && keys.iter().any(|key| key == segments[path.len()]))
        .then(|| (path.clone(), segments[path.len()].to_string()))
    })
}

fn node_at_mut<'a>(root: &'a mut Value, path: &[String]) -> Option<&'a mut Value> {
    path.iter().try_fold(root, |node, segment| match node {
        Value::Object(map) => map.get_mut(segment),
        Value::Array(items) => items.get_mut(segment.parse::<usize>().ok()?),
        _ => None,
    })
}

/// A definition name a grammar backend resolves verbatim and a URI fragment
/// carries without escaping.
fn is_plain_identifier(name: &str) -> bool {
    !name.is_empty() && name.chars().all(is_plain_char)
}

fn is_plain_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
}

/// The decoded segments of a local reference (`#/a/b`), or `None` when the
/// reference is not local or holds an invalid escape.
fn decode_local_ref(reference: &str) -> Option<Path> {
    let fragment = reference.strip_prefix('#')?;
    // URI percent-decoding precedes JSON Pointer's `~0`/`~1` decoding.
    let pointer = percent_decode(fragment)?;
    let path = pointer.strip_prefix('/')?;
    path.split('/').map(unescape_pointer_segment).collect()
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
    fn nested_definition_containers_are_renamed_with_their_full_path() {
        // JSON Schema allows `$defs` inside a subschema; the rename is keyed by the
        // container's pointer path, not only the root.
        let schema = json!({
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "$defs": {"a/b": {"type": "integer"}, "Plain": {"type": "string"}},
                    "properties": {
                        "n": {"$ref": "#/properties/outer/$defs/a~1b"},
                        "s": {"$ref": "#/properties/outer/$defs/Plain"}
                    }
                },
                "list": {"anyOf": [{"definitions": {"x y": {"type": "boolean"}}}]},
                "flag": {"$ref": "#/properties/list/anyOf/0/definitions/x%20y"}
            }
        });
        assert_eq!(
            canonicalize_local_refs(schema),
            json!({
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "$defs": {"a_b": {"type": "integer"}, "Plain": {"type": "string"}},
                        "properties": {
                            "n": {"$ref": "#/properties/outer/$defs/a_b"},
                            "s": {"$ref": "#/properties/outer/$defs/Plain"}
                        }
                    },
                    "list": {"anyOf": [{"definitions": {"x_y": {"type": "boolean"}}}]},
                    "flag": {"$ref": "#/properties/list/anyOf/0/definitions/x_y"}
                }
            })
        );
    }

    #[test]
    fn a_definition_keeps_its_name_while_an_unrewritable_reference_resolves_to_it() {
        // `#/$defs/Bad%ZZ` cannot be decoded, but XGrammar resolves it by literal
        // lookup today; renaming its target would make it dangle. The sibling with
        // only decodable references is still renamed.
        let schema = json!({
            "$defs": {"Bad%ZZ": {"type": "string"}, "x y": {"type": "integer"}},
            "properties": {
                "a": {"$ref": "#/$defs/Bad%ZZ"},
                "b": {"$ref": "#/$defs/x%20y"},
                "c": {"$ref": "#/$defs/x y"}
            }
        });
        assert_eq!(
            canonicalize_local_refs(schema),
            json!({
                "$defs": {"Bad%ZZ": {"type": "string"}, "x_y": {"type": "integer"}},
                "properties": {
                    "a": {"$ref": "#/$defs/Bad%ZZ"},
                    "b": {"$ref": "#/$defs/x_y"},
                    "c": {"$ref": "#/$defs/x_y"}
                }
            })
        );
    }

    #[test]
    fn non_local_and_malformed_references_are_left_alone() {
        let schema = json!({
            "$defs": {"Ok": {"type": "string"}},
            "properties": {
                "b": {"$ref": "#/$defs/Ok~"},
                "c": {"$ref": "https://example.com/schema.json#/$defs/Remote%20X"},
                "d": {"$ref": "other.json#/x"},
                "e": {"$ref": "#"},
                "f": {"$ref": 7}
            }
        });
        assert_eq!(canonicalize_local_refs(schema.clone()), schema);
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
    fn a_schema_too_deep_to_analyse_is_returned_unchanged() {
        // The reference sits beyond the depth bound; renaming its target while
        // leaving it as written would break a schema XGrammar resolves today.
        let mut deep = json!({"$ref": "#/$defs/a b"});
        for _ in 0..(MAX_DEPTH + 40) {
            deep = json!({"items": deep});
        }
        let schema = json!({"$defs": {"a b": {"type": "integer"}}, "properties": {"deep": deep}});
        assert_eq!(canonicalize_local_refs(schema.clone()), schema);
    }

    #[test]
    fn a_schema_too_large_to_analyse_is_returned_unchanged() {
        let branches: Vec<Value> = (0..MAX_NODES).map(|_| json!({"type": "integer"})).collect();
        let schema = json!({
            "$defs": {"a b": {"type": "integer"}},
            "properties": {
                "big": {"anyOf": branches},
                "z": {"$ref": "#/$defs/a b"}
            }
        });
        assert_eq!(canonicalize_local_refs(schema.clone()), schema);
    }

    #[test]
    fn instance_valued_keywords_are_argument_data_not_schemas() {
        // Objects under `const`, `enum`, `default` and `examples` are values the
        // model must produce; a `$defs` or `$ref` spelled inside them is data.
        let schema = json!({
            "type": "object",
            "$defs": {"a b": {"type": "integer"}},
            "properties": {
                "p": {"const": {"$defs": {"a b": 1}, "$ref": "#/$defs/a%20b"}},
                "q": {"enum": [{"$ref": "#/$defs/a~1b"}, {"$defs": {"x/y": true}}]},
                "r": {"type": "object", "default": {"$defs": {"q/r": {}}}},
                "s": {"examples": [{"$ref": "#/%24defs/a%20b"}]},
                "t": {"$ref": "#/$defs/a%20b"}
            }
        });
        let mut expected = schema.clone();
        expected["$defs"] = json!({"a_b": {"type": "integer"}});
        expected["properties"]["t"] = json!({"$ref": "#/$defs/a_b"});
        assert_eq!(canonicalize_local_refs(schema), expected);
    }

    #[test]
    fn a_literal_reference_inside_instance_data_does_not_pin_a_definition() {
        // The `const` object spells `#/$defs/a b` but is not a reference, so the
        // definition is still renamed for the real reference.
        let schema = json!({
            "$defs": {"a b": {"type": "integer"}},
            "properties": {
                "p": {"const": {"$ref": "#/$defs/a b"}},
                "t": {"$ref": "#/$defs/a b"}
            }
        });
        assert_eq!(
            canonicalize_local_refs(schema),
            json!({
                "$defs": {"a_b": {"type": "integer"}},
                "properties": {
                    "p": {"const": {"$ref": "#/$defs/a b"}},
                    "t": {"$ref": "#/$defs/a_b"}
                }
            })
        );
    }

    #[test]
    fn property_names_that_spell_keywords_are_names() {
        // `properties.$ref` and `properties.$defs` are argument names: the first
        // is not a reference, the second is not a definition container, so the
        // non-identifier property under it is not renamed. A reference into it
        // is still decoded to the literal spelling.
        let schema = json!({
            "type": "object",
            "$defs": {"x y": {"type": "boolean"}},
            "properties": {
                "$ref": {"type": "string"},
                "$defs": {
                    "type": "object",
                    "properties": {"a b": {"type": "integer"}}
                },
                "copy": {"$ref": "#/properties/%24defs/properties/a%20b"},
                "flag": {"$ref": "#/$defs/x%20y"}
            }
        });
        assert_eq!(
            canonicalize_local_refs(schema),
            json!({
                "type": "object",
                "$defs": {"x_y": {"type": "boolean"}},
                "properties": {
                    "$ref": {"type": "string"},
                    "$defs": {
                        "type": "object",
                        "properties": {"a b": {"type": "integer"}}
                    },
                    "copy": {"$ref": "#/properties/$defs/properties/a b"},
                    "flag": {"$ref": "#/$defs/x_y"}
                }
            })
        );
    }

    #[test]
    fn every_schema_valued_keyword_position_is_canonicalized() {
        let reference = json!({"$ref": "#/$defs/a%20b"});
        let schema = json!({
            "$defs": {"a b": {"type": "integer"}},
            "properties": {"p": reference},
            "patternProperties": {"^x": reference},
            "additionalProperties": reference,
            "items": [reference],
            "prefixItems": [reference],
            "contains": reference,
            "propertyNames": reference,
            "not": reference,
            "if": reference, "then": reference, "else": reference,
            "allOf": [reference], "anyOf": [reference], "oneOf": [reference],
            "dependentSchemas": {"p": reference},
            "dependencies": {"p": reference, "q": ["p"]},
            "unevaluatedProperties": reference,
            "unevaluatedItems": reference,
            "contentSchema": reference
        });
        let canonical = canonicalize_local_refs(schema);
        assert_eq!(canonical["$defs"], json!({"a_b": {"type": "integer"}}));
        let text = canonical.to_string();
        assert!(!text.contains("a%20b"), "{text}");
        assert_eq!(text.matches("#/$defs/a_b").count(), 19, "{text}");
        assert_eq!(canonical["dependencies"]["q"], json!(["p"]));
    }
}
