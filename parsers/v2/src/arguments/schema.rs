// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use percent_encoding::percent_decode_str;
use serde_json::Value;
use std::collections::HashSet;

#[derive(Clone, Copy)]
enum ReferenceSiblings {
    Intersect,
    Unknown,
}

pub(crate) fn collect_allowed_types(schema: &Value) -> HashSet<SchemaType> {
    let mut walker = SchemaWalker::new(schema);
    walker.type_policy = TypeHintPolicy::ConstrainedAlternatives;
    walker.type_constraints(schema).unwrap_or_default()
}

pub(crate) fn schema_has_type(root: &Value, schema: &Value, expected: &str) -> bool {
    let mut walker = SchemaWalker::new(root);
    walker.max_reference_depth = 16;
    walker.has_type(Some(schema), expected)
}

/// Coarse JSON-schema type category, used to resolve union (`anyOf`/`oneOf`/
/// `type: [..]`/`nullable`) schemas to the set of types they actually allow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SchemaType {
    String,
    Integer,
    Number,
    Boolean,
    Object,
    Array,
    Null,
}

/// Map a schema type name (including the aliases `convert_param_value`
/// recognizes) to its category. Unknown names return `None`.
fn categorize_type(name: &str) -> Option<SchemaType> {
    let t = name.to_lowercase();
    let t = t.as_str();
    if matches!(t, "string" | "str" | "text" | "varchar" | "char" | "enum") {
        Some(SchemaType::String)
    } else if matches!(t, "boolean" | "bool" | "binary") {
        Some(SchemaType::Boolean)
    } else if matches!(t, "null" | "none") {
        Some(SchemaType::Null)
    } else if t.starts_with("int")
        || t.starts_with("uint")
        || t.starts_with("long")
        || t.starts_with("short")
        || t.starts_with("unsigned")
    {
        Some(SchemaType::Integer)
    } else if t.starts_with("num") || t.starts_with("float") {
        Some(SchemaType::Number)
    } else if t == "object" || t.starts_with("dict") {
        Some(SchemaType::Object)
    } else if t == "array" || t == "arr" || t.starts_with("list") {
        Some(SchemaType::Array)
    } else {
        None
    }
}

// A float-backed schema literal has already passed through f64: an integral-looking
// value may have originated as a large fraction. Retain the number alternative.
// Explicit integer types still intersect this set and exclude fractional arguments.
fn literal_type_constraints(value: &Value) -> HashSet<SchemaType> {
    match value_category(value) {
        SchemaType::Number => HashSet::from([SchemaType::Integer, SchemaType::Number]),
        category => HashSet::from([category]),
    }
}

/// The storage category, without inferring mathematical integrality from f64.
pub(crate) fn value_category(v: &Value) -> SchemaType {
    match v {
        Value::String(_) => SchemaType::String,
        Value::Bool(_) => SchemaType::Boolean,
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                SchemaType::Integer
            } else {
                SchemaType::Number
            }
        }
        Value::Object(_) => SchemaType::Object,
        Value::Array(_) => SchemaType::Array,
        Value::Null => SchemaType::Null,
    }
}

// Bound graph traversal as well as recursion: shared references can expand
// exponentially even when there are no cycles. Exhaustion preserves untyped output.
pub(crate) const MAX_SCHEMA_WORK: usize = 1024;
pub(crate) const MAX_SCHEMA_DEPTH: usize = 64;

// Each lookup tracks its own schema path: recursive schemas may be revisited
// after consuming another XML child, but reference/composition cycles cannot loop.
#[derive(Clone, Copy)]
enum TypeHintPolicy {
    ConstrainedAlternatives,
    AvailableAlternatives,
}

pub(crate) struct SchemaWalker<'a> {
    root: &'a Value,
    pub(crate) path: Vec<&'a Value>,
    pub(crate) remaining_work: usize,
    pub(crate) exhausted: bool,
    max_reference_depth: usize,
    reference_depth: usize,
    type_policy: TypeHintPolicy,
}

impl<'a> SchemaWalker<'a> {
    pub(crate) fn new(root: &'a Value) -> Self {
        Self {
            root,
            path: Vec::new(),
            remaining_work: MAX_SCHEMA_WORK,
            exhausted: false,
            max_reference_depth: MAX_SCHEMA_WORK,
            reference_depth: 0,
            type_policy: TypeHintPolicy::AvailableAlternatives,
        }
    }

    fn spend_work(&mut self) -> bool {
        if self.exhausted || self.remaining_work == 0 {
            self.exhausted = true;
            return false;
        }
        self.remaining_work -= 1;
        true
    }

    // Leave unknown references and cycles untouched; do not discard sibling constraints.
    pub(crate) fn resolve_ref(&mut self, schema: &'a Value) -> Option<&'a Value> {
        let initial_depth = self.reference_depth;
        let mut current = schema;
        if has_unsupported_schema_ref_scope(current) && !std::ptr::eq(current, self.root) {
            return None;
        }
        let mut visited = Vec::new();
        while let Some(reference) = current.get("$ref").and_then(Value::as_str) {
            if !self.spend_work() || self.reference_depth >= self.max_reference_depth {
                self.exhausted = true;
                return None;
            }
            if current.as_object().is_some_and(|object| {
                object.keys().any(|key| {
                    !matches!(
                        key.as_str(),
                        "$ref" | "title" | "description" | "default" | "examples" | "$comment"
                    )
                })
            }) || visited.contains(&reference)
            {
                self.reference_depth = initial_depth;
                return Some(schema);
            }
            let Some(fragment) = reference.strip_prefix('#') else {
                self.reference_depth = initial_depth;
                return Some(schema);
            };
            // percent_decode_str preserves invalid escapes, so reject them before lookup.
            if fragment.as_bytes().iter().enumerate().any(|(index, byte)| {
                *byte == b'%'
                    && !fragment
                        .as_bytes()
                        .get(index + 1..index + 3)
                        .is_some_and(|digits| digits.iter().all(u8::is_ascii_hexdigit))
            }) {
                self.reference_depth = initial_depth;
                return Some(schema);
            }
            let Ok(pointer) = percent_decode_str(fragment).decode_utf8() else {
                self.reference_depth = initial_depth;
                return Some(schema);
            };
            let Some(target) = self.root.pointer(&pointer) else {
                self.reference_depth = initial_depth;
                return Some(schema);
            };
            if has_unsupported_schema_ref_scope(target) && !std::ptr::eq(target, self.root) {
                return None;
            }
            self.reference_depth += 1;
            visited.push(reference);
            current = target;
        }
        Some(current)
    }

    fn with_schema<T>(
        &mut self,
        schema: &'a Value,
        fallback: T,
        query: impl FnOnce(&mut Self, &'a Value) -> T,
    ) -> T {
        if !self.spend_work() || self.path.len() >= MAX_SCHEMA_DEPTH {
            self.exhausted = true;
            return fallback;
        }
        let previous_depth = self.reference_depth;
        let Some(schema) = self.resolve_ref(schema) else {
            self.reference_depth = previous_depth;
            return fallback;
        };
        if self.path.iter().any(|seen| std::ptr::eq(*seen, schema)) {
            self.reference_depth = previous_depth;
            return fallback;
        }
        self.path.push(schema);
        let result = query(self, schema);
        self.path.pop();
        self.reference_depth = previous_depth;
        if self.exhausted { fallback } else { result }
    }

    // Unknown constraints remain possible, preserving object-union ambiguity.
    pub(crate) fn may_describe_object(&mut self, schema: &'a Value) -> bool {
        self.with_schema(schema, true, |walker, schema| {
            if schema == &Value::Bool(false) {
                return false;
            }
            if let Some(ty) = schema.get("type") {
                let object = ty.as_str() == Some("object")
                    || ty
                        .as_array()
                        .is_some_and(|types| types.iter().any(|ty| ty == "object"));
                if !object {
                    return false;
                }
            }
            if schema.get("const").is_some_and(|value| !value.is_object())
                || schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(|values| !values.iter().any(Value::is_object))
            {
                return false;
            }
            for keyword in ["allOf", "anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    let possible = if keyword == "allOf" {
                        branches
                            .iter()
                            .all(|branch| walker.may_describe_object(branch))
                    } else {
                        branches
                            .iter()
                            .any(|branch| walker.may_describe_object(branch))
                    };
                    if !possible {
                        return false;
                    }
                }
            }
            true
        })
    }

    // Nested XML identifies an object, but does not choose among object variants.
    pub(crate) fn object_child(&mut self, schema: &'a Value, tag: &str) -> Option<&'a Value> {
        self.with_schema(schema, None, |walker, schema| {
            if let Some(child) = schema.get("properties").and_then(|props| props.get(tag)) {
                return Some(child);
            }
            if let Some(additional) = schema
                .get("additionalProperties")
                .filter(|value| value.is_object())
            {
                return Some(additional);
            }
            let branches = match (schema.get("anyOf"), schema.get("oneOf")) {
                (Some(branches), None) | (None, Some(branches)) => branches.as_array()?,
                _ => return None,
            };
            let mut objects = branches
                .iter()
                .filter(|branch| walker.may_describe_object(branch));
            let object = objects.next()?;
            if objects.next().is_some() {
                return None;
            }
            walker.object_child(object, tag)
        })
    }

    pub(crate) fn permits_null(&mut self, schema: &'a Value) -> Option<bool> {
        if self.exhausted {
            return None;
        }
        let result = schema_null_match(
            schema,
            self.root,
            &mut vec![schema],
            0,
            &mut self.remaining_work,
            ReferenceSiblings::Unknown,
        );
        if self.remaining_work == 0 {
            self.exhausted = true;
            return None;
        }
        result
    }

    pub(crate) fn has_type(&mut self, schema: Option<&'a Value>, expected: &str) -> bool {
        let Some(category) = categorize_type(expected) else {
            return false;
        };
        schema
            .and_then(|s| self.type_constraints(s))
            .is_some_and(|types| types.contains(&category))
    }

    // None is an absent type constraint, not an empty intersection.
    fn type_constraints(&mut self, schema: &'a Value) -> Option<HashSet<SchemaType>> {
        self.with_schema(schema, None, |walker, schema| {
            if schema == &Value::Bool(false) {
                return Some(HashSet::new());
            }
            let mut out = HashSet::new();
            if let Some(ty) = schema.get("type") {
                if let Some(name) = ty.as_str() {
                    if let Some(cat) = categorize_type(name) {
                        out.insert(cat);
                    }
                } else if let Some(arr) = ty.as_array() {
                    for item in arr.iter().filter_map(Value::as_str) {
                        if let Some(cat) = categorize_type(item) {
                            out.insert(cat);
                        }
                    }
                }
            }
            if out.contains(&SchemaType::Number) {
                out.insert(SchemaType::Integer);
            }
            if schema.get("nullable").and_then(Value::as_bool) == Some(true) {
                out.insert(SchemaType::Null);
            }
            let mut constraints = Vec::new();
            if let Some(target) = schema
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|reference| resolve_local_schema_ref(reference, walker.root))
            {
                if walker.reference_depth >= walker.max_reference_depth {
                    walker.exhausted = true;
                    return None;
                }
                walker.reference_depth += 1;
                let types = walker.type_constraints(target);
                walker.reference_depth -= 1;
                if let Some(types) = types {
                    constraints.push(types);
                }
            }
            if !out.is_empty() {
                constraints.push(out);
            }
            if let Some(value) = schema.get("const") {
                constraints.push(literal_type_constraints(value));
            }
            if let Some(values) = schema.get("enum").and_then(Value::as_array) {
                constraints.push(values.iter().flat_map(literal_type_constraints).collect());
            }
            for key in ["anyOf", "oneOf", "allOf"] {
                if let Some(options) = schema.get(key).and_then(Value::as_array) {
                    let available =
                        matches!(walker.type_policy, TypeHintPolicy::AvailableAlternatives);
                    let branches = options.iter().map(|branch| walker.type_constraints(branch));
                    if key == "allOf" {
                        constraints.extend(branches.flatten());
                    } else if available {
                        let alternatives: Vec<_> = branches.flatten().collect();
                        if !alternatives.is_empty() {
                            constraints.push(alternatives.into_iter().flatten().collect());
                        }
                    } else if let Some(alternatives) = branches.collect::<Option<Vec<_>>>() {
                        constraints.push(alternatives.into_iter().flatten().collect());
                    }
                }
            }
            constraints.into_iter().reduce(|mut left, right| {
                left.retain(|ty| right.contains(ty));
                left
            })
        })
    }
    pub(crate) fn array_item(&mut self, schema: Option<&'a Value>) -> Option<&'a Value> {
        self.with_schema(schema?, None, |walker, schema| {
            if let Some(items) = schema.get("items") {
                return Some(items);
            }
            for key in ["anyOf", "oneOf"] {
                if let Some(options) = schema.get(key).and_then(Value::as_array) {
                    for option in options {
                        if let Some(items) = walker.array_item(Some(option)) {
                            return Some(items);
                        }
                    }
                }
            }
            None
        })
    }
}

const MAX_NULL_SCHEMA_REF_DEPTH: usize = 16;
const MAX_NULL_SCHEMA_NODES: usize = 4096;

// Only a proven match may turn the model's text into JSON null. Unknown references
// must remain unknown through negation and oneOf, rather than counting as false.
pub(crate) fn schema_permits_null(schema: &Value, root: &Value) -> bool {
    let mut active_refs = vec![schema];
    let mut remaining = MAX_NULL_SCHEMA_NODES;
    schema_null_match(
        schema,
        root,
        &mut active_refs,
        0,
        &mut remaining,
        ReferenceSiblings::Intersect,
    ) == Some(true)
}

fn intersect_null_matches(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn has_unsupported_schema_ref_scope(schema: &Value) -> bool {
    ["$id", "$dynamicRef", "$recursiveRef"]
        .iter()
        .any(|keyword| schema.get(*keyword).and_then(Value::as_str).is_some())
}

fn resolve_local_schema_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    // URI percent-decoding precedes JSON Pointer's ~0/~1 decoding.
    let pointer = reference.strip_prefix('#')?;
    let decoded;
    let pointer = if pointer.contains('%') {
        let mut bytes = pointer.bytes();
        let mut result = Vec::with_capacity(pointer.len());
        while let Some(byte) = bytes.next() {
            result.push(if byte == b'%' {
                let high = char::from(bytes.next()?).to_digit(16)?;
                let low = char::from(bytes.next()?).to_digit(16)?;
                ((high << 4) | low) as u8
            } else {
                byte
            });
        }
        decoded = String::from_utf8(result).ok()?;
        decoded.as_str()
    } else {
        pointer
    };
    let mut bytes = pointer.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'~' && !matches!(bytes.next(), Some(b'0' | b'1')) {
            return None;
        }
    }
    // A nested $id changes the reference scope. Do not jump through it while
    // resolving a fragment against the original tool-parameter document.
    for (end, _) in pointer.char_indices().filter(|(_, ch)| *ch == '/').skip(1) {
        if has_unsupported_schema_ref_scope(root.pointer(&pointer[..end])?) {
            return None;
        }
    }
    let target = root.pointer(pointer)?;
    matches!(target, Value::Bool(_) | Value::Object(_)).then_some(target)
}

// Evaluate only the constraints relevant to null. Ref targets and sibling
// keywords intersect; the remaining coercion rules do not change.
fn schema_null_match<'a>(
    schema: &'a Value,
    root: &'a Value,
    active_refs: &mut Vec<&'a Value>,
    ref_depth: usize,
    remaining: &mut usize,
    siblings: ReferenceSiblings,
) -> Option<bool> {
    *remaining = (*remaining).checked_sub(1)?;
    if matches!(siblings, ReferenceSiblings::Unknown) && ref_depth >= MAX_SCHEMA_DEPTH {
        *remaining = 0;
        return None;
    }
    let schema = if matches!(siblings, ReferenceSiblings::Unknown) {
        let mut resolver = SchemaWalker::new(root);
        resolver.remaining_work = *remaining;
        let resolved = resolver.resolve_ref(schema);
        *remaining = resolver.remaining_work;
        let resolved = resolved?;
        if !std::ptr::eq(resolved, schema) {
            if active_refs
                .iter()
                .any(|active| std::ptr::eq(*active, resolved))
            {
                return None;
            }
            active_refs.push(resolved);
            let result =
                schema_null_match(resolved, root, active_refs, ref_depth, remaining, siblings);
            active_refs.pop();
            return result;
        }
        resolved
    } else {
        schema
    };
    if let Some(allowed) = schema.as_bool() {
        return Some(allowed);
    }
    if !schema.is_object()
        || schema.get("$dynamicRef").is_some()
        || schema.get("$recursiveRef").is_some()
        || (schema.get("$id").is_some() && !std::ptr::eq(schema, root))
    {
        return None;
    }
    let mut permits = Some(true);
    if let Some(reference) = schema.get("$ref") {
        let conservative_siblings = matches!(siblings, ReferenceSiblings::Unknown)
            && schema.as_object().is_some_and(|object| {
                object.keys().any(|key| {
                    !matches!(
                        key.as_str(),
                        "$ref" | "title" | "description" | "default" | "examples" | "$comment"
                    )
                })
            });
        let target = reference
            .as_str()
            .and_then(|reference| resolve_local_schema_ref(reference, root));
        permits = if let Some(target) = target.filter(|_| !conservative_siblings)
            && ref_depth
                < if matches!(siblings, ReferenceSiblings::Unknown) {
                    MAX_SCHEMA_DEPTH
                } else {
                    MAX_NULL_SCHEMA_REF_DEPTH
                }
            && !active_refs
                .iter()
                .any(|active| std::ptr::eq(*active, target))
        {
            active_refs.push(target);
            let result = schema_null_match(
                target,
                root,
                active_refs,
                ref_depth + 1,
                remaining,
                siblings,
            );
            active_refs.pop();
            result
        } else {
            None
        };
    }
    if let Some(ty) = schema.get("type") {
        let nullable = schema.get("nullable").and_then(Value::as_bool) == Some(true);
        if !nullable
            && ty.as_str() != Some("null")
            && !ty
                .as_array()
                .is_some_and(|types| types.iter().any(|ty| ty == "null"))
        {
            return Some(false);
        }
    }
    if schema.get("const").is_some_and(|value| !value.is_null())
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.iter().any(Value::is_null))
    {
        return Some(false);
    }
    let child_depth = ref_depth + usize::from(matches!(siblings, ReferenceSiblings::Unknown));
    for keyword in ["allOf", "anyOf", "oneOf"] {
        if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
            let mut matches = 0;
            let mut unknown = 0;
            for branch in branches {
                match schema_null_match(branch, root, active_refs, child_depth, remaining, siblings)
                {
                    Some(true) => matches += 1,
                    Some(false) => {}
                    None => unknown += 1,
                }
            }
            let branch_match = match keyword {
                "allOf" if matches + unknown < branches.len() => Some(false),
                "anyOf" if matches > 0 => Some(true),
                "oneOf" if matches > 1 => Some(false),
                _ if unknown > 0 => None,
                "allOf" => Some(true),
                "anyOf" => Some(false),
                _ => Some(matches == 1),
            };
            permits = intersect_null_matches(permits, branch_match);
            if permits == Some(false) {
                return permits;
            }
        }
    }
    if let Some(not) = schema.get("not") {
        let not_match = schema_null_match(not, root, active_refs, child_depth, remaining, siblings);
        permits = intersect_null_matches(permits, not_match.map(|matches| !matches));
    }
    permits
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn coarse_union_hints_differ_from_proven_constraints() {
        for kind in ["string", "object"] {
            let schema = json!({"anyOf":[{}, {"type":kind}]});
            assert!(SchemaWalker::new(&schema).has_type(Some(&schema), kind));
            assert!(schema_has_type(&schema, &schema, kind));
            assert!(collect_allowed_types(&schema).is_empty());
        }
    }

    #[test]
    fn cyclic_null_branches_do_not_erase_decisive_siblings() {
        for (schema, expected) in [
            (json!({"anyOf":[{"$ref":"#"}, {"type":"null"}]}), Some(true)),
            (json!({"allOf":[{"$ref":"#"}, false]}), Some(false)),
        ] {
            assert_eq!(SchemaWalker::new(&schema).permits_null(&schema), expected);
        }
    }

    fn reference_chain(length: usize, siblings: bool, kind: &str) -> Value {
        let mut defs = serde_json::Map::new();
        defs.insert(format!("s{length}"), json!({"type":kind}));
        for index in 0..length {
            let mut schema = json!({"$ref":format!("#/$defs/s{}", index+1)});
            if siblings {
                schema["minLength"] = json!(0);
            }
            defs.insert(format!("s{index}"), schema);
        }
        json!({"$defs":defs})
    }

    #[test]
    fn reference_budgets_include_siblings_without_reducing_iterative_null_chains() {
        let reference = json!({"$ref":"#/$defs/s0"});
        for siblings in [false, true] {
            for (length, expected) in [(15, true), (16, false), (20, false)] {
                let root = reference_chain(length, siblings, "string");
                assert_eq!(
                    schema_has_type(&root, &reference, "string"),
                    expected,
                    "{length} siblings={siblings}"
                );
            }
        }
        let root = reference_chain(71, false, "null");
        assert_eq!(
            SchemaWalker::new(&root).permits_null(&reference),
            Some(true)
        );
    }
}
