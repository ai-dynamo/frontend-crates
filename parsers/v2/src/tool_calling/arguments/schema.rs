// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use percent_encoding::percent_decode_str;
use serde_json::Value;

// Bound graph traversal as well as recursion: shared references can expand
// exponentially even when there are no cycles. Exhaustion preserves untyped output.
pub(crate) const MAX_SCHEMA_WORK: usize = 1024;
pub(crate) const MAX_SCHEMA_DEPTH: usize = 64;

// Each lookup tracks its own schema path: recursive schemas may be revisited
// after consuming another XML child, but reference/composition cycles cannot loop.
pub(crate) struct SchemaFacts<'a> {
    root: &'a Value,
    #[cfg(test)]
    pub(crate) path: Vec<&'a Value>,
    #[cfg(not(test))]
    path: Vec<&'a Value>,
    #[cfg(test)]
    pub(crate) remaining_work: usize,
    #[cfg(not(test))]
    remaining_work: usize,
    pub(crate) exhausted: bool,
}

impl<'a> SchemaFacts<'a> {
    pub(crate) fn new(root: &'a Value) -> Self {
        Self {
            root,
            path: Vec::new(),
            remaining_work: MAX_SCHEMA_WORK,
            exhausted: false,
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
        let mut current = schema;
        let mut visited = Vec::new();
        while let Some(reference) = current.get("$ref").and_then(Value::as_str) {
            if !self.spend_work() {
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
                return Some(schema);
            }
            let Some(fragment) = reference.strip_prefix('#') else {
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
                return Some(schema);
            }
            let Ok(pointer) = percent_decode_str(fragment).decode_utf8() else {
                return Some(schema);
            };
            let Some(target) = self.root.pointer(&pointer) else {
                return Some(schema);
            };
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
        let Some(schema) = self.resolve_ref(schema) else {
            return fallback;
        };
        if self.path.iter().any(|seen| std::ptr::eq(*seen, schema)) {
            return fallback;
        }
        self.path.push(schema);
        let result = query(self, schema);
        self.path.pop();
        if self.exhausted { fallback } else { result }
    }

    // Unknown constraints remain possible, preserving object-union ambiguity.
    pub(crate) fn may_describe_object(&mut self, schema: &'a Value) -> bool {
        self.may_describe_structural_type(schema, "object")
    }

    // `items` constrains arrays even when the schema omits an explicit `type`.
    pub(crate) fn may_describe_array(&mut self, schema: &'a Value) -> bool {
        self.may_describe_structural_type(schema, "array")
    }

    pub(crate) fn may_describe_array_constraints(&mut self, schemas: &[&'a Value]) -> bool {
        !schemas.is_empty() && schemas.iter().all(|schema| self.may_describe_array(schema))
    }

    fn may_describe_structural_type(&mut self, schema: &'a Value, expected: &str) -> bool {
        self.with_schema(schema, true, |walker, schema| {
            if schema == &Value::Bool(false) {
                return false;
            }
            if let Some(ty) = schema.get("type") {
                let matches_type = ty.as_str() == Some(expected)
                    || ty
                        .as_array()
                        .is_some_and(|types| types.iter().any(|ty| ty == expected));
                if !matches_type {
                    return false;
                }
            }
            let matches_value = |value: &Value| match expected {
                "object" => value.is_object(),
                "array" => value.is_array(),
                _ => false,
            };
            if schema
                .get("const")
                .is_some_and(|value| !matches_value(value))
                || schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(|values| !values.iter().any(matches_value))
            {
                return false;
            }
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
                && !walker.may_describe_structural_type(target, expected)
            {
                return false;
            }
            for keyword in ["allOf", "anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    let possible = if keyword == "allOf" {
                        branches
                            .iter()
                            .all(|branch| walker.may_describe_structural_type(branch, expected))
                    } else {
                        branches
                            .iter()
                            .any(|branch| walker.may_describe_structural_type(branch, expected))
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
    pub(crate) fn object_children(&mut self, schema: &'a Value, tag: &str) -> Vec<&'a Value> {
        self.with_schema(schema, Vec::new(), |walker, schema| {
            let mut children = Vec::new();
            if let Some(child) = schema.get("properties").and_then(|props| props.get(tag)) {
                children.push(child);
            } else if let Some(additional) = schema
                .get("additionalProperties")
                .filter(|value| value.is_object())
            {
                children.push(additional);
            }

            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
            {
                children.extend(walker.object_children(target, tag));
            }

            if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
                for branch in branches {
                    if walker.may_describe_object(branch) {
                        children.extend(walker.object_children(branch, tag));
                    }
                }
            }

            let branches = match (schema.get("anyOf"), schema.get("oneOf")) {
                (Some(branches), None) | (None, Some(branches)) => {
                    let Some(branches) = branches.as_array() else {
                        return children;
                    };
                    branches
                }
                _ => return children,
            };
            let mut objects = branches
                .iter()
                .filter(|branch| walker.may_describe_object(branch));
            let Some(object) = objects.next() else {
                return children;
            };
            if objects.next().is_none() {
                children.extend(walker.object_children(object, tag));
            }
            children
        })
    }

    pub(crate) fn permits_null(&mut self, schema: &'a Value) -> Option<bool> {
        let mut walker = NullSchemaWalker::new(
            self.root,
            schema,
            MAX_SCHEMA_DEPTH,
            self.remaining_work.min(MAX_NULL_SCHEMA_NODES),
        );
        let result = walker.matches(schema, 0, 0);
        self.remaining_work = walker.remaining;
        if walker.exhausted {
            self.exhausted = true;
            None
        } else {
            result
        }
    }

    #[cfg(test)]
    pub(crate) fn has_type(&mut self, schema: Option<&'a Value>, expected: &str) -> bool {
        schema.is_some_and(|schema| self.has_type_constraints(&[schema], expected))
    }

    pub(crate) fn has_type_constraints(&mut self, schemas: &[&'a Value], expected: &str) -> bool {
        let mut matched = false;
        for schema in schemas {
            match self.explicit_type_hint(schema, expected) {
                Some(false) => return false,
                Some(true) => matched = true,
                None => {}
            }
        }
        matched && !self.exhausted
    }

    // Scalar coercion uses explicit types and alternatives; literal constraints
    // and allOf do not add types. Reference/property siblings still intersect.
    fn explicit_type_hint(&mut self, schema: &'a Value, expected: &str) -> Option<bool> {
        self.with_schema(schema, None, |walker, schema| {
            if let Some(allowed) = schema.as_bool() {
                return (!allowed).then_some(false);
            }
            if !schema.is_object() {
                return None;
            }
            let mut hint = schema.get("type").map(|ty| {
                ty.as_str() == Some(expected)
                    || ty
                        .as_array()
                        .is_some_and(|types| types.iter().any(|ty| ty.as_str() == Some(expected)))
            });
            for keyword in ["anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    for branch in branches {
                        match walker.explicit_type_hint(branch, expected) {
                            Some(true) => hint = Some(true),
                            Some(false) if hint.is_none() => hint = Some(false),
                            _ => {}
                        }
                    }
                }
            }
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
                let reference_hint = resolve_local_schema_ref(reference, walker.root)
                    .and_then(|target| walker.explicit_type_hint(target, expected));
                hint = intersect_type_hints(hint, reference_hint);
            }
            hint
        })
    }

    pub(crate) fn array_items(&mut self, schema: &'a Value) -> Vec<&'a Value> {
        self.with_schema(schema, Vec::new(), |walker, schema| {
            let mut items = Vec::new();
            if let Some(item_schema) = schema.get("items") {
                items.push(item_schema);
            }
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
            {
                items.extend(walker.array_items(target));
            }
            if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
                for branch in branches {
                    items.extend(walker.array_items(branch));
                }
            }
            for key in ["anyOf", "oneOf"] {
                if let Some(options) = schema.get(key).and_then(Value::as_array) {
                    for option in options {
                        if !walker.may_describe_array(option) {
                            continue;
                        }
                        let branch_items = walker.array_items(option);
                        if !branch_items.is_empty() {
                            items.extend(branch_items);
                        }
                    }
                }
            }
            items
        })
    }

    // Array item unions are alternatives. Callers use this query when converting
    // a scalar item so one incompatible branch cannot veto a compatible branch.
    pub(crate) fn has_array_item_type(&mut self, schemas: &[&'a Value], expected: &str) -> bool {
        let mut hint = None;
        for schema in schemas {
            hint = intersect_type_hints(hint, self.array_item_type_hint(schema, expected));
            if hint == Some(false) {
                return false;
            }
        }
        hint == Some(true)
    }

    pub(crate) fn array_item_permits_null(&mut self, schemas: &[&'a Value]) -> bool {
        let mut hint = None;
        for schema in schemas {
            hint = intersect_type_hints(hint, self.array_item_null_hint(schema));
            if hint == Some(false) {
                return false;
            }
        }
        hint == Some(true)
    }

    fn array_item_null_hint(&mut self, schema: &'a Value) -> Option<bool> {
        self.with_schema(schema, None, |walker, schema| {
            let mut hint = schema
                .get("items")
                .and_then(|items| walker.permits_null(items));

            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
            {
                hint = intersect_type_hints(hint, walker.array_item_null_hint(target));
            }

            if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
                for branch in branches {
                    hint = intersect_type_hints(hint, walker.array_item_null_hint(branch));
                }
            }

            for keyword in ["anyOf", "oneOf"] {
                let Some(branches) = schema.get(keyword).and_then(Value::as_array) else {
                    continue;
                };
                let mut branch_hint = None;
                let mut unknown = false;
                for branch in branches {
                    if !walker.may_describe_array(branch) {
                        continue;
                    }
                    match walker.array_item_null_hint(branch) {
                        Some(true) => {
                            branch_hint = Some(true);
                            break;
                        }
                        Some(false) => {}
                        None => unknown = true,
                    }
                }
                let alternatives = match (branch_hint, unknown) {
                    (Some(true), _) => Some(true),
                    (None, true) => None,
                    (None, false) => Some(false),
                    (Some(false), _) => Some(false),
                };
                hint = intersect_type_hints(hint, alternatives);
            }
            hint
        })
    }

    fn array_item_type_hint(&mut self, schema: &'a Value, expected: &str) -> Option<bool> {
        self.with_schema(schema, None, |walker, schema| {
            let mut hint = schema
                .get("items")
                .and_then(|items| walker.explicit_type_hint(items, expected));

            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
            {
                hint = intersect_type_hints(hint, walker.array_item_type_hint(target, expected));
            }

            if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
                for branch in branches {
                    hint =
                        intersect_type_hints(hint, walker.array_item_type_hint(branch, expected));
                }
            }

            for keyword in ["anyOf", "oneOf"] {
                let Some(branches) = schema.get(keyword).and_then(Value::as_array) else {
                    continue;
                };
                let mut branch_hint = None;
                let mut unknown = false;
                for branch in branches {
                    match walker.array_item_type_hint(branch, expected) {
                        Some(true) => {
                            branch_hint = Some(true);
                            break;
                        }
                        Some(false) => {}
                        None => unknown = true,
                    }
                }
                let alternatives = match (branch_hint, unknown) {
                    (Some(true), _) => Some(true),
                    (None, true) => None,
                    (None, false) => Some(false),
                    (Some(false), _) => Some(false),
                };
                hint = intersect_type_hints(hint, alternatives);
            }
            hint
        })
    }
}

fn intersect_type_hints(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), _) | (_, Some(true)) => Some(true),
        _ => None,
    }
}

const MAX_NULL_SCHEMA_REF_DEPTH: usize = 16;
const MAX_NULL_SCHEMA_NODES: usize = 4096;

// Only a proven match may turn the model's text into JSON null. Unknown references
// must remain unknown through negation and oneOf, rather than counting as false.
pub(crate) fn schema_permits_null(schema: &Value, root: &Value) -> bool {
    let mut walker =
        NullSchemaWalker::new(root, schema, MAX_NULL_SCHEMA_NODES, MAX_NULL_SCHEMA_NODES);
    walker.matches(schema, 0, 0) == Some(true)
}

fn intersect_null_matches(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

pub(crate) fn has_unsupported_schema_ref_scope(schema: &Value) -> bool {
    ["$id", "$dynamicRef", "$recursiveRef"]
        .iter()
        .any(|keyword| schema.get(*keyword).and_then(Value::as_str).is_some())
}

pub(crate) fn resolve_local_schema_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
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
struct NullSchemaWalker<'a> {
    root: &'a Value,
    active_refs: Vec<&'a Value>,
    max_depth: usize,
    remaining: usize,
    exhausted: bool,
}

impl<'a> NullSchemaWalker<'a> {
    fn new(root: &'a Value, schema: &'a Value, max_depth: usize, remaining: usize) -> Self {
        Self {
            root,
            active_refs: vec![schema],
            max_depth,
            remaining,
            exhausted: false,
        }
    }

    fn matches(&mut self, schema: &'a Value, ref_depth: usize, depth: usize) -> Option<bool> {
        let Some(work) = self.remaining.checked_sub(1) else {
            self.exhausted = true;
            return None;
        };
        self.remaining = work;
        if depth > self.max_depth {
            self.exhausted = true;
            return None;
        }
        if let Some(allowed) = schema.as_bool() {
            return Some(allowed);
        }
        if !schema.is_object()
            || schema.get("$dynamicRef").is_some()
            || schema.get("$recursiveRef").is_some()
            || (schema.get("$id").is_some() && !std::ptr::eq(schema, self.root))
        {
            return None;
        }
        let mut permits = Some(true);
        if let Some(reference) = schema.get("$ref") {
            let target = reference
                .as_str()
                .and_then(|reference| resolve_local_schema_ref(reference, self.root));
            permits = if let Some(target) = target
                && ref_depth < MAX_NULL_SCHEMA_REF_DEPTH
                && !self
                    .active_refs
                    .iter()
                    .any(|active| std::ptr::eq(*active, target))
            {
                self.active_refs.push(target);
                let result = self.matches(target, ref_depth + 1, depth + 1);
                self.active_refs.pop();
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
        for keyword in ["allOf", "anyOf", "oneOf"] {
            if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                let mut matches = 0;
                let mut unknown = 0;
                for branch in branches {
                    match self.matches(branch, ref_depth, depth + 1) {
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
            let not_match = self.matches(not, ref_depth, depth + 1);
            permits = intersect_null_matches(permits, not_match.map(|matches| !matches));
        }
        permits
    }
}

pub(crate) fn schema_has_type(root: &Value, schema: &Value, expected: &str, raw: &str) -> bool {
    let mut remaining = 1024;
    let matched = schema_type_match(root, schema, expected, raw, 0, &mut remaining);
    remaining > 0 && matched == Some(true)
}

// None means no type hint: e.g. minLength alone must not exclude an allOf sibling's type.
pub(crate) fn schema_type_match(
    root: &Value,
    schema: &Value,
    expected: &str,
    raw: &str,
    depth: usize,
    remaining: &mut usize,
) -> Option<bool> {
    // A branching reference cycle can expand exponentially even at bounded depth.
    *remaining = remaining.checked_sub(1)?;
    // Local references are common in strict tool schemas. Limit traversal so a
    // cyclic definition cannot recurse indefinitely while deciding a type hint.
    if depth > 16
        || schema.get("$dynamicRef").is_some()
        || schema.get("$recursiveRef").is_some()
        || (schema.get("$id").is_some() && !std::ptr::eq(schema, root))
    {
        return None;
    }
    let reference_hint = schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| resolve_local_schema_ref(reference, root))
        .and_then(|target| schema_type_match(root, target, expected, raw, depth + 1, remaining));
    let matches = |ty: &Value| {
        ty.as_str() == Some(expected) || (expected == "integer" && ty.as_str() == Some("number"))
    };
    let mut hint = schema.get("type").map(|ty| {
        ty.as_array()
            .map_or_else(|| matches(ty), |types| types.iter().any(matches))
    });
    // A string constant preserves literal XML text only when this value
    // matches it. Otherwise a union's numeric/structured alternative must
    // remain available. Intersect with types, refs, and composition siblings.
    if expected == "string"
        && let Some(value) = schema.get("const")
    {
        hint = Some(hint.unwrap_or(true) && value.as_str() == Some(raw));
    }
    // Modern JSON Schema applies $ref siblings as additional constraints.
    hint = match (hint, reference_hint) {
        (Some(left), Some(right)) => Some(left && right),
        (left, right) => left.or(right),
    };
    for keyword in ["anyOf", "oneOf", "allOf"] {
        let Some(options) = schema.get(keyword).and_then(Value::as_array) else {
            continue;
        };
        let branches = options
            .iter()
            .map(|option| schema_type_match(root, option, expected, raw, depth, remaining));
        let branch_hint = if keyword == "allOf" {
            branches.flatten().reduce(|left, right| left && right)
        } else {
            branches
                .reduce(|left, right| match (left, right) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                })
                .flatten()
        };
        hint = match (hint, branch_hint) {
            (Some(left), Some(right)) => Some(left && right),
            (left, right) => left.or(right),
        };
    }
    hint
}

#[cfg(test)]
mod structural_type_tests {
    use super::SchemaFacts;
    use serde_json::json;

    #[test]
    fn referenced_type_intersects_structural_siblings() {
        let schema = json!({
            "$defs": {
                "Text": {"type": "string"},
                "Array": {"type": "array"}
            },
            "$ref": "#/$defs/Text",
            "items": {"type": "number"}
        });

        assert!(!SchemaFacts::new(&schema).may_describe_array(&schema));

        let valid = json!({
            "$defs": {"Array": {"type": "array"}},
            "$ref": "#/$defs/Array",
            "items": {"type": "number"}
        });
        assert!(SchemaFacts::new(&valid).may_describe_array(&valid));

        let unresolved = json!({
            "$ref": "#/$defs/Missing",
            "items": {"type": "number"}
        });
        assert!(SchemaFacts::new(&unresolved).may_describe_array(&unresolved));
    }
}
