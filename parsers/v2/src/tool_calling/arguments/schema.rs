// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;
pub(crate) mod types;

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
                        "$ref"
                            | "$defs"
                            | "definitions"
                            | "title"
                            | "description"
                            | "default"
                            | "examples"
                            | "$comment"
                    )
                })
            }) || visited.contains(&reference)
            {
                return Some(schema);
            }
            let Some(target) = resolve_local_schema_ref(reference, self.root) else {
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
            if (schema
                .get("const")
                .is_some_and(|value| !matches_value(value))
                || schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(|values| !values.iter().any(matches_value)))
                && !literals_contradict_type(schema)
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

    // Tool parameters may be declared at the root, under allOf, or under anyOf/oneOf
    // variants. Root and allOf declarations always apply, so they win over variant
    // declarations; a parameter declared only by several variants is typed by their union.
    pub(crate) fn parameter_schemas(&mut self, tag: &str) -> (Vec<&'a Value>, Vec<&'a Value>) {
        let mut required = Vec::new();
        let mut alternatives = Vec::new();
        self.collect_parameter_schemas(self.root, tag, false, &mut required, &mut alternatives);
        if self.exhausted {
            (Vec::new(), Vec::new())
        } else {
            (required, alternatives)
        }
    }

    fn collect_parameter_schemas(
        &mut self,
        schema: &'a Value,
        tag: &str,
        in_variant: bool,
        required: &mut Vec<&'a Value>,
        alternatives: &mut Vec<&'a Value>,
    ) {
        self.with_schema(schema, (), |walker, schema| {
            if let Some(child) = schema.get("properties").and_then(|props| props.get(tag)) {
                let declared = if in_variant {
                    &mut *alternatives
                } else {
                    &mut *required
                };
                if !declared.contains(&child) {
                    declared.push(child);
                }
            }
            for keyword in ["allOf", "anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    let in_variant = in_variant || keyword != "allOf";
                    for branch in branches {
                        walker.collect_parameter_schemas(
                            branch,
                            tag,
                            in_variant,
                            required,
                            alternatives,
                        );
                    }
                }
            }
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
        let result = walker.matches(schema, 0, 0, false);
        self.remaining_work = walker.remaining;
        if walker.exhausted {
            self.exhausted = true;
            None
        } else {
            result
        }
    }

    // Infers types from `const`/`enum` and intersects `allOf`, `not`, and `$ref` siblings.
    pub(crate) fn has_type(&mut self, schema: Option<&'a Value>, expected: &str) -> bool {
        schema.is_some_and(|schema| self.type_constraint(schema, expected, true) == Some(true))
    }

    // A string literal branch such as `const: "auto"` must not hide a numeric
    // alternative when the input is "42". Only reject known type/literal
    // contradictions here; this coercion helper is not a schema validator.
    pub(crate) fn rejects_string_literal(
        &mut self,
        schema: &'a Value,
        raw: &str,
        match_enum: bool,
    ) -> bool {
        self.with_schema(schema, false, |walker, schema| {
            if schema == &Value::Bool(false) {
                return true;
            }
            let literal_rejects = schema
                .get("const")
                .is_some_and(|value| value.as_str() != Some(raw))
                || match_enum
                    && schema
                        .get("enum")
                        .and_then(Value::as_array)
                        .is_some_and(|values| {
                            !values.iter().any(|value| value.as_str() == Some(raw))
                        });
            if literal_rejects && !literals_contradict_type(schema) {
                return true;
            }
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
                && walker.rejects_string_literal(target, raw, match_enum)
            {
                return true;
            }
            for keyword in ["allOf", "anyOf", "oneOf"] {
                if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
                    let rejected = if keyword == "allOf" {
                        branches
                            .iter()
                            .any(|branch| walker.rejects_string_literal(branch, raw, match_enum))
                    } else {
                        branches
                            .iter()
                            .all(|branch| walker.rejects_string_literal(branch, raw, match_enum))
                    };
                    if rejected {
                        return true;
                    }
                }
            }
            false
        })
    }

    // None means no known type constraint; false means this type is excluded.
    // Sibling keywords and allOf intersect, while anyOf/oneOf combine alternatives.
    pub(crate) fn type_constraint(
        &mut self,
        schema: &'a Value,
        expected: &str,
        infer_literals: bool,
    ) -> Option<bool> {
        let result = types::type_hint(
            self.root,
            schema,
            expected,
            if infer_literals {
                types::LiteralTypes::All
            } else {
                types::LiteralTypes::None
            },
            &mut self.remaining_work,
        );
        self.exhausted |= self.remaining_work == 0;
        result
    }

    // Reads only explicit `type` names, so an alias such as `type: "enum"` is visible.
    pub(crate) fn has_explicit_type(&mut self, schemas: &[&'a Value], expected: &str) -> bool {
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
            hint = intersect_item_null(hint, self.array_item_null_hint(schema));
            if hint == Some(Some(false)) {
                return false;
            }
        }
        hint == Some(Some(true))
    }

    // None when no item constraint applies; Some(None) when one applies but its
    // null answer is unknown, which an applicable sibling's proof cannot override.
    fn array_item_null_hint(&mut self, schema: &'a Value) -> Option<Option<bool>> {
        self.with_schema(schema, None, |walker, schema| {
            let mut hint = schema.get("items").map(|items| walker.permits_null(items));

            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = resolve_local_schema_ref(reference, walker.root)
            {
                hint = intersect_item_null(hint, walker.array_item_null_hint(target));
            }

            if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
                for branch in branches {
                    hint = intersect_item_null(hint, walker.array_item_null_hint(branch));
                }
            }

            for keyword in ["anyOf", "oneOf"] {
                let Some(branches) = schema.get(keyword).and_then(Value::as_array) else {
                    continue;
                };
                let mut proven = false;
                let mut unconstrained = false;
                let mut unknown = false;
                for branch in branches {
                    if !walker.may_describe_array(branch) {
                        continue;
                    }
                    match walker.array_item_null_hint(branch) {
                        Some(Some(true)) => {
                            proven = true;
                            break;
                        }
                        Some(Some(false)) => {}
                        Some(None) => unknown = true,
                        None => unconstrained = true,
                    }
                }
                // An alternative without an item constraint leaves items free, so
                // the union adds no constraint; an unknown alternative stays unknown.
                let alternatives = if proven {
                    Some(Some(true))
                } else if unconstrained {
                    None
                } else if unknown {
                    Some(None)
                } else {
                    Some(Some(false))
                };
                hint = intersect_item_null(hint, alternatives);
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

const JSON_TYPES: [&str; 7] = [
    "string", "number", "integer", "boolean", "object", "array", "null",
];

fn literal_satisfies(name: &str, value: &Value) -> bool {
    match name {
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || value.as_f64().is_some_and(|number| number.fract() == 0.0)
        }
        _ => false,
    }
}

// True when a node declares only JSON types and no `enum`/`const` value is an
// instance of any of them, as when an array's `enum` is written beside `type`
// instead of under `items`. Typing then follows the declared `type`, as stock
// SGLang does; literals that fit the declared type still narrow it.
pub(crate) fn literals_contradict_type(schema: &Value) -> bool {
    let enumerated = schema.get("enum").and_then(Value::as_array);
    let constant = schema.get("const");
    if enumerated.is_none() && constant.is_none() {
        return false;
    }
    let declared = match schema.get("type") {
        Some(name @ Value::String(_)) => std::slice::from_ref(name),
        Some(Value::Array(names)) if !names.is_empty() => names.as_slice(),
        _ => return false,
    };
    // A custom type name may admit any literal, so it never contradicts one.
    if !declared
        .iter()
        .all(|name| name.as_str().is_some_and(|name| JSON_TYPES.contains(&name)))
    {
        return false;
    }
    !constant
        .into_iter()
        .chain(enumerated.into_iter().flatten())
        .any(|value| {
            declared
                .iter()
                .filter_map(Value::as_str)
                .any(|name| literal_satisfies(name, value))
        })
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
    walker.matches(schema, 0, 0, false) == Some(true)
}

fn intersect_item_null(
    left: Option<Option<bool>>,
    right: Option<Option<bool>>,
) -> Option<Option<bool>> {
    match (left, right) {
        (None, other) | (other, None) => other,
        (Some(left), Some(right)) => Some(intersect_null_matches(left, right)),
    }
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

    // `scoped` marks a node inside a nested `$id` resource in this walk.
    fn matches(
        &mut self,
        schema: &'a Value,
        ref_depth: usize,
        depth: usize,
        scoped: bool,
    ) -> Option<bool> {
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
        {
            return None;
        }
        // A nested `$id` rebases the references inside it, not its other keywords:
        // only a reference within the resource stays unknown.
        let scoped = scoped || (schema.get("$id").is_some() && !std::ptr::eq(schema, self.root));
        let mut permits = Some(true);
        if let Some(reference) = schema.get("$ref") {
            let target = reference
                .as_str()
                .filter(|_| !scoped)
                .and_then(|reference| resolve_local_schema_ref(reference, self.root));
            permits = if let Some(target) = target
                && ref_depth < MAX_NULL_SCHEMA_REF_DEPTH
                && !self
                    .active_refs
                    .iter()
                    .any(|active| std::ptr::eq(*active, target))
            {
                self.active_refs.push(target);
                let result = self.matches(target, ref_depth + 1, depth + 1, false);
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
        // Unlike typing, null still intersects literals that contradict `type`:
        // only a proven match may produce JSON null, and GLM-4.7 shares this walker.
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
                    match self.matches(branch, ref_depth, depth + 1, scoped) {
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
            let not_match = self.matches(not, ref_depth, depth + 1, scoped);
            permits = intersect_null_matches(permits, not_match.map(|matches| !matches));
        }
        if let Some(condition) = schema.get("if") {
            let condition = self.matches(condition, ref_depth, depth + 1, scoped);
            let mut branch = |key| {
                schema.get(key).map_or(Some(true), |branch| {
                    self.matches(branch, ref_depth, depth + 1, scoped)
                })
            };
            let outcome = match condition {
                Some(true) => branch("then"),
                Some(false) => branch("else"),
                None => {
                    let left = branch("then");
                    let right = branch("else");
                    if left == right { left } else { None }
                }
            };
            permits = intersect_null_matches(permits, outcome);
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
    if depth > 16 {
        return None;
    }
    let hint = types::type_hint(
        root,
        schema,
        expected,
        if expected == "string" {
            types::LiteralTypes::StringHint(raw)
        } else {
            types::LiteralTypes::ScalarHints
        },
        remaining,
    );
    if types::categorize_type(expected) == Some(types::SchemaType::Number)
        && super::coerce_integral_number(raw.trim()).is_some()
    {
        return types::type_hint(
            root,
            schema,
            "integer",
            types::LiteralTypes::ScalarHints,
            remaining,
        );
    }
    hint
}

#[cfg(test)]
mod structural_type_tests {
    use super::{SchemaFacts, literals_contradict_type};
    use serde_json::json;

    #[test]
    fn literals_contradict_only_declared_json_types() {
        for (schema, contradicts) in [
            (json!({"type": "array", "enum": ["a", "b"]}), true),
            (json!({"type": "integer", "const": "auto"}), true),
            (json!({"type": "integer", "enum": []}), true),
            (json!({"type": "string", "enum": ["a", 1]}), false),
            (json!({"type": "custom", "enum": []}), false),
            (json!({"type": ["integer", 5], "enum": ["a"]}), false),
            (json!({"enum": ["a"]}), false),
        ] {
            assert_eq!(literals_contradict_type(&schema), contradicts, "{schema}");
        }
    }

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

    #[test]
    fn nested_ids_scope_only_their_references() {
        let id = "https://example.com/v";
        let root = json!({"$defs": {
            "Nullable": {"type": ["integer", "null"]},
            "Scoped": {"$id": "https://example.com/t", "type": ["integer", "null"]},
        }});
        for (schema, permits) in [
            (json!({"$id": id, "type": ["integer", "null"]}), Some(true)),
            (json!({"$id": id, "type": "integer"}), Some(false)),
            (json!({"$id": id, "$ref": "#/$defs/Nullable"}), None),
            (
                json!({"$id": id, "anyOf": [{"$ref": "#/$defs/Nullable"}]}),
                None,
            ),
            (
                json!({"$id": id, "not": {"$ref": "#/$defs/Nullable"}}),
                None,
            ),
            (json!({"$ref": "#/$defs/Scoped"}), Some(true)),
            (json!({"$ref": "#/$defs/Nullable"}), Some(true)),
            (json!({"$dynamicRef": "#/$defs/Nullable"}), None),
            (json!(true), Some(true)),
            (json!(false), Some(false)),
        ] {
            assert_eq!(
                SchemaFacts::new(&root).permits_null(&schema),
                permits,
                "{schema}"
            );
        }
    }

    #[test]
    fn array_item_null_needs_every_applicable_constraint() {
        let nullable =
            json!({"items": {"$id": "https://example.com/i", "type": ["string", "null"]}});
        let unknown = json!({"items": {"$dynamicRef": "#meta"}});
        let proven = json!({"type": "array", "allOf": [nullable.clone()]});
        let unproven = json!({"type": "array", "allOf": [nullable, unknown]});
        assert!(SchemaFacts::new(&proven).array_item_permits_null(&[&proven]));
        assert!(!SchemaFacts::new(&unproven).array_item_permits_null(&[&unproven]));

        // A union alternative without an item constraint adds none; an unknown
        // alternative keeps the union unknown.
        let items = json!({"type": ["integer", "null"]});
        for (schema, permits) in [
            (
                json!({"type": "array", "items": items, "anyOf": [{"type": "array"}]}),
                true,
            ),
            (
                json!({"type": "array", "items": items, "oneOf": [{"type": "array"}]}),
                true,
            ),
            (
                json!({"type": "array", "anyOf": [{"type": "array"}]}),
                false,
            ),
            (
                json!({"type": "array", "items": items, "anyOf": [{"items": {"$dynamicRef": "#m"}}]}),
                false,
            ),
        ] {
            assert_eq!(
                SchemaFacts::new(&schema).array_item_permits_null(&[&schema]),
                permits,
                "{schema}"
            );
        }
    }
}
