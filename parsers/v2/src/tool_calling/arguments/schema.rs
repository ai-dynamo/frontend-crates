// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    // The schema the current top-level query started at, and whether it lies
    // inside a nested `$id` resource of the root, found on first use.
    entry: Option<&'a Value>,
    entry_scope: Option<(&'a Value, bool)>,
}

impl<'a> SchemaFacts<'a> {
    pub(crate) fn new(root: &'a Value) -> Self {
        Self {
            root,
            path: Vec::new(),
            remaining_work: MAX_SCHEMA_WORK,
            exhausted: false,
            entry: None,
            entry_scope: None,
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
            let Some(target) = self.resolve_scoped_ref(reference) else {
                return Some(schema);
            };
            visited.push(reference);
            current = target;
        }
        Some(current)
    }

    // A nested `$id` rebases the references inside it, not its other keywords.
    // When this walk started inside or passed through such a resource, leave its
    // references unknown rather than resolving them against the root.
    fn resolve_scoped_ref(&mut self, reference: &str) -> Option<&'a Value> {
        let scoped = self
            .path
            .iter()
            .any(|node| !std::ptr::eq(*node, self.root) && node.get("$id").is_some())
            || self.entry_in_nested_resource();
        if scoped {
            return None;
        }
        resolve_local_schema_ref(reference, self.root)
    }

    // A walk that starts below the root, such as one for a nested XML child,
    // cannot see the resources enclosing its entry schema on its path.
    fn entry_in_nested_resource(&mut self) -> bool {
        let Some(entry) = self.entry else {
            return false;
        };
        match self.entry_scope {
            Some((cached, nested)) if std::ptr::eq(cached, entry) => nested,
            _ => {
                let nested = within_nested_resource(self.root, entry);
                self.entry_scope = Some((entry, nested));
                nested
            }
        }
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
        if self.path.is_empty() {
            self.entry = Some(schema);
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
                && let Some(target) = walker.resolve_scoped_ref(reference)
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
                && let Some(target) = walker.resolve_scoped_ref(reference)
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
        let nested = within_nested_resource(self.root, schema);
        let result = walker.matches(schema, 0, 0, nested);
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

    // The shared policy for native text whose wire format cannot mark strings.
    // A JSON value proven to fit the schema wins; otherwise text the schema
    // admits as a string stays verbatim; otherwise a typed JSON value whose fit
    // is unproven wins over no answer. `007` and `1e3` are not JSON integers, so
    // they stay text under `["string", "integer"]`. None leaves the choice to
    // the caller's conversions.
    pub(crate) fn native_text_type(&self, schema: &'a Value, raw: &str) -> Option<NativeText> {
        let trimmed = raw.trim();
        // Strings and null cannot become `Json`; only these bytes start other
        // values. A container is decoded only when the schema can type it, so a
        // string-only parameter never builds a JSON tree from its text.
        let value = match trimmed.as_bytes().first() {
            Some(b'{') if !self.typed(schema, "object")? => None,
            Some(b'[') if !self.typed(schema, "array")? => None,
            Some(b'{' | b'[' | b't' | b'f' | b'-' | b'0'..=b'9') => {
                serde_json::from_str::<Value>(trimmed).ok()
            }
            _ => None,
        };
        let mut unproven = None;
        if let Some(value) = &value
            && let Some(kind) = json_kind(value)
            && self.typed(schema, kind)?
        {
            match self.fit(schema, Candidate::Json(value)) {
                Fit::Fits => return Some(NativeText::Json(kind)),
                Fit::Unproven => unproven = Some(NativeText::Json(kind)),
                Fit::Rejects => {}
            }
        }
        if self.typed(schema, "string")? && self.fit(schema, Candidate::Text(raw)) != Fit::Rejects {
            return Some(NativeText::String);
        }
        unproven
    }

    // Each walk gets its own work budget, so typing one value costs no more
    // capacity than a single query. Exhaustion leaves the answer unknown.
    fn typed(&self, schema: &'a Value, kind: &str) -> Option<bool> {
        let mut facts = SchemaFacts::new(self.root);
        let typed = facts.has_type(Some(schema), kind);
        (!facts.exhausted).then_some(typed)
    }

    // An unfinished walk proves nothing, so the candidate stays Unproven: an
    // admitted string then wins over a JSON value it could not decide.
    fn fit(&self, schema: &'a Value, candidate: Candidate<'_>) -> Fit {
        let mut facts = SchemaFacts::new(self.root);
        let fit = facts.value_fit(schema, candidate);
        if facts.exhausted { Fit::Unproven } else { fit }
    }

    // A literal branch such as `const: "auto"` must not hide an alternative that
    // admits the value, such as 42. `Fits` is a proof, so `oneOf` and `not` are
    // decided exactly; this is not a validator, so any keyword it does not
    // evaluate leaves the candidate `Unproven`.
    pub(crate) fn value_fit(&mut self, schema: &'a Value, candidate: Candidate<'_>) -> Fit {
        self.with_schema(schema, Fit::Unproven, |walker, schema| {
            let Some(node) = schema.as_object() else {
                return match schema.as_bool() {
                    Some(true) => Fit::Fits,
                    Some(false) => Fit::Rejects,
                    None => Fit::Unproven,
                };
            };
            if node.get("type").is_some_and(|ty| {
                !ty.as_str().map_or_else(
                    || {
                        ty.as_array().is_some_and(|types| {
                            types
                                .iter()
                                .filter_map(Value::as_str)
                                .any(|ty| candidate.is(ty))
                        })
                    },
                    |ty| candidate.is(ty),
                )
            }) {
                return Fit::Rejects;
            }
            let mut fit = Fit::Fits;
            let constant = node.get("const").map(|literal| candidate.equals(literal));
            let enumerated = node
                .get("enum")
                .and_then(Value::as_array)
                .map(|literals| candidate.equals_any(literals));
            for matched in [constant, enumerated].into_iter().flatten() {
                match matched {
                    Some(true) => {}
                    // Literals that contradict the declared type do not decide it.
                    Some(false) if !literals_contradict_type(schema) => return Fit::Rejects,
                    Some(false) | None => fit = Fit::Unproven,
                }
            }
            if UNEVALUATED_FOR_ANY_KIND
                .iter()
                .chain(candidate.unevaluated_keywords())
                .any(|keyword| node.contains_key(*keyword))
            {
                fit = Fit::Unproven;
            }
            if let Some(reference) = node.get("$ref") {
                fit = fit.and(
                    match reference
                        .as_str()
                        .and_then(|reference| walker.resolve_scoped_ref(reference))
                    {
                        Some(target) => walker.value_fit(target, candidate),
                        None => Fit::Unproven,
                    },
                );
            }
            if let Some(branches) = node.get("allOf").and_then(Value::as_array) {
                for branch in branches {
                    fit = fit.and(walker.value_fit(branch, candidate));
                    if fit == Fit::Rejects {
                        return fit;
                    }
                }
            }
            if let Some(branches) = node.get("anyOf").and_then(Value::as_array) {
                let mut any = Fit::Rejects;
                for branch in branches {
                    any = any.max(walker.value_fit(branch, candidate));
                    if any == Fit::Fits {
                        break;
                    }
                }
                fit = fit.and(any);
            }
            if let Some(branches) = node.get("oneOf").and_then(Value::as_array) {
                let (mut fits, mut unproven) = (0, 0);
                for branch in branches {
                    match walker.value_fit(branch, candidate) {
                        Fit::Fits => fits += 1,
                        Fit::Unproven => unproven += 1,
                        Fit::Rejects => {}
                    }
                    if fits > 1 {
                        break;
                    }
                }
                fit = fit.and(match (fits, unproven) {
                    (1, 0) => Fit::Fits,
                    (0, 0) | (2.., _) => Fit::Rejects,
                    _ => Fit::Unproven,
                });
            }
            if let Some(negated) = node.get("not") {
                fit = fit.and(match walker.value_fit(negated, candidate) {
                    Fit::Fits => Fit::Rejects,
                    Fit::Rejects => Fit::Fits,
                    Fit::Unproven => Fit::Unproven,
                });
            }
            fit
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
        self.with_schema(schema, None, |walker, schema| {
            if schema == &Value::Bool(false) {
                return Some(false);
            }
            let matches =
                |name: &str| name == expected || (name == "number" && expected == "integer");
            let mut constraint = schema.get("type").and_then(|ty| {
                ty.as_str().map(matches).or_else(|| {
                    ty.as_array()
                        .map(|types| types.iter().filter_map(Value::as_str).any(matches))
                })
            });
            // A reference and its sibling constraints apply together. Keep the
            // sibling node intact and intersect the target's facts with it.
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
                && let Some(target) = walker.resolve_scoped_ref(reference)
                && let Some(permits) = walker.type_constraint(target, expected, infer_literals)
            {
                constraint = Some(constraint.unwrap_or(true) && permits);
            }
            // A plain negated type can exclude strings without declaring a
            // positive type. More complex negations do not supply a safe hint.
            if expected != "enum"
                && let Some(negated) = schema.get("not").and_then(Value::as_object)
                && negated.keys().all(|key| {
                    matches!(
                        key.as_str(),
                        "type" | "title" | "description" | "default" | "examples" | "$comment"
                    )
                })
                && let Some(ty) = negated.get("type")
                && let Some(excluded) = ty.as_str().map(matches).or_else(|| {
                    ty.as_array()
                        .map(|types| types.iter().filter_map(Value::as_str).any(matches))
                })
            {
                constraint = Some(constraint.unwrap_or(true) && !excluded);
            }
            // Literal constraints can establish a type even without `type`.
            // Intersect them with explicit types so an enum cannot widen a
            // sibling constraint, unless no literal fits the declared type at
            // all. Ambiguous unions still preserve strings.
            let literal_matches = |value: &Value| literal_satisfies(expected, value);
            let own_literals = infer_literals && !literals_contradict_type(schema);
            if own_literals && let Some(value) = schema.get("const") {
                constraint = Some(constraint.unwrap_or(true) && literal_matches(value));
            }
            if own_literals && let Some(values) = schema.get("enum").and_then(Value::as_array) {
                constraint = Some(constraint.unwrap_or(true) && values.iter().any(literal_matches));
            }
            for key in ["anyOf", "oneOf"] {
                if let Some(options) = schema.get(key).and_then(Value::as_array) {
                    // Branches without known type constraints supply no hint.
                    if let Some(permits) = options
                        .iter()
                        .filter_map(|option| {
                            walker.type_constraint(option, expected, infer_literals)
                        })
                        .reduce(|left, right| left || right)
                    {
                        constraint = Some(constraint.unwrap_or(true) && permits);
                    }
                }
            }
            if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
                for branch in branches {
                    if let Some(permits) = walker.type_constraint(branch, expected, infer_literals)
                    {
                        constraint = Some(constraint.unwrap_or(true) && permits);
                    }
                }
            }
            constraint
        })
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
                let reference_hint = walker
                    .resolve_scoped_ref(reference)
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
                && let Some(target) = walker.resolve_scoped_ref(reference)
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
                && let Some(target) = walker.resolve_scoped_ref(reference)
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
                && let Some(target) = walker.resolve_scoped_ref(reference)
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

// How the shared policy types native text; see `SchemaFacts::native_text_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeText {
    // Valid JSON for an admitted non-string value of this JSON Schema type.
    Json(&'static str),
    // Text the schema admits as a string, kept verbatim.
    String,
}

// What the evaluated keywords prove about one candidate; ordered so that
// constraints that all apply take the minimum and alternatives the maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Fit {
    Rejects,
    Unproven,
    Fits,
}

impl Fit {
    fn and(self, other: Self) -> Self {
        self.min(other)
    }
}

// A typing candidate: the verbatim text, or the JSON value it decodes to.
#[derive(Clone, Copy)]
pub(crate) enum Candidate<'v> {
    Text(&'v str),
    Json(&'v Value),
}

impl Candidate<'_> {
    fn is(self, name: &str) -> bool {
        match self {
            Self::Text(_) => name == "string",
            Self::Json(value) => literal_satisfies(name, value),
        }
    }

    // None when an equality cannot be decided exactly.
    fn equals(self, literal: &Value) -> Option<bool> {
        match self {
            Self::Text(text) => Some(literal.as_str() == Some(text)),
            Self::Json(value) => json_equal(literal, value),
        }
    }

    // Proven equal to one literal, proven unequal to all, or unknown.
    fn equals_any(self, literals: &[Value]) -> Option<bool> {
        let mut decided = true;
        for literal in literals {
            match self.equals(literal) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => decided = false,
            }
        }
        decided.then_some(false)
    }

    // Keywords that can reject this candidate's kind and that `value_fit` does
    // not evaluate. Keywords for other kinds and annotations never apply.
    fn unevaluated_keywords(self) -> &'static [&'static str] {
        match self {
            Self::Text(_) | Self::Json(Value::String(_)) => &[
                "minLength",
                "maxLength",
                "pattern",
                "format",
                "contentEncoding",
                "contentMediaType",
                "contentSchema",
            ],
            Self::Json(Value::Number(_)) => &[
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
            ],
            Self::Json(Value::Object(_)) => &[
                "properties",
                "required",
                "additionalProperties",
                "patternProperties",
                "propertyNames",
                "minProperties",
                "maxProperties",
                "dependentRequired",
                "dependentSchemas",
                "dependencies",
                "unevaluatedProperties",
            ],
            Self::Json(Value::Array(_)) => &[
                "items",
                "prefixItems",
                "additionalItems",
                "contains",
                "minContains",
                "maxContains",
                "minItems",
                "maxItems",
                "uniqueItems",
                "unevaluatedItems",
            ],
            Self::Json(Value::Bool(_) | Value::Null) => &[],
        }
    }
}

// Keywords `value_fit` does not evaluate that can reject a value of any kind.
const UNEVALUATED_FOR_ANY_KIND: [&str; 3] = ["if", "$dynamicRef", "$recursiveRef"];

// Integer kind requires an integer spelling; `1.0` and integers beyond 64 bits
// are numbers here, so their existing typed paths stay unchanged.
fn json_kind(value: &Value) -> Option<&'static str> {
    match value {
        Value::Bool(_) => Some("boolean"),
        Value::Number(number) if number.is_i64() || number.is_u64() => Some("integer"),
        Value::Number(_) => Some("number"),
        Value::Array(_) => Some("array"),
        Value::Object(_) => Some("object"),
        Value::Null | Value::String(_) => None,
    }
}

// JSON Schema compares literals by value and type: 1 equals 1.0, "1" does not
// equal 1. Integers compare exactly and floats as f64, the way JSON readers
// decode them; an integer is never rounded to compare with a float. None when
// the comparison cannot be decided exactly.
fn json_equal(left: &Value, right: &Value) -> Option<bool> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => numbers_equal(left, right),
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                return Some(false);
            }
            all_equal(left.iter().zip(right))
        }
        (Value::Object(left), Value::Object(right)) => {
            if left.len() != right.len() || !left.keys().all(|key| right.contains_key(key)) {
                return Some(false);
            }
            all_equal(left.iter().map(|(key, left)| (left, &right[key])))
        }
        _ => Some(left == right),
    }
}

// A proven difference decides; otherwise any undecided pair leaves it unknown.
fn all_equal<'v>(pairs: impl Iterator<Item = (&'v Value, &'v Value)>) -> Option<bool> {
    let mut decided = true;
    for (left, right) in pairs {
        match json_equal(left, right) {
            Some(false) => return Some(false),
            Some(true) => {}
            None => decided = false,
        }
    }
    decided.then_some(true)
}

fn numbers_equal(left: &serde_json::Number, right: &serde_json::Number) -> Option<bool> {
    match (left.is_f64(), right.is_f64()) {
        (true, true) => Some(left.as_f64() == right.as_f64()),
        (false, false) => match (exact_integer(left), exact_integer(right)) {
            (Some(left), Some(right)) => Some(left == right),
            // Beyond i128, an integer has a single JSON spelling. With arbitrary
            // precision, other numbers beyond f64, such as `1e400`, stay undecided.
            _ if integer_spelling(left) && integer_spelling(right) => Some(left == right),
            _ => None,
        },
        _ => {
            let (integer, float) = if left.is_f64() {
                (right, left)
            } else {
                (left, right)
            };
            let float = float.as_f64()?;
            if float.fract() != 0.0 {
                return Some(false);
            }
            // An integral float outside [-2^127, 2^127) equals no i128.
            exact_integer(integer).map(|integer| {
                (i128::MIN as f64..i128::MAX as f64).contains(&float) && integer == float as i128
            })
        }
    }
}

// The exact value of a JSON integer. With arbitrary precision, an integer
// beyond 64 bits is parsed from its spelling; None beyond i128.
fn exact_integer(number: &serde_json::Number) -> Option<i128> {
    number
        .as_i64()
        .map(i128::from)
        .or_else(|| number.as_u64().map(i128::from))
        .or_else(|| number.to_string().parse().ok())
}

fn integer_spelling(number: &serde_json::Number) -> bool {
    let spelling = number.to_string();
    let digits = spelling.strip_prefix('-').unwrap_or(&spelling);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

// Whether `target` lies inside a nested `$id` resource of `root`, searching
// schema positions only. An unfinished search counts as nested, which leaves
// references unknown rather than resolving them against the wrong resource.
// A schema a caller composed from copies, such as the union of several
// variant declarations, is not found by identity: it counts as nested when the
// root has any nested resource.
fn within_nested_resource(root: &Value, target: &Value) -> bool {
    fn search(
        root: &Value,
        schema: &Value,
        target: &Value,
        nested: bool,
        budget: &mut usize,
        any_nested: &mut bool,
    ) -> Option<bool> {
        if std::ptr::eq(schema, target) {
            return Some(nested);
        }
        let node = schema.as_object()?;
        if *budget == 0 {
            return Some(true);
        }
        *budget -= 1;
        let resource =
            !std::ptr::eq(schema, root) && node.get("$id").and_then(Value::as_str).is_some();
        *any_nested |= resource;
        let nested = nested || resource;
        let mut visit = |child: &Value| search(root, child, target, nested, budget, any_nested);
        node.iter()
            .find_map(|(keyword, value)| match (keyword.as_str(), value) {
                (
                    "properties" | "patternProperties" | "$defs" | "definitions"
                    | "dependentSchemas",
                    Value::Object(children),
                ) => children.values().find_map(&mut visit),
                ("allOf" | "anyOf" | "oneOf" | "prefixItems" | "items", Value::Array(children)) => {
                    children.iter().find_map(&mut visit)
                }
                (
                    "items"
                    | "additionalItems"
                    | "additionalProperties"
                    | "not"
                    | "if"
                    | "then"
                    | "else"
                    | "contains"
                    | "propertyNames"
                    | "unevaluatedItems"
                    | "unevaluatedProperties",
                    child,
                ) => visit(child),
                _ => None,
            })
    }
    let (mut budget, mut any_nested) = (MAX_SCOPE_SEARCH_NODES, false);
    search(root, root, target, false, &mut budget, &mut any_nested).unwrap_or(any_nested)
}

const MAX_SCOPE_SEARCH_NODES: usize = 4096;

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
        permits
    }
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
            // Each case is a parameter of the root, as parsers pass them.
            let root = json!({
                "$defs": {
                    "Nullable": {"type": ["integer", "null"]},
                    "Scoped": {"$id": "https://example.com/t", "type": ["integer", "null"]},
                },
                "properties": {"case": schema},
            });
            assert_eq!(
                SchemaFacts::new(&root).permits_null(&root["properties"]["case"]),
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

#[cfg(test)]
mod native_text_tests {
    use super::{NativeText, SchemaFacts, json_equal};
    use serde_json::{Value, json};

    fn typed(parameter: Value, raw: &str) -> Option<NativeText> {
        let schema = json!({"type": "object", "properties": {"value": parameter}});
        SchemaFacts::new(&schema).native_text_type(&schema["properties"]["value"], raw)
    }

    #[test]
    fn json_wins_only_when_its_fit_is_proven() {
        let integer = Some(NativeText::Json("integer"));
        let string = Some(NativeText::String);
        for (parameter, raw, expected) in [
            // `oneOf` admits a value matching exactly one branch.
            (
                json!({"oneOf": [{"type": ["string", "integer"]}, {"type": "integer"}]}),
                "42",
                string,
            ),
            (
                json!({"oneOf": [{"type": ["string", "integer"]}, {"const": 42}]}),
                "42",
                string,
            ),
            (
                json!({"oneOf": [{"const": "auto"}, {"type": "integer"}]}),
                "42",
                integer,
            ),
            // `not` excludes literals as well as types.
            (
                json!({"type": ["string", "integer"], "not": {"const": 42}}),
                "42",
                string,
            ),
            (
                json!({"type": ["string", "integer"], "not": {"enum": [42]}}),
                "42",
                string,
            ),
            (
                json!({"type": ["string", "integer"], "not": {"const": 7}}),
                "42",
                integer,
            ),
            (json!({"not": {"type": "string"}}), "-2500", integer),
            // An integer is not rounded to a neighboring float constant.
            (
                json!({"anyOf": [{"const": 9007199254740992.0}, {"type": "string"}]}),
                "9007199254740993",
                string,
            ),
            (
                json!({"anyOf": [{"type": "integer", "const": 9007199254740992.0}, {"type": "string"}]}),
                "9007199254740993",
                string,
            ),
            (
                json!({"anyOf": [{"const": 9007199254740992.0}, {"type": "string"}]}),
                "9007199254740992",
                integer,
            ),
            (
                json!({"anyOf": [{"const": {"n": 9007199254740992.0}}, {"type": "string"}]}),
                r#"{"n":9007199254740993}"#,
                string,
            ),
            // A keyword the walk does not evaluate leaves JSON unproven, so an
            // admitted string wins; without one, the typed JSON value still does.
            (
                json!({"type": ["string", "integer"], "minimum": 100}),
                "42",
                string,
            ),
            (json!({"type": "integer", "minimum": 100}), "42", integer),
            (
                json!({"anyOf": [{"type": "string"}, {"type": "object", "properties": {"x": {"type": "integer"}}}]}),
                r#"{"x":1}"#,
                string,
            ),
            (json!({"type": "integer", "enum": ["auto"]}), "5", integer),
            (
                json!({"type": ["string", "integer"], "$ref": "#/$defs/Missing"}),
                "42",
                string,
            ),
            // 2^64 equals the negated float constant, including when arbitrary
            // precision keeps the integer beyond 64 bits.
            (
                json!({"type": ["string", "number"], "not": {"const": 18446744073709551616.0}}),
                "18446744073709551616",
                string,
            ),
        ] {
            assert_eq!(typed(parameter.clone(), raw), expected, "{parameter} {raw}");
        }
    }

    // A walk cut short by the depth limit proves nothing, so the admitted
    // string wins instead of leaving the choice to type hints.
    #[test]
    fn an_unfinished_walk_leaves_json_unproven() {
        let mut negated = json!({"const": 42});
        for _ in 0..66 {
            negated = json!({"not": negated});
        }
        let parameter = json!({"type": ["string", "integer"], "not": negated});
        assert_eq!(typed(parameter, "42"), Some(NativeText::String));
    }

    // A walk for a nested child starts below the root, so it must still treat
    // the child's enclosing `$id` resource as rebasing its references.
    #[test]
    fn walks_entered_inside_a_nested_resource_leave_its_references_unknown() {
        let schema = json!({
            "type": "object",
            "$defs": {"T": {"type": "integer"}, "N": {"type": ["integer", "null"]}},
            "properties": {
                "payload": {
                    "$id": "https://example.test/payload",
                    "$defs": {"T": {"type": "boolean"}},
                    "type": "object",
                    "properties": {
                        "value": {"anyOf": [{"$ref": "#/$defs/T"}, {"type": "string"}]},
                        "nullable": {"$ref": "#/$defs/N"}
                    }
                },
                "plain": {"anyOf": [{"$ref": "#/$defs/T"}, {"type": "string"}]}
            }
        });
        let payload = &schema["properties"]["payload"]["properties"];
        let facts = SchemaFacts::new(&schema);
        assert_eq!(
            facts.native_text_type(&payload["value"], "42"),
            Some(NativeText::String)
        );
        assert_eq!(
            facts.native_text_type(&schema["properties"]["plain"], "42"),
            Some(NativeText::Json("integer"))
        );
        assert_eq!(
            SchemaFacts::new(&schema).permits_null(&payload["nullable"]),
            None
        );
        // A schema composed from copies cannot be located in the root, so any
        // nested resource there leaves its references unknown.
        let composed = json!({"allOf": [payload["value"].clone()]});
        assert_eq!(
            facts.native_text_type(&composed, "42"),
            Some(NativeText::String)
        );
        let plain_root = json!({"$defs": {"T": {"type": "integer"}}});
        assert_eq!(
            SchemaFacts::new(&plain_root).native_text_type(&composed, "42"),
            Some(NativeText::Json("integer"))
        );
    }

    #[test]
    fn numbers_compare_exactly() {
        let parsed = |text: &str| serde_json::from_str::<Value>(text).unwrap();
        for (left, right, equal) in [
            (json!(1), json!(1.0), true),
            (json!(-0.0), json!(0), true),
            (
                json!(9007199254740993_u64),
                json!(9007199254740992.0),
                false,
            ),
            (json!(u64::MAX), json!(18446744073709551615.0), false),
            (
                parsed("18446744073709551616"),
                json!(18446744073709551616.0),
                true,
            ),
            (
                parsed("-170141183460469231731687303715884105728"),
                json!(-(2f64.powi(127))),
                true,
            ),
            (json!(1), json!(1.5), false),
            (json!(1.5), json!(1.5), true),
            (json!({"n": [1]}), json!({"n": [1.0]}), true),
            (json!({"n": [1]}), json!({"n": [1], "m": 2}), false),
        ] {
            assert_eq!(json_equal(&left, &right), Some(equal), "{left} {right}");
            assert_eq!(json_equal(&right, &left), Some(equal), "{right} {left}");
        }
    }

    // Arbitrary precision keeps numbers beyond f64 that only their spelling
    // identifies; only integer spellings can be compared that way.
    #[cfg(feature = "arbitrary-precision")]
    #[test]
    fn numbers_beyond_f64_compare_only_as_integers() {
        let parsed = |text: &str| serde_json::from_str::<Value>(text).unwrap();
        assert_eq!(json_equal(&parsed("1e400"), &parsed("10e399")), None);
        let beyond_i128 = parsed("340282366920938463463374607431768211457");
        assert_eq!(json_equal(&beyond_i128, &beyond_i128.clone()), Some(true));
        assert_eq!(
            json_equal(
                &beyond_i128,
                &parsed("340282366920938463463374607431768211458")
            ),
            Some(false)
        );
    }
}
