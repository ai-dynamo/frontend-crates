// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{has_unsupported_schema_ref_scope, literals_contradict_type, resolve_local_schema_ref};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

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
pub(crate) fn categorize_type(name: &str) -> Option<SchemaType> {
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
    } else if t.starts_with("num") || t.starts_with("float") || t == "double" {
        Some(SchemaType::Number)
    } else if t == "object" || t.starts_with("dict") {
        Some(SchemaType::Object)
    } else if t == "array" || t == "arr" || t.starts_with("list") {
        Some(SchemaType::Array)
    } else {
        None
    }
}

/// Collect the set of types a (possibly union) schema allows, walking
/// `type`, `const`/`enum`, `anyOf`/`oneOf` branches, and OpenAPI `nullable`.
pub(crate) fn collect_allowed_types(
    root: &Value,
    schema: &Value,
    literals: LiteralTypes<'_>,
    remaining: &mut usize,
) -> (HashSet<SchemaType>, bool) {
    let mut active_refs = HashSet::new();
    let mut context = TypeConstraintContext {
        inherited_nullable: false,
        literals,
        refs_allowed: !has_unsupported_schema_ref_scope(root),
        has_type_constraints: false,
    };
    let allowed_types =
        collect_type_constraints(root, schema, 0, remaining, &mut active_refs, &mut context)
            .filter(|types| types.len() < ALL_SCHEMA_TYPES.len())
            .unwrap_or_default();
    (
        if *remaining == 0 {
            HashSet::new()
        } else {
            allowed_types
        },
        context.has_type_constraints,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiteralTypes<'a> {
    All,
    // Beam and MiniMax keep the declared type when no literal has that type.
    DeclaredFirst,
    // GLM keeps mixed string enums ambiguous for scalar coercion.
    ScalarHints,
    StringHint(&'a str),
    None,
}

pub(crate) fn type_hint(
    root: &Value,
    schema: &Value,
    expected: &str,
    literals: LiteralTypes<'_>,
    remaining: &mut usize,
) -> Option<bool> {
    let mut context = TypeConstraintContext {
        inherited_nullable: false,
        literals,
        refs_allowed: !has_unsupported_schema_ref_scope(root),
        has_type_constraints: false,
    };
    let types = collect_type_constraints(
        root,
        schema,
        0,
        remaining,
        &mut HashSet::new(),
        &mut context,
    )?;
    if *remaining == 0 {
        return None;
    }
    let expected = categorize_type(expected)?;
    if types.len() == ALL_SCHEMA_TYPES.len() {
        return None;
    }
    Some(types.contains(&expected))
}

struct TypeConstraintContext<'a> {
    inherited_nullable: bool,
    literals: LiteralTypes<'a>,
    refs_allowed: bool,
    has_type_constraints: bool,
}

// None is an absent type constraint, not an empty intersection.
fn collect_type_constraints(
    root: &Value,
    schema: &Value,
    depth: usize,
    remaining: &mut usize,
    active_refs: &mut HashSet<String>,
    context: &mut TypeConstraintContext<'_>,
) -> Option<HashSet<SchemaType>> {
    *remaining = remaining.checked_sub(1)?;
    if depth > 16 {
        *remaining = 0;
        return None;
    }
    if let Some(allowed) = schema.as_bool() {
        return if allowed { None } else { Some(HashSet::new()) };
    }
    if !schema.is_object() {
        return None;
    }
    let refs_allowed = context.refs_allowed && !has_unsupported_schema_ref_scope(schema);
    let mut out = HashSet::new();
    if let Some(ty) = schema.get("type") {
        if let Some(name) = ty.as_str() {
            if let Some(cat) = categorize_type(name) {
                out.insert(cat);
                context.has_type_constraints = true;
            }
        } else if let Some(arr) = ty.as_array() {
            for item in arr.iter().filter_map(Value::as_str) {
                if let Some(cat) = categorize_type(item) {
                    out.insert(cat);
                    context.has_type_constraints = true;
                }
            }
        }
    }
    if out.contains(&SchemaType::Number) {
        out.insert(SchemaType::Integer);
    }
    let nullable = schema.get("nullable").and_then(Value::as_bool) == Some(true);
    // On a $ref sibling, nullable extends the target's type alternatives; it is
    // not a separate `null` constraint to intersect with the referenced type.
    if (context.inherited_nullable && !out.is_empty())
        || (nullable && (schema.get("$ref").is_none() || schema.get("type").is_some()))
    {
        out.insert(SchemaType::Null);
        context.has_type_constraints = true;
    }
    let mut constraints = Vec::new();
    if !out.is_empty() {
        constraints.push(out);
    }
    let infer_literals = context.literals != LiteralTypes::None
        && (context.literals != LiteralTypes::DeclaredFirst || !literals_contradict_type(schema));
    if infer_literals && let Some(value) = schema.get("const") {
        context.has_type_constraints = true;
        constraints.push(literal_type_constraints(value, context.literals));
    }
    if infer_literals
        && let Some(values) = schema.get("enum").and_then(Value::as_array)
        && match context.literals {
            LiteralTypes::All | LiteralTypes::DeclaredFirst => true,
            LiteralTypes::StringHint(raw) => values.iter().any(|value| value.as_str() == Some(raw)),
            LiteralTypes::ScalarHints => values.iter().all(|value| !value.is_string()),
            LiteralTypes::None => false,
        }
    {
        context.has_type_constraints = true;
        constraints.push(
            values
                .iter()
                .flat_map(|value| literal_type_constraints(value, context.literals))
                .collect(),
        );
    }
    if ["$ref", "$dynamicRef", "$recursiveRef"]
        .iter()
        .any(|key| schema.get(key).is_some())
    {
        context.has_type_constraints = true;
    }
    if refs_allowed
        && let Some(reference) = schema.get("$ref").and_then(Value::as_str)
        && active_refs.insert(reference.to_string())
    {
        if let Some(target) = resolve_local_schema_ref(reference, root)
            && let Some(target_types) = {
                let mut target_context = TypeConstraintContext {
                    inherited_nullable: context.inherited_nullable || nullable,
                    literals: context.literals,
                    refs_allowed,
                    has_type_constraints: false,
                };
                collect_type_constraints(
                    root,
                    target,
                    depth + 1,
                    remaining,
                    active_refs,
                    &mut target_context,
                )
            }
        {
            constraints.push(target_types);
        }
        active_refs.remove(reference);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(options) = schema.get(key).and_then(Value::as_array) {
            context.has_type_constraints = true;
            let mut alternatives = Vec::new();
            let mut unknown = false;
            for option in options {
                let mut option_context = TypeConstraintContext {
                    inherited_nullable: context.inherited_nullable,
                    literals: context.literals,
                    refs_allowed,
                    has_type_constraints: false,
                };
                match collect_type_constraints(
                    root,
                    option,
                    depth + 1,
                    remaining,
                    active_refs,
                    &mut option_context,
                ) {
                    Some(types) => alternatives.push(types),
                    None if !matches!(
                        context.literals,
                        LiteralTypes::All | LiteralTypes::DeclaredFirst
                    ) && !option_context.has_type_constraints => {}
                    None => unknown = true,
                }
            }
            if !unknown && !alternatives.is_empty() {
                constraints.push(alternatives.into_iter().flatten().collect());
            }
        }
    }
    if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
        context.has_type_constraints = true;
        for branch in branches {
            let mut branch_context = TypeConstraintContext {
                inherited_nullable: context.inherited_nullable,
                literals: context.literals,
                refs_allowed,
                has_type_constraints: false,
            };
            if let Some(types) = collect_type_constraints(
                root,
                branch,
                depth + 1,
                remaining,
                active_refs,
                &mut branch_context,
            ) {
                constraints.push(types);
            }
        }
    }
    // `not`: only an EXACT type-level negation (`type`, `$ref`/composition of
    // such with a decidable type set) excludes whole categories. A value-
    // sensitive negation (`not: {const: 0}`, bounds) or an inexact one
    // (overlapping `oneOf`) still admits values of every type and constrains
    // nothing here. Integer ⊂ Number: `Number` alone means a non-integral
    // number, so excluding `integer` keeps fractional numbers.
    if schema.get("not").is_some() || schema.get("if").is_some() {
        context.has_type_constraints = true;
    }
    if let Some(negated) = schema.get("not")
        && let Some(excluded) = exact_schema_types(
            root,
            negated,
            depth + 1,
            remaining,
            active_refs,
            &mut HashMap::new(),
            refs_allowed,
        )
    {
        context.has_type_constraints = true;
        constraints.push(complement_types(&excluded));
    }
    // `if`/`then`/`else`: an exact type-level condition partitions the value
    // space, (if ∩ then) ∪ (¬if ∩ else); a value-sensitive or inexact condition
    // cannot be decided from types, so both branches stay possible and only
    // their union constrains the value. For an inexact condition, an absent or
    // untyped branch leaves one outcome unconstrained.
    if let Some(condition) = schema.get("if")
        && (schema.get("then").is_some() || schema.get("else").is_some())
    {
        let mut branch_types = |key| {
            schema.get(key).and_then(|branch| {
                let mut branch_context = TypeConstraintContext {
                    inherited_nullable: false,
                    literals: context.literals,
                    refs_allowed,
                    has_type_constraints: false,
                };
                collect_type_constraints(
                    root,
                    branch,
                    depth + 1,
                    remaining,
                    active_refs,
                    &mut branch_context,
                )
            })
        };
        let then_types = branch_types("then");
        let else_types = branch_types("else");
        let condition_types = exact_schema_types(
            root,
            condition,
            depth + 1,
            remaining,
            active_refs,
            &mut HashMap::new(),
            refs_allowed,
        );
        let outcome = match condition_types {
            Some(when) => {
                let then_branch =
                    then_types.map_or_else(|| when.clone(), |types| intersect_types(&when, &types));
                let otherwise = complement_types(&when);
                let else_branch = else_types.map_or_else(
                    || otherwise.clone(),
                    |types| intersect_types(&otherwise, &types),
                );
                Some(
                    then_branch
                        .union(&else_branch)
                        .copied()
                        .collect::<HashSet<_>>(),
                )
            }
            None => then_types
                .zip(else_types)
                .map(|(left, right)| left.union(&right).copied().collect()),
        };
        if let Some(types) = outcome {
            context.has_type_constraints = true;
            constraints.push(types);
        }
    }
    constraints.into_iter().reduce(|mut left, right| {
        left.retain(|ty| right.contains(ty));
        left
    })
}

const ALL_SCHEMA_TYPES: [SchemaType; 7] = [
    SchemaType::String,
    SchemaType::Integer,
    SchemaType::Number,
    SchemaType::Boolean,
    SchemaType::Object,
    SchemaType::Array,
    SchemaType::Null,
];

/// Keywords that never narrow a schema below a whole JSON type.
const TYPE_LEVEL_KEYWORDS: [&str; 17] = [
    "type",
    "$ref",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "nullable",
    "$defs",
    "definitions",
    "$id",
    "$schema",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
];

/// Per-walk cache of resolved reference results, so a reference shared by many
/// branches of a definition graph is expanded once.
type ExactRefMemo = HashMap<String, Option<HashSet<SchemaType>>>;

/// The EXACT set of JSON types a schema accepts when that is decided by type
/// alone, or `None` when it is not: a value-sensitive keyword (`const`, `enum`,
/// bounds, `pattern`, `properties`, `items`, `if`, …), an unknown type name,
/// an inexact composition (`oneOf` whose alternatives overlap), an
/// unresolvable or cyclic reference, excessive depth, or an exhausted visit
/// budget. Callers treat `None` as "no constraint". Only an exact set may be
/// complemented (`not`) or used to partition `if`/`then`/`else`.
///
/// `Number` in the returned set means a non-integral number; `Integer` is
/// listed separately, so `type: number` yields both and `type: integer` only
/// the latter.
pub(crate) fn exact_schema_types(
    root: &Value,
    schema: &Value,
    depth: usize,
    remaining: &mut usize,
    active_refs: &mut HashSet<String>,
    memo: &mut ExactRefMemo,
    refs_allowed: bool,
) -> Option<HashSet<SchemaType>> {
    *remaining = remaining.checked_sub(1)?;
    if depth > 16 {
        *remaining = 0;
        return None;
    }
    if let Some(allowed) = schema.as_bool() {
        return Some(if allowed {
            ALL_SCHEMA_TYPES.into_iter().collect()
        } else {
            HashSet::new()
        });
    }
    let object = schema.as_object()?;
    if !object
        .keys()
        .all(|key| TYPE_LEVEL_KEYWORDS.contains(&key.as_str()))
    {
        return None;
    }
    let refs_allowed = refs_allowed && !has_unsupported_schema_ref_scope(schema);
    let mut parts: Vec<HashSet<SchemaType>> = Vec::new();
    if let Some(ty) = object.get("type") {
        let names: Vec<&str> = match ty {
            Value::String(name) => vec![name.as_str()],
            Value::Array(names) => names
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<_>>>()?,
            _ => return None,
        };
        let mut types = HashSet::new();
        for name in names {
            types.insert(categorize_type(name)?);
        }
        if types.contains(&SchemaType::Number) {
            types.insert(SchemaType::Integer);
        }
        parts.push(types);
    }
    if let Some(reference) = object.get("$ref") {
        let reference = reference.as_str()?;
        if !refs_allowed {
            return None;
        }
        let resolved = match memo.get(reference) {
            Some(cached) => cached.clone(),
            None => {
                // A reference already being expanded is a cycle: inexact.
                if !active_refs.insert(reference.to_string()) {
                    return None;
                }
                let resolved = resolve_local_schema_ref(reference, root).and_then(|target| {
                    exact_schema_types(
                        root,
                        target,
                        depth + 1,
                        remaining,
                        active_refs,
                        memo,
                        refs_allowed,
                    )
                });
                active_refs.remove(reference);
                memo.insert(reference.to_string(), resolved.clone());
                resolved
            }
        };
        parts.push(resolved?);
    }
    for keyword in ["allOf", "anyOf", "oneOf"] {
        let Some(branches) = object.get(keyword) else {
            continue;
        };
        let branches = branches.as_array()?;
        let mut sets = Vec::with_capacity(branches.len());
        for branch in branches {
            sets.push(exact_schema_types(
                root,
                branch,
                depth + 1,
                remaining,
                active_refs,
                memo,
                refs_allowed,
            )?);
        }
        let combined = match keyword {
            "allOf" => sets
                .into_iter()
                .reduce(|left, right| intersect_types(&left, &right))?,
            "anyOf" => sets.into_iter().flatten().collect(),
            // `oneOf` is exactly one alternative: a type accepted by two
            // alternatives is rejected, so only pairwise-disjoint alternatives
            // have a type-decidable result.
            _ => {
                let overlapping = sets.iter().enumerate().any(|(index, left)| {
                    sets[index + 1..]
                        .iter()
                        .any(|right| !left.is_disjoint(right))
                });
                if overlapping {
                    return None;
                }
                sets.into_iter().flatten().collect()
            }
        };
        parts.push(combined);
    }
    if let Some(negated) = object.get("not") {
        let excluded = exact_schema_types(
            root,
            negated,
            depth + 1,
            remaining,
            active_refs,
            memo,
            refs_allowed,
        )?;
        parts.push(complement_types(&excluded));
    }
    let mut result = parts
        .into_iter()
        .reduce(|left, right| intersect_types(&left, &right))
        .unwrap_or_else(|| ALL_SCHEMA_TYPES.into_iter().collect());
    if object.get("nullable").and_then(Value::as_bool) == Some(true) {
        result.insert(SchemaType::Null);
    }
    Some(result)
}

fn complement_types(excluded: &HashSet<SchemaType>) -> HashSet<SchemaType> {
    ALL_SCHEMA_TYPES
        .iter()
        .copied()
        .filter(|ty| !excluded.contains(ty))
        .collect()
}

fn intersect_types(left: &HashSet<SchemaType>, right: &HashSet<SchemaType>) -> HashSet<SchemaType> {
    left.intersection(right).copied().collect()
}

// A float-backed schema literal has already passed through f64: an integral-looking
// value may have originated as a large fraction. Retain the number alternative.
// Explicit integer types still intersect this set and exclude fractional arguments.
fn literal_type_constraints(value: &Value, literals: LiteralTypes<'_>) -> HashSet<SchemaType> {
    if let LiteralTypes::StringHint(raw) = literals
        && value.as_str() != Some(raw)
    {
        return HashSet::new();
    }
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
