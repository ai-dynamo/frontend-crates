// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The publication wire format retains YAML key types and insertion order.
//! Resolution follows the maintained PyYAML strict loader, not YAML 1.2.

use anyhow::{Context, Result, bail, ensure};
use chrono::{NaiveDate, NaiveDateTime};
use num_bigint::BigInt;
use num_traits::{FromPrimitive, Zero};
use regex::Regex;
use serde_json::Value;
use std::{collections::HashSet, path::Path, sync::LazyLock};
use yaml_rust2::{
    parser::{Event, Parser},
    scanner::{Marker, TScalarStyle},
};

/// Owned YAML values with ordered, typed mapping keys.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Null,
    Bool(bool),
    Int(BigInt),
    Float(f64),
    Str(String),
    Date(String),
    DateTime(String),
    Binary(Vec<u8>),
    Array(Vec<Node>),
    Map(Vec<(Node, Node)>),
}

impl Node {
    /// Make an empty ordered mapping.
    pub fn map() -> Self {
        Self::Map(Vec::new())
    }

    /// Require a string scalar.
    pub fn string(&self) -> Result<&str> {
        match self {
            Self::Str(s) => Ok(s),
            _ => bail!("expected a string"),
        }
    }

    /// Require an array.
    pub fn array(&self) -> Result<&[Node]> {
        match self {
            Self::Array(v) => Ok(v),
            _ => bail!("expected an array"),
        }
    }

    /// Require a present mapping field.
    pub fn field(&self, key: &str) -> Result<&Self> {
        self.get(key)
            .with_context(|| format!("missing field {key}"))
    }

    /// Mutate a field while retaining its original mapping position.
    pub fn set(&mut self, key: &str, value: Self) -> Result<()> {
        let Self::Map(v) = self else {
            bail!("expected a mapping")
        };
        if let Some((_, old)) = v
            .iter_mut()
            .find(|(k, _)| matches!(k, Self::Str(s) if s == key))
        {
            *old = value;
        } else {
            v.push((Self::Str(key.into()), value));
        }
        Ok(())
    }

    /// Remove a field without reordering the remaining entries.
    pub fn remove(&mut self, key: &str) -> Result<Option<Self>> {
        let Self::Map(v) = self else {
            bail!("expected a mapping")
        };
        Ok(v.iter()
            .position(|(k, _)| matches!(k, Self::Str(s) if s == key))
            .map(|i| v.remove(i).1))
    }

    /// Update an ordered mapping with another ordered mapping.
    pub fn update(&mut self, other: &Self) -> Result<()> {
        for (key, value) in other.mapping()? {
            self.set(key.string()?, value.clone())?;
        }
        Ok(())
    }

    /// Recursively merge execution defaults, preserving missing keys.
    pub fn merge(&self, other: &Self) -> Result<Self> {
        let mut result = self.mapping()?.to_vec();
        for (key, value) in other.mapping()? {
            if let Some((_, old)) = result.iter_mut().find(|(k, _)| key_equal(k, key)) {
                *old = match (&*old, value) {
                    (Self::Map(_), Self::Map(_)) => old.merge(value)?,
                    _ => value.clone(),
                };
            } else {
                result.push((key.clone(), value.clone()));
            }
        }
        Ok(Self::Map(result))
    }

    /// Compare typed values with mapping order ignored, as in typed_equal.
    pub fn equivalent(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Map(a), Self::Map(b)) => {
                a.len() == b.len()
                    && a.iter().all(|(k, v)| {
                        b.iter()
                            .any(|(q, w)| k.typed() == q.typed() && v.equivalent(w))
                    })
            }
            (Self::Array(a), Self::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.equivalent(b))
            }
            _ => self.typed() == other.typed(),
        }
    }

    /// Conservative owned tree size for expansion admission (including node storage).
    pub fn heap_bytes(&self) -> u64 {
        std::mem::size_of::<Self>() as u64
            + match self {
                Self::Str(v) | Self::Date(v) | Self::DateTime(v) => v.capacity() as u64,
                Self::Binary(v) => v.capacity() as u64,
                Self::Int(v) => v.bits().div_ceil(8) + 32,
                Self::Array(v) => {
                    v.iter().map(Self::heap_bytes).sum::<u64>()
                        + (v.capacity() - v.len()) as u64 * std::mem::size_of::<Self>() as u64
                }
                Self::Map(v) => {
                    v.iter()
                        .map(|(k, v)| k.heap_bytes() + v.heap_bytes())
                        .sum::<u64>()
                        + (v.capacity() - v.len()) as u64
                            * std::mem::size_of::<(Self, Self)>() as u64
                }
                _ => 0,
            }
    }

    /// Encode the existing typed-json-v1 representation, moving each child once.
    pub fn typed(&self) -> Value {
        use base64::{Engine, engine::general_purpose::STANDARD};
        match self {
            Self::Null => Value::Array(vec![Value::String("null".into())]),
            Self::Bool(v) => tagged("bool", Value::Bool(*v)),
            Self::Int(v) => tagged("int", Value::String(v.to_string())),
            Self::Float(v) => tagged("float", Value::String(float_repr(*v))),
            Self::Str(v) => tagged("str", Value::String(v.clone())),
            Self::Date(v) => tagged("date", Value::String(v.clone())),
            Self::DateTime(v) => tagged("datetime", Value::String(v.clone())),
            Self::Binary(v) => tagged("binary", Value::String(STANDARD.encode(v))),
            Self::Array(v) => tagged("array", Value::Array(v.iter().map(Self::typed).collect())),
            Self::Map(v) => tagged(
                "object",
                Value::Array(
                    v.iter()
                        .map(|(key, value)| Value::Array(vec![key.typed(), value.typed()]))
                        .collect(),
                ),
            ),
        }
    }

    pub fn from_typed(value: &Value) -> Result<Self> {
        use base64::{Engine, engine::general_purpose::STANDARD};
        let row = value.as_array().context("typed node is not an array")?;
        let tag = row
            .first()
            .and_then(Value::as_str)
            .context("typed tag missing")?;
        if tag == "null" {
            ensure!(row.len() == 1, "invalid null");
            return Ok(Self::Null);
        }
        ensure!(row.len() == 2, "invalid typed arity");
        let payload = &row[1];
        let text = || payload.as_str().context("invalid typed scalar");
        Ok(match tag {
            "bool" => Self::Bool(payload.as_bool().context("invalid bool")?),
            "int" => Self::Int(text()?.parse()?),
            "float" => Self::Float(match text()? {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                other => other.parse()?,
            }),
            "str" => Self::Str(text()?.into()),
            "date" => Self::Date(text()?.into()),
            "datetime" => Self::DateTime(text()?.into()),
            "binary" => Self::Binary(STANDARD.decode(text()?)?),
            "array" => Self::Array(
                payload
                    .as_array()
                    .context("invalid array")?
                    .iter()
                    .map(Self::from_typed)
                    .collect::<Result<_>>()?,
            ),
            "object" => Self::Map(
                payload
                    .as_array()
                    .context("invalid object")?
                    .iter()
                    .map(|entry| {
                        let pair = entry.as_array().context("invalid entry")?;
                        ensure!(pair.len() == 2, "invalid entry arity");
                        Ok((Self::from_typed(&pair[0])?, Self::from_typed(&pair[1])?))
                    })
                    .collect::<Result<_>>()?,
            ),
            _ => bail!("invalid typed tag"),
        })
    }

    /// Borrow a mapping without coercing its keys.
    pub fn mapping(&self) -> Result<&[(Node, Node)]> {
        match self {
            Self::Map(v) => Ok(v),
            _ => bail!("expected a mapping"),
        }
    }

    /// Look up a string key, preserving missing versus null.
    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Map(v) => v
                .iter()
                .find_map(|(k, v)| matches!(k, Self::Str(s) if s == key).then_some(v)),
            _ => None,
        }
    }
}

pub(crate) fn float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v.is_sign_negative() { "-inf" } else { "inf" }.into();
    }
    let s = format!("{v:?}");
    if let Some((mantissa, exponent)) = s.split_once('e') {
        let e: i32 = exponent.parse().expect("Rust float exponent");
        format!("{mantissa}e{e:+03}")
    } else {
        s
    }
}

/// Build a typed container without serializing and copying its owned children.
pub fn tagged(tag: &str, payload: Value) -> Value {
    Value::Array(vec![Value::String(tag.into()), payload])
}

// Python dict equality conflates bool/int/integral-float keys. The strict
// loader rejects these collisions rather than silently losing a typed key.
fn integer_key(v: &Node) -> Option<BigInt> {
    match v {
        Node::Bool(v) => Some(BigInt::from(u8::from(*v))),
        Node::Int(v) => Some(v.clone()),
        Node::Float(v) if v.is_finite() && v.fract() == 0.0 => BigInt::from_f64(*v),
        _ => None,
    }
}

#[derive(Eq, PartialEq, Hash)]
enum Key {
    Null,
    Integer(BigInt),
    Float(u64),
    String(String),
    Date(String),
    AwareDateTime(i64, u32),
    NaiveDateTime(String),
    Binary(Vec<u8>),
}

fn key_identity(value: &Node) -> Option<Key> {
    if let Some(integer) = integer_key(value) {
        return Some(Key::Integer(integer));
    }
    Some(match value {
        Node::Null => Key::Null,
        Node::Float(value) => Key::Float(if value.is_nan() {
            f64::NAN.to_bits()
        } else {
            value.to_bits()
        }),
        Node::Str(value) => Key::String(value.clone()),
        Node::Date(value) => Key::Date(value.clone()),
        Node::DateTime(value) => match chrono::DateTime::parse_from_rfc3339(value) {
            Ok(value) => Key::AwareDateTime(value.timestamp(), value.timestamp_subsec_nanos()),
            Err(_) => Key::NaiveDateTime(value.clone()),
        },
        Node::Binary(value) => Key::Binary(value.clone()),
        _ => return None,
    })
}

fn key_equal(a: &Node, b: &Node) -> bool {
    match (key_identity(a), key_identity(b)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

static INT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$").unwrap()
});
static FLOAT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$").unwrap()
});
static DATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})$").unwrap());
static DATETIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?:[Tt]|[ \t]+)([0-9]{1,2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]*))?(?:[ \t]*(Z|([-+])([0-9]{1,2})(?::([0-9]{2}))?))?$").unwrap()
});

fn scalar(s: String, style: TScalarStyle) -> Result<Node> {
    if style != TScalarStyle::Plain {
        return Ok(Node::Str(s));
    }
    match s.as_str() {
        "" | "~" | "null" | "Null" | "NULL" => return Ok(Node::Null),
        "yes" | "Yes" | "YES" | "true" | "True" | "TRUE" | "on" | "On" | "ON" => {
            return Ok(Node::Bool(true));
        }
        "no" | "No" | "NO" | "false" | "False" | "FALSE" | "off" | "Off" | "OFF" => {
            return Ok(Node::Bool(false));
        }
        "<<" => bail!("unsupported YAML merge value"),
        "=" => bail!("unsupported YAML value tag"),
        _ => (),
    }
    if INT.is_match(&s) {
        let clean = s.replace('_', "");
        let negative = clean.starts_with('-');
        let digits = clean.trim_start_matches(['+', '-']);
        let v = if digits.contains(':') {
            let mut value = BigInt::zero();
            for part in digits.split(':') {
                value = value * 60 + part.parse::<BigInt>()?;
            }
            value
        } else {
            let (digits, radix) = if let Some(v) = digits.strip_prefix("0b") {
                (v, 2)
            } else if let Some(v) = digits.strip_prefix("0x") {
                (v, 16)
            } else if digits.starts_with('0') && digits.len() > 1 {
                (&digits[1..], 8)
            } else {
                (digits, 10)
            };
            BigInt::parse_bytes(digits.as_bytes(), radix).context("invalid YAML integer")?
        };
        return Ok(Node::Int(if negative { -v } else { v }));
    }
    if FLOAT.is_match(&s) {
        let clean = s.replace('_', "").to_lowercase();
        let v = match clean.as_str() {
            ".nan" => f64::NAN,
            ".inf" | "+.inf" => f64::INFINITY,
            "-.inf" => f64::NEG_INFINITY,
            _ if clean.contains(':') => {
                let mut value = 0.0;
                for part in clean.trim_start_matches(['+', '-']).split(':') {
                    value = value * 60.0 + part.parse::<f64>()?;
                }
                if clean.starts_with('-') {
                    -value
                } else {
                    value
                }
            }
            _ => clean.parse()?,
        };
        return Ok(Node::Float(v));
    }
    if DATE.is_match(&s) {
        ensure!(!s.starts_with("0000"), "invalid YAML date year");
        return Ok(Node::Date(
            NaiveDate::parse_from_str(&s, "%Y-%m-%d")?.to_string(),
        ));
    }
    if let Some(c) = DATETIME.captures(&s) {
        ensure!(&c[1] != "0000", "invalid YAML timestamp year");
        let part = |i| -> Result<u32> {
            Ok(c.get(i)
                .context("timestamp field missing")?
                .as_str()
                .parse()?)
        };
        let date = NaiveDate::from_ymd_opt(part(1)? as i32, part(2)?, part(3)?)
            .context("invalid YAML date")?;
        let fraction = c.get(7).map_or("", |m| m.as_str());
        let micros: u32 = format!("{fraction:0<6}")[..6].parse()?;
        let time = date
            .and_hms_micro_opt(part(4)?, part(5)?, part(6)?, micros)
            .context("invalid YAML time")?;
        let mut repr = NaiveDateTime::format(&time, "%Y-%m-%dT%H:%M:%S").to_string();
        if micros != 0 {
            repr.push_str(&format!(".{micros:06}"));
        }
        if let Some(zone) = c.get(8) {
            let (sign, hours, minutes) = if zone.as_str() == "Z" {
                ('+', 0, 0)
            } else {
                (
                    if &c[9] == "-" { '-' } else { '+' },
                    part(10)?,
                    c.get(11).map_or(Ok(0), |m| m.as_str().parse())?,
                )
            };
            let total = hours * 60 + minutes;
            ensure!(total < 24 * 60, "invalid YAML timezone");
            repr.push_str(&format!(
                "{}{:02}:{:02}",
                if total == 0 { '+' } else { sign },
                total / 60,
                total % 60
            ));
        }
        return Ok(Node::DateTime(repr));
    }
    Ok(Node::Str(s))
}

struct Reader<'a> {
    parser: Parser<std::str::Chars<'a>>,
}

impl Reader<'_> {
    fn next(&mut self) -> Result<(Event, Marker)> {
        Ok(self.parser.next_token()?)
    }

    fn node(&mut self, event: Event, depth: usize) -> Result<Node> {
        ensure!(depth < 256, "YAML nesting exceeds supported depth");
        match event {
            Event::Alias(_) => bail!("YAML aliases are not allowed"),
            Event::Scalar(s, style, anchor, tag) => {
                ensure!(anchor == 0, "YAML anchors are not allowed");
                ensure!(tag.is_none(), "YAML tags are not allowed");
                scalar(s, style)
            }
            Event::SequenceStart(anchor, tag) => {
                ensure!(
                    anchor == 0 && tag.is_none(),
                    "YAML anchors and tags are not allowed"
                );
                let mut v = Vec::new();
                loop {
                    let (event, mark) = self.next()?;
                    if event == Event::SequenceEnd {
                        break;
                    }
                    v.push(
                        self.node(event, depth + 1)
                            .with_context(|| format!("at {mark:?}"))?,
                    );
                }
                Ok(Node::Array(v))
            }
            Event::MappingStart(anchor, tag) => {
                ensure!(
                    anchor == 0 && tag.is_none(),
                    "YAML anchors and tags are not allowed"
                );
                let mut v: Vec<(Node, Node)> = Vec::new();
                let mut keys = HashSet::new();
                loop {
                    let (event, mark) = self.next()?;
                    if event == Event::MappingEnd {
                        break;
                    }
                    let key = self
                        .node(event, depth + 1)
                        .with_context(|| format!("at {mark:?}"))?;
                    ensure!(
                        keys.insert(
                            key_identity(&key).with_context(|| format!(
                                "unhashable YAML mapping key at {mark:?}"
                            ))?
                        ),
                        "duplicate YAML key at {mark:?}"
                    );
                    let (event, mark) = self.next()?;
                    let value = self
                        .node(event, depth + 1)
                        .with_context(|| format!("at {mark:?}"))?;
                    v.push((key, value));
                }
                Ok(Node::Map(v))
            }
            _ => bail!("unexpected YAML event: {event:?}"),
        }
    }
}

/// Load exactly one mapping, rejecting features forbidden by StrictLoader.
pub fn load_str(text: &str) -> Result<Node> {
    let mut reader = Reader {
        parser: Parser::new_from_str(text),
    };
    ensure!(
        reader.next()?.0 == Event::StreamStart,
        "expected YAML stream"
    );
    ensure!(
        reader.next()?.0 == Event::DocumentStart,
        "expected one YAML document"
    );
    let (event, mark) = reader.next()?;
    let value = reader
        .node(event, 0)
        .with_context(|| format!("at {mark:?}"))?;
    ensure!(
        matches!(value, Node::Map(_)),
        "YAML document must be a mapping"
    );
    ensure!(
        reader.next()?.0 == Event::DocumentEnd,
        "invalid YAML document"
    );
    ensure!(
        reader.next()?.0 == Event::StreamEnd,
        "expected exactly one YAML document"
    );
    Ok(value)
}

/// Load UTF-8 YAML with path and parser location diagnostics.
pub fn load(path: &Path) -> Result<Node> {
    load_str(&std::fs::read_to_string(path).with_context(|| path.display().to_string())?)
        .with_context(|| format!("invalid YAML: {}", path.display()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn typed_cache_roundtrip_preserves_keys_null_and_scalar_types() -> anyhow::Result<()> {
        let source = super::load_str(
            "null_value: null\nbool: true\nint: 9999999999999999999999999\nfloat: -0.0\ntext: string\ndate: 2026-10-23\ndatetime: 2026-10-23T12:34:56Z\narray: [1, true, null]\nmap: {1: int-key, '1': string-key}\nnan: .nan\ninf: .inf\n",
        )?;
        for source in [source, super::Node::Binary(vec![0, 1, 255])] {
            let typed = source.typed();
            assert_eq!(typed, super::Node::from_typed(&typed)?.typed());
        }
        assert!(super::Node::from_typed(&serde_json::json!(["object", [["bad"]]])).is_err());
        Ok(())
    }

    use super::*;
    use serde_json::json;

    #[test]
    fn strict_loader_semantic_probes() -> Result<()> {
        for (source, expected) in [
            ("yes", json!(["bool", true])),
            ("012", json!(["int", "10"])),
            (
                "18446744073709551616",
                json!(["int", "18446744073709551616"]),
            ),
            ("2026-10-18", json!(["date", "2026-10-18"])),
            (
                "2001-12-15T02:59:43.1Z",
                json!(["datetime", "2001-12-15T02:59:43.100000+00:00"]),
            ),
            ("1e3", json!(["str", "1e3"])),
            ("1.0e+3", json!(["float", "1000.0"])),
            ("1.0e-5", json!(["float", "1e-05"])),
            ("1:20", json!(["int", "80"])),
            ("-0.0", json!(["float", "-0.0"])),
            (".nan", json!(["float", "nan"])),
            ("'yes'", json!(["str", "yes"])),
            ("._1", json!(["str", "._1"])),
            ("._", json!(["str", "._"])),
            (
                "\u{0662}\u{0660}\u{0662}\u{0666}-10-18",
                json!(["str", "\u{0662}\u{0660}\u{0662}\u{0666}-10-18"]),
            ),
        ] {
            let value = load_str(&format!("value: {source}\n"))?;
            assert_eq!(value.get("value").unwrap().typed(), expected, "{source}");
        }
        Ok(())
    }

    #[test]
    fn rejects_unsafe_and_ambiguous_documents() {
        for source in [
            "first: &same [1]\nsecond: *same\n",
            "value: !!str 1\n",
            "value: &unused 1\n",
            "value: 1\nvalue: 2\n",
            "true: a\n1: b\n",
            "1: a\n1.0: b\n",
            "v: 1\n---\nv: 2\n",
            "[1, 2]\n",
            "? [1, 2]\n: v\n",
            "v: 2026-99-99\n",
            "v: <<\n",
            "v: =\n",
            "v: 0000-01-01\n",
            "v: 2001-12-15T02:59:60Z\n",
            "2001-12-15T02:59:43Z: one\n2001-12-15T03:59:43+01:00: two\n",
        ] {
            assert!(load_str(source).is_err(), "accepted {source}");
        }
    }

    #[test]
    fn execution_merge_uses_python_mapping_key_equality() -> Result<()> {
        let defaults = load_str("true: old\n")?;
        let exception = load_str("1: new\n")?;
        let merged = defaults.merge(&exception)?;
        assert_eq!(merged.mapping()?.len(), 1);
        assert_eq!(
            merged.mapping()?[0],
            (Node::Bool(true), Node::Str("new".into()))
        );
        Ok(())
    }
}
