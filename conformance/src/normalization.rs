// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Storage normalization shared by the native publisher and downstream runners.

pub type Result<T> = std::result::Result<T, String>;

/// Typed operations keep schema and argument scalars out of a JSON round trip.
pub trait Value: Clone {
    fn object() -> Self;
    fn text(value: &str) -> Self;
    fn integer(value: u64) -> Self;
    fn array(values: Vec<Self>) -> Self;
    fn null() -> Self;
    fn is_null(&self) -> bool;
    fn as_integer(&self) -> Option<u64>;
    fn as_bool(&self) -> Option<bool>;
    fn as_text(&self) -> Result<&str>;
    fn entries(&self) -> Result<Vec<(&str, &Self)>>;
    fn items(&self) -> Result<&[Self]>;
    fn get(&self, name: &str) -> Option<&Self>;
    fn set(&mut self, name: &str, value: Self) -> Result<()>;
    fn digest(&self) -> Result<String>;

    fn field(&self, name: &str) -> Result<&Self> {
        self.get(name)
            .ok_or_else(|| format!("missing normalized field: {name}"))
    }
}

fn fields<V: Value>(value: &V, required: &[&str], optional: &[&str]) -> Result<()> {
    for name in required {
        value.field(name)?;
    }
    for (name, _) in value.entries()? {
        if !required.contains(&name) && !optional.contains(&name) {
            return Err(format!("unknown normalized field: {name}"));
        }
    }
    Ok(())
}

fn object<V: Value>(entries: &[(&str, V)]) -> Result<V> {
    let mut result = V::object();
    for (name, value) in entries {
        result.set(name, value.clone())?;
    }
    Ok(result)
}

fn reference<V: Value>(table: &V, identity: &V) -> Result<V> {
    Ok(table.field(identity.as_text()?)?.clone())
}

pub fn is_normalized<V: Value>(document: &V) -> bool {
    document.get("schema_version").and_then(Value::as_integer) == Some(3)
}

/// Resolve the frozen request library, checking identity after all defaults.
pub fn requests<V: Value>(document: &V) -> Result<V> {
    fields(
        document,
        &[
            "schema_version",
            "family",
            "request_defaults",
            "tools",
            "requests",
            "cases",
        ],
        &[],
    )?;
    if !is_normalized(document) {
        return Err("invalid normalized authored schema".into());
    }
    let defaults = document.field("request_defaults")?;
    fields(defaults, &["init", "finish_reason"], &[])?;
    let mut result = V::object();
    for (identity, stored) in document.field("requests")?.entries()? {
        fields(
            stored,
            &["input", "tools"],
            &[
                "init",
                "finish_reason",
                "chunks",
                "chunk_sizes",
                "finish_step",
            ],
        )?;
        let mut request = defaults.clone();
        for (name, value) in stored.entries()? {
            if !["chunk_sizes", "finish_step"].contains(&name) {
                request.set(name, value.clone())?;
            }
        }
        request.set(
            "tools",
            reference(document.field("tools")?, stored.field("tools")?)?,
        )?;
        let mut chunks = Vec::new();
        if let Some(sizes) = stored.get("chunk_sizes") {
            if stored.get("chunks").is_some() {
                return Err("ambiguous normalized chunk schedule".into());
            }
            let characters: Vec<_> = stored.field("input")?.as_text()?.chars().collect();
            let mut offset: usize = 0;
            for size in sizes.items()? {
                let size = size
                    .as_integer()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("invalid normalized chunk size")?;
                let end = offset
                    .checked_add(size)
                    .filter(|end| *end <= characters.len())
                    .ok_or("invalid normalized chunk size")?;
                let text: String = characters[offset..end].iter().collect();
                chunks.push(object(&[("delta_text", V::text(&text))])?);
                offset = end;
            }
            if offset != characters.len() {
                return Err("normalized chunk sizes do not cover input".into());
            }
            if let Some(finish) = stored.get("finish_step")
                && finish.as_bool().ok_or("invalid normalized finish step")?
            {
                chunks.push(object(&[("delta_text", V::text("‹finish›"))])?);
            }
        } else if let Some(schedule) = stored.get("chunks") {
            if stored.get("finish_step").is_some() {
                return Err("finish step requires a chunk schedule".into());
            }
            for chunk in schedule.items()? {
                chunks.push(if chunk.as_text().is_ok() {
                    object(&[("delta_text", chunk.clone())])?
                } else {
                    chunk.clone()
                });
            }
        } else {
            if stored.get("finish_step").is_some() {
                return Err("finish step requires a chunk schedule".into());
            }
            chunks.push(object(&[("delta_text", stored.field("input")?.clone())])?);
        }
        request.set("chunks", V::array(chunks))?;
        if request.digest()? != identity {
            return Err("normalized request digest mismatch".into());
        }
        result.set(identity, request)?;
    }
    Ok(result)
}

/// Expand authored storage into the existing case contract.
pub fn family<V: Value>(document: &V) -> Result<V> {
    if !is_normalized(document) {
        return Ok(document.clone());
    }
    let library = requests(document)?;
    let name = document.field("family")?.clone();
    let mut cases = V::object();
    for (identity, stored) in document.field("cases")?.entries()? {
        fields(
            stored,
            &["description", "display_id", "request", "golden"],
            &["lifecycle", "scenario", "policy", "historical_ids"],
        )?;
        let mut case = object(&[
            ("lifecycle", V::text("active")),
            ("scenario", V::text(identity)),
            ("policy", V::array(vec![])),
            ("historical_ids", V::array(vec![])),
        ])?;
        for (field, value) in stored.entries()? {
            case.set(field, value.clone())?;
        }
        let request = stored.field("request")?;
        case.set(
            "request",
            if request.is_null() {
                V::null()
            } else {
                reference(&library, request)?
            },
        )?;
        let golden = stored.field("golden")?;
        case.set(
            "golden",
            if golden.is_null() {
                V::null()
            } else {
                object(&[("assembled", golden.clone())])?
            },
        )?;
        cases.set(identity, case)?;
    }
    let header = object(&[("family", name)])?;
    object(&[
        ("input_document", header.clone()),
        ("golden_document", header),
        ("cases", cases),
    ])
}

/// Reconstruct complete per-record metadata; references are never deep-merged.
pub fn metadata<V: Value>(document: &V, stored: &V) -> Result<V> {
    let defaults = document.field("record_defaults")?;
    let origins = document.field("origins")?;
    let mut producer = reference(
        document.field("producers")?,
        stored
            .get("producer")
            .unwrap_or(defaults.field("producer")?),
    )?;
    fields(&producer, &[], &["captured_with", "capture_origin"])?;
    if let Some(origin) = producer.get("capture_origin").filter(|v| !v.is_null()) {
        let origin = reference(origins, origin)?;
        producer.set("capture_origin", origin)?;
    }
    let mut context = reference(
        document.field("contexts")?,
        stored.get("context").unwrap_or(defaults.field("context")?),
    )?;
    if context.get("captured_with").is_some() || context.get("capture_origin").is_some() {
        return Err("normalized context repeats producer fields".into());
    }
    for (name, value) in producer.entries()? {
        context.set(name, value.clone())?;
    }
    Ok(context)
}

/// Expand a latest snapshot without touching historical files or literal payloads.
pub fn snapshot<V: Value>(document: &V, authored: &V) -> Result<V> {
    if !is_normalized(document) {
        return Ok(document.clone());
    }
    fields(
        document,
        &[
            "schema_version",
            "provenance",
            "document",
            "origins",
            "producers",
            "contexts",
            "record_defaults",
            "records",
        ],
        &[
            "capture_selection",
            "execution_proof",
            "historical_evidence",
        ],
    )?;
    let library = requests(authored)?;
    let origins = document.field("origins")?;
    let mut provenance = document.field("provenance")?.clone();
    if let Some(origin) = provenance.get("origin") {
        let origin = reference(origins, origin)?;
        provenance.set("origin", origin)?;
    }
    let defaults = document.field("record_defaults")?;
    fields(defaults, &["member", "producer", "context"], &[])?;
    let mut records = V::object();
    for (identity, stored) in document.field("records")?.entries()? {
        fields(
            stored,
            &["case_key"],
            &[
                "request",
                "stimulus",
                "output",
                "error",
                "unavailable",
                "producer",
                "context",
                "member",
                "suppress_inherited_from",
            ],
        )?;
        if stored.get("request").is_some() == stored.get("stimulus").is_some() {
            return Err("normalized record needs exactly one stimulus".into());
        }
        let states: Vec<_> = ["output", "error", "unavailable"]
            .into_iter()
            .filter(|key| stored.get(key).is_some())
            .collect();
        if states.len() != 1 {
            return Err("normalized record needs exactly one observation".into());
        }
        let context = metadata(document, stored)?;
        let stimulus = if let Some(request) = stored.get("request") {
            object(&[
                ("inline", reference(&library, request)?),
                ("semantic_sha256", request.clone()),
            ])?
        } else {
            stored.field("stimulus")?.clone()
        };
        let state = states[0];
        let value = object(&[
            ("case_key", stored.field("case_key")?.clone()),
            ("document", context),
            ("stimulus", stimulus),
            (
                "observation",
                object(&[(
                    if state == "output" { "value" } else { state },
                    stored.field(state)?.clone(),
                )])?,
            ),
        ])?;
        let mut wrapped = object(&[
            (
                "member",
                stored
                    .get("member")
                    .unwrap_or(defaults.field("member")?)
                    .clone(),
            ),
            ("value", value),
        ])?;
        if let Some(value) = stored.get("suppress_inherited_from") {
            wrapped.set("suppress_inherited_from", value.clone())?;
        }
        records.set(identity, wrapped)?;
    }
    let mut result = V::object();
    for (name, value) in document.entries()? {
        if ![
            "origins",
            "producers",
            "contexts",
            "record_defaults",
            "records",
        ]
        .contains(&name)
        {
            result.set(name, value.clone())?;
        }
    }
    result.set("schema_version", V::integer(2))?;
    result.set("provenance", provenance)?;
    result.set("records", records)?;
    Ok(result)
}
