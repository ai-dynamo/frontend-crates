// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Latest renderer evidence, shared by capture, replay and publication.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path},
};

pub const STORE: &str = "conformance/fixtures-renderer";
const MAX_FILE: u64 = 16 * 1024 * 1024;

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn valid_hash(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Read source bytes without following links or accepting path traversal.
pub fn read_source(root: &Path, name: &str) -> Result<Vec<u8>> {
    ensure!(
        !name.is_empty() && !name.contains(['\\', ':']),
        "invalid renderer source path"
    );
    ensure!(
        name.split('/')
            .all(|part| !part.is_empty() && part != "." && part != ".."),
        "invalid renderer source path"
    );
    ensure!(
        Path::new(name)
            .components()
            .all(|part| matches!(part, Component::Normal(_))),
        "invalid renderer source path"
    );
    let mut path = root.to_path_buf();
    for component in Path::new(name).components() {
        path.push(component);
        ensure!(
            !fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "renderer source cannot be a symlink: {}",
            path.display()
        );
    }
    let metadata = fs::metadata(&path)?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_FILE,
        "invalid renderer source file: {}",
        path.display()
    );
    fs::read(&path).with_context(|| format!("read renderer source {}", path.display()))
}

pub fn read_json(root: &Path, name: &str) -> Result<(Value, String)> {
    let bytes = read_source(root, name)?;
    let document =
        serde_json::from_slice(&bytes).with_context(|| format!("invalid renderer JSON: {name}"))?;
    Ok((document, sha256(&bytes)))
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub add_generation_prompt: bool,
    #[serde(default)]
    pub chat_template_args: std::collections::HashMap<String, Value>,
    pub request_mode: String,
    pub content_format: String,
    pub exclude_tools_when_tool_choice_none: bool,
}

pub fn validate_inputs(inputs: &Value, model: &str) -> Result<()> {
    ensure!(
        inputs["schema"] == 1 && inputs["model"] == model,
        "renderer inputs identity mismatch"
    );
    let cases = inputs["cases"]
        .as_array()
        .context("renderer inputs need cases")?;
    ensure!(!cases.is_empty(), "renderer inputs cannot be empty");
    let mut ids = BTreeSet::new();
    for case in cases {
        let id = case["id"].as_str().context("missing renderer case ID")?;
        ensure!(
            id.starts_with("RENDERER.") && ids.insert(id),
            "invalid or duplicate renderer case ID: {id}"
        );
        ensure!(
            case["group"].as_str().is_some_and(|s| !s.is_empty())
                && case["description"].as_str().is_some_and(|s| !s.is_empty()),
            "renderer case needs group and description: {id}"
        );
        let settings: Settings = serde_json::from_value(case["settings"].clone())
            .with_context(|| format!("{id}: invalid renderer settings"))?;
        ensure!(
            ["typed", "json"].contains(&settings.request_mode.as_str()),
            "{id}: invalid request mode"
        );
        ensure!(
            ["auto", "openai", "string"].contains(&settings.content_format.as_str()),
            "{id}: invalid content format"
        );
        let request = case["request"]
            .as_object()
            .context("renderer request must be an object")?;
        for key in request.keys() {
            ensure!(
                [
                    "model",
                    "messages",
                    "tools",
                    "tool_choice",
                    "response_format",
                    "reasoning_effort",
                    "template_tools",
                    "mm_processor_kwargs"
                ]
                .contains(&key.as_str()),
                "{id}: unknown renderer request field {key}; controls belong in settings"
            );
        }
        ensure!(
            request.get("model").and_then(Value::as_str).is_some(),
            "{id}: request needs model"
        );
        ensure!(
            request
                .get("messages")
                .and_then(Value::as_array)
                .is_some_and(|v| !v.is_empty()),
            "{id}: request needs messages"
        );
    }
    Ok(())
}

pub fn validate_result(result: &Value, message_count: usize) -> Result<()> {
    match result["kind"].as_str() {
        Some("prompt") => {
            let text = result["text"]
                .as_str()
                .context("renderer prompt needs text")?;
            let pending = result["pending_segments"]
                .as_u64()
                .context("renderer prompt needs pending_segments")?;
            ensure!(
                result.get("segments").is_some() && result.get("media_message_order").is_some(),
                "renderer prompt needs explicit optional metadata"
            );
            if let Some(segments) = result["segments"].as_array() {
                ensure!(
                    pending <= segments.len() as u64,
                    "pending segments exceed segment count"
                );
                let mut assembled = String::new();
                for segment in segments {
                    assembled.push_str(segment["text"].as_str().context("segment needs text")?);
                    ensure!(
                        segment["allow_special"].is_boolean(),
                        "segment needs special-token permission"
                    );
                }
                ensure!(assembled == text, "segment text disagrees with prompt");
            } else {
                ensure!(
                    result["segments"].is_null() && pending == 0,
                    "unsegmented prompt has invalid segment metadata"
                );
            }
            if !result["media_message_order"].is_null() {
                let order = result["media_message_order"]
                    .as_array()
                    .context("media order must be a list")?;
                let indices: BTreeSet<_> = order.iter().map(|item| item.as_u64()).collect();
                ensure!(
                    order.len() == message_count
                        && indices == (0..message_count as u64).map(Some).collect(),
                    "media order must be a permutation of source messages"
                );
            }
        }
        Some("error") => {
            ensure!(
                [Some("invalid_request"), Some("render_error")]
                    .contains(&result["error_type"].as_str()),
                "invalid renderer error type"
            );
            ensure!(
                result["message"].as_str().is_some_and(|s| !s.is_empty()),
                "renderer error needs message"
            );
        }
        Some("unsupported") => ensure!(
            result["reason"].as_str().is_some_and(|s| !s.is_empty()),
            "unsupported renderer result needs reason"
        ),
        _ => bail!("invalid renderer result kind"),
    }
    Ok(())
}

pub fn validate_capture(
    capture: &Value,
    inputs: &Value,
    inputs_sha256: &str,
    manifest_sha256: &str,
    implementation: &str,
) -> Result<()> {
    ensure!(
        capture["schema"] == 1 && capture["model"] == inputs["model"],
        "renderer capture identity mismatch"
    );
    ensure!(
        capture["inputs_sha256"] == inputs_sha256 && capture["manifest_sha256"] == manifest_sha256,
        "renderer capture request/assets binding mismatch"
    );
    let producer = &capture["producer"];
    ensure!(
        producer["implementation"] == implementation
            && producer["version"].as_str().is_some_and(|s| !s.is_empty())
            && producer["source_sha256"].as_str().is_some_and(valid_hash),
        "invalid renderer producer receipt"
    );
    let cases = inputs["cases"]
        .as_array()
        .context("missing renderer cases")?;
    let results = capture["results"]
        .as_object()
        .context("missing renderer capture results")?;
    let expected: BTreeSet<_> = cases
        .iter()
        .map(|case| case["id"].as_str().unwrap())
        .collect();
    ensure!(
        results.keys().map(String::as_str).collect::<BTreeSet<_>>() == expected,
        "missing or extra renderer capture cases"
    );
    for case in cases {
        let observation = &results[case["id"].as_str().unwrap()];
        validate_result(
            &observation["result"],
            case["request"]["messages"].as_array().unwrap().len(),
        )?;
        if implementation == "reference" {
            ensure!(
                observation["reference_input"].is_object()
                    && observation["transformations"].is_array(),
                "reference observation needs actual adapted input and transformations"
            );
            ensure!(
                observation["metadata_origin"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()),
                "reference metadata needs authority"
            );
            for transformation in observation["transformations"].as_array().unwrap() {
                ensure!(
                    transformation["name"]
                        .as_str()
                        .is_some_and(|value| !value.is_empty())
                        && transformation["authority"]
                            .as_str()
                            .is_some_and(|value| !value.is_empty()),
                    "reference transformation needs named authority"
                );
            }
        }
    }
    Ok(())
}

/// Validate current source evidence and build the one lazy browser resource.
pub fn load_suite(root: &Path) -> Result<Option<Value>> {
    let store = root.join(STORE);
    if !store.exists() {
        return Ok(None);
    }
    ensure!(
        !fs::symlink_metadata(&store)?.file_type().is_symlink(),
        "renderer store cannot be a symlink"
    );
    let (manifest, manifest_hash) = read_json(&store, "manifest.json")?;
    ensure!(manifest["schema"] == 1, "unsupported renderer manifest");
    let model_specs = manifest["models"]
        .as_array()
        .context("renderer manifest needs models")?;
    ensure!(!model_specs.is_empty(), "renderer manifest has no models");
    let mut ids = BTreeSet::new();
    let mut cases_by_id = std::collections::BTreeMap::new();
    let mut models = Vec::new();
    for model in model_specs {
        let id = model["id"].as_str().context("missing renderer model ID")?;
        ensure!(
            !id.is_empty()
                && id
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                && ids.insert(id),
            "invalid or duplicate renderer model ID"
        );
        ensure!(
            model["label"].as_str().is_some_and(|s| !s.is_empty())
                && model["repository"]
                    .as_str()
                    .is_some_and(|s| s.contains('/'))
                && model["revision"]
                    .as_str()
                    .is_some_and(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())),
            "renderer model needs pinned source identity"
        );
        ensure!(
            [Some("deepseek_v41"), Some("kimi_k3"), Some("hf")]
                .contains(&model["formatter"].as_str()),
            "unknown renderer formatter"
        );
        let assets = model["assets"]
            .as_array()
            .context("renderer model needs assets")?;
        ensure!(!assets.is_empty(), "renderer model has no assets");
        let mut roles = BTreeSet::new();
        for asset in assets {
            let role = asset["role"].as_str().context("asset needs role")?;
            ensure!(roles.insert(role), "duplicate renderer asset role");
            let name = asset["path"].as_str().context("asset needs path")?;
            let hash = asset["sha256"].as_str().context("asset needs hash")?;
            ensure!(
                valid_hash(hash) && sha256(&read_source(&store, name)?) == hash,
                "renderer asset hash mismatch: {name}"
            );
            ensure!(
                asset["url"]
                    .as_str()
                    .is_some_and(|s| s.starts_with("https://")),
                "asset needs source URL"
            );
        }
        ensure!(
            roles.contains("tokenizer_config"),
            "renderer model needs tokenizer_config"
        );
        ensure!(
            if model["formatter"] == "hf" {
                roles.contains("chat_template")
            } else {
                roles.contains("encoder")
            },
            "renderer reference asset missing"
        );
        let (inputs, inputs_hash) = read_json(&store, &format!("families/{id}/inputs.json"))?;
        validate_inputs(&inputs, id)?;
        for case in inputs["cases"].as_array().unwrap() {
            let case_id = case["id"].as_str().unwrap();
            let identity = (case["group"].clone(), case["description"].clone());
            if let Some(previous) = cases_by_id.insert(case_id.to_owned(), identity.clone()) {
                ensure!(
                    previous == identity,
                    "conflicting shared renderer case identity: {case_id}"
                );
            }
        }
        let (reference, _) = read_json(&store, &format!("families/{id}/reference.json"))?;
        let (dynamo, _) = read_json(&store, &format!("families/{id}/dynamo.json"))?;
        validate_capture(
            &reference,
            &inputs,
            &inputs_hash,
            &manifest_hash,
            "reference",
        )?;
        validate_capture(&dynamo, &inputs, &inputs_hash, &manifest_hash, "dynamo")?;
        models.push(json!({"model":model,"inputs":inputs,"inputs_sha256":inputs_hash,"reference":reference,"dynamo":dynamo}));
    }
    Ok(Some(
        json!({"schema":1,"manifest":manifest,"manifest_sha256":manifest_hash,"models":models}),
    ))
}

/// Compare semantic reference results without changing the recorded evidence.
pub fn reference_matches(expected: &Value, observed: &Value, message_count: usize) -> bool {
    if expected["kind"] != observed["kind"] {
        return false;
    }
    if expected["kind"] == "error" {
        return expected["error_type"] == observed["error_type"];
    }
    if expected["kind"] != "prompt" {
        return expected == observed;
    }
    let mut expected = expected.clone();
    let mut observed = observed.clone();
    for result in [&mut expected, &mut observed] {
        if result["media_message_order"].is_null() {
            result["media_message_order"] = json!((0..message_count).collect::<Vec<_>>());
        }
    }
    expected == observed
}
