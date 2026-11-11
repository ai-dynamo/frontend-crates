// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "renderer-runner")]

use dynamo_conformance_fixtures_v2::{renderer, renderer_runner};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

mod common;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}

#[test]
fn recorded_renderer_matches_live_execution() {
    let suite = renderer::load_suite(&root())
        .unwrap()
        .expect("renderer fixture store");
    let models = suite["models"].as_array().unwrap();
    assert_eq!(models.len(), 5);
    let checked = renderer_runner::verify_suite(&root(), &suite).unwrap();
    assert!(
        checked.checked >= 100,
        "all five model corpora must execute"
    );
    common::unified_capture::report_comparison(
        "renderer-reference",
        checked.checked,
        &checked.reference_differences,
        false,
    );
    common::unified_capture::report_comparison(
        "renderer-recorded",
        checked.checked,
        &checked.recorded_differences,
        true,
    );
    assert!(
        checked.recorded_differences.is_empty(),
        "unrecorded renderer output changes: {:?}",
        checked.recorded_differences
    );
}

#[test]
fn prompt_and_metadata_corruption_fail_the_actual_replay_gate() {
    let suite = renderer::load_suite(&root()).unwrap().unwrap();
    let model = suite["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["model"]["id"] == "kimi_k3")
        .unwrap();
    let case = model["inputs"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| {
            let result = &model["dynamo"]["results"][case["id"].as_str().unwrap()]["result"];
            result["kind"] == "prompt"
                && result["segments"]
                    .as_array()
                    .is_some_and(|segments| !segments.is_empty())
        })
        .unwrap();
    let id = case["id"].as_str().unwrap();
    let mut one = model.clone();
    one["inputs"]["cases"] = json!([case]);
    let baseline = json!({"models":[one]});
    assert!(
        renderer_runner::verify_suite(&root(), &baseline)
            .unwrap()
            .recorded_differences
            .is_empty()
    );
    for change in ["text", "permission", "pending", "media"] {
        let mut damaged = baseline.clone();
        let result = &mut damaged["models"][0]["dynamo"]["results"][id]["result"];
        match change {
            "text" => {
                result["text"] = format!("{}changed", result["text"].as_str().unwrap()).into()
            }
            "permission" => {
                result["segments"][0]["allow_special"] =
                    (!result["segments"][0]["allow_special"].as_bool().unwrap()).into()
            }
            "pending" => {
                result["pending_segments"] =
                    (result["pending_segments"].as_u64().unwrap() + 1).into()
            }
            "media" => result["media_message_order"] = json!([999]),
            _ => unreachable!(),
        }
        assert_eq!(
            renderer_runner::verify_suite(&root(), &damaged)
                .unwrap()
                .recorded_differences
                .len(),
            1,
            "{change} corruption must fail recorded verification"
        );
    }
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let path = entry.unwrap().path();
        let target = destination.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_tree(&path, &target);
        } else {
            fs::copy(path, target).unwrap();
        }
    }
}

#[test]
fn source_assets_request_order_and_capture_membership_are_bound() {
    for mutation in [
        "asset",
        "request",
        "order",
        "missing",
        "producer",
        "reference_input",
        "metadata_origin",
        "transformation",
        "blank_authority",
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join(renderer::STORE);
        copy_tree(&root().join(renderer::STORE), &store);
        let suite = renderer::load_suite(temporary.path()).unwrap().unwrap();
        let model = &suite["models"][0];
        let id = model["model"]["id"].as_str().unwrap();
        match mutation {
            "asset" => {
                let asset = model["model"]["assets"][0]["path"].as_str().unwrap();
                let mut bytes = fs::read(store.join(asset)).unwrap();
                bytes.push(b' ');
                fs::write(store.join(asset), bytes).unwrap();
            }
            "request" | "order" => {
                let path = store.join(format!("families/{id}/inputs.json"));
                let mut inputs: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                if mutation == "request" {
                    inputs["cases"][0]["request"]["messages"][0]["content"] = "changed".into();
                } else {
                    let request = inputs["cases"][0]["request"].as_object_mut().unwrap();
                    let value = request.shift_remove("messages").unwrap();
                    request.insert("messages".into(), value);
                }
                fs::write(path, serde_json::to_vec_pretty(&inputs).unwrap()).unwrap();
            }
            "missing" | "producer" => {
                let path = store.join(format!("families/{id}/dynamo.json"));
                let mut capture: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                if mutation == "missing" {
                    let results = capture["results"].as_object_mut().unwrap();
                    let first = results.keys().next().unwrap().clone();
                    results.remove(&first);
                } else {
                    capture["producer"]["source_sha256"] = "invalid".into();
                }
                fs::write(path, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
            }
            "reference_input" | "metadata_origin" | "transformation" | "blank_authority" => {
                let path = store.join(format!("families/{id}/reference.json"));
                let mut capture: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                let first = capture["results"]
                    .as_object_mut()
                    .unwrap()
                    .values_mut()
                    .next()
                    .unwrap();
                if mutation == "transformation" {
                    first["transformations"] = json!([{"name":"rewrite","authority":null}]);
                } else if mutation == "blank_authority" {
                    first["metadata_origin"] = " \t\n".into();
                } else {
                    first[mutation] = Value::Null;
                }
                fs::write(path, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
            }
            _ => unreachable!(),
        }
        assert!(
            renderer::load_suite(temporary.path()).is_err(),
            "{mutation} corruption must fail evidence validation"
        );
    }
}

#[test]
fn strict_controls_and_output_ownership_cannot_be_silently_ignored() {
    let suite = renderer::load_suite(&root()).unwrap().unwrap();
    let model = &suite["models"][0];
    let mut inputs = model["inputs"].clone();
    inputs["cases"][0]["settings"]["unknown_control"] = true.into();
    assert!(renderer::validate_inputs(&inputs, model["model"]["id"].as_str().unwrap()).is_err());
    let mut inputs = model["inputs"].clone();
    inputs["cases"][0]["request"]["add_generation_prompt"] = false.into();
    assert!(renderer::validate_inputs(&inputs, model["model"]["id"].as_str().unwrap()).is_err());
    assert!(renderer_runner::capture(&root(), &root().join(renderer::STORE), None).is_err());
    assert!(
        renderer_runner::capture(
            &root(),
            &root().join(renderer::STORE).join("new-directory"),
            None
        )
        .is_err()
    );
}

#[test]
fn supported_controls_change_the_actual_formatter_output() {
    let suite = renderer::load_suite(&root()).unwrap().unwrap();
    for model in suite["models"].as_array().unwrap() {
        let cases = model["inputs"]["cases"].as_array().unwrap();
        for control in ["generation", "thinking"] {
            let on_id = format!("RENDERER.controls-{control}-enabled");
            let off_id = format!("RENDERER.controls-{control}-disabled");
            let enabled = cases.iter().find(|case| case["id"] == on_id).unwrap();
            let disabled = cases.iter().find(|case| case["id"] == off_id).unwrap();
            let reference = &model["reference"]["results"];
            // Some pinned templates have no thinking switch; generation applies to all.
            if control == "generation"
                || reference[&on_id]["result"]["text"] != reference[&off_id]["result"]["text"]
            {
                assert_ne!(
                    renderer_runner::render_case(&root(), &model["model"], enabled).unwrap()["text"],
                    renderer_runner::render_case(&root(), &model["model"], disabled).unwrap()["text"],
                    "{}/{} controls must reach the public formatter",
                    model["model"]["id"],
                    control
                );
            }
        }
    }
}

#[test]
fn reference_equality_preserves_segment_boundaries_and_error_class() {
    let result = json!({"kind":"prompt","text":"Hello","segments":null,"pending_segments":0,"media_message_order":null});
    let mut identity = result.clone();
    identity["media_message_order"] = json!([0, 1]);
    assert!(renderer::reference_matches(&result, &identity, 2));
    identity["segments"] = json!([{"text":"Hello","allow_special":false}]);
    assert!(!renderer::reference_matches(&result, &identity, 2));
    assert!(renderer::reference_matches(
        &json!({"kind":"error","error_type":"invalid_request","message":"Python wording"}),
        &json!({"kind":"error","error_type":"invalid_request","message":"Rust wording"}),
        0
    ));
    assert!(!renderer::reference_matches(
        &json!({"kind":"error","error_type":"invalid_request","message":"Rejected"}),
        &json!({"kind":"error","error_type":"render_error","message":"Rejected"}),
        0
    ));
}
