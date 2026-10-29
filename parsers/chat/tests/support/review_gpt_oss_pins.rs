// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn review_f1_gpt_oss_stream_pin_rejects_unrelated_changes() {
    let record = fixture()
        .into_iter()
        .find(|record| {
            record["version"] == "auto"
                && record["name"] == "harmony+gpt_oss/metadata/reason_text/thinking_off"
        })
        .expect("GPT-OSS metadata record");
    let (output, _) = run_stream(&record).await.expect("recorded stream");
    assert_ne!(output, record["output"]);
    assert!(gpt_oss_reasoning_kept(&record, &output));
    assert!(!text_and_rest(&output).0.is_empty());
    assert!(
        output
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item.pointer("/data/usage").is_some() })
    );

    let mut accepted = Vec::new();
    for mutation in [
        "analysis leaked into content",
        "choice index changed",
        "usage dropped",
    ] {
        let mut changed = output.clone();
        for item in changed.as_array_mut().unwrap() {
            if mutation == "usage dropped"
                && let Some(data) = item.get_mut("data").and_then(Value::as_object_mut)
            {
                data.remove("usage");
            }
            for choice in item["data"]["choices"].as_array_mut().into_iter().flatten() {
                if mutation == "choice index changed" {
                    choice["index"] = json!(7);
                }
                if mutation == "analysis leaked into content" {
                    let delta = choice["delta"].as_object_mut().unwrap();
                    if let Some(reasoning) = delta.remove("reasoning_content") {
                        let content = delta.get("content").and_then(Value::as_str).unwrap_or("");
                        delta.insert(
                            "content".into(),
                            json!(format!("{}{content}", reasoning.as_str().unwrap())),
                        );
                    }
                }
            }
        }
        assert_ne!(
            changed, output,
            "mutation did not change the control: {mutation}"
        );
        if changed != record["output"] && gpt_oss_reasoning_kept(&record, &changed) {
            accepted.push(mutation);
        }
    }
    assert!(
        accepted.is_empty(),
        "GPT-OSS stream pin accepted unrelated regressions: {accepted:?}"
    );
}

#[test]
fn review_f2_gpt_oss_rule_pin_rejects_a_reverted_default_case() {
    let record = fixture()
        .into_iter()
        .find(|record| {
            record["kind"] == "predicates"
                && record["version"] == "auto"
                && record["name"] == "harmony+gpt_oss/thinking_off"
        })
        .expect("GPT-OSS thinking-off predicate record");
    let mut mismatches = check_predicates(&record);
    assert_eq!(mismatches.len(), 4);
    assert!(gpt_oss_rule_kept(&record, &mismatches));

    // Returning the recorded true for this mode removes only its mismatch.
    let reverted =
        "request_disables_reasoning_with_family(Some(\"enabled\"), family): got false, dynamo true";
    assert!(mismatches.iter().any(|mismatch| mismatch == reverted));
    mismatches.retain(|mismatch| mismatch != reverted);
    assert_eq!(mismatches.len(), 3);
    assert!(
        !gpt_oss_rule_kept(&record, &mismatches),
        "GPT-OSS rule pin accepted reasoning disabled with default_thinking_mode=enabled and enable_thinking=false; the other three mode mismatches still pin the same record"
    );
}
