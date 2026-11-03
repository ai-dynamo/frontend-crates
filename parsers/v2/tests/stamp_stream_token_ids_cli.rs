// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use dynamo_parsers_v2::decode_harmony;

#[test]
fn input_cli_preserves_eof_scalar_text_before_replacing_the_file() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after the Unix epoch")
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!(
        "stamp-stream-token-ids-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&temp_dir).expect("create isolated CLI test directory");
    let input_path = temp_dir.join("fixture.yaml");

    let scalar_forms = [
        "Hello",
        "''",
        "'Hello'",
        "|-\n        Hello",
        "|\n        Hello",
        ">-\n        Hello",
    ];
    let line_endings = ["", "\n", "\r\n"];
    let mut checked = 0;

    for scalar in scalar_forms {
        for ending in line_endings {
            let original =
                format!("cases:\n  one:\n    chunks:\n    - delta_text: {scalar}{ending}");
            fs::write(&input_path, &original).expect("write source fixture");

            let output = Command::new(env!("CARGO_BIN_EXE_stamp_stream_token_ids"))
                .arg("--input")
                .arg(&input_path)
                .output()
                .expect("run stamp_stream_token_ids --input");
            assert!(
                output.status.success(),
                "CLI failed for scalar {scalar:?} and ending {ending:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );

            let stamped = fs::read_to_string(&input_path).expect("read stamped fixture");
            let source: serde_yaml::Value =
                serde_yaml::from_str(&original).expect("parse original fixture");
            let result: serde_yaml::Value =
                serde_yaml::from_str(&stamped).expect("parse rewritten fixture");
            let source_text = source["cases"]["one"]["chunks"][0]["delta_text"]
                .as_str()
                .expect("source chunk text");
            let result_text = result["cases"]["one"]["chunks"][0]["delta_text"]
                .as_str()
                .expect("rewritten chunk text");
            assert_eq!(
                result_text, source_text,
                "scalar {scalar:?}, ending {ending:?}"
            );

            let token_ids = result["cases"]["one"]["chunks"][0]["delta_token_ids"]
                .as_sequence()
                .expect("rewritten token IDs")
                .iter()
                .map(|id| id.as_u64().expect("numeric token ID") as u32)
                .collect::<Vec<_>>();
            assert_eq!(
                decode_harmony(&token_ids).expect("decode stamped token IDs"),
                source_text,
                "scalar {scalar:?}, ending {ending:?}"
            );

            let second_run = Command::new(env!("CARGO_BIN_EXE_stamp_stream_token_ids"))
                .arg("--input")
                .arg(&input_path)
                .output()
                .expect("rerun stamp_stream_token_ids --input");
            assert!(second_run.status.success(), "CLI rerun failed");
            assert_eq!(
                fs::read_to_string(&input_path).expect("read rerun fixture"),
                stamped,
                "stamping must be idempotent"
            );
            checked += 1;
        }
    }

    fs::remove_dir_all(&temp_dir).expect("remove isolated CLI test directory");
    assert_eq!(checked, 18, "six scalar forms across three line endings");
}

#[test]
fn input_cli_preserves_case_order_and_leaves_rejected_yaml_untouched() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after the Unix epoch")
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!(
        "stamp-stream-token-ids-order-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&temp_dir).expect("create isolated CLI test directory");
    let input_path = temp_dir.join("fixture.yaml");
    let source = "cases:\n  z:\n    chunks:\n    - delta_text: Hello\n  a:\n    chunks:\n    - delta_text: World\n";
    fs::write(&input_path, source).expect("write unsorted source fixture");

    let output = Command::new(env!("CARGO_BIN_EXE_stamp_stream_token_ids"))
        .arg("--input")
        .arg(&input_path)
        .output()
        .expect("run stamp_stream_token_ids --input");
    assert!(
        output.status.success(),
        "CLI failed for unsorted cases: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stamped = fs::read_to_string(&input_path).expect("read stamped fixture");
    let fixture: serde_yaml::Value = serde_yaml::from_str(&stamped).expect("parse stamped fixture");
    let cases = fixture["cases"].as_mapping().expect("cases mapping");
    let case_ids = cases
        .keys()
        .map(|key| key.as_str().expect("string case ID"))
        .collect::<Vec<_>>();
    assert_eq!(case_ids, ["z", "a"], "preserve YAML document order");
    for (case_id, expected_text) in [("z", "Hello"), ("a", "World")] {
        let chunk = &fixture["cases"][case_id]["chunks"][0];
        let text = chunk["delta_text"].as_str().expect("chunk text");
        let token_ids = chunk["delta_token_ids"]
            .as_sequence()
            .expect("token IDs")
            .iter()
            .map(|id| id.as_u64().expect("numeric token ID") as u32)
            .collect::<Vec<_>>();
        assert_eq!(text, expected_text);
        assert_eq!(
            decode_harmony(&token_ids).expect("decode token IDs"),
            expected_text
        );
    }

    let invalid = "cases:\n  one:\n    chunks:\n    - delta_text: 42\n";
    fs::write(&input_path, invalid).expect("write rejected source fixture");
    let output = Command::new(env!("CARGO_BIN_EXE_stamp_stream_token_ids"))
        .arg("--input")
        .arg(&input_path)
        .output()
        .expect("run stamp_stream_token_ids --input");
    assert!(!output.status.success(), "CLI accepted a non-string chunk");
    assert_eq!(
        fs::read_to_string(&input_path).expect("read rejected fixture"),
        invalid,
        "rejected input must remain byte-for-byte unchanged"
    );

    fs::remove_dir_all(&temp_dir).expect("remove isolated CLI test directory");
}
