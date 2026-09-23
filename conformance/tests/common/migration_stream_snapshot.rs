// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{Fixture, common, merge_dynamo, stream_dynamo_dirs};
use serde_json::{Map, Value, json};
use std::path::PathBuf;

// Included by the real stream harness so this exercises its private deserializer
// and merge function, including fields Python resolvers do not consume.
#[test]
#[ignore = "requires MIGRATION_FIXTURES and MIGRATION_OUTPUT; run for both source revisions"]
fn export_migration_observations() {
    let root = PathBuf::from(std::env::var_os("MIGRATION_FIXTURES").expect("fixture root"));
    let output = PathBuf::from(std::env::var_os("MIGRATION_OUTPUT").expect("output path"));
    let root = root.join("toolcalling/fixtures-stream-v1");
    let mut inputs = Vec::new();
    common::collect_yaml(&root.join("inputs"), &mut inputs);
    inputs.sort();
    assert!(
        !inputs.is_empty(),
        "empty corpus cannot prove migration parity"
    );
    let mut records = Map::new();
    let mut inventory = Map::new();
    let released: Vec<_> = common::version_dirs_ascending(&root, "dynamo_v2-")
        .into_iter()
        .filter(|path| !path.file_name().unwrap().to_str().unwrap().contains('+'))
        .collect();
    for (mode, dirs) in [
        ("release", released),
        ("current", stream_dynamo_dirs(&root)),
    ] {
        let names: Vec<_> = dirs
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        // Inventory is a set; the selected observations below prove fold order.
        let mut inventory_names = names.clone();
        inventory_names.sort();
        inventory.insert(mode.to_string(), json!(inventory_names));
        for end in 1..=dirs.len() {
            let selection = &names[end - 1];
            for path in &inputs {
                let relative = path.strip_prefix(root.join("inputs")).unwrap();
                let mut fixture: Fixture =
                    serde_yaml::from_slice(&std::fs::read(path).unwrap()).unwrap();
                for directory in &dirs[..end] {
                    merge_dynamo(&mut fixture, directory, relative);
                }
                let cases: Map<String, Value> = fixture.cases.iter().map(|(id, case)| {
                    let chunks: Vec<_> = case.chunks.iter().map(|chunk| {
                        let expected: Map<String, Value> = chunk.expected.iter().map(|(implementation, events)| {
                            let events: Vec<_> = events.iter().map(|event| json!({
                                "index": event.index, "id": event.id, "name": event.name,
                                "arguments": event.arguments, "complete": event.complete,
                            })).collect();
                            (implementation.clone(), json!(events))
                        }).collect();
                        json!({"delta_text": chunk.delta_text, "delta_token_ids": chunk.delta_token_ids,
                               "finish_reason": chunk.finish_reason, "expected": expected,
                               "normal_text": chunk.normal_text})
                    }).collect();
                    (id.clone(), json!({"tools": case.tools, "unavailable": case.unavailable, "chunks": chunks}))
                }).collect();
                let key = format!("{mode}/{selection}/{}", relative.display());
                assert!(
                    records
                        .insert(
                            key,
                            json!({"family": fixture.family, "mode": fixture.mode, "cases": cases})
                        )
                        .is_none()
                );
            }
        }
    }
    assert!(!records.is_empty());
    std::fs::write(
        output,
        serde_json::to_vec(&json!({"inventory": inventory, "records": records})).unwrap(),
    )
    .unwrap();
}
