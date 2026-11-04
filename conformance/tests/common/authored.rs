// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

pub fn load(conformance: &Path, family: &str) -> serde_json::Value {
    let decoded = std::process::Command::new("python3")
        .args(["-c", "import json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[2]); import unified_history; path=Path(sys.argv[1]); print(json.dumps(unified_history._parse_yaml(path,path.read_bytes())))"])
        .arg(conformance.join("fixtures-unified-v2/families").join(family).join("inputs_and_golden.yaml"))
        .arg(conformance.join("utils/src"))
        .output()
        .expect("decode authored family with the shared storage reader");
    assert!(
        decoded.status.success(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    serde_json::from_slice(&decoded.stdout).expect("expanded authored family")
}
