// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::decode_harmony;
use serde_json::Value;

#[test]
fn authored_harmony_parallel_chunks_decode_to_their_exact_text() {
    let corpus: Value =
        serde_json::from_str(include_str!("../utils/src/parallel_boundary_cases.json")).unwrap();
    let cases = corpus["cases"]["harmony"].as_object().unwrap();
    for (case_id, case) in cases {
        for (index, chunk) in case["chunks"].as_array().unwrap().iter().enumerate() {
            let ids: Vec<u32> = chunk["delta_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
                .collect();
            let decoded = decode_harmony(&ids).unwrap();
            let authored = chunk["delta_text"].as_str().unwrap();
            // Full-response equality cannot detect punctuation delivered in
            // the wrong chunk: the decoder may join a dot to the next name.
            assert_eq!(decoded, authored, "{case_id} chunk {index}");
        }
    }
}
