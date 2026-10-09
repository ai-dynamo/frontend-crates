// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::ChatCompletionTokenLogprob;
use serde_json::json;

#[test]
fn candidate_token_ids_survive_a_wire_round_trip() {
    let wire = json!({
        "token": "a",
        "token_id": 42,
        "logprob": -0.1,
        "bytes": [97],
        "top_logprobs": [
            {"token": "\u{0}", "token_id": 115, "logprob": -0.2, "bytes": [0]},
            {"token": "\u{0}", "token_id": 120, "logprob": -0.3, "bytes": [0]}
        ]
    });
    let logprob: ChatCompletionTokenLogprob = serde_json::from_value(wire).unwrap();
    let serialized = serde_json::to_value(logprob).unwrap();

    assert_eq!(serialized["token_id"], 42);
    assert_eq!(serialized["top_logprobs"][0]["token_id"], 115);
    assert_eq!(serialized["top_logprobs"][1]["token_id"], 120);
}

#[test]
fn standard_candidates_keep_the_openai_wire_shape() {
    let wire = json!({
        "token": "a",
        "logprob": -0.5,
        "bytes": null,
        "top_logprobs": [
            {"token": "a", "logprob": -0.5, "bytes": null},
            {"token": "b", "logprob": -1.0, "bytes": [98]}
        ]
    });
    let logprob: ChatCompletionTokenLogprob = serde_json::from_value(wire.clone()).unwrap();

    assert_eq!(serde_json::to_value(logprob).unwrap(), wire);
}
