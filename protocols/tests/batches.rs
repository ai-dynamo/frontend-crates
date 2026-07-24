// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::{
    BatchCompletionWindow, BatchEndpoint, BatchRequest, BatchRequestArgs, OpenAIFile,
};

#[test]
fn batch_request_types_are_reexported() {
    let request = BatchRequestArgs::default()
        .input_file_id("file-123")
        .endpoint(BatchEndpoint::V1Completions)
        .completion_window(BatchCompletionWindow::W24H)
        .build()
        .unwrap();

    let value = serde_json::to_value(&request).unwrap();
    assert_eq!(value["input_file_id"], "file-123");
    assert_eq!(value["endpoint"], "/v1/completions");
    assert_eq!(value["completion_window"], "24h");

    let round_trip: BatchRequest = serde_json::from_value(value).unwrap();
    assert_eq!(round_trip, request);
}

#[test]
fn file_response_types_are_reexported() {
    let file: OpenAIFile = serde_json::from_value(serde_json::json!({
        "id": "file-123",
        "object": "file",
        "bytes": 42,
        "created_at": 1_700_000_000,
        "expires_at": null,
        "filename": "batch.jsonl",
        "purpose": "batch",
        "status": null,
        "status_details": null
    }))
    .unwrap();

    assert_eq!(file.id, "file-123");
    assert_eq!(file.filename, "batch.jsonl");
}
