// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::{
    ChatCompletionRequestToolMessageContentPart as Tool,
    ChatCompletionRequestUserMessageContentPart as User,
};
use serde_json::json;

#[test]
fn custom_preserves_all_fields_for_users_and_tools() {
    let value = json!({"type":"custom", "kind":"chemistry", "fields":{
        "chemistry":{"data_base64":"AA=="}, "metadata":[null, true, 3, "x"], "other":{}
    }});
    let user: User = serde_json::from_value(value.clone()).unwrap();
    let tool: Tool = serde_json::from_value(value.clone()).unwrap();
    assert!(matches!(user, User::Custom(_)));
    assert!(matches!(tool, Tool::Custom(_)));
    assert_eq!(serde_json::to_value(user).unwrap(), value);
    assert_eq!(serde_json::to_value(tool).unwrap(), value);
}

#[test]
fn malformed_envelopes_and_unknown_tags_are_rejected() {
    for value in [
        json!({"type":"chemistry","chemistry":{}}),
        json!({"type":"custom","fields":{}}),
        json!({"type":"custom","kind":"","fields":{}}),
        json!({"type":"custom","kind":"  ","fields":{}}),
        json!({"type":"custom","kind":"x","fields":null}),
        json!({"type":"custom","kind":"x","fields":[]}),
        json!({"type":"custom","kind":"x","fields":{},"lost":true}),
    ] {
        assert!(
            serde_json::from_value::<User>(value.clone()).is_err(),
            "{value}"
        );
        assert!(
            serde_json::from_value::<Tool>(value.clone()).is_err(),
            "{value}"
        );
    }
}

#[test]
fn builtin_variants_remain_typed_and_validate() {
    assert!(matches!(
        serde_json::from_value::<User>(
            json!({"type":"image_url","image_url":{"url":"https://example.com/x"}})
        )
        .unwrap(),
        User::ImageUrl(_)
    ));
    assert!(matches!(
        serde_json::from_value::<User>(
            json!({"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}})
        )
        .unwrap(),
        User::InputAudio(_)
    ));
    for value in [
        json!({"type":"image_url","image_url":{"url":123}}),
        json!({"type":"video_url","video_url":{"url":false}}),
        json!({"type":"audio_url","audio_url":[]}),
        json!({"type":"input_audio","input_audio":{"data":"AA==","format":"invalid"}}),
    ] {
        assert!(serde_json::from_value::<User>(value).is_err());
    }
}
