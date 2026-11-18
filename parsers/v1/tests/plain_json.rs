// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::{
    config::ParserConfig,
    jail::{Annotated, JailedStream},
    json::{JsonParserConfig, JsonParserType, try_tool_call_parse_basic_json},
    parsers::{detect_and_parse_tool_call_with_recovery, get_tool_parser_map},
};
use dynamo_protocols::types::{ChatCompletionMessageContent, FinishReason};
use futures::StreamExt;

#[tokio::test]
async fn ordinary_arrays_survive_basic_json_batch_and_stream() {
    for (&name, config) in get_tool_parser_map() {
        if !matches!(&config.parser_config, ParserConfig::Json(c) if matches!(c.parser_type, JsonParserType::Basic))
        {
            continue;
        }
        for text in [
            "[5,0,8]",
            "[]",
            r#"["a","b"]"#,
            r#"[{"a":1},{"b":2}]"#,
            "결과: [5,0,8]",
            " \n[5,0,8]\t ",
        ] {
            let (calls, content) = detect_and_parse_tool_call_with_recovery(text, Some(name), None)
                .await
                .unwrap();
            assert!(calls.is_empty(), "{name}");
            assert_eq!(content.as_deref(), Some(text), "{name}");
            for split in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
                let input = [&text[..split], &text[split..]].into_iter().enumerate().map(|(i, part)| Annotated {
                    data: Some(serde_json::from_value(serde_json::json!({
                        "id":"test", "object":"chat.completion.chunk", "created":0, "model":"test",
                        "choices":[{"index":0,"delta":{"content":part},"finish_reason":if i == 1 { Some("stop") } else { None }}]
                    })).unwrap()),
                    id: None, event: None, comment: None, error: None,
                });
                let output: Vec<_> = JailedStream::builder()
                    .tool_call_parser(name)
                    .build()
                    .apply_with_finish_reason(futures::stream::iter(input))
                    .collect()
                    .await;
                let mut content = String::new();
                let mut finishes = Vec::new();
                for chunk in output {
                    assert!(chunk.error.is_none());
                    for choice in chunk.data.into_iter().flat_map(|data| data.choices) {
                        assert!(choice.delta.tool_calls.as_ref().is_none_or(Vec::is_empty));
                        if let Some(ChatCompletionMessageContent::Text(part)) = choice.delta.content
                        {
                            content.push_str(&part);
                        }
                        finishes.extend(choice.finish_reason);
                    }
                }
                assert_eq!(content, text, "{name}, split={split}");
                assert_eq!(finishes, [FinishReason::Stop]);
            }
        }
    }
}

#[test]
fn framed_and_guided_empty_arrays_remain_empty_results() {
    for (text, config) in [
        (
            "[]",
            JsonParserConfig {
                bare_json_mode: true,
                ..Default::default()
            },
        ),
        ("<TOOLCALL>[]</TOOLCALL>", JsonParserConfig::default()),
    ] {
        let (calls, content) = try_tool_call_parse_basic_json(text, &config, None).unwrap();
        assert!(calls.is_empty());
        assert_eq!(content.as_deref(), Some(""));
    }
}
