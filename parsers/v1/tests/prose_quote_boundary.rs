// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::{BasicReasoningParser, ReasoningParser};

#[test]
fn prose_quotes_do_not_shield_v1_reasoning_closers() {
    for (reasoning, answer) in [
        ("analysis \"", "Answer\""),
        (
            r"Use re.split(r'(\s)', text). Let's final.",
            "The code prints ['a', ' ', 'b'].",
        ),
    ] {
        let input = format!("{reasoning}</think>{answer}");
        let mut batch = BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true);
        let result = batch.detect_and_parse_reasoning(&input, &[]);
        assert_eq!(result.reasoning_text, reasoning);
        assert_eq!(result.normal_text, answer);

        let mut stream = BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true);
        let result = stream.parse_reasoning_streaming_incremental(&input, &[]);
        assert_eq!(result.reasoning_text, reasoning);
        assert_eq!(result.normal_text, answer);
        let tail = stream.finish_reasoning_stream();
        assert!(tail.reasoning_text.is_empty());
        assert!(tail.normal_text.is_empty());
    }
}
