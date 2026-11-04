// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, assemble,
    create_unified_parser_for_family,
};

#[test]
fn unbalanced_prose_quotes_cannot_hide_the_reasoning_closer() {
    unsafe { std::env::set_var("DYNAMO_PARSERS_DISABLE_QUOTE_AWARE_PARSING", "1") };
    let reasoning = "`load_config`'s check is fine, but `open('/tmp/x')` is not.\n";
    let text = "\n\nI've found the bug in `app.py`.";
    let mut parser = create_unified_parser_for_family("qwen3", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Reasoning,
            ..UnifiedParserInit::default()
        })
        .unwrap();
    let mut events = parser.push(&format!("{reasoning}</think>{text}")).unwrap();
    events.extend(parser.finish().unwrap().events);
    assert_eq!(
        assemble(&events),
        [
            UnifiedEvent::Reasoning {
                text: reasoning.into()
            },
            UnifiedEvent::Text { text: text.into() },
        ]
    );
}
