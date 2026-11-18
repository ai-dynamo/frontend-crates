use dynamo_parsers_v2::{Tool, ToolParseResult, create_tool_parser_for_family};

#[test]
fn public_parser_preserves_literal_parameters_at_every_utf8_split() {
    let tools = vec![Tool {
        name: "echo".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"text": {"type": "string"}}
        }),
        strict: None,
    }];
    for value in [
        "  Montréal &amp; &lt;tag&gt;\n  ",
        "a</invoke>b</minimax:tool_call>c",
    ] {
        let input = format!(
            "<minimax:tool_call><invoke name=\"echo\"><parameter name=\"text\">{value}</parameter></invoke></minimax:tool_call>"
        );
        for split in (0..=input.len()).filter(|position| input.is_char_boundary(*position)) {
            let mut parser = create_tool_parser_for_family("minimax_m2", &tools).unwrap();
            let mut output = ToolParseResult::default();
            output.append(parser.push(&input[..split]).unwrap());
            output.append(parser.push(&input[split..]).unwrap());
            output.append(parser.finish().unwrap());
            let output = output.coalesce_calls();
            assert_eq!(output.calls.len(), 1, "split {split}");
            assert_eq!(output.calls[0].name.as_deref(), Some("echo"));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&output.calls[0].arguments).unwrap(),
                serde_json::json!({"text": value}),
                "split {split}"
            );
        }
    }
}
