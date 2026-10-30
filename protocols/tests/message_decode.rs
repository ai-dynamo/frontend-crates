// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a chat request message accepts and refuses on the wire.
//!
//! `ChatCompletionRequestMessage` reads `role` first and then decodes the
//! role's message in the same pass, without buffering it. These tests hold it
//! to the decoder it replaced: a `role`-tagged enum whose variants each
//! `#[serde(flatten)]` the message beside its `tools` and `partial` guard.
//! `Oracle` below is a copy of that decoder. Every input in the corpus must
//! give the same message, or the same error text, from both. The tests at the
//! end name the differences.

use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestFunctionMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage,
};
use serde::Deserialize;

/// The message decoder before the one-pass decoder, kept as the test oracle.
struct Oracle(ChatCompletionRequestMessage);

impl<'de> Deserialize<'de> for Oracle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        #[derive(Deserialize)]
        struct ForbidToolsAndPartial<T> {
            tools: Option<serde::de::IgnoredAny>,
            partial: Option<serde::de::IgnoredAny>,
            #[serde(flatten)]
            message: T,
        }

        #[derive(Deserialize)]
        struct ForbidTools<T> {
            tools: Option<serde::de::IgnoredAny>,
            #[serde(flatten)]
            message: T,
        }

        #[derive(Deserialize)]
        struct ForbidPartial<T> {
            partial: Option<serde::de::IgnoredAny>,
            #[serde(flatten)]
            message: T,
        }

        #[derive(Deserialize)]
        #[serde(tag = "role")]
        #[serde(rename_all = "lowercase")]
        enum Wire {
            Developer(ForbidToolsAndPartial<ChatCompletionRequestDeveloperMessage>),
            System(ForbidPartial<ChatCompletionRequestSystemMessage>),
            User(ForbidToolsAndPartial<ChatCompletionRequestUserMessage>),
            Assistant(ForbidTools<ChatCompletionRequestAssistantMessage>),
            Tool(ForbidToolsAndPartial<ChatCompletionRequestToolMessage>),
            Function(ForbidToolsAndPartial<ChatCompletionRequestFunctionMessage>),
        }

        fn reject_forbidden<E: Error>(
            value: Option<serde::de::IgnoredAny>,
            field: &str,
            allowed_role: &str,
            actual_role: &str,
        ) -> Result<(), E> {
            if value.is_some() {
                return Err(E::custom(format!(
                    "`{field}` is only accepted on {allowed_role} messages, not on role {actual_role}"
                )));
            }
            Ok(())
        }

        use ChatCompletionRequestMessage as M;
        let wire = Wire::deserialize(deserializer)?;
        Ok(Oracle(match wire {
            Wire::Developer(ForbidToolsAndPartial {
                tools,
                partial,
                message,
            }) => {
                reject_forbidden::<D::Error>(tools, "tools", "system", "developer")?;
                reject_forbidden::<D::Error>(partial, "partial", "assistant", "developer")?;
                M::Developer(message)
            }
            Wire::System(ForbidPartial { partial, message }) => {
                reject_forbidden::<D::Error>(partial, "partial", "assistant", "system")?;
                M::System(message)
            }
            Wire::User(ForbidToolsAndPartial {
                tools,
                partial,
                message,
            }) => {
                reject_forbidden::<D::Error>(tools, "tools", "system", "user")?;
                reject_forbidden::<D::Error>(partial, "partial", "assistant", "user")?;
                M::User(message)
            }
            Wire::Assistant(ForbidTools { tools, message }) => {
                reject_forbidden::<D::Error>(tools, "tools", "system", "assistant")?;
                M::Assistant(message)
            }
            Wire::Tool(ForbidToolsAndPartial {
                tools,
                partial,
                message,
            }) => {
                reject_forbidden::<D::Error>(tools, "tools", "system", "tool")?;
                reject_forbidden::<D::Error>(partial, "partial", "assistant", "tool")?;
                M::Tool(message)
            }
            Wire::Function(ForbidToolsAndPartial {
                tools,
                partial,
                message,
            }) => {
                reject_forbidden::<D::Error>(tools, "tools", "system", "function")?;
                reject_forbidden::<D::Error>(partial, "partial", "assistant", "function")?;
                M::Function(message)
            }
        }))
    }
}

/// One valid message body per role, without its `role` key.
const BODIES: [(&str, &str); 6] = [
    ("developer", r#""content":"be brief""#),
    ("system", r#""content":"be brief","name":"ops""#),
    (
        "user",
        r#""content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]"#,
    ),
    (
        "assistant",
        r#""content":null,"reasoning":"think","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]"#,
    ),
    ("tool", r#""content":"ok","tool_call_id":"c1""#),
    ("function", r#""content":"ok","name":"f""#),
];

/// Fields added to each body, to probe the guards and the inner decoders.
const EXTRAS: [&str; 22] = [
    "",
    r#""tools":[{"name":"lookup"}]"#,
    r#""tools":null"#,
    r#""tools":[]"#,
    r#""tools":"lookup""#,
    r#""tools":null,"tools":null"#,
    r#""tools":[{"name":"a"}],"tools":[{"name":"b"}]"#,
    r#""partial":true"#,
    r#""partial":false"#,
    r#""partial":null"#,
    r#""partial":"yes""#,
    r#""partial":null,"partial":null"#,
    r#""tools":[1],"partial":true"#,
    r#""role":"user""#,
    r#""role":null"#,
    r#""content":"again""#,
    r#""name":"n""#,
    r#""name":7"#,
    r#""x_vendor":{"a":1}"#,
    r#""reasoning_content":"r""#,
    r#""tool_call_id":5"#,
    r#""content":{"text":"t","text":"u"}"#,
];

/// Whole messages that do not fit the body and extra grid.
const ODD: [&str; 26] = [
    "{}",
    "[]",
    r#"["user"]"#,
    r#"["user","hi"]"#,
    r#""user""#,
    "null",
    "1",
    "true",
    r#"{"content":"hi"}"#,
    r#"{"role":null,"content":"hi"}"#,
    r#"{"role":5,"content":"hi"}"#,
    r#"{"role":"User","content":"hi"}"#,
    r#"{"role":"robot","content":"hi"}"#,
    r#"{"role":["user"],"content":"hi"}"#,
    r#"{"role":"user"}"#,
    r#"{"role":"user","tools":[1]}"#,
    r#"{"tools":[1],"role":"user"}"#,
    r#"{"role":"user","content":5}"#,
    r#"{"role":"system"}"#,
    r#"{"role":"system","tools":[]}"#,
    r#"{"role":"system","tools":[{"name":"lookup"}]}"#,
    r#"{"role":"tool","content":"ok"}"#,
    r#"{"role":"function","content":"ok"}"#,
    r#"{"role":"assistant"}"#,
    r#"{"role":"assistant","content":"seed","partial":true}"#,
    r#"{"role":"user","content":"hi","role":"user"}"#,
];

/// Every input the oracle comparison runs: each role's body with each extra,
/// with `role` first and with `role` last, then the odd messages.
fn corpus() -> Vec<String> {
    let mut corpus = Vec::new();
    for (role, body) in BODIES {
        for extra in EXTRAS {
            let rest = if extra.is_empty() {
                body.to_string()
            } else {
                format!("{body},{extra}")
            };
            corpus.push(format!(r#"{{"role":"{role}",{rest}}}"#));
            corpus.push(format!(r#"{{{rest},"role":"{role}"}}"#));
            corpus.push(format!(r#"{{{extra},"role":"{role}",{body}}}"#).replace("{,", "{"));
        }
    }
    corpus.extend(ODD.iter().map(|odd| odd.to_string()));
    corpus
}

/// The old decoder's private type names, which its errors for a message
/// that is not an object gave.
const PRIVATE_NAMES: [&str; 2] = ["struct Forbid", "enum Wire"];

/// The error text without serde_json's position, which depends on how far
/// each decoder reads before it fails.
fn text(error: serde_json::Error) -> String {
    let error = error.to_string();
    match error.rfind(" at line ") {
        Some(at) => error[..at].to_string(),
        None => error,
    }
}

#[test]
fn decoder_matches_the_flatten_oracle() {
    let mut accepted = 0;
    for input in corpus() {
        let oracle = serde_json::from_str::<Oracle>(&input).map(|oracle| oracle.0);
        let decoded = serde_json::from_str::<ChatCompletionRequestMessage>(&input);
        match (oracle, decoded) {
            (Ok(oracle), Ok(decoded)) => {
                assert_eq!(decoded, oracle, "{input}");
                accepted += 1;
            }
            (Err(oracle), Err(decoded)) => {
                let (oracle, decoded) = (text(oracle), text(decoded));
                // A message that is not an object named the old decoder's
                // private types in its error; any refusal will do there.
                if !PRIVATE_NAMES.iter().any(|name| oracle.contains(name)) {
                    assert_eq!(decoded, oracle, "{input}");
                }
            }
            (oracle, decoded) => {
                panic!("{input}: the oracle gave {oracle:?}, the decoder gave {decoded:?}")
            }
        }
    }
    assert!(accepted > 50, "only {accepted} inputs decoded");
}

/// Whether this build has serde_json's `arbitrary_precision`.
fn arbitrary_precision() -> bool {
    serde_json::from_value::<String>(serde_json::json!(7))
        .unwrap_err()
        .to_string()
        .starts_with("invalid type: number")
}

/// The same comparison from a `serde_json::Value`, whose keys arrive as
/// owned strings rather than borrowed from the input.
#[test]
fn decoder_matches_the_flatten_oracle_from_a_value() {
    for input in corpus() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&input) else {
            continue;
        };
        let oracle = serde_json::from_value::<Oracle>(value.clone()).map(|oracle| oracle.0);
        let decoded = serde_json::from_value::<ChatCompletionRequestMessage>(value);
        match (oracle, decoded) {
            (Ok(oracle), Ok(decoded)) => assert_eq!(decoded, oracle, "{input}"),
            (Err(oracle), Err(decoded)) => {
                let (oracle, decoded) = (oracle.to_string(), decoded.to_string());
                // With serde_json's `arbitrary_precision`, which another
                // crate can turn on, a `Value`'s number names itself
                // "number" when decoded straight, and its digits when
                // decoded through serde's buffer.
                let number = arbitrary_precision() && decoded.starts_with("invalid type: number");
                if !number && !PRIVATE_NAMES.iter().any(|name| oracle.contains(name)) {
                    assert_eq!(decoded, oracle, "{input}");
                }
            }
            (oracle, decoded) => {
                panic!("{input}: the oracle gave {oracle:?}, the decoder gave {decoded:?}")
            }
        }
    }
}

/// The one difference in what the two decoders accept. The oracle buffered
/// each message, so it parsed every value, even one no field reads. The
/// decoder skips such a value, as serde skips an unknown field anywhere
/// else in a request. So a number out of range, or a lone surrogate escape,
/// in a skipped value is dropped with it, where the oracle refused it.
#[test]
fn a_skipped_value_is_not_parsed() {
    // With serde_json's `arbitrary_precision`, which another crate can turn
    // on, a number keeps its digits and is never out of range, so only the
    // escape differs.
    let numbers = !arbitrary_precision();
    for (input, number) in [
        (r#"{"role":"user","content":"hi","x":1e400}"#, true),
        (r#"{"role":"user","content":"hi","x":{"a":[-1e400]}}"#, true),
        (r#"{"role":"assistant","audio":{"id":"a","x":1e400}}"#, true),
        (r#"{"role":"user","content":"hi","x":"\ud800"}"#, false),
    ] {
        let oracle = serde_json::from_str::<Oracle>(input);
        assert_eq!(oracle.is_err(), numbers || !number, "{input}");
        serde_json::from_str::<ChatCompletionRequestMessage>(input)
            .unwrap_or_else(|error| panic!("{input}: {error}"));
    }
    // With `role` last, the tagged enum buffers the message, and refuses it.
    let input = r#"{"content":"hi","x":"\ud800","role":"user"}"#;
    assert!(serde_json::from_str::<ChatCompletionRequestMessage>(input).is_err());
}

#[test]
/// A key that is not a string, as MessagePack can send, is skipped, as the
/// oracle's flattened guard skipped it.
fn unknown_numeric_map_keys_keep_the_oracle_value() {
    use serde::de::value::MapDeserializer;
    use serde_json::json;

    for entries in [
        vec![
            (json!("role"), json!("user")),
            (json!(42), json!("ignored")),
            (json!("content"), json!("hi")),
        ],
        vec![
            (json!(42), json!("ignored")),
            (json!("content"), json!("hi")),
            (json!("role"), json!("user")),
        ],
    ] {
        let oracle = Oracle::deserialize(MapDeserializer::<_, serde_json::Error>::new(
            entries.clone().into_iter(),
        ))
        .expect("oracle accepts an unknown numeric key")
        .0;
        let decoded = ChatCompletionRequestMessage::deserialize(MapDeserializer::<
            _,
            serde_json::Error,
        >::new(entries.into_iter()))
        .map_err(text);
        assert_eq!(decoded, Ok(oracle));
    }
}

/// A message with two errors reports the first that its decoder meets. The
/// oracle read the whole message before it decoded the role's fields, so a
/// second `partial` came first there. Both refuse the message.
#[test]
fn a_message_with_two_errors_is_still_refused() {
    for input in [
        r#"{"role":"user","content":5,"partial":null,"partial":null}"#,
        r#"{"content":5,"partial":null,"partial":null,"role":"user"}"#,
    ] {
        let oracle = text(serde_json::from_str::<Oracle>(input).err().expect(input));
        assert_eq!(oracle, "duplicate field `partial`");
        let decoded =
            text(serde_json::from_str::<ChatCompletionRequestMessage>(input).expect_err(input));
        assert_eq!(
            decoded,
            "data did not match any variant of untagged enum ChatCompletionRequestUserMessageContent"
        );
    }
}
