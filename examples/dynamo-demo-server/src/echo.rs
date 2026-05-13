//! Dummy echo backend -- the pluggable inference point.
//!
//! This module extracts text from incoming requests and echoes it back.
//! Replace these functions with calls to your own tokenizer + inference engine.

use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent, Prompt,
};

/// Extract the last user message text from a chat completion request's messages.
pub fn extract_chat_text(messages: &[ChatCompletionRequestMessage]) -> String {
    for msg in messages.iter().rev() {
        if let ChatCompletionRequestMessage::User(user_msg) = msg {
            match &user_msg.content {
                ChatCompletionRequestUserMessageContent::Text(t) => return t.clone(),
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    // Concatenate text parts
                    let texts: Vec<&str> = parts
                        .iter()
                        .filter_map(|p| {
                            use dynamo_protocols::types::ChatCompletionRequestUserMessageContentPart;
                            match p {
                                ChatCompletionRequestUserMessageContentPart::Text(t) => {
                                    Some(t.text.as_str())
                                }
                                _ => None,
                            }
                        })
                        .collect();
                    if !texts.is_empty() {
                        return texts.join(" ");
                    }
                }
            }
        }
    }
    "(no user message found)".to_string()
}

/// Extract text from a completions prompt.
pub fn extract_completion_text(prompt: &Prompt) -> Result<String, String> {
    match prompt {
        Prompt::String(s) => Ok(s.clone()),
        Prompt::StringArray(arr) => Ok(arr.join(" ")),
        _ => Err("Token-ID prompts are not supported by the echo backend".to_string()),
    }
}

/// Extract text from Anthropic messages (last user message).
pub fn extract_anthropic_text(
    messages: &[dynamo_protocols::types::anthropic::AnthropicMessage],
) -> String {
    use dynamo_protocols::types::anthropic::{AnthropicMessageContent, AnthropicRole};
    for msg in messages.iter().rev() {
        if msg.role == AnthropicRole::User {
            match &msg.content {
                AnthropicMessageContent::Text { content } => return content.clone(),
                AnthropicMessageContent::Blocks { content } => {
                    use dynamo_protocols::types::anthropic::AnthropicContentBlock;
                    let texts: Vec<&str> = content
                        .iter()
                        .filter_map(|b| match b {
                            AnthropicContentBlock::Text { text, .. } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    if !texts.is_empty() {
                        return texts.join(" ");
                    }
                }
            }
        }
    }
    "(no user message found)".to_string()
}
