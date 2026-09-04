// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Re-exports upstream async-openai realtime types and adds a narrow wrapper for
// Dynamo-specific client events without replacing the upstream public enum.

pub use async_openai::types::realtime::*;
use serde::{Deserialize, Serialize};

/// Append UTF-8 text to the current incremental input buffer.
///
/// This is a Dynamo extension for clients that receive text progressively,
/// such as cascaded ASR -> LLM pipelines. Like `input_audio_buffer.append`, an
/// append does not finalize a conversation item or request a response.
#[derive(Debug, Serialize, Deserialize)]
pub struct RealtimeClientEventInputTextBufferAppend {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    pub text: String,
}

/// Finalize the current incremental text buffer as a user conversation item.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RealtimeClientEventInputTextBufferCommit {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
}

/// Discard the current incremental text buffer without creating an item.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RealtimeClientEventInputTextBufferClear {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
}

/// Dynamo-specific extensions to the OpenAI Realtime client event set.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeClientEventExtension {
    #[serde(rename = "input_text_buffer.append")]
    InputTextBufferAppend(RealtimeClientEventInputTextBufferAppend),
    #[serde(rename = "input_text_buffer.commit")]
    InputTextBufferCommit(RealtimeClientEventInputTextBufferCommit),
    #[serde(rename = "input_text_buffer.clear")]
    InputTextBufferClear(RealtimeClientEventInputTextBufferClear),
}

/// OpenAI Realtime client events plus inference-serving extensions from Dynamo.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DynamoRealtimeClientEvent {
    OpenAI(RealtimeClientEvent),
    Extension(RealtimeClientEventExtension),
}

impl From<RealtimeClientEvent> for DynamoRealtimeClientEvent {
    fn from(event: RealtimeClientEvent) -> Self {
        Self::OpenAI(event)
    }
}

/// Returns the `type` wire-tag string for a realtime event variant — useful
/// for logging, error messages, and metric labels that need a stable name
/// without reserializing the value.
///
/// `async-openai` ships an equivalent `crate::traits::EventType` trait, but it
/// is gated on the `_api` feature, which pulls reqwest / tokio / secrecy /
/// eventsource-stream into the build. `dynamo-protocols` is types-only by
/// design (see the Cargo.toml banner), so we mirror the trait shape locally.
/// If `_api` ever becomes affordable for this crate, swap `pub use
/// async_openai::traits::EventType;` in here and remove the impls below; call
/// sites need no changes.
///
/// Implemented for the upstream and Dynamo client event sets. The
/// `RealtimeServerEvent` impl can be added when a consumer needs it.
///
/// [NOTE] Could be replaced with a serde-introspection helper (e.g. the
/// `serde_variant` crate) that reads the wire tag from `#[serde(rename)]`
/// at runtime; deferred until as clean up work.
pub trait EventType {
    fn event_type(&self) -> &'static str;
}

impl EventType for RealtimeClientEvent {
    fn event_type(&self) -> &'static str {
        // `RealtimeClientEvent` is not `#[non_exhaustive]`, so a future upstream
        // variant breaks this match at compile time rather than silently
        // returning a stale label.
        match self {
            RealtimeClientEvent::SessionUpdate(_) => "session.update",
            RealtimeClientEvent::InputAudioBufferAppend(_) => "input_audio_buffer.append",
            RealtimeClientEvent::InputAudioBufferCommit(_) => "input_audio_buffer.commit",
            RealtimeClientEvent::InputAudioBufferClear(_) => "input_audio_buffer.clear",
            RealtimeClientEvent::ConversationItemCreate(_) => "conversation.item.create",
            RealtimeClientEvent::ConversationItemRetrieve(_) => "conversation.item.retrieve",
            RealtimeClientEvent::ConversationItemTruncate(_) => "conversation.item.truncate",
            RealtimeClientEvent::ConversationItemDelete(_) => "conversation.item.delete",
            RealtimeClientEvent::ResponseCreate(_) => "response.create",
            RealtimeClientEvent::ResponseCancel(_) => "response.cancel",
            RealtimeClientEvent::OutputAudioBufferClear(_) => "output_audio_buffer.clear",
        }
    }
}

impl EventType for RealtimeClientEventExtension {
    fn event_type(&self) -> &'static str {
        match self {
            RealtimeClientEventExtension::InputTextBufferAppend(_) => "input_text_buffer.append",
            RealtimeClientEventExtension::InputTextBufferCommit(_) => "input_text_buffer.commit",
            RealtimeClientEventExtension::InputTextBufferClear(_) => "input_text_buffer.clear",
        }
    }
}

impl EventType for DynamoRealtimeClientEvent {
    fn event_type(&self) -> &'static str {
        match self {
            DynamoRealtimeClientEvent::OpenAI(event) => event.event_type(),
            DynamoRealtimeClientEvent::Extension(event) => event.event_type(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_update_accepts_null_turn_detection() {
        let event: RealtimeClientEvent = serde_json::from_value(serde_json::json!({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": { "type": "audio/pcm", "rate": 24000 },
                        "transcription": { "model": "whisper-1" },
                        "turn_detection": null
                    }
                }
            }
        }))
        .expect("null turn_detection should be accepted");

        let RealtimeClientEvent::SessionUpdate(update) = event else {
            panic!("expected session.update");
        };
        let Session::RealtimeTranscriptionSession(session) = update.session else {
            panic!("expected transcription session");
        };
        assert!(session.audio.input.turn_detection.is_none());
    }

    #[test]
    fn text_buffer_events_round_trip() {
        let values = [
            (
                serde_json::json!({
                    "type": "input_text_buffer.append",
                    "event_id": "append-1",
                    "text": "hello "
                }),
                "input_text_buffer.append",
            ),
            (
                serde_json::json!({
                    "type": "input_text_buffer.commit",
                    "event_id": "commit-1"
                }),
                "input_text_buffer.commit",
            ),
            (
                serde_json::json!({
                    "type": "input_text_buffer.clear",
                    "event_id": "clear-1"
                }),
                "input_text_buffer.clear",
            ),
        ];

        for (value, event_type) in values {
            let event: DynamoRealtimeClientEvent =
                serde_json::from_value(value.clone()).expect("event should deserialize");
            assert_eq!(event.event_type(), event_type);
            assert_eq!(serde_json::to_value(event).unwrap(), value);
        }
    }

    #[test]
    fn text_buffer_append_requires_non_null_text() {
        for value in [
            serde_json::json!({"type": "input_text_buffer.append"}),
            serde_json::json!({"type": "input_text_buffer.append", "text": null}),
        ] {
            assert!(serde_json::from_value::<DynamoRealtimeClientEvent>(value).is_err());
        }
    }
}
