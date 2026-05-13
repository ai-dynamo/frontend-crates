//! POST /v1/responses -- OpenAI Responses API
//!
//! Non-streaming only for simplicity. Constructs the response via serde_json
//! since the upstream types are complex and builder-heavy.

use axum::Json;
use axum::response::{IntoResponse, Response};

pub async fn handler(Json(req): Json<serde_json::Value>) -> Response {
    // Extract input text from the request
    let input_text = match req.get("input") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(items)) => {
            // Walk items looking for message content
            items
                .iter()
                .filter_map(|item| {
                    if item.get("type")?.as_str()? == "message" {
                        let content = item.get("content")?;
                        match content {
                            serde_json::Value::String(s) => Some(s.clone()),
                            serde_json::Value::Array(parts) => {
                                Some(parts.iter().filter_map(|p| {
                                    if p.get("type")?.as_str()? == "input_text" {
                                        p.get("text")?.as_str().map(String::from)
                                    } else {
                                        None
                                    }
                                }).collect::<Vec<_>>().join(" "))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => "(no input)".to_string(),
    };

    let model = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("echo");
    let echo_text = format!("[echo] {input_text}");
    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());

    // Build the response as JSON directly -- the upstream ResponseObject types
    // are complex with many fields. This keeps the demo simple.
    let resp = serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": chrono_now_secs(),
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": echo_text,
                        "annotations": []
                    }
                ]
            }
        ],
        "usage": {
            "input_tokens": input_text.len() / 4,
            "output_tokens": echo_text.len() / 4,
            "total_tokens": (input_text.len() + echo_text.len()) / 4
        }
    });

    Json(resp).into_response()
}

fn chrono_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
