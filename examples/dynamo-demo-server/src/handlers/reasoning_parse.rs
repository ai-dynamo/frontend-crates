//! POST /v1/reasoning-parse -- dynamo-parsers reasoning demo endpoint
//!
//! Splits raw model output into reasoning vs normal text using one of the
//! registered reasoning parsers (deepseek_r1, qwen3, gpt_oss, kimi_k25, ...).

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use dynamo_parsers::reasoning::{
    ReasoningParser, ReasoningParserType, get_available_reasoning_parsers,
};

#[derive(Deserialize)]
pub struct ReasoningParseRequest {
    /// Raw model output that may contain a reasoning block.
    pub text: String,
    /// Parser name (e.g. "deepseek_r1", "qwen3", "gpt_oss"). Defaults to "basic".
    #[serde(default)]
    pub parser: Option<String>,
}

#[derive(Serialize)]
pub struct ReasoningParseResponse {
    pub reasoning_text: String,
    pub normal_text: String,
    pub parser: String,
    pub available_parsers: Vec<String>,
}

pub async fn handler(Json(req): Json<ReasoningParseRequest>) -> Response {
    let name = req.parser.as_deref().unwrap_or("basic");
    let Some(ty) = lookup(name) else {
        let body = serde_json::json!({
            "error": format!("unknown reasoning parser: {name}"),
            "available_parsers": get_available_reasoning_parsers(),
        });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    };

    let mut parser = ty.get_reasoning_parser();
    let result = parser.detect_and_parse_reasoning(&req.text, &[]);

    Json(ReasoningParseResponse {
        reasoning_text: result.reasoning_text,
        normal_text: result.normal_text,
        parser: name.to_string(),
        available_parsers: get_available_reasoning_parsers()
            .into_iter()
            .map(String::from)
            .collect(),
    })
    .into_response()
}

fn lookup(name: &str) -> Option<ReasoningParserType> {
    // The parsers crate exposes the registry via name list + per-type constructor.
    // Match the same names supported by `get_reasoning_parser_map` upstream.
    Some(match name {
        "deepseek_r1" => ReasoningParserType::DeepseekR1,
        "basic" => ReasoningParserType::Basic,
        "gpt_oss" => ReasoningParserType::GptOss,
        "qwen3" => ReasoningParserType::Qwen,
        "deepseek_v4" | "deepseek-v4" | "deepseekv4" => ReasoningParserType::DeepSeekV4,
        "nemotron_deci" | "nemotron_nano" | "nemotron3" => ReasoningParserType::NemotronDeci,
        "kimi" => ReasoningParserType::Kimi,
        "kimi_k25" => ReasoningParserType::KimiK25,
        "step3" => ReasoningParserType::Step3,
        "mistral" => ReasoningParserType::Mistral,
        "granite" => ReasoningParserType::Granite,
        "minimax_append_think" => ReasoningParserType::MiniMaxAppendThink,
        "gemma4" | "gemma-4" => ReasoningParserType::Gemma4,
        "glm45" => ReasoningParserType::NemotronDeci,
        _ => return None,
    })
}
