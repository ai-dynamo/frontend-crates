// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Every image-capable formatter must render tool-result images in the order
//! `media_message_order` reports, because Dynamo collects media in that order.
//! Each case puts a text label before each image, so the label nearest each
//! rendered placeholder identifies the image that fills it.

use std::sync::Arc;

use dynamo_renderer::deepseek::v41::DeepSeekV41Formatter;
use dynamo_renderer::kimi_k3::KimiK3Formatter;
use dynamo_renderer::mistral::MistralFormatter;
use dynamo_renderer::{
    ChatTemplate, ContextMixins, OAIChatLikeRequest, OAIPromptFormatter, PromptFormatter,
};
use minijinja::Value;
use serde_json::json;

struct Family {
    name: &'static str,
    formatter: Arc<dyn OAIPromptFormatter>,
    placeholder: &'static str,
    /// Whether the out-of-order case renders tool results in call order.
    reorders: bool,
}

// Real chat templates, checked in verbatim from Hugging Face.
const TEMPLATES: &[(&str, &str, &str, &str, bool)] = &[
    // zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
    (
        "glm-5.3-flash",
        include_str!("fixtures/chat_templates/glm_5_3_flash.jinja"),
        "<|endoftext|>",
        "<|image|>",
        true,
    ),
    // Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
    (
        "qwen3-vl",
        include_str!("fixtures/chat_templates/qwen3_vl.jinja"),
        "<|im_end|>",
        "<|image_pad|>",
        false,
    ),
    // nvidia/Qwen3.5-397B-A17B-NVFP4@12b28061efd8145b11226e4fab2528ee4f39ef09
    (
        "qwen3.5",
        include_str!("fixtures/chat_templates/qwen3_5.jinja"),
        "<|im_end|>",
        "<|image_pad|>",
        false,
    ),
    // Qwen/Qwen3.6-35B-A3B@345d467b117d1fd5eda9b879a5988de9ae5f47a7
    (
        "qwen3.6",
        include_str!("fixtures/chat_templates/qwen3_6.jinja"),
        "<|im_end|>",
        "<|image_pad|>",
        false,
    ),
    // Qwen/Qwen3.8-Flash-Next@812d96e10a0dd82b69d749e68693698cf6c36daa
    (
        "qwen3.8-flash-next",
        include_str!("fixtures/chat_templates/qwen3_8_flash_next.jinja"),
        "<|im_end|>",
        "<|image_pad|>",
        false,
    ),
    // nvidia/MiniMax-M3-NVFP4@901464083161bf8612a29ff7ad29914cd4ab4a85
    (
        "minimax-m3",
        include_str!("fixtures/chat_templates/minimax_m3.jinja"),
        "[e~[",
        "<]image[>",
        false,
    ),
    // MiniMaxAI/MiniMax-M3.1-Flash@54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e
    (
        "minimax-m3.1-flash",
        include_str!("fixtures/chat_templates/minimax_m3_1_flash.jinja"),
        "[e~[",
        "<]image[>",
        false,
    ),
];

/// Every `OAIPromptFormatter` implementation, with how this test covers it.
/// Add new formatters here; `every_formatter_is_classified` enforces it.
const FORMATTERS: &[(&str, &str)] = &[
    (
        "HfTokenizerConfigJsonFormatter",
        "chat templates in TEMPLATES",
    ),
    ("KimiK3Formatter", "native: call order"),
    ("MistralFormatter", "native: call order"),
    ("DeepSeekV41Formatter", "native: call order"),
    (
        "InklingFormatter",
        "not applicable: rejects images in tool results",
    ),
    ("DeepSeekV4Formatter", "not applicable: renders no images"),
    ("DeepSeekV32Formatter", "not applicable: renders no images"),
    ("NoOpFormatter", "not applicable: renders no images"),
];

fn families() -> Vec<Family> {
    let mut families: Vec<Family> = TEMPLATES
        .iter()
        .map(|&(name, source, eos, placeholder, reorders)| {
            let config: ChatTemplate =
                serde_json::from_value(json!({"chat_template": source, "eos_token": eos})).unwrap();
            let PromptFormatter::OAI(formatter) =
                PromptFormatter::from_parts(config, ContextMixins::new(&[]), true).unwrap();
            Family {
                name,
                formatter,
                placeholder,
                reorders,
            }
        })
        .collect();
    let settings = json!({
        "reasoning_effort": {"type": "enum", "accepts_none": true, "default": "none", "values": ["none", "high"]},
        "json_schema": {"type": "json_schema", "accepts_none": false, "default": null}
    });
    let native: [(&str, Arc<dyn OAIPromptFormatter>, &str, bool); 3] = [
        (
            "kimi-k3",
            Arc::new(KimiK3Formatter::new(true)),
            "<|media_pad|>",
            true,
        ),
        (
            "mistral-v15",
            Arc::new(MistralFormatter::new(15, Some(&settings)).unwrap()),
            "[IMG]",
            true,
        ),
        (
            "deepseek-v4.1",
            Arc::new(DeepSeekV41Formatter),
            "<\u{ff5c}deepseek_image\u{ff5c}>",
            true,
        ),
    ];
    families.extend(
        native
            .into_iter()
            .map(|(name, formatter, placeholder, reorders)| Family {
                name,
                formatter,
                placeholder,
                reorders,
            }),
    );
    families
}

fn assistant(ids: &[&str]) -> serde_json::Value {
    json!({"role": "assistant", "content": "", "tool_calls": ids.iter().map(|id| json!({
        "id": id, "type": "function", "function": {"name": format!("fn_{id}"), "arguments": "{}"}
    })).collect::<Vec<_>>()})
}

/// One text label per image, plus a trailing label for an image-free result.
fn tool(id: &str, labels: &[&str]) -> serde_json::Value {
    let mut content: Vec<_> = labels
        .iter()
        .flat_map(|label| {
            [
                json!({"type": "text", "text": label}),
                json!({"type": "image_url", "image_url": {"url": format!("https://example.com/{label}.png")}}),
            ]
        })
        .collect();
    if content.is_empty() {
        content.push(json!({"type": "text", "text": format!("NOIMG_{id}")}));
    }
    json!({"role": "tool", "tool_call_id": id, "content": content})
}

fn cases() -> Vec<(&'static str, serde_json::Value)> {
    let user = json!({"role": "user", "content": "Call the tools."});
    let wait = json!({"role": "user", "content": "wait"});
    vec![
        // Calls [a, b, c]; results arrive c (2 images), a (1 image), b (none).
        (
            "out_of_order",
            json!([
                user,
                assistant(&["a", "b", "c"]),
                tool("c", &["IMG_C1", "IMG_C2"]),
                tool("a", &["IMG_A"]),
                tool("b", &[])
            ]),
        ),
        (
            "unknown_id",
            json!([
                user,
                assistant(&["a", "b"]),
                tool("b", &["IMG_B"]),
                tool("x", &["IMG_X"])
            ]),
        ),
        (
            "duplicate_id",
            json!([
                user,
                assistant(&["a", "b"]),
                tool("b", &["IMG_B1"]),
                tool("b", &["IMG_B2"])
            ]),
        ),
        (
            "interrupted_run",
            json!([
                user,
                assistant(&["a", "b"]),
                wait,
                tool("b", &["IMG_B"]),
                tool("a", &["IMG_A"])
            ]),
        ),
    ]
}

struct RawRequest(serde_json::Value);

impl OAIChatLikeRequest for RawRequest {
    fn model(&self) -> String {
        "test".into()
    }
    fn messages(&self) -> Value {
        Value::from_serialize(&self.0)
    }
    fn tools(&self) -> Option<Value> {
        Some(Value::from_serialize(tools()))
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
}

fn tools() -> serde_json::Value {
    json!(
        ["a", "b", "c", "x"].map(|id| json!({"type": "function", "function": {
            "name": format!("fn_{id}"), "description": "Return images.",
            "parameters": {"type": "object", "properties": {}}
        }}))
    )
}

fn image_labels(message: &serde_json::Value) -> Vec<String> {
    let parts = message["content"].as_array().cloned().unwrap_or_default();
    parts
        .windows(2)
        .filter(|pair| pair[1]["type"] == "image_url")
        .filter_map(|pair| pair[0]["text"].as_str().map(str::to_string))
        .collect()
}

/// The label rendered most recently before each placeholder.
fn rendered_labels(prompt: &str, placeholder: &str, labels: &[String]) -> Vec<String> {
    let mut events: Vec<(usize, Option<&str>)> = prompt
        .match_indices(placeholder)
        .map(|(position, _)| (position, None))
        .collect();
    for label in labels {
        events.extend(
            prompt
                .match_indices(label.as_str())
                .map(|(position, _)| (position, Some(label.as_str()))),
        );
    }
    events.sort();
    let mut current = "<none>";
    let mut output = Vec::new();
    for (_, label) in events {
        match label {
            Some(label) => current = label,
            None => output.push(current.to_string()),
        }
    }
    output
}

#[test]
fn tool_result_images_render_in_media_message_order() {
    let mut failures = Vec::new();
    for family in families() {
        for (case, messages) in cases() {
            let typed: dynamo_protocols::types::CreateChatCompletionRequest =
                serde_json::from_value(
                    json!({"model": "test", "messages": messages, "tools": tools()}),
                )
                .unwrap();
            let raw = RawRequest(messages.clone());
            for (kind, request) in [
                ("typed", &typed as &dyn OAIChatLikeRequest),
                ("raw", &raw as &dyn OAIChatLikeRequest),
            ] {
                let source = messages.as_array().unwrap();
                let order = family
                    .formatter
                    .media_message_order(request)
                    .unwrap_or_else(|| (0..source.len()).collect());
                let mut sorted = order.clone();
                sorted.sort();
                assert_eq!(
                    sorted,
                    (0..source.len()).collect::<Vec<_>>(),
                    "{}",
                    family.name
                );
                let expected: Vec<String> = order
                    .iter()
                    .flat_map(|&index| image_labels(&source[index]))
                    .collect();
                let all: Vec<String> = source.iter().flat_map(image_labels).collect();
                // A rejected request renders no prompt, so no image can be misplaced.
                let prompt = match family.formatter.render(request) {
                    Ok(prompt) => prompt,
                    Err(_) if case != "out_of_order" => continue,
                    Err(error) => {
                        failures.push(format!(
                            "{} {case} {kind}: render failed: {error:#}",
                            family.name
                        ));
                        continue;
                    }
                };
                let actual = rendered_labels(&prompt, family.placeholder, &all);
                if actual != expected {
                    failures.push(format!(
                        "{} {case} {kind}: placeholders {actual:?}, media order {expected:?}",
                        family.name
                    ));
                }
                if case == "out_of_order" && family.reorders != (expected != all) {
                    failures.push(format!(
                        "{} {case} {kind}: expected reorder={}, media order {expected:?}",
                        family.name, family.reorders
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_formatter_is_classified() {
    fn implementations(dir: &std::path::Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                implementations(&path, found);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path).unwrap();
                found.extend(
                    source
                        .lines()
                        .filter(|line| line.trim_start().starts_with("impl"))
                        .filter_map(|line| line.split("OAIPromptFormatter for ").nth(1))
                        .map(|rest| {
                            rest.split(|c: char| !c.is_alphanumeric() && c != '_')
                                .next()
                                .unwrap()
                                .to_string()
                        }),
                );
            }
        }
    }
    let mut found = Vec::new();
    implementations(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut found,
    );
    let unlisted: Vec<_> = found
        .iter()
        .filter(|name| !FORMATTERS.iter().any(|(listed, _)| listed == name))
        .collect();
    assert!(
        unlisted.is_empty(),
        "classify these formatters in FORMATTERS and cover them if they render images: {unlisted:?}"
    );
}

/// GLM's sort conditions are easy to get subtly wrong, so compare the hook
/// with the real template on generated histories.
#[test]
fn glm_media_order_matches_template_on_generated_histories() {
    let family = families()
        .into_iter()
        .find(|family| family.name == "glm-5.3-flash")
        .unwrap();
    // ponytail: fixed-seed LCG instead of a rand dependency.
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = |n: usize| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize % n
    };
    let ids = ["a", "b", "c", "d", ""];
    let mut reordered = 0;
    for case in 0..2000 {
        let mut messages = vec![json!({"role": "user", "content": "start"})];
        let mut image = 0;
        for _ in 0..1 + next(3) {
            // Mostly valid parallel calls, sometimes with an empty or repeated ID.
            let mut calls: Vec<&str> = ids[..4].to_vec();
            calls.truncate(1 + next(4));
            if next(8) == 0 {
                let slot = next(calls.len());
                calls[slot] = ids[next(ids.len())];
            }
            messages.push(assistant(&calls));
            if next(8) == 0 {
                messages.push(json!({"role": "user", "content": "interrupt"}));
            }
            // A shuffled subset of results, sometimes with a bad ID.
            let mut results = calls.clone();
            for i in (1..results.len()).rev() {
                results.swap(i, next(i + 1));
            }
            results.truncate(1 + next(results.len()));
            if next(8) == 0 {
                let slot = next(results.len());
                results[slot] = ids[next(ids.len())];
            }
            for id in results {
                let labels: Vec<String> = (0..next(3))
                    .map(|_| {
                        image += 1;
                        format!("IMG_{case}_{image}_")
                    })
                    .collect();
                let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
                messages.push(tool(id, &labels));
            }
        }
        let request = RawRequest(json!(messages));
        let order = family
            .formatter
            .media_message_order(&request)
            .unwrap_or_else(|| (0..messages.len()).collect());
        let expected: Vec<String> = order
            .iter()
            .flat_map(|&index| image_labels(&messages[index]))
            .collect();
        let all: Vec<String> = messages.iter().flat_map(image_labels).collect();
        reordered += usize::from(expected != all);
        let prompt = family.formatter.render(&request).unwrap();
        assert_eq!(
            rendered_labels(&prompt, family.placeholder, &all),
            expected,
            "{}",
            serde_json::to_string(&messages).unwrap()
        );
    }
    // Guard against a generator that never exercises the sort.
    assert!(
        reordered > 200,
        "only {reordered} histories reordered media"
    );
}
