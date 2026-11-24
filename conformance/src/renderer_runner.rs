// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invoke the published renderer interfaces without downstream HTTP normalization.

use crate::renderer::{self, Settings};
use anyhow::{Context, Result, ensure};
use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use dynamo_renderer::{
    ChatTemplateContentFormat, ContextMixins, MediaPlaceholders, OAIChatLikeRequest,
    PromptFormatter, PromptRenderError, kimi_k3_formatter_for, native_formatter_for,
};
use minijinja::Value as TemplateValue;
use serde_json::{Value, json};
use std::{collections::HashMap, fs, path::Path};

struct Request {
    raw: Value,
    typed: Option<CreateChatCompletionRequest>,
    settings: Settings,
}

impl OAIChatLikeRequest for Request {
    fn model(&self) -> String {
        self.raw["model"].as_str().unwrap().to_owned()
    }
    fn messages(&self) -> TemplateValue {
        self.typed.as_ref().map_or_else(
            || TemplateValue::from_serialize(&self.raw["messages"]),
            OAIChatLikeRequest::messages,
        )
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.typed
            .as_ref()
            .map(|request| request.messages.as_slice())
    }
    fn template_messages_json_override(&self) -> Option<Value> {
        self.typed.is_none().then(|| self.raw["messages"].clone())
    }
    fn tools(&self) -> Option<TemplateValue> {
        self.typed.as_ref().map_or_else(
            || {
                self.raw
                    .get("tools")
                    .cloned()
                    .and_then(dynamo_renderer::may_be_fix_tool_schema)
            },
            OAIChatLikeRequest::tools,
        )
    }
    fn raw_tools(&self) -> Option<TemplateValue> {
        self.typed.as_ref().map_or_else(
            || self.raw.get("tools").map(TemplateValue::from_serialize),
            OAIChatLikeRequest::raw_tools,
        )
    }
    fn template_tools(&self) -> Option<TemplateValue> {
        self.raw
            .get("template_tools")
            .map(TemplateValue::from_serialize)
    }
    fn tool_choice(&self) -> Option<TemplateValue> {
        self.typed.as_ref().map_or_else(
            || {
                self.raw
                    .get("tool_choice")
                    .map(TemplateValue::from_serialize)
            },
            OAIChatLikeRequest::tool_choice,
        )
    }
    fn response_format(&self) -> Option<TemplateValue> {
        self.typed.as_ref().map_or_else(
            || {
                self.raw
                    .get("response_format")
                    .map(TemplateValue::from_serialize)
            },
            OAIChatLikeRequest::response_format,
        )
    }
    fn reasoning_effort(&self) -> Option<TemplateValue> {
        self.typed.as_ref().map_or_else(
            || {
                self.raw
                    .get("reasoning_effort")
                    .map(TemplateValue::from_serialize)
            },
            OAIChatLikeRequest::reasoning_effort,
        )
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.settings.add_generation_prompt
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.settings.chat_template_args)
    }
    fn mm_processor_kwargs(&self) -> Option<&Value> {
        self.raw.get("mm_processor_kwargs")
    }
}

fn asset_text(root: &Path, model: &Value, role: &str) -> Result<String> {
    let asset = model["assets"]
        .as_array()
        .context("model has no assets")?
        .iter()
        .find(|asset| asset["role"] == role)
        .with_context(|| format!("missing renderer {role} asset"))?;
    let bytes = renderer::read_source(
        &root.join(renderer::STORE),
        asset["path"].as_str().context("missing asset path")?,
    )?;
    ensure!(
        asset["sha256"] == renderer::sha256(&bytes),
        "renderer asset changed before execution"
    );
    String::from_utf8(bytes).context("renderer asset is not UTF-8")
}

pub fn render_case(root: &Path, model: &Value, case: &Value) -> Result<Value> {
    let settings: Settings = serde_json::from_value(case["settings"].clone())?;
    let typed = if settings.request_mode == "typed" {
        match serde_json::from_value(case["request"].clone()) {
            Ok(request) => Some(request),
            Err(error) => {
                return Ok(
                    json!({"kind":"error","error_type":"invalid_request","message":format!("typed request: {error}")}),
                );
            }
        }
    } else {
        None
    };
    let request = Request {
        raw: case["request"].clone(),
        typed,
        settings,
    };
    let selector = model["model_type"].as_str().map(str::to_owned);
    let display_name = model["display_name"]
        .as_str()
        .or_else(|| model["repository"].as_str())
        .context("model needs display name")?
        .to_lowercase();
    ensure!(
        model
            .get("context_mixins")
            .is_none_or(|mixins| mixins.as_array().is_some_and(Vec::is_empty)),
        "MVP renderer cases use no dynamic context mixins"
    );
    let formatter = match model["formatter"].as_str() {
        Some("kimi_k3") => kimi_k3_formatter_for(
            &selector,
            &display_name,
            request.settings.exclude_tools_when_tool_choice_none,
        )
        .context("K3 selector did not select native formatter")?,
        Some("deepseek_v41") => native_formatter_for(&selector, &display_name)
            .context("DeepSeek selector did not select native formatter")?,
        Some("hf") => {
            let mut config: Value =
                serde_json::from_str(&asset_text(root, model, "tokenizer_config")?)?;
            config["chat_template"] = asset_text(root, model, "chat_template")?.into();
            let content_format = match request.settings.content_format.as_str() {
                "auto" => ChatTemplateContentFormat::Auto,
                "openai" => ChatTemplateContentFormat::OpenAi,
                "string" => ChatTemplateContentFormat::String(MediaPlaceholders::default()),
                _ => anyhow::bail!("invalid content format"),
            };
            PromptFormatter::from_parts_with_content_format(
                serde_json::from_value(config)?,
                ContextMixins::new(&[]),
                request.settings.exclude_tools_when_tool_choice_none,
                content_format,
            )?
        }
        _ => anyhow::bail!("unknown renderer formatter"),
    };
    let PromptFormatter::OAI(formatter) = formatter;
    let result = match formatter.render_prompt(&request) {
        Ok(prompt) => {
            json!({"kind":"prompt","text":prompt.as_str(),"segments":prompt.segments().map(|segments| segments.iter().map(|segment| json!({"text":segment.text,"allow_special":segment.allow_special})).collect::<Vec<_>>()),"pending_segments":prompt.pending_segments(),"media_message_order":formatter.media_message_order(&request)})
        }
        Err(error) => {
            json!({"kind":"error","error_type":if error.downcast_ref::<PromptRenderError>().is_some() { "invalid_request" } else { "render_error" },"message":format!("{error:#}")})
        }
    };
    renderer::validate_result(&result, request.raw["messages"].as_array().unwrap().len())?;
    Ok(result)
}

/// Identify the actual renderer source independently of the checkpoint commit.
pub fn producer(root: &Path) -> Result<Value> {
    fn sources(root: &Path, directory: &Path, files: &mut Vec<String>) -> Result<()> {
        for item in fs::read_dir(directory)? {
            let path = item?.path();
            if path.is_dir() {
                sources(root, &path, files)?;
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(
                    path.strip_prefix(root)?
                        .to_str()
                        .context("non-UTF8 source path")?
                        .to_owned(),
                );
            }
        }
        Ok(())
    }
    let mut files = vec!["renderer/Cargo.toml".to_owned(), "Cargo.lock".to_owned()];
    sources(root, &root.join("renderer/src"), &mut files)?;
    files.sort();
    let mut bytes = Vec::new();
    for file in files {
        let contents = renderer::read_source(root, &file)?;
        bytes.extend_from_slice(file.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&(contents.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&contents);
    }
    let manifest = fs::read_to_string(root.join("renderer/Cargo.toml"))?;
    let version = manifest
        .lines()
        .find_map(|line| {
            line.strip_prefix("version = \"")
                .and_then(|line| line.strip_suffix('"'))
        })
        .context("renderer package version missing")?;
    Ok(
        json!({"implementation":"dynamo","version":version,"source_sha256":renderer::sha256(&bytes),"adapter_source_sha256":renderer::sha256(include_bytes!("renderer_runner.rs")),"scope":"public dynamo-renderer API; downstream normalization is not exercised"}),
    )
}

pub fn capture_model(root: &Path, model: &Value, manifest_sha256: &str) -> Result<Value> {
    let id = model["id"].as_str().context("model ID missing")?;
    let (inputs, inputs_hash) = renderer::read_json(
        &root.join(renderer::STORE),
        &format!("families/{id}/inputs.json"),
    )?;
    renderer::validate_inputs(&inputs, id)?;
    for asset in model["assets"].as_array().context("missing model assets")? {
        let bytes = renderer::read_source(
            &root.join(renderer::STORE),
            asset["path"].as_str().context("asset path missing")?,
        )?;
        ensure!(
            asset["sha256"] == renderer::sha256(&bytes),
            "renderer asset hash mismatch during capture"
        );
    }
    let mut results = serde_json::Map::new();
    for case in inputs["cases"].as_array().unwrap() {
        results.insert(
            case["id"].as_str().unwrap().to_owned(),
            json!({"result":render_case(root, model, case)?}),
        );
    }
    Ok(
        json!({"schema":1,"model":id,"inputs_sha256":inputs_hash,"manifest_sha256":manifest_sha256,"producer":producer(root)?,"results":results}),
    )
}

pub fn capture(root: &Path, output: &Path, selected: Option<&str>) -> Result<()> {
    ensure!(
        !output.exists(),
        "capture output must not exist; authored/reference evidence cannot be overwritten"
    );
    let root = root.canonicalize()?;
    let output = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()?.join(output)
    };
    let mut ancestor = output.as_path();
    while !ancestor.exists() {
        ancestor = ancestor.parent().context("invalid capture output")?;
    }
    let suffix = output.strip_prefix(ancestor)?;
    ensure!(
        suffix
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_))),
        "invalid capture output"
    );
    let destination = ancestor.canonicalize()?.join(suffix);
    ensure!(
        !destination.starts_with(root.join(renderer::STORE)),
        "capture cannot overwrite renderer source evidence"
    );
    let (manifest, manifest_hash) =
        renderer::read_json(&root.join(renderer::STORE), "manifest.json")?;
    let models = manifest["models"]
        .as_array()
        .context("renderer manifest needs models")?;
    ensure!(
        selected.is_none_or(|id| models.iter().any(|model| model["id"] == id)),
        "unknown renderer model"
    );
    if selected.is_some() {
        let current = producer(&root)?;
        check_partial_versions(
            &root,
            models,
            selected,
            current["version"].as_str().unwrap(),
        )?;
    }
    // Complete every invocation before creating a publishable output directory.
    let mut captures = Vec::new();
    for model in models
        .iter()
        .filter(|model| selected.is_none_or(|id| model["id"] == id))
    {
        let captured = capture_model(&root, model, &manifest_hash)?;
        captures.push((model["id"].as_str().unwrap().to_owned(), captured));
    }
    fs::create_dir(&destination)?;
    for (id, document) in captures {
        let directory = destination.join("families").join(id);
        fs::create_dir_all(&directory)?;
        let mut bytes = serde_json::to_vec_pretty(&document)?;
        bytes.push(b'\n');
        fs::write(directory.join("dynamo.json"), bytes)?;
    }
    Ok(())
}

fn check_partial_versions(
    root: &Path,
    models: &[Value],
    selected: Option<&str>,
    version: &str,
) -> Result<()> {
    let Some(selected) = selected else {
        return Ok(());
    };
    for model in models.iter().filter(|model| model["id"] != selected) {
        let id = model["id"].as_str().context("missing model ID")?;
        let name = format!("families/{id}/dynamo.json");
        if root.join(renderer::STORE).join(&name).exists() {
            let (recorded, _) = renderer::read_json(&root.join(renderer::STORE), &name)?;
            ensure!(
                recorded["producer"]["version"] == version,
                "renderer release changed: model {id} was captured with {}; omit --model and refresh all models with {version} to avoid mixed producer versions",
                recorded["producer"]["version"]
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_recapture_across_a_release_fails_before_writing_results() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary
            .path()
            .join(renderer::STORE)
            .join("families/retained");
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("dynamo.json"),
            br#"{"producer":{"version":"10.3.2"}}"#,
        )
        .unwrap();
        let models = vec![json!({"id":"selected"}), json!({"id":"retained"})];
        assert!(
            check_partial_versions(temporary.path(), &models, Some("selected"), "10.3.2").is_ok()
        );
        let error = check_partial_versions(temporary.path(), &models, Some("selected"), "10.3.3")
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("omit --model and refresh all models")
        );
        assert!(check_partial_versions(temporary.path(), &models, None, "10.3.3").is_ok());
        assert_eq!(
            fs::read(directory.join("dynamo.json")).unwrap(),
            br#"{"producer":{"version":"10.3.2"}}"#
        );
    }
}

#[derive(Default, Debug)]
pub struct Verification {
    pub checked: usize,
    pub reference_differences: Vec<String>,
    pub recorded_differences: Vec<String>,
}

/// The strict recorded gate and advisory reference comparison share live execution.
pub fn verify_suite(root: &Path, suite: &Value) -> Result<Verification> {
    let mut verification = Verification::default();
    for model in suite["models"]
        .as_array()
        .context("missing renderer suite models")?
    {
        for case in model["inputs"]["cases"]
            .as_array()
            .context("missing renderer suite cases")?
        {
            let id = case["id"].as_str().context("missing renderer case ID")?;
            let live = render_case(root, &model["model"], case)?;
            verification.checked += 1;
            let label = format!("{}/{id}", model["model"]["id"].as_str().unwrap());
            if live != model["dynamo"]["results"][id]["result"] {
                verification.recorded_differences.push(label.clone());
            }
            if !renderer::reference_matches(
                &model["reference"]["results"][id]["result"],
                &live,
                case["request"]["messages"].as_array().unwrap().len(),
            ) {
                verification.reference_differences.push(label);
            }
        }
    }
    Ok(verification)
}
