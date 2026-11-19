// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The llm-multimodal column of the mm_parity tables: the worker protocol of
//! `multimodal/tests/fixtures/mm_parity/README.md` over the published crate
//! `llm-multimodal`.

use std::{
    collections::HashMap,
    fmt, fs,
    io::Read,
    panic::{self, AssertUnwindSafe},
    path::Path,
};

use llm_multimodal::{
    encoder_inputs::{EncoderDtype, ModelSpecificValue},
    jpeg_turbo, webp_native, ImageFetchConfig, MediaConnector, MediaConnectorConfig, MediaSource,
    Modality, ModelMetadata, ModelProcessorSpec, ModelRegistry, PlaceholderRange,
    PreProcessorConfig, Tokenizer,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const VERSION: &str = "1.15.0";
/// The files this binary is built from; `frameworks/llm_multimodal.py` checks their hash.
const SOURCE: [&[u8]; 3] = [
    include_bytes!("../Cargo.toml"),
    include_bytes!("../Cargo.lock"),
    include_bytes!("main.rs"),
];
const OMITS: [&str; 2] = ["mrope_sha256", "mrope_delta"];

/// The one bicubic kernel each spec's image processor runs (read from the
/// crate source); the crate has no switch, so every case runs it.
fn resize_kernel(spec: &str) -> &'static str {
    match spec {
        "glm53_flash" => "resize_bicubic_pil_rgb",
        "qwen_vl" => "resize_bicubic_torch_aa",
        _ => "unknown",
    }
}

/// The tokenizer's vocabulary only: image replacements need special-token
/// ids, never encoded text.
struct Vocab(HashMap<String, u32>);

impl Vocab {
    /// `tokenizer.json`, then `tokenizer_config.json`'s `added_tokens_decoder`
    /// as transformers adds it (Qwen2-VL's `<|image_pad|>` is only there).
    fn load(model_dir: &Path) -> Result<Self, String> {
        let json = read_json(&model_dir.join("tokenizer.json"))?;
        let mut ids = HashMap::new();
        for (token, id) in json["model"]["vocab"].as_object().into_iter().flatten() {
            if let Some(id) = id.as_u64() {
                ids.insert(token.clone(), id as u32);
            }
        }
        for added in json["added_tokens"].as_array().into_iter().flatten() {
            if let (Some(token), Some(id)) = (added["content"].as_str(), added["id"].as_u64()) {
                ids.insert(token.to_string(), id as u32);
            }
        }
        let config = read_json(&model_dir.join("tokenizer_config.json"))?;
        for (id, added) in config["added_tokens_decoder"]
            .as_object()
            .into_iter()
            .flatten()
        {
            if let (Some(token), Ok(id)) = (added["content"].as_str(), id.parse()) {
                ids.insert(token.to_string(), id);
            }
        }
        Ok(Self(ids))
    }
}

impl Tokenizer for Vocab {
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0
            .iter()
            .find(|(_, &v)| v == id)
            .map(|(k, _)| k.clone())
    }

    fn encode_text(&self, _text: &str) -> Option<Vec<u32>> {
        None
    }
}

struct Context<'a> {
    spec: &'a dyn ModelProcessorSpec,
    metadata: ModelMetadata<'a>,
    processor_config: Value,
    images_dir: &'a Path,
    connector: MediaConnector,
    runtime: tokio::runtime::Runtime,
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The image section a serving frontend loads: `preprocessor_config.json`,
/// else the `image_processor` section of `processor_config.json`.
fn image_processor_config(model_dir: &Path) -> Result<Value, String> {
    let dedicated = model_dir.join("preprocessor_config.json");
    if dedicated.exists() {
        return read_json(&dedicated);
    }
    let combined = read_json(&model_dir.join("processor_config.json"))?;
    combined
        .get("image_processor")
        .cloned()
        .ok_or_else(|| "processor_config.json has no image_processor section".to_string())
}

fn reject(error: impl fmt::Debug + fmt::Display) -> String {
    let debug = format!("{error:?}");
    let variant: String = debug
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    format!("{variant}: {error}")
}

fn int_tensor(value: Option<&ModelSpecificValue>, key: &str) -> Result<Vec<i64>, String> {
    match value {
        Some(ModelSpecificValue::IntTensor { data, .. }) => Ok(data.clone()),
        _ => Err(format!(
            "MissingField: preprocessor output has no int tensor {key}"
        )),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn process(ctx: &Context, case: &Value) -> Result<Value, String> {
    let mut config = ctx.processor_config.clone();
    if let (Some(config), Some(kwargs)) = (config.as_object_mut(), case["image_kwargs"].as_object())
    {
        config.extend(kwargs.clone());
    }
    let config = PreProcessorConfig::from_value(config).map_err(reject)?;
    let processor = ctx
        .spec
        .vision_processor(ctx.metadata.config, &config, Modality::Image)
        .map_err(reject)?;

    let mut images = Vec::new();
    for name in case["images"].as_array().into_iter().flatten() {
        let path = ctx.images_dir.join(name.as_str().unwrap_or_default());
        let bytes =
            fs::read(&path).map_err(|e| format!("{:?}: {}: {e}", e.kind(), path.display()))?;
        let fetch = ctx
            .connector
            .fetch_image(MediaSource::InlineBytes(bytes), ImageFetchConfig::default());
        let frame = ctx.runtime.block_on(fetch).map_err(reject)?;
        images.push(frame.image().map_err(reject)?.clone());
    }
    let preprocessed = processor.preprocess(&images).map_err(reject)?;
    let replacements = ctx
        .spec
        .prompt_replacements_for(&ctx.metadata, &preprocessed, Modality::Image)
        .map_err(reject)?;
    let placeholder = ctx
        .spec
        .placeholder_token_for(&ctx.metadata, Modality::Image)
        .map_err(reject)?;
    let marker = i64::from(ctx.metadata.token_id(&placeholder).map_err(reject)?);

    let mut input_ids = Vec::new();
    let mut offsets = Vec::new();
    let mut pending = replacements.iter();
    for id in case["input_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_i64)
    {
        let Some(replacement) = (id == marker).then(|| pending.next()).flatten() else {
            input_ids.push(id);
            continue;
        };
        let start = input_ids.len();
        let whole = [PlaceholderRange {
            offset: 0,
            length: replacement.tokens.len(),
        }];
        let ranges = replacement.feature_ranges.as_deref().unwrap_or(&whole);
        offsets.extend(
            ranges
                .iter()
                .map(|r| [start + r.offset, start + r.offset + r.length - 1]),
        );
        input_ids.extend(replacement.tokens.iter().map(|&t| i64::from(t)));
    }
    if pending.next().is_some() || replacements.len() != images.len() {
        return Err(format!(
            "ExpansionMismatch: {} replacements for {} images, not all placed",
            replacements.len(),
            images.len()
        ));
    }

    let grid = int_tensor(
        preprocessed.model_specific.get("image_grid_thw"),
        "image_grid_thw",
    )?;
    let rows = int_tensor(
        preprocessed.model_specific.get("patches_per_image"),
        "patches_per_image",
    )?;
    let columns: usize = preprocessed.encoder_input.shape()[1..].iter().product();
    let features = preprocessed
        .encoder_input
        .view()
        .try_write_as(EncoderDtype::Float32)?;
    let mut shapes = Vec::new();
    let mut hashes = Vec::new();
    let mut start = 0;
    for &count in &rows {
        let end = start + count as usize * columns * 4;
        let mut shape = vec![count as usize];
        shape.extend(&preprocessed.encoder_input.shape()[1..]);
        shapes.push(shape);
        hashes.push(sha256_hex(&features[start..end]));
        start = end;
    }
    Ok(json!({
        "input_ids": input_ids,
        "offsets": offsets,
        "image_grid_thw": grid.chunks(3).collect::<Vec<_>>(),
        "feature_shapes": shapes,
        "feature_sha256": hashes,
    }))
}

fn run_case(ctx: &Context, case: &Value) -> Value {
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| process(ctx, case)));
    match outcome {
        Ok(Ok(output)) => json!({"output": output}),
        Ok(Err(message)) => json!({"rejects": message}),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            json!({"rejects": format!("Panic: {message}")})
        }
    }
}

fn run(request: &Value, cases: &[Value]) -> Result<(String, Map<String, Value>), String> {
    let model_dir = Path::new(
        request["model_dir"]
            .as_str()
            .ok_or("request has no model_dir")?,
    );
    let images_dir = Path::new(
        request["images_dir"]
            .as_str()
            .ok_or("request has no images_dir")?,
    );
    let config = read_json(&model_dir.join("config.json"))?;
    let vocab = Vocab::load(model_dir)?;
    let model_id = model_dir.to_string_lossy();
    let metadata = ModelMetadata {
        model_id: &model_id,
        tokenizer: &vocab,
        config: &config,
    };
    let registry = ModelRegistry::new();
    let spec = registry.lookup(&metadata).ok_or_else(|| {
        format!(
            "ModelRegistry has no spec for {model_id} (specs: {})",
            registry.spec_names().join(", ")
        )
    })?;
    let connector =
        MediaConnector::new(reqwest::Client::builder(), MediaConnectorConfig::default())
            .map_err(|e| e.to_string())?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| e.to_string())?;
    let ctx = Context {
        spec,
        metadata,
        processor_config: image_processor_config(model_dir)?,
        images_dir,
        connector,
        runtime,
    };
    let jpeg = if jpeg_turbo::turbojpeg_available() {
        "libturbojpeg"
    } else {
        "image crate (no libturbojpeg.so.0)"
    };
    let webp =
        webp_native::webp_version().map_or("image crate".to_string(), |v| format!("libwebp {v}"));
    let source = format!(
        "ModelRegistry::lookup -> {spec}.vision_processor(image processor config + image_kwargs) -> \
         MediaConnector::fetch_image(InlineBytes, KeepAlpha) -> ImageFrame::image -> preprocess -> \
         prompt_replacements_for + placeholder splice; {spec} uses {kernel} for every backend; \
         JPEG via {jpeg}, WebP via {webp}",
        spec = spec.name(),
        kernel = resize_kernel(spec.name())
    );
    eprintln!("spec {}; {source}", spec.name());
    eprintln!("M-RoPE: the crate has no position code (no mrope or rope_index function in src); omits {OMITS:?}");

    let mut out = Map::new();
    for case in cases {
        let name = case["name"].as_str().unwrap_or_default().to_string();
        let result = run_case(&ctx, case);
        match &result["output"] {
            Value::Null => eprintln!("{name}: {result}"),
            output => eprintln!("{name}: output grid {}", output["image_grid_thw"]),
        }
        out.insert(name, result);
    }
    Ok((source, out))
}

fn main() {
    let mut text = String::new();
    let request: Value = std::io::stdin()
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())
        .and_then(|_| serde_json::from_str(&text).map_err(|e| e.to_string()))
        .unwrap_or_else(|e| panic!("request on stdin: {e}"));
    let cases = request["cases"].as_array().cloned().unwrap_or_default();
    let (source, cases) = run(&request, &cases).unwrap_or_else(|reason| {
        eprintln!("setup: {reason}");
        let na = cases
            .iter()
            .map(|c| {
                (
                    c["name"].as_str().unwrap_or_default().to_string(),
                    json!({"n/a": reason}),
                )
            })
            .collect();
        (
            "llm-multimodal ModelRegistry (setup failed)".to_string(),
            na,
        )
    });
    let versions =
        json!({"llm-multimodal": VERSION, "harness_sha256": sha256_hex(&SOURCE.concat())});
    let response = json!({
        "version": VERSION,
        "source": source,
        "omits": OMITS,
        "versions": versions,
        "cases": cases,
    });
    println!("{response}");
}
