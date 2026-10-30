// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family registry: each family implements
//! [`crate::processor::MmFamilyProcessor`] in `src/models/<model>.rs`; a
//! consumer selects one by its typed [`ProcessorSpec`] or by serializing a
//! spec (`{"family": ..., resolved processor params}`).

use crate::models::deepseek_v41::{DeepseekV41GeometrySpec, DeepseekV41Processor, DeepseekV41Spec};
use crate::models::glm5_next::{Glm5NextGeometrySpec, Glm5NextProcessor, Glm5NextSpec};
use crate::models::qwen_vl::{QwenVlSpec, Resampler};
use crate::{MmError, Result};
use serde_json::Value;

/// Resolved parameters of one family processor, one variant per family.
/// Built directly or deserialized via [`processor_from_spec`], where the
/// `family` key selects the variant.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(try_from = "Value")]
#[non_exhaustive]
pub enum ProcessorSpec {
    QwenVl(QwenVlSpec),
    /// Metadata-only image accounting and layout; pixel processing is unsupported.
    DeepseekV41(DeepseekV41Spec),
    /// Metadata-only GLM-5.3-Flash image accounting and layout; pixel
    /// processing and video are unsupported.
    Glm5Next(Glm5NextSpec),
}

// serde's internally tagged enum buffer treats arbitrary-precision decimals
// as maps before typed f32 fields can see them. Dispatch from Value instead,
// preserving numbers until each family's own deserializer consumes them.
impl TryFrom<Value> for ProcessorSpec {
    type Error = serde_json::Error;

    fn try_from(value: Value) -> std::result::Result<Self, Self::Error> {
        match value.get("family").and_then(Value::as_str) {
            Some("qwen_vl") => serde_json::from_value(value).map(Self::QwenVl),
            Some("deepseek_v41") => serde_json::from_value(value).map(Self::DeepseekV41),
            Some("glm5_next") => serde_json::from_value(value).map(Self::Glm5Next),
            _ => Err(serde::de::Error::custom(
                "unknown or missing processor family",
            )),
        }
    }
}

/// Build a family processor from a typed spec. `Err` when the family
/// rejects its parameters (e.g. a zero patch size).
pub fn build_processor(
    spec: ProcessorSpec,
) -> Result<Box<dyn crate::processor::MmFamilyProcessor>> {
    match spec {
        ProcessorSpec::QwenVl(spec) => Ok(Box::new(crate::models::qwen_vl::QwenVlProcessor::new(
            spec,
        )?)),
        ProcessorSpec::DeepseekV41(spec) => Ok(Box::new(DeepseekV41Processor::new(spec)?)),
        ProcessorSpec::Glm5Next(spec) => Ok(Box::new(Glm5NextProcessor::new(spec)?)),
    }
}

/// Build a family processor from the consumer-side spec JSON
/// (`{"family": ..., resolved processor params}`). `Err` on an unknown family
/// or malformed spec — the caller treats that as "no native processor".
pub fn processor_from_spec(json: &str) -> Result<Box<dyn crate::processor::MmFamilyProcessor>> {
    let spec: ProcessorSpec = serde_json::from_str(json)
        .map_err(|error| MmError::invalid_input_with_source("invalid mm spec", error))?;
    build_processor(spec)
}

/// Resolve a [`ProcessorSpec`] from the HF config files — the
/// `AutoProcessor.from_pretrained` equivalent, resolved once per model at
/// boot. `config.json` selects the family (`model_type`) and carries the
/// token ids; `preprocessor_config.json` carries the processor knobs. For
/// `glm5_next` the second argument may instead be the nested transformers v5
/// `processor_config.json`, the only processor file GLM-5.3-Flash ships.
///
/// Deliberately conservative: an unknown `model_type`, an unrecognized knob,
/// or one the Rust pipeline cannot honor bit-exactly (e.g. `do_normalize:
/// false`) is an `Err` — "no native processor" — never a silent
/// approximation.
pub fn spec_from_hf_configs(
    config_json: &str,
    preprocessor_config_json: &str,
) -> Result<ProcessorSpec> {
    let config: Value = serde_json::from_str(config_json)
        .map_err(|error| MmError::invalid_input_with_source("invalid config.json", error))?;
    let pre: Value = serde_json::from_str(preprocessor_config_json).map_err(|error| {
        MmError::invalid_input_with_source("invalid preprocessor_config.json", error)
    })?;
    match config["model_type"].as_str() {
        Some(
            "qwen2_vl" | "qwen2_5_vl" | "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe",
        ) => Ok(ProcessorSpec::QwenVl(resolve_qwen_vl(&config, &pre)?)),
        Some("deepseek_v4" | "deepseek_v41") => Ok(ProcessorSpec::DeepseekV41(
            resolve_deepseek_v41(&config, &pre)?,
        )),
        Some("glm5_next") => Ok(ProcessorSpec::Glm5Next(resolve_glm5_next(&config, &pre)?)),
        Some(other) => Err(MmError::unsupported(format!(
            "no native processor for model_type {other:?}"
        ))),
        None => Err(MmError::invalid_input("config.json is missing model_type")),
    }
}

fn resolve_deepseek_v41(config: &Value, pre: &Value) -> Result<DeepseekV41Spec> {
    let knobs = pre
        .as_object()
        .ok_or_else(|| MmError::invalid_input("preprocessor_config.json is not an object"))?;
    // This planner mirrors the checkpoint-config-driven SGLang processor, not
    // a pixel pipeline selected by preprocessor_config.json. Refuse overrides
    // rather than silently applying unrelated resize/crop arithmetic.
    for (name, value) in knobs {
        if name != "processor_class" && !value.is_null() {
            return Err(MmError::unsupported(format!(
                "deepseek_v41 metadata planner cannot honor preprocessor knob {name}"
            )));
        }
    }
    // Match the checkpoint adapter's flat-vs-nested discriminator. Optional
    // root-level fields alone must not mask a complete nested vision config.
    let vision = if config.get("vision_patch_size").is_some() {
        config
    } else {
        config.get("vision_config").unwrap_or(config)
    };
    if !vision.is_object() {
        return Err(MmError::invalid_input(
            "deepseek_v41 vision_config must be an object",
        ));
    }
    let geometry: DeepseekV41GeometrySpec =
        serde_json::from_value(vision.clone()).map_err(|error| {
            MmError::invalid_input_with_source("invalid deepseek_v41 vision config", error)
        })?;
    geometry.validate()?;
    let image_token_id = config["image_token_id"]
        .as_i64()
        .and_then(|id| i32::try_from(id).ok())
        .filter(|id| *id >= 0)
        .ok_or_else(|| {
            MmError::invalid_input("config.json needs a nonnegative i32 image_token_id")
        })?;
    Ok(DeepseekV41Spec {
        image_token_id,
        geometry,
    })
}

/// Geometry knobs [`resolve_glm5_next`] consumes (`Glm5NextImageProcessorKwargs`).
const GLM5_NEXT_GEOMETRY: &[&str] = &[
    "patch_size",
    "merge_size",
    "temporal_patch_size",
    "patch_expand_factor",
    "min_image_tokens",
    "max_image_tokens",
];

/// Knobs whose value must hold for the planned canvas to match HF's: resizing
/// stays on, the crop/pad stages this processor never runs stay off, and the
/// processor is one of the two GLM-5.3-Flash image processors, which share
/// `smart_resize`.
const GLM5_NEXT_PINNED: &[(&str, Predicate)] = &[
    ("do_resize", |v| v.as_bool() == Some(true)),
    ("do_center_crop", |v| {
        v.is_null() || v.as_bool() == Some(false)
    }),
    ("do_pad", |v| v.is_null() || v.as_bool() == Some(false)),
    ("image_processor_type", |v| {
        matches!(
            v.as_str(),
            Some("Glm5NextImageProcessor" | "Glm5NextImageProcessorPil")
        )
    }),
];

/// Knobs that only shape pixel values or output packaging, never the canvas,
/// so a metadata-only planner can ignore them. `size` is declared by the HF
/// class but ignored by its `_preprocess`; the token bounds replace it.
const GLM5_NEXT_INERT: &[&str] = &[
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "do_convert_rgb",
    "resample",
    "size",
    "default_to_square",
    "data_format",
    "input_data_format",
    "device",
    "disable_grouping",
    "return_tensors",
    "crop_size",
    "processor_class",
];

fn resolve_glm5_next(config: &Value, pre: &Value) -> Result<Glm5NextSpec> {
    let top = pre
        .as_object()
        .ok_or_else(|| MmError::invalid_input("processor config is not an object"))?;
    // Transformers v5 checkpoints ship `processor_config.json`, which nests
    // the image knobs under `image_processor` beside the video processor; a
    // `preprocessor_config.json` dump is the flat image-processor object.
    let knobs = match top.get("image_processor") {
        Some(nested) => {
            if let Some((name, _)) = top.iter().find(|(name, value)| {
                !matches!(
                    name.as_str(),
                    "image_processor" | "video_processor" | "processor_class"
                ) && !value.is_null()
            }) {
                return Err(MmError::unsupported(format!(
                    "unrecognized processor_config key {name}"
                )));
            }
            nested.as_object().ok_or_else(|| {
                MmError::invalid_input("processor_config image_processor is not an object")
            })?
        }
        None => top,
    };
    for (knob, value) in knobs {
        let knob = knob.as_str();
        if GLM5_NEXT_GEOMETRY.contains(&knob) || GLM5_NEXT_INERT.contains(&knob) {
            continue;
        }
        match GLM5_NEXT_PINNED.iter().find(|(name, _)| *name == knob) {
            Some((_, honored)) if honored(value) => {}
            Some(_) => {
                return Err(MmError::unsupported(format!(
                    "glm5_next preprocessor knob {knob} = {value} cannot be honored"
                )));
            }
            None if value.is_null() => {}
            None => {
                return Err(MmError::unsupported(format!(
                    "unrecognized glm5_next preprocessor knob {knob}"
                )));
            }
        }
    }
    // Every geometry knob must be present and non-null: no serving defaults.
    let geometry: Glm5NextGeometrySpec = serde_json::from_value(Value::Object(knobs.clone()))
        .map_err(|error| {
            MmError::invalid_input_with_source("invalid glm5_next image processor geometry", error)
        })?;
    geometry.validate()?;
    // The vision tower must agree with the processor; a drifted checkpoint is
    // refused rather than planned approximately.
    if let Some(vision) = config
        .get("vision_config")
        .filter(|value| value.is_object())
    {
        for (knob, field, value) in [
            ("patch_size", "patch_size", geometry.patch_size),
            ("merge_size", "spatial_merge_size", geometry.merge_size),
            (
                "temporal_patch_size",
                "temporal_patch_size",
                geometry.temporal_patch_size,
            ),
        ] {
            if let Some(declared) = vision.get(field).and_then(Value::as_u64)
                && declared != u64::from(value)
            {
                return Err(MmError::invalid_input(format!(
                    "glm5_next: processor {knob} = {value} disagrees with vision_config.{field} = {declared}"
                )));
            }
        }
    }
    let token_id = |field: &str| -> Result<Option<i32>> {
        match config.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_i64()
                .and_then(|id| i32::try_from(id).ok())
                .filter(|id| *id >= 0)
                .map(Some)
                .ok_or_else(|| {
                    MmError::invalid_input(format!("config.json {field} must be a nonnegative i32"))
                }),
        }
    };
    let image_token_id = token_id("image_token_id")?
        .ok_or_else(|| MmError::invalid_input("config.json is missing image_token_id"))?;
    Ok(Glm5NextSpec {
        image_token_id,
        video_token_id: token_id("video_token_id")?,
        geometry,
    })
}

/// [`spec_from_hf_configs`] over a local model directory: reads `config.json`
/// and `preprocessor_config.json`, or `processor_config.json` when the
/// checkpoint ships only the nested transformers v5 file (GLM-5.3-Flash).
/// Downloading from a hub stays the consumer's concern — hand this the
/// resolved local dir.
pub fn spec_from_model_dir(dir: &std::path::Path) -> Result<ProcessorSpec> {
    let read = |name: &str| {
        std::fs::read_to_string(dir.join(name)).map_err(|error| {
            MmError::invalid_input_with_source(format!("cannot read {name}"), error)
        })
    };
    let config = read("config.json")?;
    let pre = match std::fs::read_to_string(dir.join("preprocessor_config.json")) {
        Ok(json) => json,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::read_to_string(dir.join("processor_config.json")).map_err(|error| {
                MmError::invalid_input_with_source(
                    "cannot read preprocessor_config.json or processor_config.json",
                    error,
                )
            })?
        }
        Err(error) => {
            return Err(MmError::invalid_input_with_source(
                "cannot read preprocessor_config.json",
                error,
            ));
        }
    };
    spec_from_hf_configs(&config, &pre)
}

const CONSUMED: &[&str] = &[
    "patch_size",
    "merge_size",
    "temporal_patch_size",
    "min_pixels",
    "max_pixels",
    "size",
    "image_mean",
    "image_std",
];

/// Knobs whose value must hold for the Rust arithmetic to match HF's: the
/// stages this pipeline hardcodes stay enabled, the ones it omits stay
/// disabled, and PIL BICUBIC (3) is the only mirrored kernel.
type Predicate = fn(&Value) -> bool;
const PINNED: &[(&str, Predicate)] = &[
    ("do_resize", |v| v.as_bool() == Some(true)),
    ("do_rescale", |v| v.as_bool() == Some(true)),
    ("do_normalize", |v| v.as_bool() == Some(true)),
    ("do_convert_rgb", |v| v.as_bool() == Some(true)),
    // HF treats null as disabled for these optional stages.
    ("do_center_crop", |v| {
        v.is_null() || v.as_bool() == Some(false)
    }),
    ("do_pad", |v| v.is_null() || v.as_bool() == Some(false)),
    ("resample", |v| v.as_i64() == Some(3)),
    ("rescale_factor", |v| v.as_f64() == Some(1.0 / 255.0)),
    ("data_format", |v| v.as_str() == Some("channels_first")),
    ("input_data_format", |v| {
        v.is_null() || v.as_str() == Some("channels_first")
    }),
    ("image_processor_type", |v| {
        matches!(
            v.as_str(),
            Some("Qwen2VLImageProcessor" | "Qwen2VLImageProcessorFast")
        )
    }),
];

/// Knobs recognized but irrelevant to the output: batching, output
/// packaging, naming, or gated behind a [`PINNED`] flag that stays off.
const INERT: &[&str] = &[
    "crop_size",
    "default_to_square",
    "disable_grouping",
    "return_tensors",
    "processor_class",
];

fn resolve_qwen_vl(config: &Value, pre: &Value) -> Result<QwenVlSpec> {
    let knobs = pre
        .as_object()
        .ok_or_else(|| MmError::invalid_input("preprocessor_config.json is not an object"))?;
    let set = |knob: &str| knobs.get(knob).filter(|value| !value.is_null());
    for (knob, value) in knobs {
        let knob = knob.as_str();
        if CONSUMED.contains(&knob) || INERT.contains(&knob) {
            continue;
        }
        match PINNED.iter().find(|(name, _)| *name == knob) {
            Some((_, honored)) if honored(value) => {}
            Some(_) => {
                return Err(MmError::unsupported(format!(
                    "preprocessor knob {knob} = {value} cannot be honored bit-exactly"
                )));
            }
            None if value.is_null() => {}
            None => {
                return Err(MmError::unsupported(format!(
                    "unrecognized preprocessor knob {knob}"
                )));
            }
        }
    }

    let to_usize = |knob: &str, value: Option<&Value>| {
        let value = value.and_then(Value::as_u64).ok_or_else(|| {
            MmError::invalid_input(format!("preprocessor knob {knob} is missing"))
        })?;
        usize::try_from(value).map_err(|_| {
            MmError::invalid_input(format!(
                "preprocessor knob {knob} = {value} does not fit usize"
            ))
        })
    };
    let usize_knob = |knob: &str| to_usize(knob, set(knob));
    let rgb_knob = |knob: &str| -> Result<[f32; 3]> {
        set(knob)
            .and_then(|value| serde_json::from_value::<[f32; 3]>(value.clone()).ok())
            .ok_or_else(|| {
                MmError::invalid_input(format!("preprocessor knob {knob} must be 3 numbers"))
            })
    };
    // Qwen2/2.5 carry the pixel bounds at the top level; Qwen3 as
    // `size.{shortest,longest}_edge` (pixel counts despite the names).
    let size = set("size")
        .map(|value| {
            let size = value.as_object().ok_or_else(|| {
                MmError::invalid_input("preprocessor knob size must be an object")
            })?;
            for key in size.keys() {
                if !matches!(key.as_str(), "shortest_edge" | "longest_edge") {
                    return Err(MmError::unsupported(format!(
                        "unrecognized preprocessor size key {key}"
                    )));
                }
            }
            Ok(size)
        })
        .transpose()?;
    let pixels = |knob: &str, edge: &str| {
        let value = set(knob)
            .or_else(|| size?.get(edge))
            .filter(|value| !value.is_null());
        to_usize(knob, value)
    };

    Ok(QwenVlSpec {
        image_token_id: config["image_token_id"]
            .as_i64()
            .and_then(|id| i32::try_from(id).ok())
            .ok_or_else(|| MmError::invalid_input("config.json is missing image_token_id"))?,
        patch_size: usize_knob("patch_size")?,
        merge_size: usize_knob("merge_size")?,
        temporal_patch_size: usize_knob("temporal_patch_size")?,
        min_pixels: pixels("min_pixels", "shortest_edge")?,
        max_pixels: pixels("max_pixels", "longest_edge")?,
        image_mean: rgb_knob("image_mean")?,
        image_std: rgb_knob("image_std")?,
        // Both HF processor names run torchvision's uint8 path on a default
        // server; `Resampler::Pil` stays reachable through the spec JSON.
        resample: Resampler::AtenU8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trimmed from Qwen/Qwen2.5-VL-7B-Instruct.
    const QWEN25_CONFIG: &str = r#"{
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "image_token_id": 151655, "video_token_id": 151656,
        "model_type": "qwen2_5_vl", "vision_config": {"patch_size": 14}
    }"#;
    const QWEN25_PREPROCESSOR: &str = r#"{
        "min_pixels": 3136, "max_pixels": 12845056,
        "patch_size": 14, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "image_processor_type": "Qwen2VLImageProcessor",
        "processor_class": "Qwen2_5_VLProcessor"
    }"#;
    // Trimmed from Qwen/Qwen3-VL-8B-Instruct (a fast-processor dump: pixel
    // bounds under `size`, inert knobs serialized as null).
    const QWEN3_CONFIG: &str = r#"{
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "image_token_id": 151655, "model_type": "qwen3_vl"
    }"#;
    const QWEN3_PREPROCESSOR: &str = r#"{
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "do_resize": true, "do_rescale": true, "do_normalize": true,
        "do_convert_rgb": true, "do_center_crop": null, "do_pad": null,
        "resample": 3, "rescale_factor": 0.00392156862745098,
        "crop_size": null, "data_format": "channels_first", "device": null,
        "default_to_square": true, "disable_grouping": null,
        "input_data_format": null, "return_tensors": null,
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast"
    }"#;

    // Trimmed from Qwen/Qwen3.5-27B: Qwen3-VL's processor under a new model_type.
    const QWEN35_CONFIG: &str = r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056, "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text"}
    }"#;
    const QWEN35_PREPROCESSOR: &str = r#"{
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast"
    }"#;

    fn qwen_spec(config: &str, pre: &str) -> Result<QwenVlSpec> {
        spec_from_hf_configs(config, pre).map(|spec| match spec {
            ProcessorSpec::QwenVl(spec) => spec,
            _ => panic!("expected Qwen spec"),
        })
    }

    #[test]
    fn resolves_qwen25_vl() {
        let spec = qwen_spec(QWEN25_CONFIG, QWEN25_PREPROCESSOR).unwrap();
        assert_eq!(spec.image_token_id, 151655);
        assert_eq!(
            (spec.patch_size, spec.merge_size, spec.temporal_patch_size),
            (14, 2, 2)
        );
        assert_eq!((spec.min_pixels, spec.max_pixels), (3136, 12845056));
        assert_eq!(spec.image_mean, [0.48145466, 0.4578275, 0.40821073]);
        assert_eq!(spec.resample, Resampler::AtenU8);
        assert!(build_processor(ProcessorSpec::QwenVl(spec)).is_ok());
    }

    #[test]
    fn resolves_qwen3_vl_size_edges() {
        let spec = qwen_spec(QWEN3_CONFIG, QWEN3_PREPROCESSOR).unwrap();
        assert_eq!((spec.min_pixels, spec.max_pixels), (65536, 16777216));
        assert_eq!(spec.patch_size, 16);
    }

    #[test]
    fn resolves_qwen35() {
        let spec = qwen_spec(QWEN35_CONFIG, QWEN35_PREPROCESSOR).unwrap();
        assert_eq!(spec.image_token_id, 248056);
        assert_eq!((spec.min_pixels, spec.max_pixels), (65536, 16777216));
        let moe = QWEN35_CONFIG.replace("qwen3_5", "qwen3_5_moe");
        assert!(qwen_spec(&moe, QWEN35_PREPROCESSOR).is_ok());
    }

    /// `size` is type-checked even when the top-level pixel bounds make it
    /// redundant, as HF does.
    #[test]
    fn malformed_size_is_invalid() {
        let pre = QWEN25_PREPROCESSOR.replacen('{', r#"{"size": "invalid","#, 1);
        assert!(matches!(
            qwen_spec(QWEN25_CONFIG, &pre),
            Err(MmError::InvalidInput { .. })
        ));
    }

    #[test]
    fn null_required_settings_are_unsupported() {
        for knob in [
            "do_resize",
            "do_rescale",
            "do_normalize",
            "do_convert_rgb",
            "resample",
            "rescale_factor",
            "data_format",
            "image_processor_type",
        ] {
            let mut pre: Value = serde_json::from_str(QWEN25_PREPROCESSOR).unwrap();
            pre[knob] = Value::Null;
            assert!(
                matches!(
                    qwen_spec(QWEN25_CONFIG, &pre.to_string()),
                    Err(MmError::Unsupported { .. })
                ),
                "{knob}: null must not select a pipeline with different behavior"
            );
        }
    }

    #[test]
    fn null_optional_settings_preserve_the_spec() {
        let mut pre: Value = serde_json::from_str(QWEN25_PREPROCESSOR).unwrap();
        for knob in ["do_center_crop", "do_pad", "input_data_format", "device"] {
            pre[knob] = Value::Null;
        }
        let spec = qwen_spec(QWEN25_CONFIG, &pre.to_string()).unwrap();
        assert_eq!(spec.resample, Resampler::AtenU8);
        assert_eq!((spec.min_pixels, spec.max_pixels), (3136, 12845056));
        assert!(build_processor(ProcessorSpec::QwenVl(spec)).is_ok());
    }

    #[test]
    fn normalization_requires_exactly_three_numeric_entries() {
        for knob in ["image_mean", "image_std"] {
            for values in [
                serde_json::json!([0.48, null, 0.45, 0.40]),
                serde_json::json!([0.48, null, 0.40]),
                serde_json::json!([0.48, 0.45]),
                serde_json::json!([0.48, 0.45, 0.40, 0.50]),
            ] {
                let mut pre: Value = serde_json::from_str(QWEN25_PREPROCESSOR).unwrap();
                pre[knob] = values;
                assert!(
                    matches!(
                        qwen_spec(QWEN25_CONFIG, &pre.to_string()),
                        Err(MmError::InvalidInput { .. })
                    ),
                    "{knob} = {} must be rejected",
                    pre[knob]
                );
            }
        }
    }

    #[test]
    fn unsupported_size_keys_are_rejected_even_with_top_level_bounds() {
        let mut pre: Value = serde_json::from_str(QWEN25_PREPROCESSOR).unwrap();
        pre["size"] = serde_json::json!({
            "shortest_edge": 65536, "longest_edge": 16777216, "alien": 512
        });
        assert!(matches!(
            qwen_spec(QWEN25_CONFIG, &pre.to_string()),
            Err(MmError::Unsupported { .. })
        ));
    }

    #[test]
    fn unknown_model_type_is_unsupported() {
        let config = r#"{"model_type": "llava", "image_token_id": 32000}"#;
        assert!(matches!(
            spec_from_hf_configs(config, QWEN25_PREPROCESSOR),
            Err(MmError::Unsupported { .. })
        ));
    }

    /// Any knob the pipeline cannot mirror bit-exactly must refuse resolution,
    /// whether it is a disabled stage, a foreign kernel, or a knob this crate
    /// has never seen.
    #[test]
    fn unhonorable_or_unknown_knobs_are_refused() {
        let mut refused =
            vec![QWEN25_PREPROCESSOR.replace("Qwen2VLImageProcessor", "LlavaImageProcessor")];
        for knob in [
            r#""do_normalize": false"#,
            r#""do_center_crop": true"#,
            r#""resample": 2"#,
            r#""rescale_factor": 0.5"#,
            r#""device": "cuda""#,
            r#""pad_size": {"height": 512}"#,
        ] {
            refused.push(QWEN25_PREPROCESSOR.replacen('{', &format!("{{{knob},"), 1));
        }
        for pre in refused {
            assert!(
                matches!(
                    qwen_spec(QWEN25_CONFIG, &pre),
                    Err(MmError::Unsupported { .. })
                ),
                "{pre} should refuse resolution"
            );
        }
    }

    #[test]
    fn model_dir_resolution_reads_both_configs() {
        let dir = std::env::temp_dir().join(format!("mm-registry-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), QWEN25_CONFIG).unwrap();
        std::fs::write(dir.join("preprocessor_config.json"), QWEN25_PREPROCESSOR).unwrap();
        let ProcessorSpec::QwenVl(spec) = spec_from_model_dir(&dir).unwrap() else {
            panic!("expected Qwen spec");
        };
        assert_eq!(spec.max_pixels, 12845056);
        std::fs::remove_dir_all(&dir).unwrap();

        assert!(spec_from_model_dir(std::path::Path::new("/nonexistent")).is_err());
    }
}
