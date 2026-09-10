// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family registry: each family implements
//! [`crate::processor::MmFamilyProcessor`] in `src/models/<model>.rs`; a
//! consumer selects one by its typed [`ProcessorSpec`] or by serializing a
//! spec (`{"family": ..., resolved processor params}`).

use crate::models::qwen_vl::{QwenVlSpec, Resampler};
use crate::{MmError, Result};
use serde_json::Value;

/// Resolved parameters of one family processor, one variant per family.
/// Built directly or deserialized via [`processor_from_spec`], where the
/// `family` key selects the variant.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProcessorSpec {
    QwenVl(QwenVlSpec),
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
/// token ids; `preprocessor_config.json` carries the processor knobs.
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
        Some("qwen2_vl" | "qwen2_5_vl" | "qwen3_vl" | "qwen3_vl_moe") => {
            Ok(ProcessorSpec::QwenVl(resolve_qwen_vl(&config, &pre)?))
        }
        Some(other) => Err(MmError::unsupported(format!(
            "no native processor for model_type {other:?}"
        ))),
        None => Err(MmError::invalid_input("config.json is missing model_type")),
    }
}

/// [`spec_from_hf_configs`] over a local model directory (reads `config.json`
/// and `preprocessor_config.json`). Downloading from a hub stays the
/// consumer's concern — hand this the resolved local dir.
pub fn spec_from_model_dir(dir: &std::path::Path) -> Result<ProcessorSpec> {
    let read = |name: &str| {
        std::fs::read_to_string(dir.join(name)).map_err(|error| {
            MmError::invalid_input_with_source(format!("cannot read {name}"), error)
        })
    };
    spec_from_hf_configs(&read("config.json")?, &read("preprocessor_config.json")?)
}

/// Knobs the qwen resolution reads: patch geometry, pixel bounds,
/// normalization.
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
    ("do_center_crop", |v| v.as_bool() == Some(false)),
    ("do_pad", |v| v.as_bool() == Some(false)),
    ("resample", |v| v.as_i64() == Some(3)),
    ("rescale_factor", |v| v.as_f64() == Some(1.0 / 255.0)),
    ("data_format", |v| v.as_str() == Some("channels_first")),
    ("input_data_format", |v| {
        v.as_str() == Some("channels_first")
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
        if value.is_null() || CONSUMED.contains(&knob) || INERT.contains(&knob) {
            continue;
        }
        match PINNED.iter().find(|(name, _)| *name == knob) {
            Some((_, honored)) if honored(value) => {}
            Some(_) => {
                return Err(MmError::unsupported(format!(
                    "preprocessor knob {knob} = {value} cannot be honored bit-exactly"
                )));
            }
            None => {
                return Err(MmError::unsupported(format!(
                    "unrecognized preprocessor knob {knob}"
                )));
            }
        }
    }

    let usize_knob = |knob: &str| {
        set(knob)
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| MmError::invalid_input(format!("preprocessor knob {knob} is missing")))
    };
    let rgb_knob = |knob: &str| -> Result<[f32; 3]> {
        let values = set(knob)
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
            .filter(|values| values.len() == 3)
            .ok_or_else(|| {
                MmError::invalid_input(format!("preprocessor knob {knob} must be 3 numbers"))
            })?;
        Ok([values[0] as f32, values[1] as f32, values[2] as f32])
    };
    // Qwen2/2.5 carry the pixel bounds at the top level; Qwen3 as
    // `size.{shortest,longest}_edge` (pixel counts despite the names).
    let pixels = |knob: &str, edge: &str| {
        set(knob)
            .or_else(|| pre["size"].get(edge).filter(|value| !value.is_null()))
            .or_else(|| pre["size"].get(knob).filter(|value| !value.is_null()))
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| MmError::invalid_input(format!("preprocessor {knob} is missing")))
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

    fn qwen_spec(config: &str, pre: &str) -> Result<QwenVlSpec> {
        spec_from_hf_configs(config, pre).map(|ProcessorSpec::QwenVl(spec)| spec)
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
        let ProcessorSpec::QwenVl(spec) = spec_from_model_dir(&dir).unwrap();
        assert_eq!(spec.max_pixels, 12845056);
        std::fs::remove_dir_all(&dir).unwrap();

        assert!(spec_from_model_dir(std::path::Path::new("/nonexistent")).is_err());
    }
}
