// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::glm5_next::Glm5NextSpec;
use dynamo_multimodal::processor::{DecodedMedia, MediaMetadata};
use dynamo_multimodal::registry::{
    ProcessorSpec, build_processor, processor_from_spec, spec_from_hf_configs, spec_from_model_dir,
};
use dynamo_multimodal::token_layout::apply_layout;

// Trimmed from zai-org/GLM-5.3-Flash config.json.
const GLM53_CONFIG: &str = r#"{
    "architectures": ["Glm5NextForConditionalGeneration"],
    "model_type": "glm5_next",
    "image_token_id": 154854, "image_start_token_id": 154830, "image_end_token_id": 154831,
    "video_token_id": 154855, "video_start_token_id": 154832, "video_end_token_id": 154833,
    "text_config": {"model_type": "glm5_next_text"},
    "vision_config": {
        "model_type": "glm5_next_vision", "image_size": 448, "patch_size": 14,
        "spatial_merge_size": 2, "temporal_patch_size": 2, "depth": 24
    }
}"#;
// zai-org/GLM-5.3-Flash processor_config.json: the checkpoint ships no
// preprocessor_config.json, only this nested transformers v5 file.
const GLM53_PROCESSOR_CONFIG: &str = r#"{
    "image_processor": {
        "do_rescale": true, "patch_expand_factor": 1, "merge_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "temporal_patch_size": 2, "patch_size": 14,
        "min_image_tokens": 16, "max_image_tokens": 8000,
        "image_processor_type": "Glm5NextImageProcessor"
    },
    "video_processor": {
        "do_rescale": true, "video_processor_type": "Glm5NextVideoProcessor",
        "patch_expand_factor": 1, "merge_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "temporal_patch_size": 2, "patch_size": 14,
        "min_image_tokens": 16, "max_image_tokens": 240000, "fps": 2
    },
    "processor_class": "Glm5NextProcessor"
}"#;

const BEGIN_OF_IMAGE: i32 = 154830;
const IMAGE: i32 = 154854;
const END_OF_IMAGE: i32 = 154831;

fn glm_spec(config: &str, pre: &str) -> dynamo_multimodal::Result<Glm5NextSpec> {
    spec_from_hf_configs(config, pre).map(|spec| match spec {
        ProcessorSpec::Glm5Next(spec) => spec,
        _ => panic!("expected GLM spec"),
    })
}

fn image_processor_object() -> serde_json::Value {
    serde_json::from_str::<serde_json::Value>(GLM53_PROCESSOR_CONFIG).unwrap()["image_processor"]
        .clone()
}

#[test]
fn resolves_the_checkpoint_from_nested_and_flat_processor_configs() {
    let flat = image_processor_object().to_string();
    for pre in [GLM53_PROCESSOR_CONFIG, flat.as_str()] {
        let spec = glm_spec(GLM53_CONFIG, pre).unwrap();
        assert_eq!(spec.image_token_id, IMAGE);
        assert_eq!(spec.video_token_id, Some(154855));
        assert_eq!(
            (
                spec.geometry.patch_size,
                spec.geometry.merge_size,
                spec.geometry.temporal_patch_size,
                spec.geometry.patch_expand_factor,
            ),
            (14, 2, 2, 1)
        );
        assert_eq!(
            (
                spec.geometry.min_image_tokens,
                spec.geometry.max_image_tokens
            ),
            (16, 8000)
        );
        let family = build_processor(ProcessorSpec::Glm5Next(spec)).unwrap();
        let image = MediaMetadata::Image {
            width: 1024,
            height: 768,
        };
        assert_eq!(family.num_media_tokens(&image).unwrap(), 1036);
    }
}

#[test]
fn layout_expands_only_the_image_placeholder() {
    let family = build_processor(ProcessorSpec::Glm5Next(
        glm_spec(GLM53_CONFIG, GLM53_PROCESSOR_CONFIG).unwrap(),
    ))
    .unwrap();
    let media =
        [(448, 448), (13, 15)].map(|(width, height)| MediaMetadata::Image { width, height });
    let counts = media
        .iter()
        .map(|item| family.num_media_tokens(item).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(counts, [256, 20]);
    // Two chat-template image blocks around text, as the renderer emits them.
    let ids = [
        7,
        BEGIN_OF_IMAGE,
        IMAGE,
        END_OF_IMAGE,
        8,
        BEGIN_OF_IMAGE,
        IMAGE,
        END_OF_IMAGE,
        9,
    ];
    let layout = family.layout_from_metadata(&ids, &media).unwrap();
    let expanded = apply_layout(&ids, &layout, &counts, 512).unwrap();
    let expected = [
        vec![7, BEGIN_OF_IMAGE],
        vec![IMAGE; 256],
        vec![END_OF_IMAGE, 8, BEGIN_OF_IMAGE],
        vec![IMAGE; 20],
        vec![END_OF_IMAGE, 9],
    ]
    .concat();
    assert_eq!(expanded.input_ids, expected);
    assert_eq!(expanded.offsets, [(2, 257), (261, 280)]);
    for (ranges, expected) in expanded.feature_ranges.iter().zip([2..258, 261..281]) {
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], expected);
    }
    // A placeholder count that does not match the media list is an error,
    // not a best-effort assignment.
    assert!(
        family
            .layout_from_metadata(&[BEGIN_OF_IMAGE, IMAGE, END_OF_IMAGE], &media)
            .is_err()
    );
}

#[test]
fn prompts_with_the_video_placeholder_are_refused() {
    let family = build_processor(ProcessorSpec::Glm5Next(
        glm_spec(GLM53_CONFIG, GLM53_PROCESSOR_CONFIG).unwrap(),
    ))
    .unwrap();
    let media = [MediaMetadata::Image {
        width: 448,
        height: 448,
    }];
    let ids = [BEGIN_OF_IMAGE, IMAGE, END_OF_IMAGE, 154832, 154855, 154833];
    assert!(matches!(
        family.layout_from_metadata(&ids, &media),
        Err(MmError::Unsupported { .. })
    ));
    let video = MediaMetadata::Video {
        width: 448,
        height: 448,
        num_frames: 16,
    };
    assert!(matches!(
        family.num_media_tokens(&video),
        Err(MmError::Unsupported { .. })
    ));
    // Without a declared video id the planner cannot detect video prompts;
    // the layout still succeeds and the consumer owns that gate.
    let mut config: serde_json::Value = serde_json::from_str(GLM53_CONFIG).unwrap();
    config.as_object_mut().unwrap().remove("video_token_id");
    let spec = glm_spec(&config.to_string(), GLM53_PROCESSOR_CONFIG).unwrap();
    assert_eq!(spec.video_token_id, None);
}

#[test]
fn serialized_spec_keeps_pixel_execution_and_other_modalities_unsupported() {
    let family = processor_from_spec(
        r#"{
            "family": "glm5_next", "image_token_id": 154854, "video_token_id": 154855,
            "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
            "patch_expand_factor": 1, "min_image_tokens": 16, "max_image_tokens": 8000
        }"#,
    )
    .unwrap();
    let image = DecodedMedia::Image {
        rgb: vec![],
        height: 42,
        width: 42,
    };
    assert!(matches!(
        family.process_item(&image),
        Err(MmError::Unsupported { .. })
    ));
    assert!(matches!(
        family.layout(&[IMAGE], &[]),
        Err(MmError::Unsupported { .. })
    ));
    let audio = MediaMetadata::Audio {
        num_samples: 16000,
        sample_rate: 16000,
    };
    assert!(matches!(
        family.layout_from_metadata(&[IMAGE], &[audio]),
        Err(MmError::Unsupported { .. })
    ));
    // Every geometry knob is required in the serialized spec too.
    assert!(
        processor_from_spec(
            r#"{"family": "glm5_next", "image_token_id": 154854, "patch_size": 14,
                "merge_size": 2, "temporal_patch_size": 2, "patch_expand_factor": 1,
                "min_image_tokens": 16}"#
        )
        .is_err()
    );
}

#[test]
fn refuses_knobs_that_change_or_hide_the_canvas() {
    let unsupported = |pre: &serde_json::Value| {
        assert!(
            matches!(
                glm_spec(GLM53_CONFIG, &pre.to_string()),
                Err(MmError::Unsupported { .. })
            ),
            "{pre}"
        );
    };
    let mut off = image_processor_object();
    off["do_resize"] = false.into();
    unsupported(&off);
    let mut null_resize = image_processor_object();
    null_resize["do_resize"] = serde_json::Value::Null;
    unsupported(&null_resize);
    let mut foreign = image_processor_object();
    foreign["image_processor_type"] = "Qwen2VLImageProcessor".into();
    unsupported(&foreign);
    let mut pixel_bounds = image_processor_object();
    pixel_bounds["max_pixels"] = 12845056.into();
    unsupported(&pixel_bounds);
    let mut cropped = image_processor_object();
    cropped["do_center_crop"] = true.into();
    unsupported(&cropped);
    let mut nested: serde_json::Value = serde_json::from_str(GLM53_PROCESSOR_CONFIG).unwrap();
    nested["audio_processor"] = serde_json::json!({"sampling_rate": 16000});
    unsupported(&nested);

    // Pixel-only knobs and the PIL processor variant do not move the canvas.
    let mut accepted = image_processor_object();
    accepted["resample"] = 2.into();
    accepted["do_convert_rgb"] = true.into();
    accepted["do_center_crop"] = serde_json::Value::Null;
    accepted["size"] = serde_json::json!({"longest_edge": 1});
    accepted["image_processor_type"] = "Glm5NextImageProcessorPil".into();
    assert!(glm_spec(GLM53_CONFIG, &accepted.to_string()).is_ok());
}

#[test]
fn refuses_missing_geometry_bad_ids_and_vision_config_drift() {
    let invalid = |config: &str, pre: &serde_json::Value| {
        assert!(
            matches!(
                glm_spec(config, &pre.to_string()),
                Err(MmError::InvalidInput { .. })
            ),
            "{pre}"
        );
    };
    let mut null_budget = image_processor_object();
    null_budget["max_image_tokens"] = serde_json::Value::Null;
    invalid(GLM53_CONFIG, &null_budget); // no inferred HF class defaults
    let mut missing = image_processor_object();
    missing.as_object_mut().unwrap().remove("min_image_tokens");
    invalid(GLM53_CONFIG, &missing);
    let mut zero = image_processor_object();
    zero["max_image_tokens"] = 0.into();
    invalid(GLM53_CONFIG, &zero);

    let pre = image_processor_object();
    let mut config: serde_json::Value = serde_json::from_str(GLM53_CONFIG).unwrap();
    config.as_object_mut().unwrap().remove("image_token_id");
    invalid(&config.to_string(), &pre);
    let mut config: serde_json::Value = serde_json::from_str(GLM53_CONFIG).unwrap();
    config["video_token_id"] = (-1).into();
    invalid(&config.to_string(), &pre);
    let mut config: serde_json::Value = serde_json::from_str(GLM53_CONFIG).unwrap();
    config["vision_config"]["patch_size"] = 16.into();
    invalid(&config.to_string(), &pre);
    let mut config: serde_json::Value = serde_json::from_str(GLM53_CONFIG).unwrap();
    config["vision_config"]["spatial_merge_size"] = 1.into();
    invalid(&config.to_string(), &pre);

    let mut renamed: serde_json::Value = serde_json::from_str(GLM53_CONFIG).unwrap();
    renamed["model_type"] = "glm4v".into();
    assert!(matches!(
        spec_from_hf_configs(&renamed.to_string(), GLM53_PROCESSOR_CONFIG),
        Err(MmError::Unsupported { .. })
    ));
}

#[test]
fn model_dir_resolution_falls_back_to_processor_config() {
    let dir = std::env::temp_dir().join(format!("mm-glm5-next-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), GLM53_CONFIG).unwrap();
    std::fs::write(dir.join("processor_config.json"), GLM53_PROCESSOR_CONFIG).unwrap();
    let ProcessorSpec::Glm5Next(spec) = spec_from_model_dir(&dir).unwrap() else {
        panic!("expected GLM spec");
    };
    assert_eq!(spec.geometry.max_image_tokens, 8000);
    // A flat preprocessor_config.json takes precedence when present.
    std::fs::write(
        dir.join("preprocessor_config.json"),
        image_processor_object().to_string(),
    )
    .unwrap();
    assert!(matches!(
        spec_from_model_dir(&dir),
        Ok(ProcessorSpec::Glm5Next(_))
    ));
    std::fs::remove_dir_all(&dir).unwrap();
}
