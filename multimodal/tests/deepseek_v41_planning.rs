// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::deepseek_v41::{DeepseekV41Processor, DeepseekV41Spec};
use dynamo_multimodal::processor::{
    DecodedMedia, MediaMetadata, MmFamilyProcessor, Modality, PositionOutput,
};
use dynamo_multimodal::registry::{
    ProcessorSpec, build_processor, processor_from_spec, spec_from_hf_configs,
};
use dynamo_multimodal::token_layout::apply_layout;

// Mathematical fixture settings, not serving defaults. Expected geometry also
// appears in geometry.json, generated from the pinned SGLang implementation.
fn config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v41", "image_token_id": 100,
        "vision_patch_size": 14, "vision_downsample_ratio": 3,
        "vision_min_pixels": 0, "vision_max_n_token": 1024,
        "vision_max_wh_ratio": null
    })
}

fn spec(config: &serde_json::Value) -> DeepseekV41Spec {
    let ProcessorSpec::DeepseekV41(spec) = spec_from_hf_configs(&config.to_string(), "{}").unwrap()
    else {
        panic!("expected DSV4.1 spec");
    };
    spec
}

#[test]
fn resolves_model_aliases_and_nested_vision_config() {
    for model_type in ["deepseek_v4", "deepseek_v41"] {
        let mut flat = config();
        flat["model_type"] = model_type.into();
        let nested = serde_json::json!({
            "model_type": model_type, "image_token_id": 100,
            // Optional root metadata must not suppress nested geometry.
            "vision_max_wh_ratio": null,
            "vision_config": {
                "patch_size": 14, "downsample_ratio": 3, "min_pixels": 0,
                "max_image_tokens": 1024, "max_wh_ratio": null
            }
        });
        for config in [flat, nested] {
            let processor = DeepseekV41Processor::new(spec(&config)).unwrap();
            let plan = processor.plan_image(84, 126).unwrap();
            assert_eq!(
                (plan.llm_height, plan.llm_width, plan.num_image_tokens),
                (3, 2, 11)
            );
            assert_eq!(
                plan.token_types().collect::<Vec<_>>(),
                [0, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3]
            );
        }
    }
}

#[test]
fn explicit_flat_patch_size_requires_complete_flat_settings() {
    let config = serde_json::json!({
        "model_type": "deepseek_v41", "image_token_id": 100,
        "vision_patch_size": 14,
        "vision_config": {
            "patch_size": 14, "downsample_ratio": 3,
            "min_pixels": 0, "max_image_tokens": 1024
        }
    });
    // Refuse an incomplete selected config; do not guess cross-level inheritance.
    assert!(matches!(
        spec_from_hf_configs(&config.to_string(), "{}"),
        Err(MmError::InvalidInput { .. })
    ));
}

#[test]
fn registry_expands_full_spans_in_prompt_order_before_context_checks() {
    let processor = build_processor(ProcessorSpec::DeepseekV41(spec(&config()))).unwrap();
    let media = [
        MediaMetadata::Image {
            width: 84,
            height: 126,
        },
        MediaMetadata::Image {
            width: 42,
            height: 42,
        },
    ];
    let ids = [7, 100, 8, 100, 9];
    let counts = media
        .iter()
        .map(|item| processor.num_media_tokens(item).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(counts, [11, 4]);
    let layout = processor.layout_from_metadata(&ids, &media).unwrap();
    let expanded = apply_layout(&ids, &layout, &counts, 18).unwrap();
    assert_eq!(
        expanded.input_ids,
        [vec![7], vec![100; 11], vec![8], vec![100; 4], vec![9]].concat()
    );
    assert_eq!(expanded.offsets, [(1, 11), (13, 16)]);
    // DSV includes learned boundary and row-separator embeddings in the feature
    // span; these positions are not inserted as Literal vocabulary ids.
    let ranges = expanded
        .feature_ranges
        .iter()
        .map(|item| {
            item.iter()
                .map(|range| (range.start, range.end))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(ranges, [vec![(1, 12)], vec![(13, 17)]]);
    assert!(matches!(
        processor.positions(18, &expanded.offsets, &[]).unwrap(),
        PositionOutput::Rope1D
    ));
    assert!(matches!(
        apply_layout(&ids, &layout, &counts, 17),
        Err(MmError::LimitExceeded { .. })
    ));
    assert!(matches!(
        processor.layout_from_metadata(&[7, 100], &media),
        Err(MmError::InvalidInput { .. })
    ));
}

#[test]
fn adjacent_images_keep_distinct_item_offsets() {
    let processor = DeepseekV41Processor::new(spec(&config())).unwrap();
    let media = [MediaMetadata::Image {
        width: 42,
        height: 42,
    }; 2];
    let layout = processor.layout_from_metadata(&[100, 100], &media).unwrap();
    let expanded = apply_layout(&[100, 100], &layout, &[4, 4], 8).unwrap();
    assert_eq!(expanded.input_ids, [100; 8]);
    assert_eq!(expanded.offsets, [(0, 3), (4, 7)]);
}

#[test]
fn serialized_spec_activates_only_metadata_planning() {
    let mut json = config();
    json.as_object_mut().unwrap().remove("model_type");
    json["family"] = "deepseek_v41".into();
    let processor = processor_from_spec(&json.to_string()).unwrap();
    assert!(processor.capabilities().supports(Modality::Image));
    assert!(!processor.capabilities().supports(Modality::Video));
    assert!(!processor.capabilities().supports(Modality::Audio));
    assert!(matches!(
        processor.process_item(&DecodedMedia::Image {
            rgb: vec![],
            height: 42,
            width: 42
        }),
        Err(MmError::Unsupported { .. })
    ));
    assert!(matches!(
        processor.layout(&[100], &[]),
        Err(MmError::Unsupported { .. })
    ));
    let video = MediaMetadata::Video {
        width: 42,
        height: 42,
        num_frames: 1,
    };
    let audio = MediaMetadata::Audio {
        num_samples: 16000,
        sample_rate: 16000,
    };
    for item in [video, audio] {
        assert!(matches!(
            processor.num_media_tokens(&item),
            Err(MmError::Unsupported { .. })
        ));
        assert!(matches!(
            processor.layout_from_metadata(&[100], &[item]),
            Err(MmError::Unsupported { .. })
        ));
    }
}

#[test]
fn config_resolution_refuses_missing_invalid_or_unhonorable_settings() {
    for (key, value) in [
        ("vision_patch_size", serde_json::json!(0)),
        ("vision_downsample_ratio", serde_json::json!(0)),
        ("vision_max_n_token", serde_json::json!(3)),
        ("vision_max_wh_ratio", serde_json::json!(0)),
        ("image_token_id", serde_json::json!(-1)),
        ("image_token_id", serde_json::json!(i64::from(i32::MAX) + 1)),
    ] {
        let mut config = config();
        config[key] = value;
        assert!(
            matches!(
                spec_from_hf_configs(&config.to_string(), "{}"),
                Err(MmError::InvalidInput { .. })
            ),
            "{config}"
        );
    }
    for key in [
        "vision_patch_size",
        "vision_downsample_ratio",
        "vision_min_pixels",
        "vision_max_n_token",
        "image_token_id",
    ] {
        let mut config = config();
        config.as_object_mut().unwrap().remove(key);
        assert!(
            matches!(
                spec_from_hf_configs(&config.to_string(), "{}"),
                Err(MmError::InvalidInput { .. })
            ),
            "{key}"
        );
    }
    for pre in [r#"{"do_resize":false}"#, r#"{"max_pixels":3136}"#] {
        assert!(matches!(
            spec_from_hf_configs(&config().to_string(), pre),
            Err(MmError::Unsupported { .. })
        ));
    }
    assert!(matches!(
        spec_from_hf_configs(&config().to_string(), "[]"),
        Err(MmError::InvalidInput { .. })
    ));
    let mut config = config();
    config["model_type"] = "deepseek_v3".into();
    assert!(matches!(
        spec_from_hf_configs(&config.to_string(), "{}"),
        Err(MmError::Unsupported { .. })
    ));
    config["model_type"] = "deepseek_v41".into();
    for key in [
        "vision_patch_size",
        "vision_downsample_ratio",
        "vision_min_pixels",
        "vision_max_n_token",
        "vision_max_wh_ratio",
    ] {
        config.as_object_mut().unwrap().remove(key);
    }
    config["vision_config"] = "invalid".into();
    assert!(matches!(
        spec_from_hf_configs(&config.to_string(), "{}"),
        Err(MmError::InvalidInput { .. })
    ));
}
