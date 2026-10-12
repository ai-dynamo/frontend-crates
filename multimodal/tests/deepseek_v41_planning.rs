// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::deepseek_v41::{DeepseekV41Processor, DeepseekV41Spec};
use dynamo_multimodal::processor::{DecodedMedia, MediaMetadata};
use dynamo_multimodal::registry::{
    ProcessorSpec, build_processor, processor_from_spec, spec_from_hf_configs,
};
use dynamo_multimodal::token_layout::apply_layout;

// Mathematical settings, not serving defaults; geometry is covered by the
// existing pinned SGLang fixtures in deepseek_v41_geometry.rs.
fn config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "deepseek_v41", "image_token_id": 100,
        "vision_patch_size": 14, "vision_downsample_ratio": 3,
        "vision_min_pixels": 0, "vision_max_n_token": 1024
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
fn resolves_flat_and_nested_configs_for_both_model_aliases() {
    for model_type in ["deepseek_v4", "deepseek_v41"] {
        let mut flat = config();
        flat["model_type"] = model_type.into();
        let nested = serde_json::json!({
            "model_type": model_type, "image_token_id": 100,
            // Optional root metadata must not suppress nested geometry.
            "vision_max_wh_ratio": null,
            "vision_config": {
                "patch_size": 14, "downsample_ratio": 3,
                "min_pixels": 0, "max_image_tokens": 1024
            }
        });
        for config in [flat, nested] {
            let processor = DeepseekV41Processor::new(spec(&config)).unwrap();
            let plan = processor.plan_image(84, 126).unwrap();
            assert_eq!(
                plan.token_types().collect::<Vec<_>>(),
                [0, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3]
            );
        }
    }
}

#[test]
fn layouts_include_boundaries_and_keep_adjacent_items_distinct() {
    let processor = build_processor(ProcessorSpec::DeepseekV41(spec(&config()))).unwrap();
    let media = [(84, 126), (42, 42)].map(|(width, height)| MediaMetadata::Image { width, height });
    let counts = media
        .iter()
        .map(|item| processor.num_media_tokens(item).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(counts, [11, 4]);
    for (ids, expected, offsets) in [
        (
            vec![7, 100, 8, 100, 9],
            [vec![7], vec![100; 11], vec![8], vec![100; 4], vec![9]].concat(),
            vec![(1, 11), (13, 16)],
        ),
        (vec![100, 100], vec![100; 15], vec![(0, 10), (11, 14)]),
    ] {
        let layout = processor.layout_from_metadata(&ids, &media).unwrap();
        let expanded = apply_layout(&ids, &layout, &counts, expected.len()).unwrap();
        assert_eq!(expanded.input_ids, expected);
        assert_eq!(expanded.offsets, offsets);
        // Learned boundary/separator embeddings occupy the entire image span.
        for (ranges, (start, end)) in expanded.feature_ranges.iter().zip(offsets) {
            assert_eq!(ranges.len(), 1);
            assert_eq!(ranges[0], start..end + 1);
        }
    }
}

#[test]
fn serialized_spec_keeps_pixel_execution_and_other_modalities_unsupported() {
    let mut json = config();
    json["family"] = "deepseek_v41".into();
    let processor = processor_from_spec(&json.to_string()).unwrap();
    let image = DecodedMedia::Image {
        rgb: vec![],
        height: 42,
        width: 42,
    };
    assert!(matches!(
        processor.process_item(&image),
        Err(MmError::Unsupported { .. })
    ));
    assert!(matches!(
        processor.layout(&[100], &[]),
        Err(MmError::Unsupported { .. })
    ));
    let audio = MediaMetadata::Audio {
        num_samples: 16000,
        sample_rate: 16000,
    };
    assert!(matches!(
        processor.layout_from_metadata(&[100], &[audio]),
        Err(MmError::Unsupported { .. })
    ));
}

#[test]
fn refuses_missing_settings_bad_ids_and_ambiguous_config_inheritance() {
    let invalid = |config: &serde_json::Value| {
        assert!(
            matches!(
                spec_from_hf_configs(&config.to_string(), "{}"),
                Err(MmError::InvalidInput { .. })
            ),
            "{config}"
        );
    };
    for value in [
        serde_json::json!(-1),
        serde_json::json!(i64::from(i32::MAX) + 1),
    ] {
        let mut config = config();
        config["image_token_id"] = value;
        invalid(&config);
    }
    let mut missing = config();
    missing.as_object_mut().unwrap().remove("vision_min_pixels");
    invalid(&missing); // no inferred checkpoint defaults
    let mut mixed = config();
    mixed["vision_config"] = mixed.clone();
    mixed
        .as_object_mut()
        .unwrap()
        .remove("vision_downsample_ratio");
    invalid(&mixed); // an explicit flat patch size forbids cross-level inheritance
    assert!(matches!(
        spec_from_hf_configs(&config().to_string(), r#"{"do_resize":false}"#),
        Err(MmError::Unsupported { .. })
    ));
    let mut renamed = config();
    renamed["model_type"] = "deepseek_v3".into();
    assert!(matches!(
        spec_from_hf_configs(&renamed.to_string(), "{}"),
        Err(MmError::Unsupported { .. })
    ));
}
