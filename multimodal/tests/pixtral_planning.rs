// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Spec resolution from the Mistral Large 4 checkpoint files, planning
//! against HF's sizing rule, and the refusals that the parity table in
//! `mm_parity.rs` does not hold. `tests/fixtures/mm_parity/generate.py
//! --family pixtral` copies the config files and runs the sizing sweep.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::pixtral::{PixtralProcessor, PixtralSpec};
use dynamo_multimodal::models::qwen_vl::Resampler;
use dynamo_multimodal::processor::{DecodedMedia, MediaMetadata, MmFamilyProcessor};
use dynamo_multimodal::registry::{
    ProcessorSpec, build_processor, spec_from_hf_configs, spec_from_model_dir,
};
use dynamo_multimodal::token_layout::apply_layout;
use serde_json::{Value, json};

const IMG: i32 = 10;
const BREAK: i32 = 12;
const END: i32 = 13;

fn geometry() -> Value {
    serde_json::from_str(include_str!("fixtures/pixtral/geometry.json")).unwrap()
}

/// The spec that the parity table replays on the default (torchvision) backend.
fn table_spec() -> PixtralSpec {
    let spec = &common::table("mistral-large-4.json")["cases"][0]["spec"];
    serde_json::from_value(spec.clone()).expect("the table spec deserializes")
}

/// The three config files of `mistralai/Mistral-Large-4-1T-A52B`, as
/// `[params.json, processor_config.json, config.json]`.
fn checkpoint() -> [Value; 3] {
    let files = &geometry()["checkpoint"];
    ["params.json", "processor_config.json", "config.json"].map(|name| files[name].clone())
}

fn resolve(files: &[(&str, &Value)]) -> dynamo_multimodal::Result<PixtralSpec> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "mm-pixtral-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, value) in files {
        std::fs::write(dir.join(name), value.to_string()).unwrap();
    }
    let spec = spec_from_model_dir(&dir);
    std::fs::remove_dir_all(&dir).unwrap();
    spec.map(|spec| match spec {
        ProcessorSpec::Pixtral(spec) => spec,
        _ => panic!("expected a pixtral spec"),
    })
}

fn resolve_checkpoint(
    params: &Value,
    processor_config: &Value,
) -> dynamo_multimodal::Result<PixtralSpec> {
    resolve(&[
        ("params.json", params),
        ("processor_config.json", processor_config),
    ])
}

fn family() -> Box<dyn MmFamilyProcessor> {
    let [params, processor_config, _] = checkpoint();
    let spec = resolve_checkpoint(&params, &processor_config).unwrap();
    build_processor(ProcessorSpec::Pixtral(spec)).unwrap()
}

/// The FP8 repo ships `config.json`; the NVFP4 repo ships the same files
/// without it. Both resolve to what the loaded HF processor reports.
#[test]
fn resolves_both_checkpoint_layouts() {
    let [params, processor_config, config] = checkpoint();
    let expected = table_spec();
    let nvfp4 = resolve_checkpoint(&params, &processor_config).unwrap();
    let fp8 = resolve(&[
        ("params.json", &params),
        ("processor_config.json", &processor_config),
        ("config.json", &config),
    ])
    .unwrap();
    assert_eq!(nvfp4, expected);
    assert_eq!(fp8, expected);
    assert_eq!(
        (
            expected.image_token_id,
            expected.image_break_token_id,
            expected.image_end_token_id
        ),
        (IMG, BREAK, END)
    );
    assert_eq!(expected.resample, Resampler::AtenU8);
    assert!(build_processor(ProcessorSpec::Pixtral(fp8)).is_ok());
}

/// `config.json` and `processor_config.json` alone lack the `[IMG_BREAK]` and
/// `[IMG_END]` ids.
#[test]
fn the_two_file_api_refuses_the_checkpoint() {
    let [params, processor_config, config] = checkpoint();
    assert!(matches!(
        spec_from_hf_configs(&config.to_string(), &processor_config.to_string()),
        Err(MmError::Unsupported { .. })
    ));
    // A text-only params.json leaves the directory to the config.json path.
    assert!(matches!(
        resolve(&[
            ("params.json", &json!({"dim": 6144})),
            ("processor_config.json", &processor_config),
            ("config.json", &config),
        ]),
        Err(MmError::Unsupported { .. })
    ));
    assert!(matches!(
        resolve(&[("params.json", &params)]),
        Err(MmError::InvalidInput { .. })
    ));
}

#[test]
fn refuses_knobs_the_pipeline_cannot_honor() {
    let [params, processor_config, _] = checkpoint();
    let with = |edit: &dyn Fn(&mut Value)| {
        let mut processor_config = processor_config.clone();
        edit(&mut processor_config);
        resolve_checkpoint(&params, &processor_config)
    };
    let image_knob = |knob: &'static str, value: Value| {
        with(&move |config| config["image_processor"][knob] = value.clone())
    };
    for (knob, value) in [
        ("do_resize", json!(false)),
        ("do_resize", Value::Null),
        ("do_rescale", json!(false)),
        ("do_normalize", json!(false)),
        ("do_convert_rgb", json!(false)),
        // HF cannot batch two images of different sizes without padding.
        ("do_pad", json!(false)),
        ("do_pad", Value::Null),
        ("do_center_crop", json!(true)),
        ("resample", json!(2)),
        ("rescale_factor", json!(0.5)),
        ("image_processor_type", json!("Qwen2VLImageProcessor")),
        ("image_processor_type", Value::Null),
        ("size", json!({"height": 1540, "width": 1540})),
        ("size", json!({"longest_edge": 1024})),
        ("pad_size", json!({"height": 1540})),
    ] {
        assert!(
            matches!(
                image_knob(knob, value.clone()),
                Err(MmError::Unsupported { .. })
            ),
            "image_processor {knob} = {value} must refuse resolution"
        );
    }
    for (key, value) in [
        ("processor_class", json!("LlavaProcessor")),
        ("image_token", json!("<image>")),
        ("image_break_token", json!("[IMG]")),
        ("image_end_token", Value::Null),
        ("video_processor", json!({})),
    ] {
        assert!(
            matches!(
                with(&|config| config[key] = value.clone()),
                Err(MmError::Unsupported { .. })
            ),
            "{key} = {value} must refuse resolution"
        );
    }

    // HF raises "Unrecognized image processor" without the key.
    assert!(matches!(
        with(&|config| {
            let knobs = config["image_processor"].as_object_mut().unwrap();
            knobs.remove("image_processor_type");
        }),
        Err(MmError::Unsupported { .. })
    ));

    // The processor call overrides the image processor's own patch_size, and
    // every Pixtral processor name loads the torchvision class by default.
    let expected = resolve_checkpoint(&params, &processor_config).unwrap();
    for (knob, value) in [
        ("patch_size", json!(16)),
        ("image_processor_type", json!("PixtralImageProcessorPil")),
        ("image_processor_type", json!("PixtralImageProcessorFast")),
        ("do_center_crop", Value::Null),
        ("crop_size", Value::Null),
    ] {
        assert_eq!(image_knob(knob, value).unwrap(), expected, "{knob}");
    }
}

#[test]
fn refuses_config_files_that_disagree() {
    let [params, processor_config, config] = checkpoint();
    let invalid = |params: &Value, processor_config: &Value| {
        assert!(
            matches!(
                resolve_checkpoint(params, processor_config),
                Err(MmError::InvalidInput { .. })
            ),
            "{params} {processor_config}"
        );
    };
    for (key, value) in [("patch_size", json!(16)), ("spatial_merge_size", json!(1))] {
        let mut drifted = processor_config.clone();
        drifted[key] = value;
        invalid(&params, &drifted);
        drifted.as_object_mut().unwrap().remove(key);
        invalid(&params, &drifted);
    }
    // The error source names the spec field that a file lacks.
    for (field, spec_field) in [
        ("image_token_id", "image_token_id"),
        ("image_break_token_id", "image_break_token_id"),
        ("image_end_token_id", "image_end_token_id"),
        ("patch_size", "patch_size"),
        ("spatial_merge_size", "spatial_merge_size"),
        ("image_size", "longest_edge"),
    ] {
        let mut missing = params.clone();
        missing["vision_encoder"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        let error = resolve_checkpoint(&missing, &processor_config).unwrap_err();
        assert!(matches!(error, MmError::InvalidInput { .. }), "{field}");
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            format!("missing field `{spec_field}`")
        );
    }
    let mut no_image_processor = processor_config.clone();
    no_image_processor
        .as_object_mut()
        .unwrap()
        .remove("image_processor");
    invalid(&params, &no_image_processor);
    let mut two_means = processor_config.clone();
    two_means["image_processor"]["image_mean"] = json!([0.5, 0.5]);
    invalid(&params, &two_means);

    let with_config = |config: &Value| {
        resolve(&[
            ("params.json", &params),
            ("processor_config.json", &processor_config),
            ("config.json", config),
        ])
    };
    let mut other_id = config.clone();
    other_id["image_token_index"] = json!(11);
    assert!(matches!(
        with_config(&other_id),
        Err(MmError::InvalidInput { .. })
    ));
    let mut other_model = config.clone();
    other_model["model_type"] = json!("mistral3");
    assert!(matches!(
        with_config(&other_model),
        Err(MmError::Unsupported { .. })
    ));
}

/// `[width, height, rows, cols]` from HF's `get_resize_output_image_size`. A
/// zero row or column count is a size whose short side floors to 0 pixels:
/// the HF processor call raises there.
#[test]
fn grid_matches_the_hf_sizing_rule() {
    let family = PixtralProcessor::new(table_spec()).unwrap();
    let cases: Vec<[u32; 4]> = serde_json::from_value(geometry()["cases"].clone()).unwrap();
    let mut rejected = 0;
    for [width, height, rows, cols] in &cases {
        let image = MediaMetadata::Image {
            width: *width,
            height: *height,
        };
        if *rows == 0 || *cols == 0 {
            rejected += 1;
            for error in [
                family.grid(*width, *height).err(),
                family.num_media_tokens(&image).err(),
                family.num_feature_tokens(&image).err(),
                family.layout_from_metadata(&[IMG], &[image]).err(),
            ] {
                assert!(
                    matches!(error, Some(MmError::InvalidInput { .. })),
                    "{width}x{height}"
                );
            }
        } else {
            assert_eq!(
                family.grid(*width, *height).unwrap(),
                (*rows, *cols),
                "{width}x{height}"
            );
            assert_eq!(
                family.num_media_tokens(&image).unwrap(),
                (*rows * (*cols + 1)) as usize,
                "{width}x{height}"
            );
            let features = family.num_feature_tokens(&image).unwrap();
            assert_eq!(features, (*rows * *cols) as usize, "{width}x{height}");
            // The literal ids are the one that closes each row.
            let layout = family.layout_from_metadata(&[IMG], &[image]).unwrap();
            assert_eq!(
                common::layout_counts(&layout),
                [(features, *rows as usize)],
                "{width}x{height}"
            );
        }
    }
    assert_eq!(
        (cases.len(), rejected),
        (140, 18),
        "the fixture changed size"
    );
}

/// Adjacent placeholders keep one item each: `[IMG_END]` separates them, and
/// each item reports one feature run per row.
#[test]
fn metadata_layout_expands_each_placeholder_into_its_own_grid() {
    let family = family();
    let media = [(100, 60), (28, 28), (29, 29)]
        .map(|(width, height)| MediaMetadata::Image { width, height });
    let spans = media
        .each_ref()
        .map(|item| family.num_media_tokens(item).unwrap());
    assert_eq!(spans, [15, 2, 6]);
    let features = media
        .each_ref()
        .map(|item| family.num_feature_tokens(item).unwrap());
    assert_eq!(features, [12, 1, 4]);
    let ids = [1, IMG, 7, IMG, IMG, 4];
    let layout = family.layout_from_metadata(&ids, &media).unwrap();
    let expanded = apply_layout(&ids, &layout, &features, 64).unwrap();
    let row = |cols: usize, close: i32| [vec![IMG; cols], vec![close]].concat();
    let expected = [
        vec![1],
        row(4, BREAK),
        row(4, BREAK),
        row(4, END),
        vec![7],
        row(1, END),
        row(2, BREAK),
        row(2, END),
        vec![4],
    ]
    .concat();
    assert_eq!(expanded.input_ids, expected);
    assert_eq!(expanded.offsets, [(1, 15), (17, 18), (19, 24)]);
    let runs: Vec<Vec<(u32, u32)>> = (expanded.feature_ranges.iter())
        .map(|ranges| ranges.iter().map(|run| (run.start, run.end)).collect())
        .collect();
    assert_eq!(
        runs,
        [
            vec![(1, 5), (6, 10), (11, 15)],
            vec![(17, 18)],
            vec![(19, 21), (22, 24)]
        ]
    );
    // The embeddings are the [IMG] positions only, not the whole span.
    assert!(apply_layout(&ids, &layout, &spans, 64).is_err());
    assert!(matches!(
        apply_layout(&ids, &layout, &[12, 1, 4], 23),
        Err(MmError::LimitExceeded { .. })
    ));
}

fn image(width: usize, height: usize) -> DecodedMedia {
    DecodedMedia::Image {
        rgb: vec![0; width * height * 3],
        height,
        width,
    }
}

/// HF raises `StopIteration` for more placeholders than images. For fewer,
/// SGLang drops the extra images, which is a known gap: the crate refuses
/// every count mismatch.
#[test]
fn placeholder_and_image_counts_must_match() {
    let family = family();
    let items = [(100, 60), (60, 200)]
        .map(|(width, height)| family.process_item(&image(width, height)).unwrap());
    for item in &items {
        let [(name, tensor)] = &item.aux[..] else {
            panic!("one model kwarg per item");
        };
        assert_eq!(
            (name.as_str(), &tensor.shape[..]),
            ("image_sizes", &[1, 2][..])
        );
    }
    for ids in [&[1, 3, IMG, 4][..], &[1, 3, 4], &[1, 3, IMG, IMG, IMG, 4]] {
        assert!(matches!(
            family.layout(ids, &items),
            Err(MmError::InvalidInput { .. })
        ));
    }
    let media = [MediaMetadata::Image {
        width: 100,
        height: 60,
    }];
    assert!(matches!(
        family.layout_from_metadata(&[1, 3, 4], &media),
        Err(MmError::InvalidInput { .. })
    ));
}

/// The HF call raises when the short side fits to 0 pixels. The engine entry
/// point refuses the size; `grid_matches_the_hf_sizing_rule` checks the router entry points.
#[test]
fn process_item_refuses_zero_side_sizes() {
    let family = family();
    for (width, height) in [(3080, 1), (1541, 1), (2, 5000)] {
        let error = family.process_item(&image(width, height)).err();
        assert!(
            matches!(error, Some(MmError::InvalidInput { .. })),
            "{width}x{height}: {error:?}"
        );
    }
}

#[test]
fn rejects_bad_buffers_and_other_modalities() {
    let family = family();
    for (height, width, rgb) in [(0, 1, vec![]), (1, 0, vec![]), (2, 2, vec![0; 11])] {
        assert!(matches!(
            family.process_item(&DecodedMedia::Image { rgb, height, width }),
            Err(MmError::InvalidInput { .. })
        ));
    }
    let video = MediaMetadata::Video {
        width: 448,
        height: 448,
        num_frames: 8,
    };
    let audio = MediaMetadata::Audio {
        num_samples: 16000,
        sample_rate: 16000,
    };
    for media in [video, audio] {
        assert!(matches!(
            family.num_media_tokens(&media),
            Err(MmError::Unsupported { .. })
        ));
        assert!(matches!(
            family.num_feature_tokens(&media),
            Err(MmError::Unsupported { .. })
        ));
        assert!(matches!(
            family.layout_from_metadata(&[IMG], &[media]),
            Err(MmError::Unsupported { .. })
        ));
    }
}
