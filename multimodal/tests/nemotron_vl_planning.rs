// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nemotron 3.5 Super VL spec resolution, tile planning against both
//! references (`tests/fixtures/nemotron_vl/sizes.json`, which
//! `tests/fixtures/mm_parity/generate.py --family nemotron_vl` writes) and
//! prompt layout. `tests/mm_parity.rs` replays the pixel cases.

mod common;

use std::path::Path;

use dynamo_multimodal::models::nemotron_vl::{
    NemotronVlImagePlan, NemotronVlProcessor, NemotronVlSizing, NemotronVlSpec,
};
use dynamo_multimodal::processor::{
    DecodedMedia, MediaMetadata, MmFamilyProcessor, Modality, ProcessedItem, Tensor, TensorData,
};
use dynamo_multimodal::registry::{
    ProcessorSpec, build_processor, processor_from_spec, spec_from_hf_configs, spec_from_model_dir,
};
use dynamo_multimodal::token_layout::apply_layout;
use dynamo_multimodal::{MmError, Result};
use serde_json::{Value, json};

// Trimmed from nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 config.json.
const CONFIG: &str = r#"{
    "architectures": ["NemotronH_Omni_Reasoning_V3"],
    "model_type": "nemotron_h_omni",
    "downsample_ratio": 0.5, "force_image_size": 512, "image_tag_type": "internvl",
    "img_context_token": "<image>", "img_context_token_id": 18,
    "norm_mean": [0.48145466, 0.4578275, 0.40821073],
    "norm_std": [0.26862954, 0.26130258, 0.27577711],
    "patch_size": 16, "sound_config": null, "use_thumbnail": false,
    "video_context_token": "<video>", "video_context_token_id": null,
    "vision_config": {
        "image_size": 224, "max_num_patches": 13312, "min_num_patches": 1024,
        "model_type": "radio", "patch_size": 16, "video_temporal_patch_size": 2
    }
}"#;
// The checkpoint's preprocessor_config.json.
const PREPROCESSOR: &str = r#"{
    "auto_map": {
        "AutoImageProcessor": "image_processing_nemotron_h_omni.NemotronH_Omni_Reasoning_V3ImageProcessor",
        "AutoProcessor": "processing_nemotron_h_omni.NemotronH_Omni_Reasoning_V3Processor"
    },
    "downsample_ratio": 0.5,
    "image_processor_type": "NemotronH_Omni_Reasoning_V3ImageProcessor",
    "max_model_len": 16384, "max_num_patches": 13312, "min_num_patches": 1024,
    "norm_mean": [0.48145466, 0.4578275, 0.40821073],
    "norm_std": [0.26862954, 0.26130258, 0.27577711],
    "patch_size": 16, "rescale_factor": 0.00392156862745098,
    "video_maintain_aspect_ratio": true, "video_target_num_patches": 1024
}"#;
// Trimmed from tokenizer.json, which lists 1000 added tokens beside the vocabulary.
const TOKENIZER: &str = r#"{
    "version": "1.0",
    "added_tokens": [
        {"id": 0, "content": "<unk>", "special": true},
        {"id": 18, "content": "<image>", "special": true},
        {"id": 19, "content": "<img>", "special": true},
        {"id": 20, "content": "</img>", "special": true}
    ],
    "model": {"type": "BPE"}
}"#;

const IMAGE: i32 = 18;
const TEXT: i32 = 1000;

fn sizes_fixture() -> Value {
    common::read_json(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nemotron_vl/sizes.json"),
    )
}

fn planner(spec: &Value, max_model_len: u64) -> NemotronVlProcessor {
    let mut spec = spec.clone();
    spec["max_model_len"] = max_model_len.into();
    NemotronVlProcessor::new(serde_json::from_value(spec).unwrap()).unwrap()
}

fn resolve(name: &str, config: &Value, pre: &Value, tokenizer: &Value) -> Result<NemotronVlSpec> {
    resolve_files(
        name,
        &[
            ("config.json", config),
            ("preprocessor_config.json", pre),
            ("tokenizer.json", tokenizer),
        ],
    )
}

fn resolve_files(name: &str, files: &[(&str, &Value)]) -> Result<NemotronVlSpec> {
    let dir = std::env::temp_dir().join(format!("mm-nemotron-vl-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (file, json) in files {
        std::fs::write(dir.join(file), json.to_string()).unwrap();
    }
    let spec = spec_from_model_dir(&dir);
    std::fs::remove_dir_all(&dir).unwrap();
    spec.map(|spec| match spec {
        ProcessorSpec::NemotronVl(spec) => spec,
        _ => panic!("expected a Nemotron VL spec"),
    })
}

fn parsed() -> [Value; 3] {
    [CONFIG, PREPROCESSOR, TOKENIZER].map(|json| serde_json::from_str(json).unwrap())
}

#[test]
fn model_dir_resolves_the_hf_spec_and_sglang_differs_in_two_fields() {
    let fixture = sizes_fixture();
    let [config, pre, tokenizer] = parsed();
    let spec = resolve("hf", &config, &pre, &tokenizer).unwrap();
    assert_eq!(
        spec,
        serde_json::from_value(fixture["hf_spec"].clone()).unwrap()
    );

    // What SGLang's processor resolved under `--context-length 16384`.
    let sglang = NemotronVlSpec {
        sizing: NemotronVlSizing::Sglang,
        max_model_len: 16384,
        ..spec.clone()
    };
    assert_eq!(
        sglang,
        serde_json::from_value(fixture["sglang_spec"].clone()).unwrap()
    );
    assert!(build_processor(ProcessorSpec::NemotronVl(sglang)).is_ok());

    // The NVFP4 checkpoint's preprocessor_config.json adds one key.
    let mut nvfp4 = pre.clone();
    nvfp4["_downsample_factor"] = 2.into();
    assert_eq!(resolve("nvfp4", &config, &nvfp4, &tokenizer).unwrap(), spec);

    // The ids are in no config file, so the two-file entry point names the other one.
    let Err(MmError::Unsupported { message, .. }) = spec_from_hf_configs(CONFIG, PREPROCESSOR)
    else {
        panic!("the two-file entry point must refuse");
    };
    assert!(message.contains("spec_from_model_dir"), "{message}");
}

/// transformers 5.19.0 loads the `image_processor` object of processor_config.json
/// when there is one, and `save_pretrained` writes only that file.
#[test]
fn model_dir_reads_the_processor_file_that_transformers_loads() {
    let fixture = sizes_fixture();
    let [config, pre, tokenizer] = parsed();
    let hf_spec: NemotronVlSpec = serde_json::from_value(fixture["hf_spec"].clone()).unwrap();
    let nested = |image_processor: &Value| {
        json!({
            "audio_hop_length": 160, "audio_sampling_rate": 16000, "audio_subsampling_factor": 8,
            "image_processor": image_processor,
            "processor_class": "NemotronH_Omni_Reasoning_V3Processor",
            "video_temporal_patch_dim": 2
        })
    };
    let files = |processor: &Value, flat: bool| {
        let mut files = vec![
            ("config.json", config.clone()),
            ("processor_config.json", processor.clone()),
            ("tokenizer.json", tokenizer.clone()),
        ];
        if flat {
            files.push(("preprocessor_config.json", pre.clone()));
        }
        files
    };
    let resolve = |name: &str, processor: &Value, flat: bool| {
        let files = files(processor, flat);
        let files: Vec<(&str, &Value)> = files.iter().map(|(file, json)| (*file, json)).collect();
        resolve_files(name, &files)
    };

    // The shipped shape, and the shape of a saved processor.
    assert_eq!(resolve("both", &nested(&pre), true).unwrap(), hf_spec);
    assert_eq!(resolve("nested", &nested(&pre), false).unwrap(), hf_spec);

    // HF with this nested value gives 1280x720 a 672x384 tile of 252 tokens.
    let mut short = pre.clone();
    short["max_model_len"] = 260.into();
    let spec = resolve("nested-260", &nested(&short), false).unwrap();
    assert_eq!(spec.max_model_len, 260);
    let plan = NemotronVlProcessor::new(spec).unwrap();
    assert_eq!(
        triple(&plan.plan_images(&[], &[(1280, 720)]).unwrap()[0]),
        json!([672, 384, 252])
    );
    // A flat file that says 16384 beside it: the two HF entry points would disagree.
    assert!(matches!(
        resolve("differ", &nested(&short), true),
        Err(MmError::InvalidInput { .. })
    ));
    // The nested object is the one checked for knobs that cannot be honored.
    let mut rescaled = pre.clone();
    rescaled["rescale_factor"] = 0.5.into();
    assert!(matches!(
        resolve("rescaled", &nested(&rescaled), true),
        Err(MmError::Unsupported { .. })
    ));
}

/// SGLang's `internvl_utils.py`: two values that no fixture holds, and `sglang_requests`.
#[test]
fn sglang_sizes_at_the_edges_of_the_budget() {
    let mut spec = sizes_fixture()["sglang_spec"].clone();
    // 3 text ids would leave the 1024 tokens the image needs; the fourth shrinks it.
    let plans = planner(&spec, 1027).plan_images(&[IMAGE, TEXT, TEXT, TEXT, TEXT], &[(1024, 1024)]);
    assert_eq!(triple(&plans.unwrap()[0]), json!([992, 992, 961]));
    // A thin image over its budget: the loop raises its maximum above `max_num_patches`.
    for request in sizes_fixture()["sglang_requests"].as_array().unwrap() {
        let sizes: Vec<(u32, u32)> = serde_json::from_value(request["sizes"].clone()).unwrap();
        let text = vec![TEXT; request["text_tokens"].as_u64().unwrap() as usize];
        let ids = [vec![IMAGE; sizes.len()], text].concat();
        let plans = planner(&spec, request["max_model_len"].as_u64().unwrap())
            .plan_images(&ids, &sizes)
            .unwrap();
        let plans: Value = plans.iter().map(triple).collect();
        assert_eq!(plans, request["plans"], "{request}");
    }
    // A maximum below the minimum: the minimum sets the first scale, the maximum the last.
    spec["max_num_patches"] = 64.into();
    let plans = planner(&spec, 16384).plan_images(&[], &[(1059, 1684)]);
    assert_eq!(triple(&plans.unwrap()[0]), json!([96, 128, 12]));
}

#[test]
fn refuses_knobs_it_cannot_honor() {
    let [config, pre, tokenizer] = parsed();
    for (knob, value) in [
        ("do_resize", json!(true)),
        ("rescale_factor", json!(0.5)),
        ("downsample_ratio", json!(0.25)),
        ("_downsample_factor", json!(4)),
        (
            "image_processor_type",
            json!("NemotronH_Nano_Omni_Reasoning_V3ImageProcessor"),
        ),
    ] {
        let mut pre = pre.clone();
        pre[knob] = value;
        assert!(
            matches!(
                resolve("knob", &config, &pre, &tokenizer),
                Err(MmError::Unsupported { .. })
            ),
            "{knob}"
        );
    }
    for knob in ["max_model_len", "min_num_patches", "norm_mean"] {
        let mut pre = pre.clone();
        pre.as_object_mut().unwrap().remove(knob);
        assert!(
            matches!(
                resolve("missing", &config, &pre, &tokenizer),
                Err(MmError::InvalidInput { .. })
            ),
            "{knob}"
        );
    }
}

/// SGLang sizes from config.json and HF from preprocessor_config.json, so a
/// checkpoint where they differ has no spec that serves both.
#[test]
fn refuses_checkpoints_whose_files_disagree() {
    let [config, pre, tokenizer] = parsed();
    let invalid = |config: &Value, tokenizer: &Value| {
        assert!(
            matches!(
                resolve("drift", config, &pre, tokenizer),
                Err(MmError::InvalidInput { .. })
            ),
            "{config} {tokenizer}"
        );
    };
    for (field, value) in [
        ("patch_size", json!(14)),
        ("min_num_patches", json!(256)),
        ("max_num_patches", Value::Null),
    ] {
        let mut config = config.clone();
        config["vision_config"][field] = value;
        invalid(&config, &tokenizer);
    }
    for (field, value) in [
        ("downsample_ratio", json!(0.25)),
        ("norm_std", json!([0.5, 0.5, 0.5])),
        ("img_context_token_id", json!(17)),
    ] {
        let mut config = config.clone();
        config[field] = value;
        invalid(&config, &tokenizer);
    }
    let mut no_end_token = tokenizer.clone();
    no_end_token["added_tokens"].as_array_mut().unwrap().pop();
    invalid(&config, &no_end_token);

    let mut renamed = config.clone();
    renamed["img_end_token"] = "<different>".into();
    assert!(matches!(
        resolve("renamed", &renamed, &pre, &tokenizer),
        Err(MmError::Unsupported { .. })
    ));
}

fn triple(plan: &NemotronVlImagePlan) -> Value {
    json!([
        plan.resized_width,
        plan.resized_height,
        plan.num_image_tokens
    ])
}

/// `_preprocess` of the HF remote code on blank images. Its sizing ignores the prompt.
#[test]
fn hf_requests_match_the_remote_code() {
    let fixture = sizes_fixture();
    for request in fixture["hf_requests"].as_array().unwrap() {
        let planner = planner(
            &fixture["hf_spec"],
            request["max_model_len"].as_u64().unwrap(),
        );
        let sizes: Vec<(u32, u32)> = serde_json::from_value(request["sizes"].clone()).unwrap();
        match (
            planner.plan_images(&[TEXT; 9000], &sizes),
            request.get("error"),
        ) {
            (Ok(plans), None) => {
                let triples: Value = plans.iter().map(triple).collect();
                assert_eq!(triples, request["plans"], "{request}");
                // The span of a lone image: its feature tokens, `<img>` and `</img>`.
                if let [(width, height)] = sizes[..] {
                    let image = MediaMetadata::Image { width, height };
                    let span = plans[0].num_image_tokens as usize + 2;
                    assert_eq!(planner.num_media_tokens(&image).unwrap(), span);
                    assert_eq!(planner.num_feature_tokens(&image).unwrap(), span - 2);
                }
            }
            (Err(MmError::InvalidInput { .. }), Some(_)) => {}
            (result, _) => panic!("{request}: {:?}", result.map_err(|e| e.to_string())),
        }
    }
}

/// The LCG `generate.py` draws its sweeps from.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 = (self.0.wrapping_mul(6364136223846793005)).wrapping_add(1442695040888963407);
        (self.0 >> 33) % bound
    }

    fn side(&mut self, max: &Value) -> u32 {
        1 + self.below(max.as_u64().unwrap()) as u32
    }
}

fn line(size: (u32, u32), plan: Option<&NemotronVlImagePlan>) -> String {
    let (width, height) = size;
    match plan {
        Some(plan) => format!(
            "{width} {height} {} {} {}",
            plan.resized_width, plan.resized_height, plan.num_image_tokens
        ),
        None => format!("{width} {height} error"),
    }
}

fn digest(lines: &[String]) -> Value {
    common::sha256(lines.join("\n").as_bytes()).into()
}

/// Random sizes and requests, digested: SGLang's `compute_budgeted_image_sizes`,
/// and HF's `_compute_target_patches` under the budget `_preprocess` gives it.
#[test]
fn sweeps_match_both_references() {
    let fixture = sizes_fixture();
    let at = |sizing: &str, max_model_len| planner(&fixture[sizing], max_model_len);
    let (sglang, hf) = (at("sglang_spec", 16384), at("hf_spec", 16384));
    let lone = |planner: &NemotronVlProcessor, size| {
        let plan = planner.plan_images(&[], &[size]).ok();
        line(size, plan.as_ref().map(|plans| &plans[0]))
    };

    // Lone images: all sizes, then thin ones, where SGLang clamps a side and HF raises.
    for name in ["single_image_sweep", "thin_image_sweep"] {
        let sweep = &fixture[name];
        let mut rng = Lcg(sweep["seed"].as_u64().unwrap());
        let (mut sglang_lines, mut hf_lines) = (Vec::new(), Vec::new());
        for _ in 0..sweep["count"].as_u64().unwrap() {
            let (long, short) = (
                rng.side(&sweep["max_long_side"]),
                rng.side(&sweep["max_short_side"]),
            );
            let size = if rng.below(2) == 1 {
                (short, long)
            } else {
                (long, short)
            };
            sglang_lines.push(lone(&sglang, size));
            hf_lines.push(lone(&hf, size));
        }
        assert_eq!(digest(&sglang_lines), sweep["sglang_sha256"], "{name}");
        assert_eq!(digest(&hf_lines), sweep["hf_sha256"], "{name}");
    }

    let sweep = &fixture["request_sweep"];
    let contexts: Vec<u64> = serde_json::from_value(sweep["contexts"].clone()).unwrap();
    let text = vec![TEXT; 16384 + 2000];
    let mut rng = Lcg(sweep["seed"].as_u64().unwrap());
    let (mut sglang_lines, mut hf_lines) = (Vec::new(), Vec::new());
    for _ in 0..sweep["count"].as_u64().unwrap() {
        let count = 1 + rng.below(sweep["max_images"].as_u64().unwrap());
        let sizes: Vec<(u32, u32)> = (0..count)
            .map(|_| (rng.side(&sweep["max_side"]), rng.side(&sweep["max_side"])))
            .collect();
        let max_model_len = contexts[rng.below(contexts.len() as u64) as usize];
        let text_tokens = rng.below(max_model_len + 2000) as usize;
        for (lines, planner, input_ids) in [
            (
                &mut sglang_lines,
                at("sglang_spec", max_model_len),
                &text[..text_tokens],
            ),
            (&mut hf_lines, at("hf_spec", max_model_len), &[][..]),
        ] {
            let plans = planner.plan_images(input_ids, &sizes).unwrap();
            let images: Vec<String> = (sizes.iter().zip(&plans))
                .map(|(&size, plan)| line(size, Some(plan)))
                .collect();
            lines.push(format!(
                "{max_model_len} {text_tokens} | {}",
                images.join(" | ")
            ));
        }
    }
    assert_eq!(digest(&sglang_lines), sweep["sglang_sha256"]);
    assert_eq!(digest(&hf_lines), sweep["hf_sha256"]);
}

/// SGLang's processor on blank images (`sglang_cases`): what a router computes
/// from the sizes alone, with `layout_from_metadata` and `apply_layout`.
#[test]
fn sglang_cases_match_from_sizes() {
    let mut failures = Vec::new();
    for case in sizes_fixture()["sglang_cases"].as_array().unwrap() {
        let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
            .flat_map(|run| {
                let id = run[0].as_i64().unwrap() as i32;
                std::iter::repeat_n(id, run[1].as_u64().unwrap() as usize)
            })
            .collect();
        let sizes: Vec<(u32, u32)> = serde_json::from_value(case["images"].clone()).unwrap();
        match (
            from_sizes(&case["spec"], &input_ids, &sizes),
            case.get("error"),
        ) {
            (Ok(actual), None) if actual == case["expected"] => {}
            // Every refusal of these cases is a placeholder count that differs from the images.
            (Err(MmError::InvalidInput { .. }), Some(_)) => {}
            (outcome, _) => failures.push(format!(
                "{}: {:?}",
                case["name"],
                outcome.map_err(|error| error.to_string())
            )),
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn from_sizes(spec: &Value, input_ids: &[i32], sizes: &[(u32, u32)]) -> Result<Value> {
    let family = processor_from_spec(&spec.to_string())?;
    let metadata: Vec<MediaMetadata> = (sizes.iter())
        .map(|&(width, height)| MediaMetadata::Image { width, height })
        .collect();
    let layout = family.layout_from_metadata(input_ids, &metadata)?;
    let counts: Vec<usize> = (common::layout_counts(&layout).iter())
        .map(|&(features, _)| features)
        .collect();
    // With no text, the request is the one `num_media_tokens` sizes: the image and its markers.
    if let ([image], 1) = (&metadata[..], input_ids.len()) {
        assert_eq!(family.num_media_tokens(image)?, counts[0] + 2);
        assert_eq!(family.num_feature_tokens(image)?, counts[0]);
    }
    let plans =
        planner(spec, spec["max_model_len"].as_u64().unwrap()).plan_images(input_ids, sizes)?;
    let planned: Vec<usize> = (plans.iter())
        .map(|plan| plan.num_image_tokens as usize)
        .collect();
    assert_eq!(planned, counts);
    let expanded = apply_layout(input_ids, &layout, &counts, usize::MAX)?;
    // SGLang's offsets are the `<image>` run; the crate's also span `<img>` and `</img>`.
    let offsets: Vec<[u32; 2]> = (expanded.feature_ranges.iter())
        .zip(&expanded.offsets)
        .map(|(ranges, &(start, end))| {
            assert_eq!((ranges.len(), &ranges[0]), (1, &(start + 1..end)));
            [start + 1, end - 1]
        })
        .collect();
    Ok(json!({
        "input_ids_sha256": common::sha256_le(expanded.input_ids.iter().map(|id| id.to_le_bytes())),
        "num_input_ids": expanded.input_ids.len(),
        "offsets": offsets,
        "num_tokens": counts,
        "feature_shapes": (plans.iter())
            .map(|plan| json!([1, 3, plan.resized_height, plan.resized_width]))
            .collect::<Vec<_>>(),
    }))
}

/// 1280x720 alone, with no text: 880 feature tokens in SGLang's tile and 920
/// in HF's. `<img>` and `</img>` are the literal ids of the span.
#[test]
fn num_feature_tokens_is_the_span_without_the_markers() {
    let fixture = sizes_fixture();
    let image = MediaMetadata::Image {
        width: 1280,
        height: 720,
    };
    for (spec, features) in [("sglang_spec", 880), ("hf_spec", 920)] {
        let family = processor_from_spec(&fixture[spec].to_string()).unwrap();
        assert_eq!(family.num_feature_tokens(&image).unwrap(), features);
        assert_eq!(family.num_media_tokens(&image).unwrap(), features + 2);
        let layout = family.layout_from_metadata(&[IMAGE], &[image]).unwrap();
        assert_eq!(common::layout_counts(&layout), [(features, 2)], "{spec}");
    }
}

fn item(feature_token_count: usize) -> ProcessedItem {
    ProcessedItem {
        modality: Modality::Image,
        feature_token_count,
        feature: Tensor {
            shape: vec![],
            data: TensorData::Bf16(vec![]),
        },
        aux: vec![],
        geometry: None,
    }
}

/// SGLang's own vector, `test_nano_nemotron_vl_processor.py`: two images of
/// two tokens each, with ids before, between and after them.
#[test]
fn layout_wraps_each_image_in_its_markers() {
    let family = processor_from_spec(&sizes_fixture()["sglang_spec"].to_string()).unwrap();
    let ids = [0, 101, IMAGE, 7, IMAGE, 102, 103];
    let layout = family.layout(&ids, &[item(2), item(2)]).unwrap();
    let expanded = apply_layout(&ids, &layout, &[2, 2], usize::MAX).unwrap();
    assert_eq!(
        expanded.input_ids,
        [
            0, 101, 19, IMAGE, IMAGE, 20, 7, 19, IMAGE, IMAGE, 20, 102, 103
        ]
    );
    // SGLang reports the feature runs, (3, 4) and (8, 9).
    assert_eq!(expanded.feature_ranges.concat(), [3..5, 8..10]);
    assert_eq!(expanded.offsets, [(2, 5), (7, 10)]);

    for (ids, items) in [
        (&ids[..3], vec![item(2), item(2)]),
        (&ids[..], vec![item(2)]),
        (&ids[..2], vec![item(2)]),
    ] {
        assert!(matches!(
            family.layout(ids, &items),
            Err(MmError::InvalidInput { .. })
        ));
    }
}

#[test]
fn hf_sizing_is_metadata_only_and_other_modalities_are_unsupported() {
    let fixture = sizes_fixture();
    let hf = processor_from_spec(&fixture["hf_spec"].to_string()).unwrap();
    let image = DecodedMedia::Image {
        rgb: vec![0; 64 * 32 * 3],
        height: 32,
        width: 64,
    };
    assert!(matches!(
        hf.process_items(&[IMAGE], std::slice::from_ref(&image)),
        Err(MmError::Unsupported { .. })
    ));
    let metadata = MediaMetadata::Image {
        width: 64,
        height: 32,
    };
    assert!(hf.layout_from_metadata(&[IMAGE], &[metadata]).is_ok());

    let sglang = processor_from_spec(&fixture["sglang_spec"].to_string()).unwrap();
    // One image alone has no request to size it by, so both sizings refuse it.
    for family in [&hf, &sglang] {
        let Err(MmError::Unsupported { message, .. }) = family.process_item(&image) else {
            panic!("process_item must be unsupported");
        };
        assert!(message.contains("process_items"), "{message}");
    }
    assert_eq!(sglang.process_items(&[IMAGE], &[image]).unwrap().len(), 1);
    for media in [
        MediaMetadata::Video {
            width: 640,
            height: 360,
            num_frames: 8,
        },
        MediaMetadata::Audio {
            num_samples: 16000,
            sample_rate: 16000,
        },
    ] {
        assert!(matches!(
            sglang.num_media_tokens(&media),
            Err(MmError::Unsupported { .. })
        ));
        assert!(matches!(
            sglang.layout_from_metadata(&[IMAGE], &[media]),
            Err(MmError::Unsupported { .. })
        ));
    }
}

/// A driver that calls `process_items` for every family gets what
/// `process_item` gave it, unless the family sizes by request.
#[test]
fn process_items_maps_process_item_by_default() {
    let qwen = processor_from_spec(
        r#"{"family": "qwen_vl", "image_token_id": 151655, "patch_size": 14, "merge_size": 2,
            "temporal_patch_size": 2, "min_pixels": 3136, "max_pixels": 12845056,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}"#,
    )
    .unwrap();
    let media = [(40, 30), (64, 96)].map(|(width, height)| DecodedMedia::Image {
        rgb: (0..width * height * 3).map(|i| i as u8).collect(),
        height,
        width,
    });
    let pixels = |item: &ProcessedItem| match &item.feature.data {
        TensorData::F32(values) => (item.feature.shape.clone(), values.clone()),
        _ => panic!("Qwen features are f32"),
    };
    let together = qwen.process_items(&[151655, 7, 151655], &media).unwrap();
    assert_eq!(together.len(), 2);
    for (item, alone) in together.iter().zip(&media) {
        assert_eq!(pixels(item), pixels(&qwen.process_item(alone).unwrap()));
    }
}

#[test]
fn rejects_invalid_specs_sizes_and_buffers() {
    let spec = sizes_fixture()["sglang_spec"].clone();
    let with = |field: &str, value: Value| {
        let mut spec = spec.clone();
        spec[field] = value;
        processor_from_spec(&spec.to_string()).map(|_| ())
    };
    for (field, value) in [
        ("downsample_factor", json!(4)),
        // SGLang's InternVL tile path.
        ("min_num_patches", json!(0)),
    ] {
        assert!(
            matches!(with(field, value), Err(MmError::Unsupported { .. })),
            "{field}"
        );
    }
    for (field, value) in [
        ("patch_size", json!(0)),
        // SGLang itself uses 8192 when `--context-length` is not set.
        ("max_model_len", json!(0)),
        ("image_end_token_id", json!(-1)),
        ("image_std", json!([0.0, 1.0, 1.0])),
        ("sizing", json!("vllm")),
    ] {
        assert!(
            matches!(with(field, value), Err(MmError::InvalidInput { .. })),
            "{field}"
        );
    }

    let family = processor_from_spec(&spec.to_string()).unwrap();
    // SGLang checks the placeholder count before it sizes anything.
    let image = DecodedMedia::Image {
        rgb: vec![0; 64 * 64 * 3],
        height: 64,
        width: 64,
    };
    for input_ids in [&[IMAGE, IMAGE][..], &[TEXT]] {
        assert!(matches!(
            family.process_items(input_ids, std::slice::from_ref(&image)),
            Err(MmError::InvalidInput { .. })
        ));
    }
    for (height, width, rgb) in [(0, 1, vec![]), (1, 0, vec![]), (2, 2, vec![0; 11])] {
        assert!(matches!(
            family.process_items(&[IMAGE], &[DecodedMedia::Image { rgb, height, width }]),
            Err(MmError::InvalidInput { .. })
        ));
    }
    let empty = MediaMetadata::Image {
        width: 0,
        height: 7,
    };
    assert!(matches!(
        family.num_media_tokens(&empty),
        Err(MmError::InvalidInput { .. })
    ));
    // Above 2^53 patches Python's int division is not an f64 division.
    let mut spec = spec;
    spec["patch_size"] = 1.into();
    let family = processor_from_spec(&spec.to_string()).unwrap();
    let huge = MediaMetadata::Image {
        width: u32::MAX,
        height: u32::MAX,
    };
    assert!(matches!(
        family.num_media_tokens(&huge),
        Err(MmError::LimitExceeded { .. })
    ));
}
