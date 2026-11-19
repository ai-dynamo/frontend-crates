// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Numerical kernel checks, deliberately distinct from the bit-exact model
//! parity tables. These do not qualify a family/backend for the registry.
#![cfg(feature = "audio")]

use dynamo_multimodal::audio::features::{FeatureLimits, LogMelConfig, LogMelExtractor};

#[test]
fn generic_log_mel_against_versioned_numpy_reference() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/audio/log_mel.json")).unwrap();
    let cases = fixtures["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let config: LogMelConfig = serde_json::from_value(case["config"].clone()).unwrap();
        let n = case["samples"].as_u64().unwrap() as usize;
        let valid = case["valid_samples"].as_u64().unwrap() as usize;
        let samples = (0..n)
            .map(|i| {
                if i < valid {
                    (((i * 7919 + 17) % 65536) as i32 - 32768) as f32 / 32768.
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>();
        let limits = FeatureLimits::default();
        let validated = config.clone().validate().unwrap();
        let plan = validated.plan(n, valid, &limits).unwrap();
        let extractor = LogMelExtractor::from_validated(validated);
        let actual = extractor
            .extract(&samples, config.sample_rate, valid, &limits)
            .unwrap();
        assert_eq!(actual.plan, plan);
        assert_eq!(
            serde_json::json!([actual.plan.num_mels, actual.plan.num_frames]),
            case["shape"]
        );
        assert_eq!(
            serde_json::json!(actual.plan.frame_anchor_mask()),
            case["frame_anchor_mask"]
        );
        let expected = case["features"].as_array().unwrap();
        assert_eq!(actual.data.len(), expected.len());
        let mut exact = 0;
        let mut max_error = 0.0f32;
        for (&a, e) in actual.data.iter().zip(expected) {
            let e = e.as_f64().unwrap() as f32;
            exact += usize::from(a.to_bits() == e.to_bits());
            max_error = max_error.max((a - e).abs());
            assert_eq!(a.to_bits(), e.to_bits(), "{}: {a} vs {e}", case["name"]);
        }
        eprintln!(
            "{}: {exact}/{} bit-exact, max_abs_error={max_error:e}",
            case["name"],
            expected.len()
        );
    }
}
