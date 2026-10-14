// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::deepseek_v41::{
    DeepseekV41GeometrySpec, DeepseekV41ImagePlan, plan_image_grid,
};

fn spec() -> DeepseekV41GeometrySpec {
    DeepseekV41GeometrySpec {
        vision_patch_size: 14,
        vision_downsample_ratio: 3,
        vision_min_pixels: 0,
        vision_max_n_token: 1024,
        vision_max_wh_ratio: None,
    }
}

#[test]
fn matches_pinned_sglang_geometry() {
    #[derive(serde::Deserialize)]
    struct Group {
        spec: DeepseekV41GeometrySpec,
        // Original width/height and expected plan.
        cases: Vec<(u32, u32, DeepseekV41ImagePlan)>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        groups: Vec<Group>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/deepseek_v41/geometry.json")).unwrap();
    for group in fixture.groups {
        for (width, height, expected) in group.cases {
            let actual = plan_image_grid(width, height, &group.spec).unwrap();
            assert_eq!(actual, expected, "{width}x{height}, {:?}", group.spec);
        }
    }
}

#[test]
fn accounts_for_boundaries_and_row_separators() {
    let plan = plan_image_grid(84, 126, &spec()).unwrap();
    assert_eq!((plan.llm_height, plan.llm_width), (3, 2));
    assert_eq!(plan.num_image_tokens, 11); // six image cells + three rows + two boundaries
}

#[test]
fn rejects_invalid_inputs_and_configuration() {
    for (w, h) in [(0, 1), (1, 0), (0, 0)] {
        assert!(matches!(
            plan_image_grid(w, h, &spec()),
            Err(MmError::InvalidInput { .. })
        ));
    }
    let reject = |config: DeepseekV41GeometrySpec| {
        assert!(matches!(
            plan_image_grid(1, 1, &config),
            Err(MmError::InvalidInput { .. })
        ));
    };
    reject(DeepseekV41GeometrySpec {
        vision_patch_size: 0,
        ..spec()
    });
    reject(DeepseekV41GeometrySpec {
        vision_downsample_ratio: 0,
        ..spec()
    });
    for budget in 0..4 {
        reject(DeepseekV41GeometrySpec {
            vision_max_n_token: budget,
            ..spec()
        });
    }
    for ratio in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        reject(DeepseekV41GeometrySpec {
            vision_max_wh_ratio: Some(ratio),
            ..spec()
        });
    }
}

#[test]
fn rejects_planned_dimension_overflow() {
    let mut config = spec();
    config.vision_patch_size = u32::MAX - 1;
    config.vision_downsample_ratio = 1;
    assert!(matches!(
        plan_image_grid(u32::MAX, 1, &config),
        Err(MmError::LimitExceeded { .. })
    ));
}
