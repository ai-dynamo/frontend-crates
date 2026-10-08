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
    struct Case {
        width: u32,
        height: u32,
        spec: DeepseekV41GeometrySpec,
        plan: DeepseekV41ImagePlan,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/deepseek_v41/geometry.json")).unwrap();
    for (index, case) in fixture.cases.iter().enumerate() {
        let actual = plan_image_grid(case.width, case.height, &case.spec)
            .unwrap_or_else(|error| panic!("case {index}: {error}"));
        assert_eq!(actual, case.plan, "case {index}: {:?}", case.spec);
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
    let mut invalid = spec();
    invalid.vision_patch_size = 0;
    assert!(matches!(
        plan_image_grid(1, 1, &invalid),
        Err(MmError::InvalidInput { .. })
    ));
    invalid = spec();
    invalid.vision_downsample_ratio = 0;
    assert!(matches!(
        plan_image_grid(1, 1, &invalid),
        Err(MmError::InvalidInput { .. })
    ));
    for budget in 0..4 {
        invalid = spec();
        invalid.vision_max_n_token = budget;
        assert!(matches!(
            plan_image_grid(1, 1, &invalid),
            Err(MmError::InvalidInput { .. })
        ));
    }
    for ratio in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        invalid = spec();
        invalid.vision_max_wh_ratio = Some(ratio);
        assert!(matches!(
            plan_image_grid(1, 1, &invalid),
            Err(MmError::InvalidInput { .. })
        ));
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
