// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::glm5_next::{
    Glm5NextGeometrySpec, Glm5NextImagePlan, plan_image_grid, smart_resize,
};

/// zai-org/GLM-5.3-Flash `processor_config.json` image settings.
fn spec() -> Glm5NextGeometrySpec {
    Glm5NextGeometrySpec {
        patch_size: 14,
        merge_size: 2,
        temporal_patch_size: 2,
        patch_expand_factor: 1,
        min_image_tokens: 16,
        max_image_tokens: 8000,
    }
}

#[test]
fn matches_pinned_transformers_geometry() {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Expected {
        Plan(Glm5NextImagePlan),
        Error { error: String },
    }
    #[derive(serde::Deserialize)]
    struct Group {
        spec: Glm5NextGeometrySpec,
        // Original width/height and the expected plan.
        cases: Vec<(u32, u32, Expected)>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        groups: Vec<Group>,
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/glm5_next/geometry.json")).unwrap();
    let mut checked = 0;
    for group in fixture.groups {
        for (width, height, expected) in group.cases {
            let actual = plan_image_grid(width, height, &group.spec);
            match expected {
                Expected::Plan(plan) => {
                    assert_eq!(actual.unwrap(), plan, "{width}x{height}, {:?}", group.spec);
                }
                Expected::Error { error } => assert!(
                    matches!(actual, Err(MmError::InvalidInput { .. })),
                    "{width}x{height}, {:?}: Python raised {error:?}",
                    group.spec
                ),
            }
            checked += 1;
        }
    }
    assert!(checked >= 400, "fixture shrank to {checked} cases");
}

/// The merged token count is the prompt expansion the HF processor emits:
/// `image_grid_thw.prod() // merge_size**2` copies of `<|image|>`.
#[test]
fn token_count_is_the_merged_patch_grid() {
    let plan = plan_image_grid(1024, 768, &spec()).unwrap();
    assert_eq!((plan.grid_h, plan.grid_w), (56, 74));
    assert_eq!(plan.num_image_tokens, 56 * 74 / 4);
    // 448x448 is already aligned: 32x32 patches, 256 tokens.
    assert_eq!(
        plan_image_grid(448, 448, &spec()).unwrap().num_image_tokens,
        256
    );
}

/// Python's `round` is ties-to-even: 3 and 5 frames both align to 4 temporal
/// frames, which tightens the budget identically; 2 frames keep the full one.
#[test]
fn frame_alignment_rounds_ties_to_even() {
    let canvas = |frames| smart_resize(frames, 4000, 4000, 2, 28, 16, 8000).unwrap();
    assert_eq!(canvas(3), canvas(5));
    assert_eq!(canvas(4), canvas(5));
    assert_ne!(canvas(2), canvas(3));
    // Fewer frames than one temporal group still fill the group.
    assert_eq!(canvas(1), canvas(2));
}

#[test]
fn budget_too_small_for_one_patch_is_invalid() {
    // 6 frames need 6 * 28² pixels for one aligned patch; a 1-token budget
    // covers only 2 * 28².
    assert!(matches!(
        smart_resize(6, 448, 448, 2, 28, 16, 1),
        Err(MmError::InvalidInput { .. })
    ));
}

#[test]
fn rejects_invalid_inputs_and_configuration() {
    for (w, h) in [(0, 1), (1, 0), (0, 0)] {
        assert!(matches!(
            plan_image_grid(w, h, &spec()),
            Err(MmError::InvalidInput { .. })
        ));
    }
    let reject = |config: Glm5NextGeometrySpec| {
        assert!(matches!(
            plan_image_grid(28, 28, &config),
            Err(MmError::InvalidInput { .. })
        ));
    };
    reject(Glm5NextGeometrySpec {
        patch_size: 0,
        ..spec()
    });
    reject(Glm5NextGeometrySpec {
        merge_size: 0,
        ..spec()
    });
    reject(Glm5NextGeometrySpec {
        temporal_patch_size: 0,
        ..spec()
    });
    reject(Glm5NextGeometrySpec {
        patch_expand_factor: 0,
        ..spec()
    });
    reject(Glm5NextGeometrySpec {
        max_image_tokens: 0,
        ..spec()
    });
}

#[test]
fn rejects_dimensions_outside_the_exact_arithmetic_domain() {
    assert!(matches!(
        plan_image_grid(u32::MAX, u32::MAX, &spec()),
        Err(MmError::LimitExceeded { .. })
    ));
    assert!(matches!(
        plan_image_grid(1 << 27, 1 << 27, &spec()),
        Err(MmError::LimitExceeded { .. })
    ));
    // 2**26 per side is the largest square inside the domain and is covered
    // by the pinned fixture.
    assert!(plan_image_grid(1 << 26, 1 << 26, &spec()).is_ok());
}
