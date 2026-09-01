// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family registry: each family implements
//! [`crate::pipeline::MmFamilyProcessor`] in `src/models/<model>.rs`; a
//! consumer selects one by its typed [`PipelineSpec`] or by serializing a
//! spec (`{"family": ..., resolved processor params}`).

/// The resolved parameters of one family pipeline — the typed form of the
/// consumer-side spec, one variant per family arm. A consumer builds it
/// directly, or reaches it through [`pipeline_from_spec`], where the `family`
/// key selects the variant.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PipelineSpec {
    QwenVl(crate::models::qwen_vl::QwenVlSpec),
}

/// Build a family processor from a typed spec. `Err` when the family
/// rejects its parameters (e.g. a zero patch size).
pub fn build_pipeline(
    spec: PipelineSpec,
) -> Result<Box<dyn crate::pipeline::MmFamilyProcessor>, String> {
    match spec {
        PipelineSpec::QwenVl(spec) => Ok(Box::new(crate::models::qwen_vl::QwenVlProcessor::new(
            spec,
        )?)),
    }
}

/// Build a family processor from the consumer-side spec JSON
/// (`{"family": ..., resolved processor params}`). `Err` on an unknown family
/// or malformed spec — the caller treats that as "no native pipeline".
pub fn pipeline_from_spec(
    json: &str,
) -> Result<Box<dyn crate::pipeline::MmFamilyProcessor>, String> {
    let spec: PipelineSpec = serde_json::from_str(json).map_err(|e| format!("mm spec: {e}"))?;
    build_pipeline(spec)
}
