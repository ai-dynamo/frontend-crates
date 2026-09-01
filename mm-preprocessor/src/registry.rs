// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-family registry: each family implements
//! [`crate::processor::MmFamilyProcessor`] in `src/models/<model>.rs`; a
//! consumer selects one by its typed [`ProcessorSpec`] or by serializing a
//! spec (`{"family": ..., resolved processor params}`).

/// The resolved parameters of one family processor — the typed form of the
/// consumer-side spec, one variant per family arm. A consumer builds it
/// directly, or reaches it through [`processor_from_spec`], where the `family`
/// key selects the variant.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProcessorSpec {
    QwenVl(crate::models::qwen_vl::QwenVlSpec),
}

/// Build a family processor from a typed spec. `Err` when the family
/// rejects its parameters (e.g. a zero patch size).
pub fn build_processor(
    spec: ProcessorSpec,
) -> Result<Box<dyn crate::processor::MmFamilyProcessor>, String> {
    match spec {
        ProcessorSpec::QwenVl(spec) => Ok(Box::new(crate::models::qwen_vl::QwenVlProcessor::new(
            spec,
        )?)),
    }
}

/// Build a family processor from the consumer-side spec JSON
/// (`{"family": ..., resolved processor params}`). `Err` on an unknown family
/// or malformed spec — the caller treats that as "no native processor".
pub fn processor_from_spec(
    json: &str,
) -> Result<Box<dyn crate::processor::MmFamilyProcessor>, String> {
    let spec: ProcessorSpec = serde_json::from_str(json).map_err(|e| format!("mm spec: {e}"))?;
    build_processor(spec)
}

/// Resolve a [`ProcessorSpec`] from the HF config files themselves — the
/// `AutoProcessor.from_pretrained` equivalent for consumers with no Python
/// side (a router resolves once per model at boot). `config.json` selects the
/// family (`model_type`) and carries the token ids; `preprocessor_config.json`
/// carries the processor knobs.
///
/// Deliberately conservative, like an engine's Python gate: an unknown
/// `model_type`, an unrecognized knob, or a knob the Rust pipeline cannot
/// honor bit-exactly (e.g. `do_normalize: false`) is an `Err` — "no native
/// processor" — never a silent approximation.
pub fn spec_from_hf_configs(
    config_json: &str,
    preprocessor_config_json: &str,
) -> Result<ProcessorSpec, String> {
    let _ = (config_json, preprocessor_config_json);
    todo!("model_type → family arm; validate + map processor knobs")
}

/// [`spec_from_hf_configs`] over a local model directory (reads `config.json`
/// and `preprocessor_config.json`). Downloading from a hub stays the
/// consumer's concern — hand this the resolved local dir.
pub fn spec_from_model_dir(dir: &std::path::Path) -> Result<ProcessorSpec, String> {
    let _ = dir;
    todo!("read the two config files, delegate to spec_from_hf_configs")
}
