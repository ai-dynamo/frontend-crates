// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendor registry behaviour, in a binary of its own.
//!
//! These tests mutate PROCESS-GLOBAL state: registering a parser changes what
//! `create_unified_parser_for_family` returns for every other caller in the same
//! process. They lived in the lib test binary and a comment claimed that was safe
//! because unrelated tests "never look up these family names". That was false —
//! the override test replaces `qwen3`, and ordinary parser tests construct `qwen3`
//! through the same registry, so with `--test-threads` high enough an unrelated
//! test could observe the vendor's parser and fail. It was reproduced at 1-in-~33
//! runs at 32 threads.
//!
//! A separate integration binary is a separate PROCESS, so the global here cannot
//! reach the lib tests at all. Within this file the mutex still serializes, and
//! `Restore` puts the registry back even if a test panics — an early `?` or failed
//! assertion must not leave a global override installed for the tests that follow.

use anyhow::Result;
use dynamo_parsers_v2::{
    Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserExt, UnifiedParserOutput,
};

static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn descriptor_and_factory_resolve_atomically_and_remain_pinned() {
    use dynamo_parsers_v2::{
        UnifiedParserDescriptor, register_unified_parser_with_descriptor, resolve_unified_parser,
    };
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let mut descriptor = UnifiedParserDescriptor::new("descriptor_vendor");
    descriptor.aliases.push("descriptor_alias".into());
    register_unified_parser_with_descriptor(descriptor, vendor_factory).unwrap();
    let _restore = Restore("descriptor_vendor");
    let pinned = resolve_unified_parser("descriptor_alias").unwrap();
    assert_eq!(pinned.descriptor.family, "descriptor_vendor");
    let mut stolen = UnifiedParserDescriptor::new("other_vendor");
    stolen.aliases.push("descriptor_alias".into());
    assert!(register_unified_parser_with_descriptor(stolen, vendor_factory).is_err());
    dynamo_parsers_v2::unregister_unified_parser("descriptor_alias").unwrap();
    assert!(resolve_unified_parser("descriptor_alias").is_none());
    assert_eq!(
        pinned
            .create(&[])
            .unwrap()
            .push("retained factory")
            .unwrap(),
        vec![UnifiedParserEvent::Text("retained factory".into())]
    );

    let mut tool = UnifiedParserDescriptor::new("vendor_tool");
    tool.aliases.push("vendor_tool_alias".into());
    register_unified_parser_with_descriptor(tool, vendor_factory).unwrap();
    let _restore_tool = Restore("vendor_tool");
    let mut reasoner = UnifiedParserDescriptor::new("vendor_reasoner");
    reasoner
        .compatible_tool_selectors
        .push("vendor_tool_alias".into());
    register_unified_parser_with_descriptor(reasoner, vendor_factory).unwrap();
    let _restore_reasoner = Restore("vendor_reasoner");
    for selector in ["vendor_tool", "vendor_tool_alias"] {
        assert_eq!(
            dynamo_parsers_v2::resolve_unified_parser_pair(Some(selector), Some("vendor_reasoner"))
                .unwrap()
                .descriptor
                .family,
            "vendor_reasoner"
        );
    }

    let mut replacement = UnifiedParserDescriptor::new("qwen3_coder");
    replacement.aliases.push("vendor_qwen".into());
    register_unified_parser_with_descriptor(replacement, vendor_factory).unwrap();
    let _restore_qwen = Restore("qwen3");
    assert_eq!(
        dynamo_parsers_v2::resolve_unified_parser_pair(Some("vendor_qwen"), Some("nemotron_v3"))
            .unwrap()
            .descriptor
            .family,
        "nemotron_v3"
    );
    for alias in ["qwen3", "qwen3_coder", "vendor_qwen"] {
        let resolved = resolve_unified_parser(alias).unwrap();
        assert_eq!(resolved.descriptor.family, "qwen3");
        assert!(!resolved.descriptor.preserve_special_tokens);
        assert_eq!(
            resolved
                .create(&[])
                .unwrap()
                .push("<think>literal</think>")
                .unwrap(),
            vec![UnifiedParserEvent::Text("<think>literal</think>".into())]
        );
    }
}

/// Removes a registration on drop, including while unwinding from a panic.
struct Restore(&'static str);
impl Drop for Restore {
    fn drop(&mut self) {
        dynamo_parsers_v2::unregister_unified_parser(self.0);
    }
}

/// The smallest possible vendor parser: every byte is visible text.
#[derive(Default)]
struct VendorEverythingIsText;

impl UnifiedParser for VendorEverythingIsText {
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        output.push_text(delta);
        Ok(())
    }
    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput::default())
    }
}

fn vendor_factory(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(VendorEverythingIsText))
}

/// The fixture name `acme_v1` becomes selectable only after runtime registration.
#[test]
fn vendor_can_register_a_new_family() {
    let _g = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    assert!(dynamo_parsers_v2::create_unified_parser_for_family("acme_v1", &[]).is_err());

    dynamo_parsers_v2::register_unified_parser("acme_v1", vendor_factory);
    let _restore = Restore("acme_v1");
    assert!(dynamo_parsers_v2::vendor_unified_families().contains(&"acme_v1".to_string()));

    let events = dynamo_parsers_v2::create_unified_parser_for_family("acme_v1", &[])
        .expect("registered family must construct")
        .push("<think>hi</think>")
        .unwrap();
    assert_eq!(
        events,
        vec![UnifiedParserEvent::Text("<think>hi</think>".into())],
        "the VENDOR parser must run, not a built-in that happens to know <think>"
    );
}

/// THE case the registry exists for: replace a family this crate already ships.
#[test]
fn vendor_can_override_a_builtin_and_restore_it() {
    let _g = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let family = "qwen3";
    assert!(dynamo_parsers_v2::builtin_unified_families().contains(&family));

    let baseline = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
        .unwrap()
        .push("<think>hi</think>")
        .unwrap();
    assert!(
        baseline
            .iter()
            .any(|e| matches!(e, UnifiedParserEvent::Reasoning(_))),
        "built-in qwen3 should emit reasoning, got {baseline:?}"
    );

    {
        dynamo_parsers_v2::register_unified_parser(family, vendor_factory);
        let _restore = Restore(family);
        let overridden = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
            .unwrap()
            .push("<think>hi</think>")
            .unwrap();
        assert_eq!(
            overridden,
            vec![UnifiedParserEvent::Text("<think>hi</think>".into())],
            "the vendor override must win over the built-in of the same name"
        );
    }

    let restored = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
        .unwrap()
        .push("<think>hi</think>")
        .unwrap();
    assert_eq!(
        restored, baseline,
        "removing the override must restore the built-in exactly"
    );
}

/// Registering ONE name of a family must shadow EVERY name that family answers to.
///
/// `qwen3` and `qwen3_coder` are one grammar with two routing names. Keying the
/// registry on the caller's spelling shadowed only that spelling, so the same
/// family silently ran the vendor's parser or ours depending on how the request
/// happened to be routed — the advertised "replace a family we ship" contract was
/// true for one name and false for its sibling.
#[test]
fn registering_one_alias_shadows_every_alias_of_that_family() {
    let _g = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (name, alias) = ("qwen3", "qwen3_coder");
    assert!(dynamo_parsers_v2::builtin_unified_families().contains(&alias));

    dynamo_parsers_v2::register_unified_parser(name, vendor_factory);
    let _restore = Restore(name);

    let by_name = dynamo_parsers_v2::create_unified_parser_for_family(name, &[])
        .unwrap()
        .push("<think>x</think>")
        .unwrap();
    let by_alias = dynamo_parsers_v2::create_unified_parser_for_family(alias, &[])
        .unwrap()
        .push("<think>x</think>")
        .unwrap();
    assert_eq!(
        by_name, by_alias,
        "registering {name} must also shadow {alias}; they are one family"
    );
    assert_eq!(
        by_alias,
        vec![UnifiedParserEvent::Text("<think>x</think>".into())]
    );
}

/// ...and unregistering by either spelling must remove it for both.
#[test]
fn unregistering_by_alias_removes_the_registration() {
    let _g = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    dynamo_parsers_v2::register_unified_parser("qwen3", vendor_factory);
    assert!(dynamo_parsers_v2::unregister_unified_parser("qwen3_coder").is_some());

    let after = dynamo_parsers_v2::create_unified_parser_for_family("qwen3", &[])
        .unwrap()
        .push("<think>x</think>")
        .unwrap();
    assert!(
        after
            .iter()
            .any(|e| matches!(e, UnifiedParserEvent::Reasoning(_))),
        "unregistering via the alias must restore the built-in for the canonical name too"
    );
}

/// The built-in list is what conformance iterates, so a vendor registration must
/// not silently enrol itself into a corpus that has no cases for it.
#[test]
fn registering_does_not_change_the_builtin_list() {
    let _g = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let before = dynamo_parsers_v2::builtin_unified_families().to_vec();
    dynamo_parsers_v2::register_unified_parser("acme_not_in_corpus", vendor_factory);
    let _restore = Restore("acme_not_in_corpus");
    assert_eq!(
        dynamo_parsers_v2::builtin_unified_families().to_vec(),
        before
    );
}

/// An unknown family must tell the caller how to supply one.
#[test]
fn unknown_family_error_explains_how_to_register() {
    let _g = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    // `Box<dyn UnifiedParser>` is not `Debug`, so `unwrap_err` is unavailable.
    let err = match dynamo_parsers_v2::create_unified_parser_for_family("no_such_family", &[]) {
        Ok(_) => panic!("an unregistered family must not construct"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("register_unified_parser"), "{err}");
    assert!(err.contains("no_such_family"), "{err}");
}

#[test]
fn factory_override_keeps_aliases_without_claiming_harmony_capabilities() {
    use dynamo_parsers_v2::{register_unified_parser, resolve_unified_parser};

    let _guard = SERIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    assert!(
        resolve_unified_parser("gpt_oss")
            .unwrap()
            .descriptor
            .token_id_input
    );
    register_unified_parser("harmony", vendor_factory);
    let _restore = Restore("gpt_oss");
    for selector in ["gpt_oss", "harmony"] {
        let registration = resolve_unified_parser(selector).unwrap();
        let descriptor = &registration.descriptor;
        assert_eq!(descriptor.family, "gpt_oss");
        assert!(descriptor.text_input);
        assert!(!descriptor.token_id_input);
        assert!(!descriptor.continuation);
        assert!(descriptor.token_prompt_prefix.is_none());
        assert!(!registration.preserves_special_tokens().unwrap());
        let mut parser = registration.create(&[]).unwrap();
        assert_eq!(
            parser.push("plain text").unwrap(),
            [UnifiedParserEvent::Text("plain text".into())]
        );
    }
}

#[test]
fn loaded_registration_validates_the_entire_batch_before_inserting() {
    use dynamo_parsers_v2::{
        BatchPrefill, UnifiedParserDescriptor, UnifiedParserRegistration, resolve_unified_parser,
        unified::register_loaded_unified_parsers,
    };

    let _guard = SERIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    let _restore_valid = Restore("loaded_valid");
    let _restore_invalid = Restore("loaded_invalid");
    let mut valid = UnifiedParserDescriptor::new("loaded_valid");
    valid.guidance.structural_tag = true;
    let valid = UnifiedParserRegistration::loaded(
        valid,
        "valid".into(),
        Some(std::sync::Arc::new(|_| Ok(None))),
        None,
        vendor_factory,
    );
    for error in [
        "marker-pair prefill requires markers",
        "parser must accept text or token IDs",
    ] {
        let mut invalid = UnifiedParserDescriptor::new("loaded_invalid");
        if error.starts_with("marker") {
            invalid.batch_prefill = BatchPrefill::MarkerPair;
        } else {
            invalid.text_input = false;
        }
        let invalid = UnifiedParserRegistration::loaded(
            invalid,
            "invalid".into(),
            None,
            None,
            vendor_factory,
        );
        assert_eq!(
            register_loaded_unified_parsers(vec![valid.clone(), invalid])
                .unwrap_err()
                .to_string(),
            error
        );
        assert!(resolve_unified_parser("loaded_valid").is_none());
        assert!(resolve_unified_parser("loaded_invalid").is_none());
    }
    register_loaded_unified_parsers(vec![valid]).unwrap();
    let registration = resolve_unified_parser("loaded_valid").unwrap();
    assert!(registration.descriptor.guidance.structural_tag);
    assert!(registration.plugin_guidance().is_some());
}

#[test]
fn legacy_factory_retains_host_and_standalone_compatibility() {
    use dynamo_parsers_v2::{
        create_tool_parser_for_family, register_unified_parser, resolve_unified_parser,
        resolve_unified_parser_pair, structural_tag_builder_for_family,
    };

    fn with_tokens(_: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
        struct PreservationProbe;
        impl UnifiedParser for PreservationProbe {
            fn preserve_special_tokens(&self) -> bool {
                true
            }
            fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
                output.push_text(delta);
                Ok(())
            }
            fn finish(&mut self) -> Result<UnifiedParserOutput> {
                Ok(UnifiedParserOutput::default())
            }
        }
        Ok(Box::new(PreservationProbe))
    }

    fn needs_tools(tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
        anyhow::ensure!(!tools.is_empty(), "tools are required");
        vendor_factory(tools)
    }

    let _guard = SERIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    assert!(
        resolve_unified_parser("gemma4")
            .unwrap()
            .preserves_special_tokens()
            .unwrap()
    );
    register_unified_parser("gemma4", with_tokens);
    let _restore_gemma = Restore("gemma4");
    assert!(
        resolve_unified_parser("gemma4")
            .unwrap()
            .preserves_special_tokens()
            .unwrap()
    );

    register_unified_parser("nemotron_v3", vendor_factory);
    let _restore_nemotron = Restore("nemotron_v3");
    assert_eq!(
        resolve_unified_parser_pair(Some("qwen3_coder"), Some("nemotron_v3"))
            .unwrap()
            .descriptor
            .family,
        "nemotron_v3",
    );

    let builder = structural_tag_builder_for_family("qwen3_coder").unwrap();
    register_unified_parser("qwen3", needs_tools);
    let _restore_qwen = Restore("qwen3");
    let registration = resolve_unified_parser("qwen3_coder").unwrap();
    assert!(registration.descriptor.structural_tag_builder.is_none());
    assert_eq!(
        registration
            .preserves_special_tokens()
            .unwrap_err()
            .to_string(),
        "tools are required"
    );
    assert!(std::ptr::eq(
        structural_tag_builder_for_family("qwen3_coder").unwrap(),
        builder
    ));
    assert!(create_tool_parser_for_family("qwen3_coder", &[]).is_ok());
}

#[test]
fn loaded_plugins_cannot_claim_a_legacy_host_selector() {
    use dynamo_parsers_v2::{
        UnifiedParserDescriptor, UnifiedParserRegistration, resolve_unified_parser,
        resolve_unified_parser_pair, unified::register_loaded_unified_parsers,
    };

    let _guard = SERIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    let mut descriptor = UnifiedParserDescriptor::new("mistral");
    for family in ["mistral", "attempted_override"] {
        descriptor.family = family.into();
        if family != "mistral" {
            descriptor.aliases.push("mistral".into());
        }
        let registration = UnifiedParserRegistration::loaded(
            descriptor.clone(),
            "test".into(),
            None,
            None,
            vendor_factory,
        );
        assert!(register_loaded_unified_parsers(vec![registration]).is_err());
        assert!(resolve_unified_parser(family).is_none());
    }
    assert_eq!(
        resolve_unified_parser_pair(Some("mistral"), None)
            .unwrap()
            .descriptor
            .family,
        "mistral_4",
    );
}
