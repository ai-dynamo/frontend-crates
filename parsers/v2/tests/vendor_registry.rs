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
use dynamo_parsers_v2::{Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// A vendor can implement `UnifiedParser` from outside and have it selected.
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
