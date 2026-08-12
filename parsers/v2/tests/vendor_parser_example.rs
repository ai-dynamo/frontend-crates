// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worked example from `CUSTOM_PARSERS.md`, compiled.
//!
//! This is an INTEGRATION test on purpose: it sees exactly what a vendor crate sees
//! — the public API of `dynamo_parsers_v2` and nothing else. If a symbol stops being
//! re-exported, or the trait's required set changes, this fails to compile and the
//! documented instructions are known to be wrong before a vendor discovers it.
//!
//! Keep this file and the code blocks in `CUSTOM_PARSERS.md` in step.

use anyhow::Result;
use dynamo_parsers_v2::{Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

/// The smallest complete vendor parser: one required method plus the flush.
#[derive(Default)]
struct AcmeParser {
    /// Anything not yet safe to emit. A real parser holds a partial marker here.
    buffered: String,
}

impl UnifiedParser for AcmeParser {
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        // A real grammar decides per byte. This one keeps a trailing '<' back,
        // standing in for "might be the start of a marker", so the example exercises
        // buffering and the flush rather than pretending neither exists.
        self.buffered.push_str(delta);
        if let Some(cut) = self.buffered.rfind('<') {
            let emit: String = self.buffered[..cut].to_string();
            self.buffered = self.buffered[cut..].to_string();
            output.push_text(emit);
        } else {
            let all = std::mem::take(&mut self.buffered);
            output.push_text(all);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut out = UnifiedParserOutput::default();
        // Whatever is still held back is ordinary text once the stream has ended.
        out.push_text(std::mem::take(&mut self.buffered));
        Ok(out)
    }

    fn reset(&mut self) -> String {
        // MANDATORY for a parser that buffers: the default returns an empty string and
        // clears nothing, so a caller recovering after an error would resume on this
        // parser's stale held-back bytes. Hand them back and return to a fresh stream.
        std::mem::take(&mut self.buffered)
    }
}

/// Removes a registration on drop, including while unwinding from a panic, so a
/// failing test cannot leave a global override installed for the next one.
struct Restore(&'static str);
impl Drop for Restore {
    fn drop(&mut self) {
        dynamo_parsers_v2::unregister_unified_parser(self.0);
    }
}

fn acme_factory(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(AcmeParser::default()))
}

/// A vendor registers a NEW family and it is selected by name.
#[test]
fn vendor_family_is_selected_through_the_public_registry() {
    dynamo_parsers_v2::register_unified_parser("acme_doc_example", acme_factory);
    let _restore = Restore("acme_doc_example");

    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("acme_doc_example", &[])
        .expect("registered family must construct");
    let mut events = parser.push("hello ").unwrap();
    events.extend(parser.push("world").unwrap());
    events.extend(parser.finish().unwrap().events);

    let text: String = events
        .iter()
        .map(|e| match e {
            UnifiedParserEvent::Text(t) => t.as_str(),
            _ => "",
        })
        .collect();
    assert_eq!(text, "hello world");
}

/// Nothing is lost across an arbitrary split — the property the contract section of
/// `CUSTOM_PARSERS.md` asks a vendor to test, demonstrated on the example itself.
#[test]
fn example_parser_is_split_invariant() {
    let input = "alpha <not-a-marker> beta";
    let whole = {
        let mut p = AcmeParser::default();
        let mut out = UnifiedParserOutput::default();
        p.parse_into(input, &mut out).unwrap();
        let mut tail = p.finish().unwrap();
        out.append(&mut tail);
        out.assembled()
    };

    for cut in 1..input.len() {
        if !input.is_char_boundary(cut) {
            continue;
        }
        let mut p = AcmeParser::default();
        let mut out = UnifiedParserOutput::default();
        p.parse_into(&input[..cut], &mut out).unwrap();
        p.parse_into(&input[cut..], &mut out).unwrap();
        let mut tail = p.finish().unwrap();
        out.append(&mut tail);
        assert_eq!(
            out.assembled(),
            whole,
            "split at {cut} produced a different result than the whole input"
        );
    }
}

/// A vendor replaces a family this crate ships, then gives it back.
#[test]
fn vendor_can_shadow_a_builtin_family() {
    let family = "qwen3";
    assert!(dynamo_parsers_v2::builtin_unified_families().contains(&family));

    let builtin = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
        .unwrap()
        .push("<think>hi</think>")
        .unwrap();

    dynamo_parsers_v2::register_unified_parser(family, acme_factory);
    let restore = Restore(family);
    let shadowed = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
        .unwrap()
        .push("<think>hi</think>")
        .unwrap();
    assert_ne!(
        shadowed, builtin,
        "the vendor parser must be the one that ran"
    );

    drop(restore);
    let restored = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
        .unwrap()
        .push("<think>hi</think>")
        .unwrap();
    assert_eq!(restored, builtin, "unregistering must restore the built-in");
}

/// The Markdown example and this file must be the SAME implementation.
///
/// `CUSTOM_PARSERS.md` claimed that if its instructions stopped being true "the build
/// fails here first". That claim was false: the documented example double-emitted its
/// buffer (`push("hello") + finish()` produced `hellohello`) while this file buffered
/// differently, so nothing compared the two and nothing failed. Matching them by hand
/// restores the invariant but leaves it resting on the manual comparison that already
/// failed once. This is the comparison, as a test.
#[test]
fn doc_example_matches_compiled_example() {
    const DOC: &str = include_str!("../CUSTOM_PARSERS.md");
    const SRC: &str = include_str!("vendor_parser_example.rs");

    /// First fenced ```rust block in the doc.
    fn first_rust_block(md: &str) -> &str {
        let start = md
            .find("```rust")
            .expect("CUSTOM_PARSERS.md has no rust block");
        let after = start + "```rust".len();
        let end = md[after..].find("```").expect("unterminated rust block") + after;
        &md[after..end]
    }

    /// Body of `fn <name>` up to the matching close brace, whitespace- and
    /// comment-insensitive so formatting or a reworded comment cannot fail this.
    fn body(src: &str, name: &str) -> String {
        let sig = format!("fn {name}(");
        let at = src.find(&sig).unwrap_or_else(|| panic!("no `{sig}` found"));
        let open = src[at..].find('{').expect("no body") + at;
        let (mut depth, mut end) = (0usize, open);
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        src[open + 1..end]
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join("")
    }

    // Derive the method set from the sources instead of hardcoding it. A hardcoded
    // ["parse_into", "finish"] let `reset` diverge between the two copies with the test
    // still green — the guide's own MANDATORY rule was violated in the doc for exactly
    // as long as this loop decided what counted.
    let doc = first_rust_block(DOC);
    // Scope to the trait impl: the compiled file also holds the tests themselves.
    let impl_block = |src: &str| -> String {
        let start = src
            .find("impl UnifiedParser for AcmeParser {")
            .expect("the example must implement UnifiedParser for AcmeParser");
        let rest = &src[start..];
        let end = rest.find("\n}").expect("unterminated impl block");
        rest[..end].to_string()
    };
    let methods = |src: &str| -> Vec<String> {
        impl_block(src)
            .lines()
            .filter_map(|l| l.trim().strip_prefix("fn "))
            .filter_map(|l| l.split('(').next())
            .map(str::to_string)
            .collect()
    };
    assert_eq!(
        methods(doc),
        methods(SRC),
        "CUSTOM_PARSERS.md and vendor_parser_example.rs implement different methods. \
         A vendor copies the doc, so a method present in only one is a trap."
    );
    for f in methods(doc) {
        let f = f.as_str();
        assert_eq!(
            body(doc, f),
            body(SRC, f),
            "CUSTOM_PARSERS.md and vendor_parser_example.rs disagree on `{f}`. \
             The documented example is what a vendor copies; keep them identical."
        );
    }
}

/// The example must obey the rule the guide states as MANDATORY: a parser that buffers
/// overrides `reset`. It held a trailing `<` back with no override, so a vendor copying
/// it would inherit the default that clears nothing and silently resume on stale bytes.
#[test]
fn the_worked_example_honours_the_reset_rule_it_documents() {
    let mut p = AcmeParser::default();
    let mut out = UnifiedParserOutput::default();
    p.parse_into("visible<par", &mut out).expect("parse");

    assert_eq!(
        out.events,
        vec![UnifiedParserEvent::Text("visible".to_string())],
        "the partial marker must be held back, not emitted"
    );
    assert_eq!(
        p.reset(),
        "<par",
        "reset must hand back the held-back bytes, not the default empty string"
    );
    assert_eq!(
        p.reset(),
        "",
        "after reset the parser is a fresh stream and holds nothing"
    );
}
