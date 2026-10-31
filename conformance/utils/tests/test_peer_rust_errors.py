# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from impls import unified_starts_in_reasoning
from capture_vllm_rust_unified import _unsupported_request, _vllm_rust_version, public_profiles, rust_main


@pytest.mark.parametrize("modern", [False, True])
def test_public_family_aliases_and_release_specific_glm_reasoner(tmp_path, modern):
    api = tmp_path / "src/parser/src"
    exports = {
        "unified": "",
        "tool": "pub use parsers::{KimiK2ToolParser, Qwen3CoderToolParser, Glm47MoeToolParser};",
        "reasoning": "pub type KimiK2ReasoningParser = Shared; pub type NemotronV3ReasoningParser = Shared; "
                     + ("pub type Glm47ReasoningParser = Shared;" if modern else "pub type Glm45ReasoningParser = Shared;"),
    }
    for module, source in exports.items():
        path = api / module / "mod.rs"
        path.parent.mkdir(parents=True)
        path.write_text(source)
    profiles = public_profiles(tmp_path)
    generated = rust_main("0.31.0" if modern else "0.25.1", "", profiles)
    assert set(profiles) == {"kimi_k2", "nemotron_v3", "glm47"}
    assert "reasoning::KimiK2ReasoningParser::create" in generated
    assert "reasoning::NemotronV3ReasoningParser::create" in generated
    assert f'reasoning::{"Glm47" if modern else "Glm45"}ReasoningParser::create' in generated
    assert "reasoning::KimiReasoningParser::create" not in generated
    assert '.with_regular_token("<think>"' in generated


def test_response_prefill_uses_public_optional_reasoner_or_persistent_native_mode():
    for family in ("kimi_k2", "qwen3", "glm47", "nemotron_v3", "deepseek_v4", "deepseek_v41", "minimax_m3", "kimi_k3"):
        assert _unsupported_request({"family": family, "init": {"starting_state": "Response"}}) is None
    for family in ("gemma4", "unknown"):
        assert _unsupported_request({"family": family, "init": {"starting_state": "Response"}}) is not None


def test_generated_public_context_preserves_defaults_and_response_for_both_schedules():
    assert unified_starts_in_reasoning("nemotron_v3")
    assert not unified_starts_in_reasoning("qwen3")
    assert unified_starts_in_reasoning("qwen3", "Reasoning")
    assert not unified_starts_in_reasoning("nemotron_v3", "Response")
    generated = rust_main("0.31.0", "")
    nemotron = generated.split('"nemotron_v3" => match case.init.starting_state.as_str()', 1)[1].split('},', 1)[0]
    qwen = generated.split('"qwen3" => match case.init.starting_state.as_str()', 1)[1].split('},', 1)[0]
    assert '\"None\" | \"\" => \"<think>\\n\"' in nemotron
    assert '\"None\" | \"\" => \"\"' in qwen
    assert 'CombinedParser::new(if starting_state == "Response" { None } else { Some(' in generated
    assert generated.count('make_parser(&case.family, &case.tools, &case.init.starting_state)') == 2


def test_native_parser_output_cannot_qualify_a_generation_schema():
    native = {"input": '{"answer":42}', "init": {"tool_output_mode": "Native"}}
    assert _unsupported_request(native) is None
    constrained = {**native, "init": {**native["init"], "structural_tag": {
        "schema": {"type": "object"}, "rejected": ["[]"],
    }}}
    code, reason = _unsupported_request(constrained)
    assert code == "peer_request_unsupported"
    assert "generation constraints" in reason


@pytest.mark.parametrize("dirname", ["vllm-99.88.77", "source"])
def test_rust_capture_rejects_unverified_release_identity(tmp_path, dirname):
    source = tmp_path / dirname / "rust"
    parser = source / "src/parser"
    parser.mkdir(parents=True)
    (parser / "Cargo.toml").write_text('[package]\nversion = "0.1.0"\n')
    with pytest.raises(ValueError, match="own Git checkout"):
        _vllm_rust_version(source, parser)


@pytest.mark.parametrize("tag", ["v0.26.0", "0.26.0"])
def test_rust_capture_uses_exact_release_tag(tmp_path, tag):
    root = tmp_path / "vllm-99.88.77"
    parser = root / "rust/src/parser"
    parser.mkdir(parents=True)
    manifest = parser / "Cargo.toml"
    manifest.write_text('[package]\nversion = "0.1.0"\n')

    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
             "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
            cwd=root, check=True, capture_output=True, text=True,
        )

    git("init")
    git("add", ".")
    git("commit", "-m", "release fixture", "--no-verify")
    with pytest.raises(ValueError, match="exact semantic release tag"):
        _vllm_rust_version(root / "rust", parser)
    git("tag", tag)
    assert _vllm_rust_version(root / "rust", parser) == "0.26.0"
    nested = root / "nested/rust"
    nested.mkdir(parents=True)
    with pytest.raises(ValueError, match="own Git checkout"):
        _vllm_rust_version(nested, nested / "src/parser")
    manifest.write_text('[package]\nversion = "0.2.0"\n')
    with pytest.raises(ValueError, match="clean Rust source"):
        _vllm_rust_version(root / "rust", parser)
    manifest.write_text('[package]\nversion = "0.1.0"\n')
    extra = parser / "extra.rs"
    extra.write_text("// untracked source\n")
    with pytest.raises(ValueError, match="clean Rust source"):
        _vllm_rust_version(root / "rust", parser)


@pytest.mark.parametrize("decoded_text", [False, True])
def test_streaming_peer_failures_are_preserved_for_both_terminal_schedules(tmp_path, decoded_text):
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("Rust compiler required to execute the generated peer capture loop")
    # Execute the producer's loop with an injected parser failure, without needing
    # a vLLM checkout. Mocking build_and_run would bypass the error being tested.
    api = "fn parse_into(&mut self, delta: DecodedText);" if decoded_text else "fn parse_into(&mut self, delta: &str);"
    loop = rust_main("0.30.0" if decoded_text else "0.25.1", api).split("// Streaming: fresh parser, per-chunk deltas.", 1)[1].split("results.insert(", 1)[0]
    source = r'''
type Value = String;
#[derive(Default)]
struct UnifiedParserOutput { events: Vec<String> }
struct Init { starting_state: String }
struct Case { family: String, tools: Vec<String>, chunks: Vec<String>, terminal_step: bool, init: Init }
struct Parser { fail_finish: bool }
impl Parser {
    fn initialize(&mut self, _ids: &[u32]) -> Result<(), &'static str> { Ok(()) }
    fn parse_into(&mut self, text: &str, out: &mut UnifiedParserOutput) -> Result<(), &'static str> {
        out.events.push(text.to_string());
        if text == "bad" { Err("push failure") } else { Ok(()) }
    }
    fn finish(&mut self) -> Result<UnifiedParserOutput, &'static str> {
        if self.fail_finish { Err("finish failure") }
        else { Ok(UnifiedParserOutput { events: vec!["finished".into()] }) }
    }
}
fn make_parser(family: &str, _tools: &[String], _starting_state: &str) -> (Parser, (), ()) {
    (Parser { fail_finish: family == "fail_finish" }, (), ())
}
fn deltas_to_json(events: &[String]) -> Vec<Value> { events.to_vec() }
fn run(case: Case, mut error: Option<String>) -> (Option<String>, Vec<Vec<Value>>) {
    let prompt_ids: Vec<u32> = Vec::new();
''' + loop + r'''
    (error, chunk_rows)
}
fn main() {
    for terminal_step in [false, true] {
        for fail_push in [false, true] {
            for fail_finish in [false, true] {
                let case = || Case {
                    family: if fail_finish { "fail_finish" } else { "ok" }.into(),
                    init: Init { starting_state: "None".into() },
                    tools: Vec::new(),
                    chunks: vec![if fail_push { "bad" } else { "good" }.into()],
                    terminal_step,
                };
                let (error, rows) = run(case(), None);
                assert_eq!(error.is_some(), fail_push || fail_finish,
                    "terminal={terminal_step}, push={fail_push}, finish={fail_finish}");
                if fail_push { assert!(error.unwrap().contains("push failure")); }
                else if fail_finish { assert!(error.unwrap().contains("finish failure")); }
                assert_eq!(rows.len(), if terminal_step { 2 } else { 1 });
                assert_eq!(run(case(), Some("batch failure".into())).0.as_deref(), Some("batch failure"));
            }
        }
    }
}
'''
    if decoded_text:
        source = """
struct DecodedText { text: String }
impl DecodedText {
    fn unattributed(text: impl Into<String>) -> Self { Self { text: text.into() } }
}
""" + source.replace("text: &str, out:", "text: DecodedText, out:").replace(
            "out.events.push(text.to_string());", "let text = &text.text; out.events.push(text.to_string());"
        )
    path = tmp_path / "peer_errors.rs"
    path.write_text(source)
    subprocess.run(["rustfmt", "--edition=2021", str(path)], check=True, capture_output=True, text=True)
    binary = tmp_path / "peer_errors"
    subprocess.run([rustc, "--edition=2021", str(path), "-o", str(binary)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
