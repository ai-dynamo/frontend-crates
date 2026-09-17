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

from capture_vllm_rust_unified import RUST_MAIN


def test_streaming_peer_failures_are_preserved_for_both_terminal_schedules(tmp_path):
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("Rust compiler required to execute the generated peer capture loop")
    # Execute the producer's loop with an injected parser failure, without needing
    # a vLLM checkout. Mocking build_and_run would bypass the error being tested.
    loop = RUST_MAIN.split("// Streaming: fresh parser, per-chunk deltas.", 1)[1].split("results.insert(", 1)[0]
    source = r'''
type Value = String;
#[derive(Default)]
struct UnifiedParserOutput { events: Vec<String> }
struct Case { family: String, chunks: Vec<String>, terminal_step: bool }
struct Parser { fail_finish: bool }
impl Parser {
    fn parse_into(&mut self, text: &str, out: &mut UnifiedParserOutput) -> Result<(), &'static str> {
        out.events.push(text.to_string());
        if text == "bad" { Err("push failure") } else { Ok(()) }
    }
    fn finish(&mut self) -> Result<UnifiedParserOutput, &'static str> {
        if self.fail_finish { Err("finish failure") }
        else { Ok(UnifiedParserOutput { events: vec!["finished".into()] }) }
    }
}
fn make_parser(family: &str) -> (Parser, ()) {
    (Parser { fail_finish: family == "fail_finish" }, ())
}
fn deltas_to_json(events: &[String]) -> Vec<Value> { events.to_vec() }
fn run(case: Case, mut error: Option<String>) -> (Option<String>, Vec<Vec<Value>>) {
''' + loop + r'''
    (error, chunk_rows)
}
fn main() {
    for terminal_step in [false, true] {
        for fail_push in [false, true] {
            for fail_finish in [false, true] {
                let case = || Case {
                    family: if fail_finish { "fail_finish" } else { "ok" }.into(),
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
    path = tmp_path / "peer_errors.rs"
    path.write_text(source)
    binary = tmp_path / "peer_errors"
    subprocess.run([rustc, "--edition=2021", str(path), "-o", str(binary)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
