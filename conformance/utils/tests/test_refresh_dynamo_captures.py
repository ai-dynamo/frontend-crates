# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_stream_fixtures
import fill_streamv2
import refresh_dynamo_captures as refresh


def test_refresh_stream_preserves_canonical_dynamo_unavailable(tmp_path, monkeypatch):
    tree = tmp_path / "fixtures-stream-v2"
    source = tree / "inputs" / "deepseek_v4" / "TOOLCALLING.streamv2.11.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        "family: deepseek_v4\n"
        "mode: streamv2\n"
        "cases:\n"
        "  TOOLCALLING.streamv2.11.a:\n"
        "    unavailable:\n"
        "      dynamo_v2: DSML has one tool-name owner.\n"
    )

    monkeypatch.setattr(refresh, "ensure_tree", lambda _name: tree)
    monkeypatch.setattr(refresh, "V2_FAMILIES", ["deepseek_v4"])
    monkeypatch.setattr(refresh, "run_bin", lambda *_args: json.dumps({}))

    refresh.refresh_stream("0.5.1")

    output = tree / "dynamo_v2-0.5.1" / "deepseek_v4" / source.name
    assert "unavailable: DSML has one tool-name owner." in output.read_text()


def test_build_sources_preserves_reasoned_unavailable_cases(tmp_path):
    fixtures = tmp_path / "batch"
    source = fixtures / "deepseek_v4" / "TOOLCALLING.batch.11.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        "family: deepseek_v4\n"
        "mode: batch\n"
        "cases:\n"
        "  TOOLCALLING.batch.11.a:\n"
        "    description: One name owner\n"
        "    explanation: DSML cannot express conflicting names.\n"
        "    unavailable:\n"
        "      dynamo_v2: DSML has one tool-name owner.\n"
    )
    out = tmp_path / "stream"
    out.mkdir()

    generated = fill_streamv2.build_sources("deepseek_v4", fixtures, out)
    doc = yaml.safe_load(open(generated["11"]))
    case = doc["cases"]["TOOLCALLING.streamv2.11.a"]

    assert case["explanation"] == "DSML cannot express conflicting names."
    assert case["unavailable"]["dynamo_v2"] == "DSML has one tool-name owner."
    assert "chunks" not in case


def test_write_sources_writes_versioned_input_tree(tmp_path):
    source = (
        tmp_path
        / "conformance/toolcalling/fixtures-batch-v1/inputs/deepseek_v4/TOOLCALLING.batch.1.yaml"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        "family: deepseek_v4\n"
        "mode: batch\n"
        "cases:\n"
        "  TOOLCALLING.batch.1:\n"
        "    description: One call\n"
        "    model_text: call\n"
    )
    generated = fill_streamv2.build_sources(
        "deepseek_v4",
        tmp_path / "conformance/toolcalling/fixtures-batch-v1/inputs",
        tmp_path / "work",
    )
    fill_streamv2.write_sources(
        {"deepseek_v4": generated},
        tmp_path / "conformance/toolcalling/fixtures-stream-v2",
    )

    output = (
        tmp_path
        / "conformance/toolcalling/fixtures-stream-v2/inputs/deepseek_v4/TOOLCALLING.streamv2.1.yaml"
    )
    assert yaml.safe_load(output.read_text())["cases"]["TOOLCALLING.streamv2.1"]["chunks"]


def test_stream_builder_preserves_source_unavailable(tmp_path, monkeypatch):
    source = tmp_path / "source.yaml"
    source.write_text(
        "family: deepseek_v4\n"
        "mode: streamv2\n"
        "cases:\n"
        "  TOOLCALLING.streamv2.11.a:\n"
        "    description: One name owner\n"
        "    explanation: DSML cannot express conflicting names.\n"
        "    unavailable:\n"
        "      dynamo_v2: DSML has one tool-name owner.\n"
    )
    output = tmp_path / "output.yaml"
    monkeypatch.setattr(
        "sys.argv",
        ["build_stream_fixtures.py", "--source", str(source), "--out", str(output)],
    )
    monkeypatch.setattr(build_stream_fixtures.capture_driver, "_vllm_rust_source_version", lambda _: None)
    monkeypatch.setattr(build_stream_fixtures.capture_driver, "_vllm_rust_unavailable", lambda _: "not captured")

    build_stream_fixtures.main()

    case = yaml.safe_load(output.read_text())["cases"]["TOOLCALLING.streamv2.11.a"]
    assert case["explanation"] == "DSML cannot express conflicting names."
    assert case["unavailable"]["dynamo_v2"] == "DSML has one tool-name owner."
