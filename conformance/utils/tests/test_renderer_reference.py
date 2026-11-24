# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference refresh preserves independent model sources and exact input bindings."""

from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path
import shutil

import pytest

import renderer_reference as reference


ROOT = Path(__file__).resolve().parents[3]
STORE = ROOT / reference.STORE
MANIFEST_BYTES = (STORE / "manifest.json").read_bytes()
MANIFEST = json.loads(MANIFEST_BYTES)


def model_named(name: str) -> dict:
    return next(model for model in MANIFEST["models"] if model["id"] == name)


def case_named(model: str, name: str) -> dict:
    inputs = reference.load_json(STORE / "families" / model / "inputs.json")
    return next(case for case in inputs["cases"] if case["id"] == "RENDERER." + name)


def require_template_runtime() -> None:
    for distribution, expected in (("transformers", "5.12.1"), ("Jinja2", "3.1.6")):
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("manual reference environment is not installed")
        if actual != expected:
            pytest.skip("manual reference environment has a different pinned version")


def test_all_model_assets_are_hash_verified() -> None:
    assert {model["id"] for model in MANIFEST["models"]} == {
        "deepseek_v41", "glm53", "kimi_k3", "qwen38", "minimax_m31"
    }
    for model in MANIFEST["models"]:
        assets = reference.assets_for(STORE, model)
        assert {"config", "tokenizer_config", "license"} <= assets.keys()
        assert len(model["revision"]) == 40


def test_asset_damage_and_path_escape_are_rejected(tmp_path: Path) -> None:
    asset = copy.deepcopy(MANIFEST["models"][0]["assets"][0])
    destination = tmp_path / asset["path"]
    destination.parent.mkdir(parents=True)
    destination.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        reference.checked_asset(tmp_path, asset)
    asset["path"] = "../outside.json"
    with pytest.raises(ValueError, match="inside"):
        reference.checked_asset(tmp_path, asset)


def test_capture_bindings_cover_original_utf8_inputs() -> None:
    for model in MANIFEST["models"]:
        directory = STORE / "families" / model["id"]
        inputs = reference.load_json(directory / "inputs.json")
        observed = reference.load_json(directory / "reference.json")
        assert observed["inputs_sha256"] == reference.sha256((directory / "inputs.json").read_bytes())
        assert observed["manifest_sha256"] == reference.sha256(MANIFEST_BYTES)
        assert set(observed["results"]) == {case["id"] for case in inputs["cases"]}
        assert "東京" in (directory / "inputs.json").read_text()


def test_reference_normalization_records_authority_without_mutating_request() -> None:
    original = [{"role": "assistant", "content": "", "reasoning_content": ["first", "", "second"],
                 "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"東京"}'}}]}]
    messages = copy.deepcopy(original)
    transforms: list[dict[str, str]] = []
    reference.normalize_hf_messages(messages, transforms, "auto")
    assert messages[0]["reasoning_content"] == "first\nsecond"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"city": "東京"}
    assert original[0]["reasoning_content"] == ["first", "", "second"]
    assert any("independently-authored" in item["authority"] for item in transforms)
    assert {item["name"] for item in transforms} == {
        "decode_historical_tool_arguments", "join_reasoning_segments_with_newlines"
    }


def test_k3_literal_controls_preserve_independent_segment_permissions() -> None:
    observed = reference.reference_result(STORE, model_named("kimi_k3"), case_named("kimi_k3", "kimi-literal-controls"))
    result = observed["result"]
    assert result["kind"] == "prompt"
    assert result["pending_segments"] == 3
    assert any(segment["text"] == "literal <|open|>think<|sep|> text" and not segment["allow_special"]
               for segment in result["segments"])
    assert any(segment["text"] == "<|open|>" and segment["allow_special"] for segment in result["segments"])


def test_reference_execution_leaves_vendored_assets_unchanged() -> None:
    assets = STORE / "assets"
    before = {str(path.relative_to(assets)): reference.sha256(path.read_bytes())
              for path in assets.rglob("*") if path.is_file()}
    reference._ENCODERS.clear()
    reference.reference_result(STORE, model_named("kimi_k3"), case_named("kimi_k3", "conversation-single"))
    after = {str(path.relative_to(assets)): reference.sha256(path.read_bytes())
             for path in assets.rglob("*") if path.is_file()}
    assert before == after


def test_k3_partial_is_explicitly_unsupported_by_official_encoder() -> None:
    result = reference.reference_result(STORE, model_named("kimi_k3"), case_named("kimi_k3", "kimi-partial-assistant"))
    assert result["result"]["kind"] == "unsupported"
    assert "no Partial Mode" in result["result"]["reason"]
    assert "reference_input" in result and "metadata_origin" in result


def test_reversed_k3_images_have_authored_media_order_not_copied_metadata() -> None:
    observed = reference.reference_result(STORE, model_named("kimi_k3"), case_named("kimi_k3", "images-reversed-tool-results"))
    assert observed["result"]["media_message_order"] == [0, 1, 3, 2]
    assert observed["reference_input"]["kwargs"]["image_prompts"] == ["<|media_pad|>", "<|media_pad|>"]
    assert "authored-contract" in observed["metadata_origin"]


def test_capture_output_cannot_overwrite_sources(tmp_path: Path) -> None:
    for output in (ROOT, STORE, STORE / "assets", STORE / "families"):
        with pytest.raises(ValueError, match="separate"):
            reference.output_directory(ROOT, output)
    assert reference.output_directory(ROOT, tmp_path) == tmp_path.resolve()


def test_atomic_json_preserves_unicode(tmp_path: Path) -> None:
    target = tmp_path / "capture.json"
    reference.atomic_json(target, {"text": "東京 中文 🦀"})
    assert "東京" in target.read_text()
    assert reference.load_json(target) == {"text": "東京 中文 🦀"}


def test_atomic_json_rejects_unpaired_surrogate_but_accepts_a_json_pair(tmp_path: Path) -> None:
    target = tmp_path / "capture.json"
    with pytest.raises(UnicodeEncodeError):
        reference.atomic_json(target, {"text": "\ud800"})
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
    paired = json.loads('{"text":"\\ud83e\\udd80"}')
    reference.atomic_json(target, paired)
    assert reference.load_json(target) == {"text": "🦀"}


def test_deepseek_medium_effort_records_pinned_template_authority() -> None:
    observed = reference.reference_result(STORE, model_named("deepseek_v41"), case_named("deepseek_v41", "controls-effort-medium"))
    assert observed["reference_input"]["kwargs"]["reasoning_effort"] == 62
    transform = next(item for item in observed["transformations"] if item["name"] == "map_named_reasoning_effort_to_numeric")
    assert "medium=62" in transform["authority"]
    assert "314113f8a7dadb388215396ba5fbee2c610a04d6/chat_template.jinja#L19-L21" in transform["authority"]


def test_deepseek_thinking_alias_records_pinned_template_authority() -> None:
    observed = reference.reference_result(STORE, model_named("deepseek_v41"), case_named("deepseek_v41", "controls-thinking-disabled"))
    assert observed["reference_input"]["kwargs"]["thinking_mode"] == "chat"
    transform = next(item for item in observed["transformations"] if item["name"] == "resolve_enable_thinking_as_thinking_mode")
    assert "pinned official template enable_thinking alias" in transform["authority"]
    assert "314113f8a7dadb388215396ba5fbee2c610a04d6/chat_template.jinja#L8-L10" in transform["authority"]


def test_deepseek_precedence_labels_authored_api_rule_separately() -> None:
    observed = reference.reference_result(STORE, model_named("deepseek_v41"), case_named("deepseek_v41", "controls-effort-precedence"))
    assert observed["reference_input"]["kwargs"]["reasoning_effort"] == 62
    transform = next(item for item in observed["transformations"] if item["name"] == "top_level_reasoning_effort_precedes_template")
    assert "independently-authored renderer API contract" in transform["authority"]
    assert "top-level" in transform["authority"]


def test_conflicting_effort_case_applies_to_every_formatter() -> None:
    descriptions = set()
    for model in MANIFEST["models"]:
        case = case_named(model["id"], "controls-effort-precedence")
        descriptions.add(case["description"])
        assert case["group"] == "controls"
        assert case["request"]["reasoning_effort"] == "medium"
        assert list(case["settings"]["chat_template_args"].values()) == ["low"]
    assert descriptions == {"Resolve conflicting top-level and template reasoning effort."}


def test_unsupported_content_format_retains_metadata_authority() -> None:
    case = copy.deepcopy(case_named("glm53", "conversation-single"))
    case["settings"]["content_format"] = "string"
    observed = reference.reference_result(STORE, model_named("glm53"), case)
    assert observed["result"]["kind"] == "unsupported"
    assert observed["metadata_origin"] == case["metadata_origin"]


def test_all_independent_reference_results_are_reproducible() -> None:
    require_template_runtime()
    for model in MANIFEST["models"]:
        observed = reference.capture_model(STORE, MANIFEST_BYTES, model)
        recorded = reference.load_json(STORE / "families" / model["id"] / "reference.json")
        assert observed["results"] == recorded["results"]


def test_reference_source_damage_is_not_captured_as_request_error(tmp_path: Path) -> None:
    model = model_named("kimi_k3")
    for asset in model["assets"]:
        destination = tmp_path / asset["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(STORE / asset["path"], destination)
    encoder = next(asset for asset in model["assets"] if asset["role"] == "encoder")
    (tmp_path / encoder["path"]).write_text("damaged")
    with pytest.raises(ValueError, match="hash mismatch"):
        reference.reference_result(tmp_path, model, case_named("kimi_k3", "conversation-single"))
