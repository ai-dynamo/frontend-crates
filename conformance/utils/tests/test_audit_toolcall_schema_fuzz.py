# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Planted defects must fail the independent schema-fuzz audit."""

from __future__ import annotations

import copy
import hashlib
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")

from jsonschema.exceptions import SchemaError, ValidationError

from conformance.utils.src import audit_toolcall_schema_fuzz as audit

pytestmark = []


def entry(
    arguments: dict[str, Any], value_schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "key": "test-shape",
        "name": "record",
        "parameters": {
            "type": "object",
            "properties": {"value": value_schema or {}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "args": [arguments, copy.deepcopy(arguments)],
    }


def case(
    arguments: dict[str, Any], value_schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    return next(audit.cases([entry(arguments, value_schema)]))


def capture(request_case: dict[str, Any], raw: Any | None = None) -> dict[str, Any]:
    raw = audit.json_text(request_case["expected"]) if raw is None else raw
    rows = [{"expected": []} for _ in request_case["request"]["chunks"]]
    rows.append(
        {
            "expected": [
                {
                    "kind": "tool_call",
                    "name": request_case["name"],
                    "arguments": raw,
                    "complete": True,
                    "tool_index": 0,
                }
            ]
        }
    )
    return {
        "assembled": [
            {"kind": "tool_call", "name": request_case["name"], "arguments": raw}
        ],
        "chunks": rows,
    }


def corpus(
    directory: Path, entries: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    entries = [entry({"value": 7}, {"type": "integer"})] if entries is None else entries
    raw = audit.json_text({"version": 6, "entries": entries}).encode("utf-8")
    manifest = {
        "dataset_file": "wordings.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_version": 6,
        "expected_entry_count": len(entries),
        "expected_argument_count": sum(len(item["args"]) for item in entries),
    }
    (directory / "wordings.json").write_bytes(raw)
    audit.write_json(directory / "manifest.json", manifest)
    return manifest


@pytest.mark.parametrize(
    "kind,expected,actual",
    [
        ("integer", 7, "7"),
        ("string", "7", 7),
        ("boolean", True, 1),
        ("integer", 1, True),
        ("null", None, "null"),
        ("number", Decimal("1.25"), "1.25"),
    ],
)
def test_wrong_scalar_type_fails(kind: str, expected: Any, actual: Any) -> None:
    request_case = case({"value": expected}, {"type": kind})
    result = audit.score(
        request_case, capture(request_case, audit.json_text({"value": actual}))
    )
    assert set(result["failures"]) == {"argument_mismatch", "schema_violation"}
    assert result["different_paths"] == ["/value"]
    assert not result["schema_valid"]


def test_schema_valid_value_change_still_fails() -> None:
    request_case = case({"value": 7}, {"type": "integer"})
    result = audit.score(request_case, capture(request_case, '{"value":8}'))
    assert result["failures"] == ["argument_mismatch"]
    assert result["schema_valid"]
    assert result["labels"] == []


def test_schema_valid_wire_ambiguity_remains_a_failure() -> None:
    request_case = case(
        {"value": 30}, {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    )
    result = audit.score(request_case, capture(request_case, '{"value":"30"}'))
    assert result["failures"] == ["argument_mismatch"]
    assert result["schema_valid"]
    assert result["labels"] == ["demonstrated_wire_ambiguity"]


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_extra_calls_fail(count: int) -> None:
    request_case = case({"value": 7})
    observed = capture(request_case)
    observed["assembled"] *= count
    assert "call_count" in audit.score(request_case, observed)["failures"]


def test_wrong_tool_name_fails() -> None:
    request_case = case({"value": 7})
    observed = capture(request_case)
    observed["assembled"][0]["name"] = "other"
    observed["chunks"][-1]["expected"][0]["name"] = "other"
    assert audit.score(request_case, observed)["failures"] == ["wrong_tool"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"value":',
        '{"value":1,"value":2}',
        '{"value":{"x":1,"x":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        {"value": 7},
    ],
)
def test_malformed_or_non_raw_arguments_fail(raw: Any) -> None:
    request_case = case({"value": 7})
    result = audit.score(request_case, capture(request_case, raw))
    assert result["failures"] == ["invalid_json"]
    assert "error" in result


def test_missing_completion_and_finish_row_fail_independently() -> None:
    request_case = case({"value": 7})
    observed = capture(request_case)
    observed["chunks"][-1]["expected"][0]["complete"] = False
    assert audit.score(request_case, observed)["failures"] == ["completion_count"]
    observed = capture(request_case)
    observed["chunks"].pop(0)
    assert audit.score(request_case, observed)["failures"] == ["missing_finish_row"]


def test_multiple_completions_and_unexpected_events_fail() -> None:
    request_case = case({"value": 7})
    observed = capture(request_case)
    observed["chunks"][-1]["expected"].append(
        {"kind": "tool_call", "arguments": "", "complete": True, "tool_index": 0}
    )
    observed["assembled"].insert(0, {"kind": "text", "text": "leaked markup"})
    assert set(audit.score(request_case, observed)["failures"]) == {
        "unexpected_event",
        "completion_count",
    }


def test_completion_before_final_argument_fragment_fails_with_valid_raw_json() -> None:
    request_case = case({"value": 7})
    observed = capture(request_case)
    observed["chunks"][0]["expected"] = [
        {
            "kind": "tool_call",
            "name": "record",
            "arguments": '{"value":',
            "complete": True,
            "tool_index": 0,
        }
    ]
    observed["chunks"][-1]["expected"] = [
        {
            "kind": "tool_call",
            "name": None,
            "arguments": "7}",
            "complete": False,
            "tool_index": 0,
        }
    ]
    result = audit.score(request_case, observed)
    assert result["failures"] == ["completion_before_end"]
    assert result["actual_arguments"] == request_case["expected"]
    assert result["schema_valid"]


def test_raw_only_unexpected_event_cannot_hide_behind_assembled_projection() -> None:
    request_case = case({"value": 7})
    observed = capture(request_case)
    observed["chunks"][0]["expected"] = [{"kind": "text", "text": "leaked markup"}]
    assert audit.score(request_case, observed)["failures"] == ["unexpected_event"]


def test_invalid_raw_delta_arguments_cannot_hide_behind_valid_assembled_arguments() -> (
    None
):
    request_case = case({"value": 7})
    observed = capture(request_case)
    observed["chunks"][-1]["expected"][0]["arguments"] = "THIS IS NOT JSON"
    result = audit.score(request_case, observed)
    assert result["failures"]
    assert result["actual_arguments_raw"] == "THIS IS NOT JSON"


def test_raw_delta_fragments_are_scored_independently_of_assembled_arguments() -> None:
    request_case = case({"value": 7})
    observed = capture(request_case)
    observed["chunks"][0]["expected"] = [
        {
            "kind": "tool_call",
            "name": "record",
            "arguments": '{"value":',
            "complete": False,
            "tool_index": 0,
        }
    ]
    observed["chunks"][-1]["expected"] = [
        {"kind": "tool_call", "arguments": "8}", "complete": True, "tool_index": 0}
    ]
    result = audit.score(request_case, observed)
    assert "argument_mismatch" in result["failures"]
    assert result["actual_arguments"] == {"value": 8}
    assert result["different_paths"] == ["/value"]


@pytest.mark.parametrize("failure", ["error", "unavailable"])
def test_parser_failures_cannot_pass(failure: str) -> None:
    result = audit.score(case({"value": 7}), {failure: "unsupported"})
    assert result["failures"] == [
        "parser_error" if failure == "error" else "unavailable"
    ]


def test_exact_decimal_json_round_trip_and_large_numeric_comparison(
    tmp_path: Path,
) -> None:
    raw = '{"large":9007199254740993,"decimal":9007199254740993.0000000000000001,"tiny":1e-500,"unicode":"你好"}'
    value = audit.load_json(raw)
    assert value["large"] == 9007199254740993
    assert value["decimal"] == Decimal("9007199254740993.0000000000000001")
    assert value["tiny"] == Decimal("1e-500")
    output = tmp_path / "roundtrip.json"
    audit.write_json(output, value)
    assert audit.differences(value, audit.load_json(output.read_text())) == []
    assert audit.differences(9007199254740993, Decimal("9007199254740993.0")) == []
    assert audit.differences(9007199254740993, 9007199254740992) == [""]
    assert audit.differences(
        value["decimal"], Decimal("9007199254740993.0000000000000002")
    ) == [""]


@pytest.mark.parametrize(
    "value", [Decimal("NaN"), Decimal("Infinity"), float("nan"), float("inf")]
)
def test_serialization_rejects_non_json_numbers(value: Any) -> None:
    with pytest.raises(ValueError):
        audit.json_text(value)


def test_structural_differences_keep_json_pointer_paths_and_boolean_types() -> None:
    assert audit.differences(
        {"a/b": {"~x": [True, 2]}, "missing": 1}, {"a/b": {"~x": [1, 2, 3]}, "extra": 1}
    ) == [
        "/extra",
        "/missing",
        "/a~1b/~0x/0",
        "/a~1b/~0x/2",
    ]
    assert audit.differences({"a": 1, "b": [Decimal("2.0")]}, {"b": [2], "a": 1}) == []


def test_dataset_preflight_accepts_valid_source(tmp_path: Path) -> None:
    manifest = corpus(tmp_path)
    loaded_manifest, entries = audit.load_dataset(tmp_path)
    assert loaded_manifest == manifest
    assert len(entries) == 1
    assert entries[0]["args"] == [{"value": 7}, {"value": 7}]


def test_dataset_preflight_rejects_checksum_change(tmp_path: Path) -> None:
    corpus(tmp_path)
    with (tmp_path / "wordings.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.load_dataset(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_version", 7),
        ("expected_entry_count", 2),
        ("expected_argument_count", 3),
    ],
)
def test_dataset_preflight_rejects_manifest_count_or_version_change(
    tmp_path: Path, field: str, value: int
) -> None:
    manifest = corpus(tmp_path)
    manifest[field] = value
    audit.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="mismatch"):
        audit.load_dataset(tmp_path)


def test_dataset_preflight_rejects_duplicate_entry_keys(tmp_path: Path) -> None:
    repeated = entry({"value": 7})
    corpus(tmp_path, [repeated, copy.deepcopy(repeated)])
    with pytest.raises(ValueError, match="duplicate or invalid dataset key"):
        audit.load_dataset(tmp_path)


def test_dataset_preflight_rejects_duplicate_json_keys_even_with_matching_checksum(
    tmp_path: Path,
) -> None:
    manifest = corpus(tmp_path)
    raw = (
        (tmp_path / "wordings.json")
        .read_bytes()
        .replace(b'"version":6', b'"version":6,"version":6', 1)
    )
    (tmp_path / "wordings.json").write_bytes(raw)
    manifest["sha256"] = hashlib.sha256(raw).hexdigest()
    audit.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit.load_dataset(tmp_path)


@pytest.mark.parametrize(
    "defect", ["schema", "arguments", "argument_pair", "object", "name"]
)
def test_dataset_preflight_rejects_bad_schemas_and_arguments(
    tmp_path: Path, defect: str
) -> None:
    source = entry({"value": 7}, {"type": "integer"})
    if defect == "schema":
        source["parameters"]["properties"]["value"] = {"type": "not-a-type"}
    elif defect == "arguments":
        source["args"][1]["value"] = "7"
    elif defect == "argument_pair":
        source["args"].pop()
    elif defect == "object":
        source["args"][1] = []
    else:
        source["name"] = ""
    corpus(tmp_path, [source])
    with pytest.raises((ValueError, TypeError, SchemaError, ValidationError)):
        audit.load_dataset(tmp_path)


def test_dataset_preflight_requires_pinned_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus(tmp_path)
    monkeypatch.setattr(audit.importlib.metadata, "version", lambda package: "0.0.0")
    with pytest.raises(ValueError, match="install jsonschema=="):
        audit.load_dataset(tmp_path)


def test_native_and_inline_json_encodings_preserve_values() -> None:
    arguments = {
        "text": "你好 🌟",
        "value": {"items": [None, {}, []]},
        "empty": {},
        "flag": False,
    }
    marker = "]<]minimax[>["
    expected_prefix = marker + "<tool_call>" + marker + '<invoke name="record">'
    expected_suffix = marker + "</invoke>" + marker + "</tool_call>"
    nested = (
        marker
        + "<text>你好 🌟"
        + marker
        + "</text>"
        + marker
        + "<value>"
        + marker
        + "<items>"
        + marker
        + "<item>null"
        + marker
        + "</item>"
        + marker
        + "<item>"
        + marker
        + "</item>"
        + marker
        + "<item>"
        + marker
        + "</item>"
        + marker
        + "</items>"
        + marker
        + "</value>"
        + marker
        + "<empty>"
        + marker
        + "</empty>"
        + marker
        + "<flag>false"
        + marker
        + "</flag>"
    )
    assert (
        audit.encode_call("record", arguments, "nested")
        == expected_prefix + nested + expected_suffix
    )
    inline = audit.encode_call("record", arguments, "json_values")
    assert marker + '<value>{"items":[null,{},[]]}' + marker + "</value>" in inline
    assert marker + "<empty>{}" + marker + "</empty>" in inline
    assert marker + "<text>你好 🌟" + marker + "</text>" in inline
    assert inline.startswith(expected_prefix) and inline.endswith(expected_suffix)
    assert audit.native_value(None) == "null"
    assert audit.native_value(Decimal("1.00000000000000001")) == "1.00000000000000001"


@pytest.mark.parametrize("schedule", audit.SCHEDULES)
def test_chunk_schedules_are_deterministic_and_preserve_unicode(schedule: str) -> None:
    text = audit.encode_call(
        "record", {"value": "你好 🌟", "nested": [None, {}]}, "nested"
    )
    chunks = audit.chunk_input(text, schedule, "stable-case")
    assert chunks == audit.chunk_input(text, schedule, "stable-case")
    assert "".join(chunks) == text
    assert all(chunks)
    if schedule == "characters":
        assert chunks == list(text)
        assert "🌟" in chunks
    if schedule == "markers":
        assert audit.NS + "<tool_call>" in chunks
        assert "你好 🌟" in chunks
    if schedule == "seeded":
        assert all(len(chunk) <= 17 for chunk in chunks)
        assert chunks != audit.chunk_input(text, schedule, "different-case")


def test_cases_cover_each_argument_encoding_and_schedule_once() -> None:
    source = entry({"value": {"text": "你好", "items": [None, {}]}})
    generated = list(audit.cases([source]))
    assert generated == list(audit.cases([source]))
    assert len(generated) == 16
    assert len({item["id"] for item in generated}) == 16
    assert {
        (item["argument_index"], item["encoding"], item["schedule"])
        for item in generated
    } == {
        (index, encoding, schedule)
        for index in range(2)
        for encoding in ("nested", "json_values")
        for schedule in audit.SCHEDULES
    }
    for item in generated:
        assert (
            "".join(chunk["delta_text"] for chunk in item["request"]["chunks"])
            == item["request"]["input"]
        )
        assert item["expected"] == source["args"][item["argument_index"]]
        assert audit.score(item, capture(item))["failures"] == []
    assert len(list(audit.cases([entry({"value": "scalar"})]))) == 8


def test_unrepresentable_parameter_names_are_labeled() -> None:
    request_case = case({"value": {">bad": 1, "/bad": 2, "": 3}})
    assert audit.unrepresentable_names(request_case["expected"]) == [
        "/value/>bad",
        "/value/~1bad",
        "/value/",
    ]
    assert audit.score(request_case, capture(request_case))["labels"] == [
        "unrepresentable_parameter_name"
    ]
    inline = next(
        item
        for item in audit.cases([entry(request_case["expected"])])
        if item["encoding"] == "json_values"
    )
    assert audit.score(inline, capture(inline))["labels"] == []
    root_invalid = case({">bad": 1})
    assert (
        "unrepresentable_parameter_name"
        in audit.score(root_invalid, capture(root_invalid))["labels"]
    )


@pytest.mark.parametrize("outcome", ["error", "no_call", "invalid_json"])
def test_unrepresentable_names_keep_labels_on_early_failures(outcome: str) -> None:
    request_case = case({"value": {">bad": 7}})
    observed = capture(request_case)
    if outcome == "error":
        observed = {"error": "parser could not consume malformed tag"}
    elif outcome == "no_call":
        observed["assembled"] = []
    else:
        observed["assembled"][0]["arguments"] = "{"
    result = audit.score(request_case, observed)
    assert result["failures"]
    assert "unrepresentable_parameter_name" in result["labels"]


@pytest.mark.parametrize("operation", ["encoding", "schedule"])
def test_unknown_encodings_or_schedules_fail(operation: str) -> None:
    with pytest.raises(ValueError, match="unknown"):
        if operation == "encoding":
            audit.encode_call("record", {"value": 7}, "unknown")
        else:
            audit.chunk_input("text", "unknown", "case")
