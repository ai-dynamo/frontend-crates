import copy
import importlib.util
import sys
from pathlib import Path


def _load_table_module():
    path = Path(__file__).parents[1] / "src" / "generate_conformance_table.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("generate_conformance_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_incomplete_tool_deltas_are_not_final_unified_events():
    module = _load_table_module()
    rows = [[{
        "kind": "tool_call",
        "name": "get_weather",
        "arguments": '{"city":"Par',
        "complete": False,
    }], []]

    assert module._assemble_stream(rows) == []


def test_complete_tool_delta_keeps_prior_argument_fragments():
    module = _load_table_module()
    rows = [
        [{
            "kind": "tool_call",
            "name": "get_weather",
            "arguments": '{"city":"Par',
            "complete": False,
        }],
        [{
            "kind": "tool_call",
            "name": None,
            "arguments": 'is"}',
            "complete": True,
        }],
    ]

    assert module._assemble_stream(rows) == [{
        "kind": "tool_call",
        "name": "get_weather",
        "arguments": {"city": "Paris"},
    }]


def test_dynamo_malformed_arguments_use_p3_without_hiding_peer_bytes():
    module = _load_table_module()
    for raw in ('{"x":"unfinished', '{', '[invalid'):
        for fragments in ([raw], [raw[:2], raw[2:]]):
            rows = [[{
                "kind": "tool_call", "name": "bad" if i == 0 else None,
                "arguments": fragment, "complete": i == len(fragments) - 1,
            }] for i, fragment in enumerate(fragments)]
            rows.append([
                {"kind": "tool_call", "name": "echo", "arguments": '{"value":"é"}', "complete": True},
                {"kind": "tool_call", "name": "echo", "arguments": '{"value":"Café"}', "complete": True},
            ])
            expected = [
                {"kind": "tool_call", "name": "bad", "arguments": {}},
                {"kind": "tool_call", "name": "echo", "arguments": {"value": "é"}},
                {"kind": "tool_call", "name": "echo", "arguments": {"value": "Café"}},
            ]
            assert module._assemble_stream(rows, recorded_assembly=expected) == expected
            expected[0]["arguments"] = raw
            assert module._assemble_stream(rows) == expected


def test_dynamo_preserves_valid_literal_string_fallback():
    module = _load_table_module()
    rows = [[{"kind": "tool_call", "name": "bad", "arguments": '{"value":"unfinished"}', "complete": True}]]
    assert module._assemble_stream(rows) == [
        {"kind": "tool_call", "name": "bad", "arguments": {"value": "unfinished"}},
    ]


def test_recorded_fallback_requires_matching_call_sequence():
    module = _load_table_module()
    rows = [[
        {"kind": "tool_call", "name": "bad", "arguments": '{"x":"unfinished', "complete": True},
        {"kind": "text", "text": "between"},
        {"kind": "tool_call", "name": "echo", "arguments": '{"value":"é"}', "complete": True},
    ]]
    raw = module._assemble_stream(rows)
    assert module._assemble_stream(rows, recorded_assembly=raw) == raw
    native = copy.deepcopy(raw)
    native[0]["arguments"] = {}
    assert module._assemble_stream(rows, recorded_assembly=native) == native
    for mismatch in (native[:1], list(reversed(native)), [native[0], native[1], {**native[2], "arguments": {"value": "different"}}]):
        assert module._assemble_stream(rows, recorded_assembly=mismatch) == raw
    batch_order = [native[1], native[0], native[2]]
    assert module._assemble_stream(rows, recorded_assembly=batch_order) == native
