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
