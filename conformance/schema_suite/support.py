# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real grammar-compiler and conformance-producer checks for groups G/H."""

import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch
from cases import j, k3_arg, k3_wire, root

SRC = Path(__file__).resolve().parents[1] / "utils" / "src"


def grammar_cases(group):
    out = []

    def add(label, field, samples, **kw):
        schema = kw.pop(
            "schema",
            {
                "type": "object",
                "properties": {"x": field},
                "required": ["x"],
                "additionalProperties": False,
            },
        )
        out.append(
            {
                "id": group + "." + label,
                "group": group,
                "family": "kimi_k3",
                "choice": "required",
                "schema_mode": "strict",
                "parallel": False,
                "schema": schema,
                "samples": [
                    {
                        "input": "<|close|>response<|sep|>"
                        + k3_wire(k3_arg("x", raw, ty)),
                        "expected": want,
                        "label": str(i),
                    }
                    for i, (raw, ty, want) in enumerate(samples)
                ],
                **kw,
            }
        )

    if group == "G01":
        for label, field in [
            ("list", {"type": ["string", "null"]}),
            ("anyof", {"anyOf": [{"type": "string"}, {"type": "null"}]}),
        ]:
            add(
                label,
                field,
                [
                    ("hello", "string", True),
                    ("null", "null", True),
                    ("42", "number", False),
                ],
            )
            out[-1]["samples"].append(
                {
                    "input": "<|close|>response<|sep|>"
                    + k3_wire(k3_arg("unknown", "hello")),
                    "expected": False,
                    "label": "unknown_key",
                }
            )
    elif group == "G02":
        for kw in ["enum", "const"]:
            add(
                kw,
                {"type": ["integer", "null"], kw: [1, None] if kw == "enum" else 1},
                [
                    ("1", "number", True),
                    ("42", "number", False),
                    ("null", "null", kw == "enum"),
                ],
            )
    elif group == "G03":
        for kw in ["enum", "const"]:
            f = {"type": "string", kw: ["safe"] if kw == "enum" else "safe"}
            add(kw, f, [("safe", "string", True), ("unsafe", "string", False)])
            add(
                kw + "_ref",
                None,
                [("safe", "string", True), ("unsafe", "string", False)],
                schema={
                    "$defs": {"S": f},
                    "properties": {"x": {"$ref": "#/$defs/S"}},
                    "required": ["x"],
                },
            )
            add(
                kw + "_union",
                {"anyOf": [f, {"type": "null"}]},
                [
                    ("safe", "string", True),
                    ("unsafe", "string", False),
                    ("null", "null", True),
                ],
            )
    elif group == "G04":
        add(
            "length",
            {"type": "string", "minLength": 2, "maxLength": 3},
            [
                ("ab", "string", True),
                ("a", "string", False),
                ("abcd", "string", False),
                ("猫猫", "string", True),
            ],
        )
        add("empty", {"type": "string", "maxLength": 3}, [("", "string", True)])
        add(
            "pattern",
            {"type": "string", "pattern": "^ab+$"},
            [("abb", "string", True), ("ac", "string", False)],
        )
    elif group == "G05":
        add(
            "escaped_dollar",
            {"type": "string", "pattern": r"^price\$"},
            [("price$", "string", True), ("priceX", "string", False)],
        )
        add(
            "escaped_backslash",
            {"type": "string", "pattern": r"^path\\$"},
            [("path\\", "string", True), ("path", "string", False)],
        )
    elif group == "G06":
        add(
            "bounds",
            {"type": "integer", "minimum": 2, "maximum": 8, "multipleOf": 2},
            [
                ("4", "number", True),
                ("1", "number", False),
                ("3", "number", False),
                ("10", "number", False),
            ],
        )
        add(
            "nested",
            {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 2,
                    }
                },
                "required": ["a"],
                "additionalProperties": False,
            },
            [
                ('{"a":[1]}', "object", True),
                ('{"a":[]}', "object", False),
                ('{"a":["1"]}', "object", False),
                ('{"b":1}', "object", False),
            ],
        )
        add(
            "oneof_overlap",
            {"oneOf": [{"type": "integer"}, {"type": "number"}]},
            [("7", "number", False)],
        )
    elif group == "G07":
        add("required", {"type": "string"}, [("yes", "string", True)])
        out[-1]["samples"] += [
            {
                "input": "<|close|>response<|sep|>" + k3_wire(""),
                "expected": False,
                "label": "required_omitted",
            },
            {
                "input": "<|close|>response<|sep|>" + k3_wire(k3_arg("y", "yes")),
                "expected": False,
                "label": "unknown",
            },
        ]
        add(
            "optional",
            {"type": "string"},
            [],
            schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": False,
            },
        )
        out[-1]["samples"] = [
            {
                "input": "<|close|>response<|sep|>" + k3_wire(""),
                "expected": True,
                "label": "omitted",
            }
        ]
        add(
            "zero_fields",
            None,
            [],
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        )
        out[-1]["samples"] = [
            {
                "input": "<|close|>response<|sep|>" + k3_wire(""),
                "expected": True,
                "label": "empty",
            },
            {
                "input": "<|close|>response<|sep|>" + k3_wire(k3_arg("x", "oops")),
                "expected": False,
                "label": "extra",
            },
        ]
    elif group == "G08":
        add(
            "object_ref",
            None,
            [
                ('{"name":"Ada"}', "object", True),
                ('"{\\"name\\":\\"Ada\\"}"', "object", False),
                ("wrong", "string", False),
            ],
            schema={
                "$defs": {
                    "O": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    }
                },
                "properties": {"x": {"$ref": "#/$defs/O"}},
                "required": ["x"],
            },
        )
        add(
            "deep_enum",
            None,
            [("safe", "string", True), ("wrong", "string", False)],
            schema={
                "$defs": {
                    "A": {"$ref": "#/$defs/B"},
                    "B": {"type": "string", "enum": ["safe"]},
                },
                "properties": {"x": {"$ref": "#/$defs/A"}},
                "required": ["x"],
            },
        )
        for key, ref in [("a b", "#/$defs/a%20b"), ("a/b~c", "#/$defs/a~1b~0c")]:
            add(
                "pointer_" + ref,
                None,
                [("safe", "string", True), ("wrong", "string", False)],
                schema={
                    "$defs": {key: {"type": "string", "enum": ["safe"]}},
                    "properties": {"x": {"$ref": ref}},
                    "required": ["x"],
                },
            )
    elif group == "G09":
        add(
            "recursive",
            None,
            [('{"child":{}}', "object", True), ("text", "string", False)],
            schema={
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {"child": {"$ref": "#/$defs/node"}},
                    }
                },
                "properties": {
                    "x": {
                        "type": "object",
                        "properties": {"child": {"$ref": "#/$defs/node"}},
                    }
                },
            },
        )
        add(
            "property_ref",
            None,
            [('{"child":{}}', "object", True)],
            schema={
                "type": "object",
                "properties": {
                    "node": {"type": "object"},
                    "x": {
                        "type": "object",
                        "properties": {"child": {"$ref": "#/properties/node"}},
                    },
                },
            },
        )
    elif group == "G10":
        for kw in ["const", "enum", "default", "examples"]:
            v = {"$ref": "literal"}
            add(
                kw,
                {"type": "object", kw: [v] if kw in ["enum", "examples"] else v},
                [
                    (j(v), "object", True),
                    ("{}", "object", kw in ["default", "examples"]),
                ],
            )
    elif group == "G11":
        for family in ["kimi_k3", "generic"]:
            for strict in [None, True, False]:
                for mode in ["auto", "strict"]:
                    enforce = mode == "strict" or (
                        strict is not False if family == "kimi_k3" else strict is True
                    )
                    add(
                        f"{family}_{strict}_{mode}",
                        {"type": "integer"},
                        [("42", "number", True), ("wrong", "string", not enforce)],
                        family=family,
                        strict=strict,
                        schema_mode=mode,
                    )
                    if family == "generic":
                        out[-1]["samples"] = [
                            {
                                "input": "<tool_call>\n<function=f>\n<parameter=x>"
                                + v
                                + "</parameter>\n</function>\n</tool_call>",
                                "expected": w,
                                "label": str(i),
                            }
                            for i, (v, w) in enumerate(
                                [("42", True), ("wrong", not enforce)]
                            )
                        ]
    elif group == "G12":
        for choice in ["auto", "required", "named"]:
            for parallel in [False, True]:
                add(
                    choice + str(parallel),
                    {"type": "string"},
                    [("ok", "string", True)],
                    choice=choice,
                    parallel=parallel,
                )
                one = k3_wire(k3_arg("x", "ok"))
                call = one.removeprefix("<|open|>tools<|sep|>").removesuffix(
                    "<|close|>tools<|sep|>"
                )
                out[-1]["samples"] += [
                    {
                        "input": "<|close|>response<|sep|><|open|>tools<|sep|>"
                        + call
                        + call.replace('index="1"', 'index="2"')
                        + "<|close|>tools<|sep|>",
                        "expected": parallel,
                        "label": "two_calls",
                    },
                    {
                        "input": "hello<|close|>response<|sep|>",
                        "expected": choice == "auto",
                        "label": "plain_response",
                    },
                    {
                        "input": "<|close|>response<|sep|>"
                        + one.replace('tool="f"', 'tool="other"'),
                        "expected": False,
                        "label": "unknown_tool",
                    },
                ]
        add("duplicates", {"type": "string"}, [])
        out[-1]["samples"] = [
            {
                "input": "<|close|>response<|sep|>"
                + k3_wire(k3_arg("x", "first") + k3_arg("x", "second")),
                "expected": False,
                "label": "duplicate_required",
            }
        ]
    elif group == "G13":
        for field in [
            {"$ref": "https://example.invalid/schema"},
            {"$id": "https://example.invalid/", "type": "object", "$ref": "#other"},
            {"anyOf": [{}, {}]},
        ]:
            add(
                str(len(out)),
                field,
                [("fallback", "string", True)],
                contract="compatibility",
            )
        defs = {f"D{i}": {"$ref": f"#/$defs/D{i+1}"} for i in range(64)}
        defs["D64"] = {"type": "object"}
        add(
            "bounded_chain",
            None,
            [("fallback", "string", True)],
            schema={"$defs": defs, "properties": {"x": {"$ref": "#/$defs/D0"}}},
            contract="compatibility",
        )
    elif group == "G14":
        for ty, value in [
            ("string", "hello"),
            ("number", 42),
            ("boolean", False),
            ("null", None),
            ("object", {"a": 1}),
            ("array", [1, None]),
        ]:
            add(
                ty,
                {"type": ty},
                [(value if ty == "string" else j(value), ty, True)],
                parse_expected={"x": value},
            )
    else:
        raise ValueError(group)
    return out


def grammar_check(payload):
    import xgrammar as xgr
    from xgrammar.testing import _is_grammar_accept_string

    out = []
    for c in payload["cases"]:
        row = {
            "id": c["id"],
            "group": payload["group"],
            "family": c["family"],
            "surface": "grammar",
            "contract": c.get("contract", "strict_guidance"),
            "tools": [
                {"name": "f", "parameters": c["schema"], "strict": c.get("strict")}
            ],
            "expected": c["samples"],
            "grammar": c.get("grammar"),
        }
        try:
            if c.get("build_error"):
                raise RuntimeError(c["build_error"])
            grammar = xgr.Grammar.from_structural_tag(c["grammar"])
            actual = []
            for s in c["samples"]:
                # _is_grammar_accept_string tests complete recognition (not prefix acceptance).
                got = _is_grammar_accept_string(grammar, s["input"])
                actual.append(
                    {
                        "label": s["label"],
                        "input": s["input"],
                        "expected": s["expected"],
                        "actual": got,
                    }
                )
            row.update(
                status="pass"
                if all(v["actual"] == v["expected"] for v in actual)
                else "fail",
                checks=len(actual),
                failed_checks=sum(v["actual"] != v["expected"] for v in actual),
                actual=actual,
            )
            if "parse_expected" in c:
                row.update(
                    parse_expected=c["parse_expected"],
                    parse_input=c["samples"][0]["input"],
                )
        except Exception as e:
            row.update(
                status="fail",
                failure_kind="grammar_compile_error",
                reason=f"{type(e).__name__}: {e}",
                checks=1,
                failed_checks=1,
            )
        out.append(row)
    return out


def harness_check(payload):
    sys.path.insert(0, str(SRC))
    import yaml
    import gen_unified_golden as G
    import capture_stimulus as S

    group = payload["group"]
    row = {
        "id": group + ".harness",
        "group": group,
        "family": "harness",
        "surface": "conformance",
        "contract": "harness",
        "checks": 1,
    }
    try:
        if group in ["H01", "H02", "H03"]:
            source = copy.deepcopy(next(iter(G.build_cases("glm47").values())))
            schema = {
                "type": "object",
                "$defs": {"S": {"type": ["string", "null"]}},
                "properties": {"x": {"$ref": "#/$defs/S"}},
                "required": ["x"],
            }
            ts = [
                {
                    "name": "f",
                    "description": None,
                    "parameters": schema,
                    "strict": False,
                }
            ]
            if group == "H02":
                ts = []
            source["tools"] = ts
            source["input"] = (
                "<tool_call>f<arg_key>x</arg_key><arg_value>null</arg_value></tool_call>"
            )
            source["golden"] = [
                {"kind": "tool_call", "name": "f", "arguments": {"x": None}}
            ]
            # Replace authored input only; exercise the real production YAML emitter.
            with patch.object(
                G, "build_cases", return_value={"UNIFIED.schema_probe.glm47": source}
            ):
                emitted = G.emit_yaml("glm47")
            actual = yaml.safe_load(emitted)["cases"]["UNIFIED.schema_probe.glm47"]
            row.update(
                input=source,
                expected={"tools": ts},
                actual=actual,
                emitted_yaml=emitted,
            )
            assert (
                actual.get("tools") == ts
            ), "Production Unified YAML emitter dropped per-case tools"
            if group == "H01":
                other = copy.deepcopy(source)
                other["tools"][0]["parameters"] = root({"type": "string"})
                with patch.object(
                    G, "build_cases", return_value={"UNIFIED.schema_probe.glm47": other}
                ):
                    second = yaml.safe_load(G.emit_yaml("glm47"))["cases"][
                        "UNIFIED.schema_probe.glm47"
                    ]
                assert (
                    actual["tools"] != second["tools"]
                ), "Different schemas collapsed to one inventory"
            if group == "H03":
                row["rust_roundtrip_tools"] = ts
        elif group == "H04":
            current = {
                "input": "same",
                "init": {},
                "tools": [{"name": "f", "parameters": root({"type": "string"})}],
                "chunks": [],
            }
            captured = {"capture_input": S.capture_input(current)}
            assert S.comparison_failure(captured, current, b"", "case.yaml", {}) is None
            wrong = copy.deepcopy(current)
            wrong["tools"][0]["parameters"] = root({"type": "integer"})
            reason = S.comparison_failure(captured, wrong, b"", "case.yaml", {})
            row.update(expected="reject mismatched tools", actual=reason)
            assert reason and "tools" in reason, "Mismatched tool schema was accepted"
        elif group == "H06":
            current = {"input": "old", "init": {}, "tools": [], "chunks": []}
            captured = {"capture_input": S.capture_input(current), "assembled": []}
            before = copy.deepcopy(captured)
            changed = {**current, "input": "new"}
            reason = S.comparison_failure(captured, changed, b"", "case.yaml", {})
            assert reason and "input" in reason
            assert captured == before, "Historical record was changed"
            row.update(
                expected="reject stale input without mutating capture", actual=reason
            )
        else:
            raise ValueError("Rust-owned group " + group)
        row["status"] = "pass"
    except AssertionError as e:
        row.update(status="fail", reason=str(e), failed_checks=1)
    except Exception as e:
        row.update(status="error", reason=f"{type(e).__name__}: {e}")
    return [row]


if __name__ == "__main__":
    if sys.argv[1] == "grammar_cases":
        print(j(grammar_cases(sys.argv[2])))
    else:
        payload = json.load(sys.stdin)
        print(
            j(
                grammar_check(payload)
                if payload["group"].startswith("G")
                else harness_check(payload)
            )
        )
