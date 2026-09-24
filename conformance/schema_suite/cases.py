# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored schema contracts. Encoding is test-only; expected values never use parser output."""

import copy
import html
import json

FAMILIES = [
    "kimi_k2",
    "kimi_k3",
    "deepseek_v3",
    "deepseek_v3_1",
    "deepseek_v3_2",
    "deepseek_v4",
    "deepseek_v41",
    "glm47",
    "minimax_m2",
    "minimax_m3",
    "qwen25",
    "qwen3_coder",
    "muse_glimmer",
    "gemma4",
    "harmony",
]
SCHEMA = ["glm47", "qwen3_coder", "minimax_m2", "minimax_m3"]
UNIFIED = [
    "kimi_k2",
    "kimi_k3",
    "deepseek_v4",
    "deepseek_v41",
    "glm47",
    "qwen3_coder",
    "muse_glimmer",
    "gemma4",
]
CAPABILITIES = {
    f: {
        "v1_batch": f not in ["deepseek_v41", "muse_glimmer"],
        "v1_jail": f not in ["deepseek_v41", "muse_glimmer"],
        "v2_tool": f not in ["deepseek_v3", "deepseek_v3_1", "deepseek_v3_2", "qwen25"],
        "unified": f in UNIFIED,
    }
    for f in FAMILIES
}
NS = "]<]minimax[>["


def j(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def kind(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "number"


def gemma(value):
    if isinstance(value, str):
        return '<|"|>' + value + '<|"|>'
    if isinstance(value, list):
        return "[" + ",".join(map(gemma, value)) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(k + ":" + gemma(v) for k, v in value.items()) + "}"
    return j(value)


def k3_arg(key, raw, ty="string"):
    return (
        f'<|open|>argument key="{key}" type="{ty}"<|sep|>{raw}<|close|>argument<|sep|>'
    )


def k3_wire(body, name="f", index=1):
    return f'<|open|>tools<|sep|><|open|>call tool="{name}" index="{index}"<|sep|>{body}<|close|>call<|sep|><|close|>tools<|sep|>'


def wire(family, pairs, name="f", raws=None, index=0):
    """Encode independently specified values; raw overrides only model text, never the oracle."""
    raws = raws or {}
    values = dict(pairs)

    def raw(k, v):
        return raws.get(k, v if isinstance(v, str) else j(v))

    if family == "kimi_k2":
        return f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{j(values)}<|tool_call_end|><|tool_calls_section_end|>"
    if family == "kimi_k3":
        return k3_wire(
            "".join(k3_arg(k, raw(k, v), kind(v)) for k, v in pairs), name, index + 1
        )
    if family in ["deepseek_v3", "deepseek_v3_1"]:
        if family == "deepseek_v3_1":
            return f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{j(values)}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        body = f"```json\n{j(values)}\n```"
        return f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n{body}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    if family in ["deepseek_v3_2", "deepseek_v4"]:
        block = "function_calls" if family == "deepseek_v3_2" else "tool_calls"
        body = "".join(
            f'<｜DSML｜parameter name="{k}" string="{str(isinstance(v,str)).lower()}">{raw(k,v)}</｜DSML｜parameter>'
            for k, v in pairs
        )
        return f'<｜DSML｜{block}><｜DSML｜invoke name="{name}">{body}</｜DSML｜invoke></｜DSML｜{block}>'
    if family == "deepseek_v41":
        return f'<｜DSML｜ calls><｜DSML｜ invoke name="{name}">{j(values)}</｜DSML｜ invoke></｜DSML｜ calls>'
    if family == "glm47":
        return (
            f"<tool_call>{name}"
            + "".join(
                f"<arg_key>{k}</arg_key><arg_value>{raw(k,v)}</arg_value>"
                for k, v in pairs
            )
            + "</tool_call>"
        )
    if family == "qwen3_coder":
        return (
            f"<tool_call><function={name}>"
            + "".join(
                f"<parameter={k}>{html.escape(raw(k,v),quote=False)}</parameter>"
                for k, v in pairs
            )
            + "</function></tool_call>"
        )
    if family == "minimax_m2":
        return (
            f'<minimax:tool_call><invoke name="{name}">'
            + "".join(
                f'<parameter name="{k}">{html.escape(raw(k,v),quote=False)}</parameter>'
                for k, v in pairs
            )
            + "</invoke></minimax:tool_call>"
        )
    if family == "minimax_m3":
        return (
            NS
            + f'<tool_call>{NS}<invoke name="{name}">'
            + "".join(
                f"{NS}<{k}>{html.escape(raw(k,v),quote=False)}{NS}</{k}>"
                for k, v in pairs
            )
            + NS
            + "</invoke>"
            + NS
            + "</tool_call>"
        )
    if family == "qwen25":
        return "<tool_call>" + j({"name": name, "arguments": values}) + "</tool_call>"
    if family == "muse_glimmer":
        return (
            f' to={name}<|message|><atem:function_calls><atem:invoke name="{name}">'
            + "".join(
                f'<atem:parameter name="{k}">{raws.get(k,j(v))}</atem:parameter>'
                for k, v in pairs
            )
            + "</atem:invoke></atem:function_calls><|eom|>"
        )
    if family == "gemma4":
        return (
            "<|tool_call>call:"
            + name
            + "{"
            + ",".join(k + ":" + raws.get(k, gemma(v)) for k, v in pairs)
            + "}<tool_call|>"
        )
    if family == "harmony":
        return f"<|start|>assistant<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{j(values)}<|call|>"
    raise ValueError(family)


def join_calls(family, first, second):
    if family == "muse_glimmer":
        return first + "<|start|>assistant" + second
    wrappers = {
        "kimi_k2": ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
        "kimi_k3": ("<|open|>tools<|sep|>", "<|close|>tools<|sep|>"),
        "deepseek_v3": ("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),
        "deepseek_v3_1": ("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),
        "deepseek_v3_2": ("<｜DSML｜function_calls>", "</｜DSML｜function_calls>"),
        "deepseek_v4": ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"),
        "deepseek_v41": ("<｜DSML｜ calls>", "</｜DSML｜ calls>"),
        "minimax_m2": ("<minimax:tool_call>", "</minimax:tool_call>"),
        "minimax_m3": (NS + "<tool_call>", NS + "</tool_call>"),
    }
    if family in wrappers:
        start, end = wrappers[family]
        return first.removesuffix(end) + second.removeprefix(start)
    return first + second


def schema_for(value):
    return {
        "type": "integer"
        if isinstance(value, int) and not isinstance(value, bool)
        else kind(value)
    }


def tool(schema, name="f", strict=None):
    return {"name": name, "description": None, "parameters": schema, "strict": strict}


def root(field):
    return {"type": "object", "properties": {"x": field}}


def build_cases():
    cases = []

    def add(
        group, label, family, value="42", field=None, raw=None, schema=None, **extra
    ):
        fields = [("x", value)]
        schema = copy.deepcopy(
            schema
            if schema is not None
            else root(field if field is not None else schema_for(value))
        )
        c = {
            "id": f"{group}.{label}.{family}",
            "group": group,
            "family": family,
            "tools": [tool(schema)],
            "input": wire(family, fields, raws={"x": raw} if raw is not None else None),
            "expected": [{"name": "f", "arguments": j(dict(fields))}],
            "strategies": ["whole", "chars", "seeded"],
            "contract": "regression",
        }
        c.update(extra)
        cases.append(c)
        return c

    def all_values(group, vals, families=FAMILIES):
        for label, value in vals:
            for f in families:
                add(group, label, f, value)

    all_values("A01", [("text", "hello"), ("empty", ""), ("whitespace", " \t\n")])
    all_values(
        "A02",
        [(str(i), v) for i, v in enumerate(["null", "true", "false", "42", "1.25"])],
    )
    all_values(
        "A03",
        [
            (str(i), v)
            for i, v in enumerate(["{}", '{"a":1}', "[]", "[1,true]", '"quoted"'])
        ],
    )
    all_values("A04", [("escapes", '  a\t"b"\\c\r\n')])
    all_values("A05", [("unicode", "café 猫 e\u0301 🐈")])
    all_values("A06", [(str(v), v) for v in [0, -1, 42]])
    all_values("A07", [("fraction", -1.25), ("integral_float", 42.0)])
    for f in SCHEMA + ["kimi_k3", "deepseek_v4", "gemma4"]:
        add("A07", "exponent", f, 100, field={"type": "number"}, raw="1e2")
    for v in [
        9007199254740993,
        -9223372036854775808,
        9223372036854775807,
        9223372036854775808,
        18446744073709551615,
    ]:
        for f in FAMILIES:
            add("A08", str(v), f, v, raw_contains=[str(v)], contract="target")
    for f in FAMILIES:
        add(
            "A09",
            "beyond_u64",
            f,
            10**30 + 1,
            raw_contains=[str(10**30 + 1)],
            contract="target",
        )
    all_values("A10", [("true", True), ("false", False)])
    all_values("A11", [("null", None), ("string_null", "null")])
    all_values(
        "A12",
        [
            ("object", {}),
            ("nested", {"n": 42, "s": "42", "a": [None, True, {"x": 1}]}),
            ("array", []),
            ("mixed", [1, "1", False, None]),
        ],
    )
    for f in SCHEMA:
        for raw in ["1x", "NaN", "Infinity", "1e9999"]:
            add(
                "A13", raw, f, raw, field={"type": "number"}, raw=raw, contract="target"
            )
        for raw in ["+1", "01", "TRUE", "1", "yes", "None"]:
            # Explicit string is unambiguous even where untyped compatibility rules differ.
            add(
                "A14",
                raw,
                f,
                raw,
                field={"type": "string"},
                raw=raw,
                contract="compatibility",
            )
    for f in FAMILIES:
        c = add(
            "A15",
            "missing_null_empty",
            f,
            None,
            schema={
                "type": "object",
                "properties": {
                    "missing": {"type": "string"},
                    "n": {"type": "null"},
                    "s": {"type": "string"},
                },
            },
        )
        c.update(
            input=wire(f, [("n", None), ("s", "")]),
            expected=[{"name": "f", "arguments": j({"n": None, "s": ""})}],
        )
        add(
            "A16",
            "typed_json",
            f,
            {"s": "42", "n": 42},
            field={
                "type": "object",
                "properties": {"s": {"type": "integer"}, "n": {"type": "string"}},
            },
        )

    rows = [
        ("B01", "string", {"type": "string"}, "null", "null"),
        ("B02", "nullable", {"type": ["string", "null"]}, "null", None),
        (
            "B04",
            "nullable_extension",
            {"type": "string", "nullable": True},
            "null",
            None,
        ),
        (
            "B06",
            "ambiguous",
            {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "42",
            "42",
        ),
        (
            "B07",
            "enum_union",
            {"anyOf": [{"type": "integer", "minimum": 100}, {"enum": ["30"]}]},
            "30",
            "30",
        ),
        (
            "B08",
            "const_union",
            {"anyOf": [{"type": "integer", "minimum": 100}, {"const": "30"}]},
            "30",
            "30",
        ),
        ("B09", "unconstrained", {"anyOf": [{"type": "integer"}, {}]}, "30", "30"),
        (
            "B11",
            "intersection",
            {"allOf": [{"type": ["string", "integer"]}, {"type": "integer"}]},
            "7",
            7,
        ),
        (
            "B12",
            "sibling",
            {"type": "string", "anyOf": [{"type": "string"}, {"type": "null"}]},
            "null",
            "null",
        ),
        (
            "B13",
            "sibling_oneof",
            {
                "type": ["string", "null"],
                "oneOf": [{"type": "string"}, {"type": "integer"}],
            },
            "null",
            "null",
        ),
        (
            "B14",
            "number_integer",
            {"allOf": [{"type": "number"}, {"type": "integer"}]},
            "7",
            7,
        ),
        (
            "B15",
            "nested_annotation",
            {
                "allOf": [
                    {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    {"description": "x"},
                ]
            },
            "7",
            7,
        ),
        (
            "B17",
            "enum_narrows_null",
            {"type": ["string", "null"], "enum": ["null"]},
            "null",
            "null",
        ),
    ]
    for kw in ["anyOf", "oneOf"]:
        for rev in [False, True]:
            branches = [{"type": "string"}, {"type": "null"}][:: (-1 if rev else 1)]
            rows.append(("B03", kw + str(rev), {kw: branches}, "null", None))
        for ty, v in [("integer", 42), ("number", 1.25), ("boolean", False)]:
            for raw, want in [("null", None), (j(v), v)]:
                rows.append(
                    (
                        "B05",
                        kw + ty + raw,
                        {kw: [{"type": ty}, {"type": "null"}]},
                        raw,
                        want,
                    )
                )
        for raw, want in [("auto", "auto"), ("42", 42)]:
            rows.append(
                (
                    "B10",
                    kw + raw,
                    {kw: [{"const": "auto"}, {"type": "integer"}]},
                    raw,
                    want,
                )
            )
    for ty, v in [("integer", 42), ("number", 1.25), ("boolean", False)]:
        for raw, want in [("null", None), (j(v), v)]:
            rows.append(("B05", "list" + ty + raw, {"type": [ty, "null"]}, raw, want))
    for kw in ["enum", "const"]:
        for v in [42.0, None, "null"]:
            rows.append(
                (
                    "B16",
                    kw + j(v),
                    {kw: [v] if kw == "enum" else v},
                    v if isinstance(v, str) else j(v),
                    v,
                )
            )
    for g, label, field, raw, want in rows:
        families = (
            ["qwen3_coder"]
            if g == "B04"
            else (["glm47"] if g in ["B06", "B09"] else SCHEMA)
        )
        for f in families:
            add(
                g,
                label,
                f,
                want,
                field=field,
                raw=raw,
                contract="compatibility"
                if g in ["B04", "B06", "B09"]
                else "regression",
            )
    for field in [
        {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        {"anyOf": [{"type": "null"}, {"type": "integer"}]},
        {"allOf": [{"type": "integer"}, {}]},
        {"type": "integer", "title": "x"},
    ]:
        for f in SCHEMA:
            add("B18", str(len(cases)), f, 7, field=field, raw="7", contract="target")
    for field in [
        False,
        {"allOf": [{"type": "string"}, {"type": "integer"}]},
        {"enum": []},
    ]:
        for f in SCHEMA:
            add(
                "B19",
                str(len(cases)),
                f,
                "unrepresentable",
                field=field,
                raw="unrepresentable",
                contract="compatibility",
            )
    for f in SCHEMA:
        add(
            "B20",
            "overlap_is_not_validation",
            f,
            7,
            field={"oneOf": [{"type": "number"}, {"type": "integer"}]},
            raw="7",
            contract="compatibility",
        )

    for f in SCHEMA:
        for group, ty, value, raw in [
            ("C01", "string", '{"a":1}', '{"a":1}'),
            ("C02", "object", {"a": 1}, '{"a":1}'),
            ("C03", "array", [1, None], "[1,null]"),
        ]:
            add(
                group,
                "direct",
                f,
                value,
                raw=raw,
                schema={
                    "type": "object",
                    "$defs": {"S": {"type": ty}},
                    "properties": {"x": {"$ref": "#/$defs/S"}},
                },
            )
        add(
            "C03",
            "nullable",
            f,
            None,
            raw="null",
            schema={
                "$defs": {"S": {"type": ["integer", "null"]}},
                "properties": {"x": {"$ref": "#/$defs/S"}},
            },
        )
        add(
            "C04",
            "chain",
            f,
            '{"a":1}',
            raw='{"a":1}',
            schema={
                "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"type": "string"}},
                "properties": {"x": {"$ref": "#/$defs/A"}},
            },
        )
        for key, ref in [("a/b~c", "#/$defs/a~1b~0c"), ("猫", "#/$defs/猫")]:
            add(
                "C05",
                key,
                f,
                "42",
                raw="42",
                schema={
                    "$defs": {key: {"type": "string"}},
                    "properties": {"x": {"$ref": ref}},
                },
            )
        add(
            "C06",
            "percent",
            f,
            {"a": 1},
            raw='{"a":1}',
            schema={
                "$defs": {"a b": {"type": "object"}},
                "properties": {"x": {"$ref": "#/$defs/a%20b"}},
            },
            contract="target",
        )
        for defs, ref in [
            ("definitions", "#/definitions/S"),
            ("properties", "#/properties/S"),
        ]:
            schema = {
                defs: {"S": {"type": "integer"}},
                "properties": {
                    "x": {"$ref": ref},
                    **({"S": {"type": "integer"}} if defs == "properties" else {}),
                },
            }
            add("C07", defs, f, 42, raw="42", schema=schema, contract="target")
        add(
            "C08",
            "sibling",
            f,
            42,
            raw="42",
            schema={
                "$defs": {"S": {"type": ["string", "integer"]}},
                "properties": {"x": {"$ref": "#/$defs/S", "type": "integer"}},
            },
            contract="compatibility",
        )
        for ref in ["#/$defs/missing", "#/~2bad"]:
            add(
                "C09",
                ref,
                f,
                "text",
                raw="text",
                field={"$ref": ref},
                contract="compatibility",
            )
        for label, defs in [
            ("self", {"A": {"$ref": "#/$defs/A"}}),
            ("cycle", {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}),
            (
                "branch",
                {"A": {"anyOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/A"}]}},
            ),
        ]:
            add(
                "C10",
                label,
                f,
                "text",
                raw="text",
                schema={"$defs": defs, "properties": {"x": {"$ref": "#/$defs/A"}}},
                contract="compatibility",
            )
        for depth in [4, 64]:
            defs = {f"D{i}": {"$ref": f"#/$defs/D{i+1}"} for i in range(depth)}
            defs[f"D{depth}"] = {"type": "string"}
            text = "text" if depth == 64 else '{"a":1}'
            add(
                "C11",
                str(depth),
                f,
                text,
                raw=text,
                schema={"$defs": defs, "properties": {"x": {"$ref": "#/$defs/D0"}}},
                contract="target",
            )
        for kw in ["const", "enum", "default", "examples"]:
            value = {"$ref": "literal"}
            add(
                "C12",
                kw,
                f,
                value,
                field={
                    "type": "object",
                    kw: [value] if kw in ["enum", "examples"] else value,
                },
            )
        for ref in ["https://example.invalid/schema", "#anchor"]:
            add(
                "C13",
                ref,
                f,
                "text",
                field={"$ref": ref},
                raw="text",
                contract="compatibility",
            )
        c = add("C14", "two_roots", f, "42")
        c["tools"] = [
            tool(
                {
                    "$defs": {"S": {"type": ty}},
                    "properties": {"x": {"$ref": "#/$defs/S"}},
                },
                name,
            )
            for name, ty in [("f", "string"), ("g", "integer")]
        ]
        c["input"] = join_calls(
            f, wire(f, [("x", "42")]), wire(f, [("x", 42)], name="g", index=1)
        )
        c["expected"] = [
            {"name": "f", "arguments": '{"x":"42"}'},
            {"name": "g", "arguments": '{"x":42}'},
        ]
        add(
            "C15",
            "root_ref",
            f,
            42,
            raw="42",
            schema={"$defs": {"S": root({"type": "integer"})}, "$ref": "#/$defs/S"},
            contract="target",
        )

    nested = {
        "type": "object",
        "properties": {"page": {"type": "integer"}, "per_page": {"type": "integer"}},
    }
    raw = NS + "<page>2" + NS + "</page>" + NS + "<per_page>10" + NS + "</per_page>"
    for group, field, want in [
        ("D01", {"anyOf": [nested, {"type": "null"}]}, {"page": 2, "per_page": 10}),
        (
            "D01",
            {"type": ["object", "null"], "properties": nested["properties"]},
            {"page": 2, "per_page": 10},
        ),
        (
            "D02",
            {"anyOf": [{"type": "string"}, {"type": "array"}, nested]},
            {"page": 2, "per_page": 10},
        ),
        (
            "D03",
            {
                "anyOf": [
                    nested,
                    {"type": "object", "properties": {"page": {"type": "string"}}},
                ]
            },
            {"page": "2", "per_page": "10"},
        ),
        ("D04", {"anyOf": [nested, {}]}, {"page": "2", "per_page": "10"}),
    ]:
        c = add(group, str(len(cases)), "minimax_m3", want, field=field)
        c["input"] = wire("minimax_m3", [("x", want)]).replace(
            html.escape(j(want), quote=False), raw
        )
    for group, field, value, body in [
        (
            "D05",
            {
                "type": "array",
                "items": {"type": "object", "properties": {"n": {"type": "integer"}}},
            },
            [{"n": 2}],
            NS + "<item>" + NS + "<n>2" + NS + "</n>" + NS + "</item>",
        ),
        (
            "D05",
            {"type": "array", "items": {"type": ["integer", "null"]}},
            [1, None],
            NS + "<item>1" + NS + "</item>" + NS + "<item>null" + NS + "</item>",
        ),
        (
            "D06",
            {"type": "object", "additionalProperties": {"type": "integer"}},
            {"extra": 2},
            NS + "<extra>2" + NS + "</extra>",
        ),
    ]:
        c = add(
            group, str(len(cases)), "minimax_m3", value, field=field, contract="target"
        )
        c["input"] = wire("minimax_m3", [("x", value)]).replace(
            html.escape(j(value), quote=False), body
        )
    for f in FAMILIES:
        c = add("D07", "empty", f)
        c.update(
            input=wire(f, []),
            tools=[tool({"type": "object", "properties": {}})],
            expected=[{"name": "f", "arguments": "{}"}],
        )
        c = add("D08", "order", f)
        pairs = [("z", "last"), ("a", "first"), ("m", "middle")]
        c.update(
            input=wire(f, pairs),
            tools=[tool({"properties": {k: {"type": "string"} for k, v in pairs}})],
            expected=[{"name": "f", "arguments": j(dict(pairs))}],
            raw_exact=[j(dict(pairs))],
        )
        if f in SCHEMA + ["kimi_k3"]:
            c = add("D09", "duplicates", f)
            pairs = [("z", "1"), ("a", "a"), ("z", "2")]
            c.update(
                input=wire(f, pairs),
                tools=[
                    tool(
                        {
                            "properties": {
                                "z": {"type": "string"},
                                "a": {"type": "string"},
                            }
                        }
                    )
                ],
                expected=[{"name": "f", "arguments": '{"z":"2","a":"a"}'}],
                raw_exact=['{"z":"2","a":"a"}'],
            )
        c = add("D10", "tool_ownership", f)
        c.update(
            input=join_calls(
                f, wire(f, [("x", "42")]), wire(f, [("x", 42)], name="g", index=1)
            ),
            tools=[
                tool(root({"type": "string"})),
                tool(root({"type": "integer"}), "g"),
            ],
            expected=[
                {"name": "f", "arguments": '{"x":"42"}'},
                {"name": "g", "arguments": '{"x":42}'},
            ],
        )
        for ty, value in [("string", "42"), ("integer", 42)]:
            add("D11", ty, f, value, field={"type": ty}, contract="target")
        for label, tools in [
            ("no_tools", []),
            ("empty_schema", [tool({})]),
            ("true_schema", [tool(True)]),
            ("false_schema", [tool(False)]),
            ("no_parameters", [{"name": "f", "description": None, "strict": None}]),
        ]:
            c = add("D12", label, f, "text", contract="compatibility")
            c["tools"] = tools
        c = add("D12", "unknown_tool", f, "text", contract="compatibility")
        c["input"] = wire(f, [("x", "text")], name="unknown")
        c["expected"][0]["name"] = "unknown"
        add(
            "D13",
            "schema_immutable",
            f,
            {"a": 1},
            field={"type": "object", "const": {"a": 1}},
            check_immutable=True,
        )
    # Dedicated native-format witnesses add distinct boundary or fallback claims.
    e_fams = {
        "E01": ["kimi_k2"],
        "E02": ["kimi_k3"],
        "E03": ["deepseek_v3", "deepseek_v3_1"],
        "E04": ["deepseek_v3_2", "deepseek_v4"],
        "E05": ["deepseek_v41"],
        "E06": ["glm47"],
        "E07": ["minimax_m2"],
        "E08": ["minimax_m3"],
        "E09": ["qwen25", "qwen3_coder"],
        "E10": ["muse_glimmer"],
        "E11": ["gemma4"],
        "E12": ["harmony"],
    }
    for group, fs in e_fams.items():
        for f in fs:
            marker = {
                "kimi_k2": "<|tool_call_end|>",
                "kimi_k3": "<|open|>think<|sep|>",
                "deepseek_v41": "</｜DSML｜ invoke>",
                "gemma4": "call:fake{} <tool_call|>",
                "harmony": "<|not_a_control_token|>",
            }.get(f, "&lt;tag&gt; &amp; &quot;")
            value = (
                marker
                if f not in ["deepseek_v3", "deepseek_v3_1"]
                else {"s": "42", "v": 42, "nested": [None, False]}
            )
            add(
                group,
                "literal",
                f,
                value,
                strategies=["whole", "chars", "seeded", "all_splits"],
            )
    for value in ["  café\n", "\t\r\n ", "", "null"]:
        for f in ["deepseek_v3_2", "deepseek_v4"]:
            add("E04", str(len(cases)), f, value)
    for val in ["null", "42", "{not json}", " just words "]:
        # Muse JSON-first decoding has no schema-directed scalar coercion.
        expected = json.loads(val) if val in ["null", "42"] else val
        add("E10", str(len(cases)), "muse_glimmer", expected, raw=val)
    c = add("E02", "missing_type", "kimi_k3", "42")
    c["input"] = c["input"].replace(' type="string"', "")
    c = add("E02", "malformed_typed", "kimi_k3", "not_json")
    c["input"] = c["input"].replace('type="string"', 'type="number"')
    c = add(
        "E05",
        "mixed_dialects",
        "deepseek_v41",
        {"s": "</｜DSML｜ invoke>", "v": [1, None]},
    )
    c["surfaces"] = ["v2_tool"]
    c["input"] += wire("deepseek_v4", [("x", 42)], name="g")
    c["tools"].append(tool(root({"type": "integer"}), "g"))
    c["expected"].append({"name": "g", "arguments": '{"x":42}'})
    # Streaming dimensions with independently authored arguments and ordered events.
    for f in FAMILIES:
        add(
            "F01",
            "all_boundaries",
            f,
            "café 🐈",
            strategies=["whole", "chars", "seeded", "all_splits"],
        )
        add(
            "F02",
            "escaping_boundaries",
            f,
            '\\"\n猫',
            strategies=["chars", "seeded", "all_splits"],
        )
        add("F03", "delta_lifecycle", f, 42, strategies=["chars"], check_lifecycle=True)
        c = add("F05", "mid_value", f, "unfinished_value")
        i = c["input"].index("unfinished_value") + 5
        c["input"] = c["input"][:i]
        c["expected"] = []
        c["surfaces"] = ["v1_jail", "v2_tool", "unified"]
        c["contract"] = "target"
        c = add("F06", "complete_then_partial", f, "first")
        full = c["input"]
        bad = wire(f, [("x", "unfinished_value")], index=1)
        c["input"] = join_calls(f, full, bad[: bad.index("unfinished_value") + 5])
        c["surfaces"] = ["v1_jail", "v2_tool", "unified"]
        c["contract"] = "target"
        c = add("F12", "two_same_name", f, "first")
        c["input"] = join_calls(f, c["input"], wire(f, [("x", "second")], index=1))
        c["expected"].append({"name": "f", "arguments": '{"x":"second"}'})
        c["strategies"] = ["whole", "chars", "seeded"]
    for nullable in [False, True]:
        c = add(
            "F04",
            "nullable" if nullable else "eager",
            "qwen3_coder",
            None if nullable else "stream me",
            field={"type": ["string", "null"]} if nullable else {"type": "string"},
            raw="null" if nullable else "stream me",
            surfaces=["v2_tool", "unified"],
            strategies=["chars"],
            early="defer" if nullable else "string",
        )
    for f in UNIFIED:
        c = add("F07", "ordered_channels", f, "42", surfaces=["unified"])
        # Native channel wrappers differ from plain think markers.
        if f == "kimi_k3":
            pre = "<|open|>think<|sep|>before<|close|>think<|sep|>"
            post = "<|open|>think<|sep|>after<|close|>think<|sep|><|open|>response<|sep|>done<|close|>response<|sep|>"
        elif f == "gemma4":
            pre = "<|channel>thought\nbefore<channel|>"
            post = "<|channel>thought\nafter<channel|>done"
        elif f == "muse_glimmer":
            pre = " to=self<|message|>before<|eom|><|start|>assistant"
            post = "<|start|>assistant to=self<|message|>after<|eom|><|start|>assistant to=user<|message|>done<|eom|>"
        else:
            pre = "<think>before</think>"
            post = "<think>after</think>done"
        c["input"] = pre + c["input"] + post
        c["events"] = [
            {"kind": "reasoning", "text": "before"},
            {"kind": "tool_call", "name": "f", "arguments": {"x": "42"}},
            {"kind": "reasoning", "text": "after"},
            {"kind": "text", "text": "done"},
        ]
        for start in ["none", "reasoning", "response"]:
            add(
                "F08",
                start,
                f,
                42,
                surfaces=["unified"],
                init={"start": start, "mode": "required", "policy": "reject"},
                input='[{"name":"f","arguments":{"x":42}}]'
                if start != "reasoning"
                else (
                    "<|close|>think<|sep|>"
                    if f == "kimi_k3"
                    else "<channel|>"
                    if f == "gemma4"
                    else "<|eom|>"
                    if f == "muse_glimmer"
                    else "</think>"
                )
                + '[{"name":"f","arguments":{"x":42}}]',
            )
        for mode in ["named", "required"]:
            for policy in ["reject", "recover", "stream"]:
                add(
                    "F09",
                    mode + policy,
                    f,
                    {"s": "42", "n": 42},
                    surfaces=["unified"],
                    input='{"x":{"s":"42","n":42}}'
                    if mode == "named"
                    else '[{"name":"f","arguments":{"x":{"s":"42","n":42}}}]',
                    init={"mode": mode, "policy": policy},
                    strategies=["chars", "whole"],
                )
        for policy in ["reject", "recover", "stream"]:
            payload = (
                '[{"name":"f","arguments":{"x":42}},{"name":"f","arguments":null}]'
            )
            c = add(
                "F10",
                policy,
                f,
                42,
                input=payload,
                surfaces=["unified"],
                init={"mode": "required", "policy": policy},
                strategies=["whole", "chars"],
                expect_error=policy == "reject",
            )
            if policy != "stream":
                c["expected"] = []
            if policy == "recover":
                c["expected_text"] = payload
        add(
            "F11",
            "reset_prefix",
            f,
            "42",
            surfaces=["unified"],
            reset_prefix="<",
            strategies=["chars"],
        )
    for f in SCHEMA:
        for raw in ["+1", "01"]:
            add(
                "A14",
                "numeric_" + raw,
                f,
                1,
                field={"type": "number"},
                raw=raw,
                contract="compatibility",
            )
        add(
            "A14",
            "uppercase_bool",
            f,
            True,
            field={"type": "boolean"},
            raw="TRUE",
            contract="compatibility",
        )
        if f == "glm47":
            add(
                "A14",
                "yes_alias",
                f,
                True,
                field={"type": "boolean"},
                raw="yes",
                contract="compatibility",
            )
        for extra in [False, None]:
            field = {"type": "object"}
            if extra is not None:
                field["additionalProperties"] = extra
            if f == "minimax_m3":
                c = add(
                    "D06",
                    "fallback_" + str(extra),
                    f,
                    {"extra": "2"},
                    field=field,
                    contract="compatibility",
                )
                c["input"] = wire(f, [("x", {"extra": "2"})]).replace(
                    html.escape(j({"extra": "2"}), quote=False),
                    NS + "<extra>2" + NS + "</extra>",
                )
    for f in FAMILIES:
        c = add(
            "F05",
            "before_value",
            f,
            "unfinished_value",
            surfaces=["v1_jail", "v2_tool", "unified"],
            expected=[],
            contract="target",
        )
        c["input"] = c["input"][: c["input"].index("unfinished_value")]
        c = add(
            "F05",
            "mid_escape",
            f,
            "escaped\\value",
            surfaces=["v1_jail", "v2_tool", "unified"],
            expected=[],
            contract="target",
        )
        at = c["input"].index("escaped") + len("escaped") + 1
        c["input"] = c["input"][:at]
    for f, end in [
        ("deepseek_v4", "</｜DSML｜tool_calls>"),
        ("deepseek_v41", "</｜DSML｜ calls>"),
        ("qwen3_coder", "</tool_call>"),
        ("minimax_m2", "</minimax:tool_call>"),
        ("gemma4", "<tool_call|>"),
        ("harmony", "<|call|>"),
    ]:
        c = add(
            "F05",
            "complete_inner_at_eof",
            f,
            42,
            surfaces=["v1_jail", "v2_tool", "unified"],
            contract="compatibility",
        )
        c["input"] = c["input"].removesuffix(end)
    # Alias smoke stays small: the canonical implementation owns exhaustive coverage.
    for family, alias, surfaces in [
        ("deepseek_v4", "deepseek-v4", ["v1_batch", "v1_jail"]),
        ("deepseek_v4", "deepseekv4", ["v1_batch", "v1_jail"]),
        ("minimax_m3", "minimax-m3", ["v1_batch", "v1_jail"]),
        ("minimax_m3", "minimax_m3_nom", ["v1_batch", "v1_jail"]),
        ("minimax_m3", "minimax-m3-nom", ["v1_batch", "v1_jail"]),
        ("kimi_k3", "kimi-k3", ["v1_batch", "v1_jail", "unified"]),
        ("gemma4", "gemma-4", ["v1_batch", "v1_jail"]),
        ("qwen3_coder", "qwen3", ["unified"]),
        ("harmony", "harmony_text", ["v2_tool"]),
    ]:
        group = next(g for g, fs in e_fams.items() if family in fs)
        add(group, "alias_" + alias, family, 42, selector=alias, surfaces=surfaces)
    for family, group, index, ident in [
        ("kimi_k2", "E01", 17, "functions.f:17"),
        ("kimi_k3", "E02", 16, "f:16"),
    ]:
        add(
            group,
            "native_id",
            family,
            "42",
            input=wire(family, [("x", "42")], index=index),
            expected_ids=[ident],
        )
    # Wire types take precedence over incompatible caller schema.
    for family in [f for f in FAMILIES if f not in SCHEMA]:
        for value, conflict in [
            ("42", "integer"),
            (42, "string"),
            (None, "string"),
            (False, "string"),
        ]:
            add(
                "A16",
                "conflict_" + kind(value),
                family,
                value,
                field={"type": conflict},
            )
    for family in SCHEMA:
        add(
            "C03",
            "nullable_object",
            family,
            {"a": 1},
            raw='{"a":1}',
            schema={
                "$defs": {"S": {"type": ["object", "null"]}},
                "properties": {"x": {"$ref": "#/$defs/S"}},
            },
            contract="target",
        )
        add(
            "C07",
            "root_pointer",
            family,
            {"x": {}},
            raw='{"x":{}}',
            schema={"type": "object", "properties": {"x": {"$ref": "#"}}},
            contract="target",
        )
        add(
            "C13",
            "nested_id_scope",
            family,
            "42",
            raw="42",
            schema={
                "$defs": {"S": {"type": "integer"}},
                "properties": {
                    "x": {
                        "$id": "https://example.invalid/nested",
                        "type": "string",
                        "$ref": "#/$defs/S",
                    }
                },
            },
            contract="compatibility",
        )
        add(
            "C11",
            "unrelated_survives",
            family,
            "text",
            schema={
                "$defs": {"A": {"$ref": "#/$defs/A"}},
                "properties": {"x": {"$ref": "#/$defs/A"}, "y": {"type": "integer"}},
            },
            input=wire(family, [("x", "text"), ("y", 7)]),
            expected=[{"name": "f", "arguments": '{"x":"text","y":7}'}],
            contract="target",
        )
    # Reverse tool order exercises state borrowed from the preceding numeric tool.
    for family in FAMILIES:
        add(
            "D10",
            "reversed",
            family,
            schema=root({"type": "string"}),
            tools=[
                tool(root({"type": "string"})),
                tool(root({"type": "integer"}), "g"),
            ],
            input=join_calls(
                family,
                wire(family, [("x", 42)], name="g"),
                wire(family, [("x", "42")], index=1),
            ),
            expected=[
                {"name": "g", "arguments": '{"x":42}'},
                {"name": "f", "arguments": '{"x":"42"}'},
            ],
            contract="target",
        )
        add(
            "D12",
            "unknown_argument",
            family,
            "text",
            schema={"type": "object", "properties": {"known": {"type": "integer"}}},
            contract="compatibility",
        )
        add(
            "D13",
            "malformed_schema_immutable",
            family,
            "text",
            field={"$ref": "#/$defs/missing"},
            contract="target",
        )
        for label, value in [("null", None), ("negative", -17), ("exponent", 0.000001)]:
            add(
                "F02",
                label,
                family,
                value,
                strategies=["chars", "seeded", "all_splits"],
            )
    # Complete values with missing native closures must not be confused with complete calls.
    parameter_end = {
        "kimi_k3": "<|close|>argument<|sep|>",
        "deepseek_v3_2": "</｜DSML｜parameter>",
        "deepseek_v4": "</｜DSML｜parameter>",
        "glm47": "</arg_value>",
        "qwen3_coder": "</parameter>",
        "minimax_m2": "</parameter>",
        "minimax_m3": NS + "</x>",
        "muse_glimmer": "</atem:parameter>",
    }
    for family, end in parameter_end.items():
        for after in [False, True]:
            c = add(
                "F05",
                "after_value_" + str(after),
                family,
                42,
                surfaces=["v1_jail", "v2_tool", "unified"],
                expected=[],
                contract="target",
            )
            c["input"] = c["input"][
                : c["input"].index(end) + (len(end) if after else 0)
            ]
    for family in UNIFIED:
        for invalid in [[], "bad"]:
            payload = j(
                [
                    {"name": "f", "arguments": {"x": 42}},
                    {"name": "f", "arguments": invalid},
                ]
            )
            add(
                "F10",
                "reject_" + kind(invalid),
                family,
                42,
                input=payload,
                surfaces=["unified"],
                init={"mode": "required", "policy": "reject"},
                expect_error=True,
                expected=[],
            )
    for val in [True, {}, [1, None], "42"]:
        add("E10", "json_" + kind(val), "muse_glimmer", val)
    add(
        "E06", "literal_entities_object", "glm47", {"text": "&quot; &#34; &#x22; &amp;"}
    )
    # Spaced V4.1 XML and raw JSON bodies are separate grammar alternatives.
    add(
        "E05",
        "spaced_parameters",
        "deepseek_v41",
        "42",
        input='<｜DSML｜ calls><｜DSML｜ invoke name="f"><｜DSML｜ parameter name="x" string="true">42</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>',
    )
    add(
        "E05",
        "root_array",
        "deepseek_v41",
        input='<｜DSML｜ calls><｜DSML｜ invoke name="f">[1,2]</｜DSML｜ invoke></｜DSML｜ calls>',
        expected=[],
        surfaces=["v2_tool", "unified"],
        contract="target",
    )
    add(
        "E11",
        "missing_string_delimiter",
        "gemma4",
        input='<|tool_call>call:f{x:<|"|>unfinished}<tool_call|>',
        expected=[],
        surfaces=["v2_tool", "unified"],
        contract="target",
    )
    for channel in ["analysis", "final"]:
        add(
            "E12",
            channel + "_not_tool",
            "harmony",
            input="<|start|>assistant<|channel|>"
            + channel
            + '<|message|>{"x":42}<|end|>',
            expected=[],
        )
    add(
        "D09",
        "replacement_changes_type",
        "kimi_k3",
        schema={"type": "object"},
        input=wire("kimi_k3", [("z", "one"), ("a", "middle"), ("z", 2)]),
        expected=[{"name": "f", "arguments": '{"z":2,"a":"middle"}'}],
        raw_exact=['{"z":2,"a":"middle"}'],
    )
    for family in UNIFIED:
        add(
            "F11",
            "reset_after_call",
            family,
            "42",
            surfaces=["unified"],
            reset_prefix=wire(family, [("x", "previous")]),
            strategies=["chars"],
        )
    for c in cases:
        if c["id"] == "F05.complete_inner_at_eof.harmony":
            # Commentary EOF recovery was deliberately removed; analysis recovery
            # belongs to the reasoning parser (dynamo #10366).
            c["expected"] = []
        if c["id"] == "E05.root_array.deepseek_v41":
            c["allow_error_contains"] = "invalid DeepSeek V4.1"
        if c["family"] in ["qwen3_coder", "minimax_m2"] and c["group"] in [
            "A01",
            "A04",
        ]:
            if "whitespace" in c["id"] or c["group"] == "A04":
                # Exact preservation is stronger than the documented XML trimming policy.
                c["contract"] = "target"
    return cases


if __name__ == "__main__":
    print(j({"cases": build_cases(), "capabilities": CAPABILITIES}))
