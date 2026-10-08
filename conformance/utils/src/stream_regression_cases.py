# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-native stream-v1 fixtures for shared parser regression invariants."""


# These capture IDs are immutable; fixture_disposition owns their display aliases.
SCALAR_CASE = "TOOLCALLING.streamv1.7.g"
STRING_CASE = "TOOLCALLING.streamv1.7.h"
REASONING_CASE = "TOOLCALLING.streamv1.51.a"
ENTITY_CASE = "TOOLCALLING.streamv1.7.i"
NESTED_UNION_CASE = "TOOLCALLING.streamv1.7.j"
REFERENCE_TYPE_CASE = "TOOLCALLING.streamv1.7.k"
OBJECT_REFERENCE_CASE = "TOOLCALLING.streamv1.7.l"
SELECTOR_CASE = "TOOLCALLING.streamv1.51.b"


def scalar_tools():
    return [{"name": "inspect", "parameters": {"type": "object", "properties": {
        "count": {"anyOf": [{"type": "integer"}]},
        "ratio": {"type": ["number"]},
        "enabled": {"type": ["boolean"]},
    }}}]


def string_tools():
    return [{"name": "inspect", "parameters": {"type": "object", "properties": {
        "spaced": {"type": "string"},
        "blank": {"type": "string"},
        "empty": {"type": "string"},
    }}}]


def stream_case(description, reference, tools, chunks):
    return {
        "description": description,
        "ref": f"https://github.com/ai-dynamo/frontend-crates/pull/{reference}",
        "tools": tools,
        "chunks": chunks + [finish()],
    }


def entity_tools():
    return [{"name": "inspect", "parameters": {"type": "object", "properties": {
        "text": {"type": "string"},
        "payload": {"type": "object", "properties": {"text": {"type": "string"}}},
    }}}]


def nested_union_tools():
    return [{
        "name": "list_notes",
        "parameters": {
            "type": "object",
            "properties": {
                "pagination": {"anyOf": [
                    {"type": "object", "properties": {
                        "page": {"type": "integer", "minimum": 1},
                        "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                    }},
                    {"type": "null"},
                ]},
            },
            "required": ["pagination"],
            "additionalProperties": False,
        },
        "strict": True,
    }]


def glm_reference_tools():
    return [{"name": "capture_payload", "parameters": {
        "type": "object",
        "$defs": {
            "Text": {"type": "string"},
            "TextAlias": {"$ref": "#/$defs/Text"},
            "Count": {"type": "integer"},
        },
        "properties": {
            "payload": {"$ref": "#/$defs/TextAlias", "allOf": [{"type": "string"}]},
            "count": {"$ref": "#/$defs/Count", "minimum": 1},
        },
        "required": ["payload", "count"],
    }}]


def object_reference_tools():
    return [{"name": "authenticate_first_name", "parameters": {
        "type": "object",
        "$defs": {"Schema": {
            "type": "object",
            "properties": {"input": {"type": "string"}, "notes": {"type": "string"}},
            "required": ["input", "notes"],
            "additionalProperties": False,
        }},
        "properties": {"data": {"$ref": "#/$defs/Schema"}},
        "required": ["data"],
        "additionalProperties": False,
    }}]


WEATHER_TOOLS = [{"name": "get_weather", "parameters": {
    "type": "object", "properties": {"location": {"type": "string"}},
}}]


def finish(reason="tool_calls"):
    return {"delta_text": "", "finish_reason": reason}


def scalar_case(chunks):
    return stream_case(
        "Composed scalar schemas preserve integer, number, and boolean values",
        248, scalar_tools(), chunks,
    )


def string_case(chunks):
    return stream_case(
        "Family-native string arguments preserve whitespace and empty strings",
        247, string_tools(), chunks,
    )


def reasoning_case(chunks):
    return stream_case(
        "Tool-only projection preserves caller-usable reasoning information around a tool call",
        253, WEATHER_TOOLS, chunks,
    )


ENTITY_CASES = {
    "glm47": stream_case(
        "GLM argument strings and object values preserve literal XML entity text",
        249,
        entity_tools(),
        [
            {"delta_text": "<tool_call>inspect<arg_key>text</arg_key><arg_value>"},
            {"delta_text": "&lt;tag&gt; &amp; &quot;x&quot; &apos;y&apos; &#65; &amp;lt;</arg_value><arg_key>payload</arg_key><arg_value>{"},
            {"delta_text": '"text":"&quot; &amp;"}</arg_value></tool_call>'},
        ],
    ),
}


NESTED_UNION_CASES = {
    "minimax_m3": stream_case(
        "MiniMax M3 keeps nested integer values when an object wins a nullable union",
        270,
        nested_union_tools(),
        [
            {"delta_text": ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="list_notes">'},
            {"delta_text": "]<]minimax[>[<pagination>]<]minimax[>[<page>2]<]minimax[>[</page>"},
            {"delta_text": "]<]minimax[>[<per_page>25]<]minimax[>[</per_page>]<]minimax[>[</pagination>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"},
        ],
    ),
}


REFERENCE_TYPE_CASES = {
    "glm47": stream_case(
        "GLM resolves local schema references and intersects sibling constraints",
        271,
        glm_reference_tools(),
        [
            {"delta_text": "<tool_call>capture_payload<arg_key>payload</arg_key><arg_value>{"},
            {"delta_text": '"x":1}</arg_value><arg_key>count</arg_key><arg_value>42</arg_value></tool_call>'},
        ],
    ),
}


OBJECT_REFERENCE_CASES = {
    "minimax_m3": stream_case(
        "MiniMax M3 resolves a local parameter reference before parsing object arguments",
        273,
        object_reference_tools(),
        [
            {"delta_text": ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="authenticate_first_name">'},
            {"delta_text": ']<]minimax[>[<data>{"input":"Alex","notes":"first name supplied"}]<]minimax[>[</data>'},
            {"delta_text": "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"},
        ],
    ),
}


SELECTOR_CASES = {
    "deepseek_v4": stream_case(
        "DeepSeek tool-only selection accepts both DSML dialects in one stream",
        255,
        [{"name": "inspect", "parameters": {"type": "object", "properties": {
            "value": {"type": "string"},
        }}}],
        [
            {"delta_text": '<｜DSML｜tool_calls><｜DSML｜invoke name="inspect">'},
            {"delta_text": '<｜DSML｜parameter name="value" string="true">compact</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls><｜DSML｜ calls><｜DSML｜ invoke name="inspect">'},
            {"delta_text": '<｜DSML｜ parameter name="value" string="true">spaced</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>'},
        ],
    ),
}


SCALAR_CASES = {
    "glm47": scalar_case([
        {"delta_text": "<tool_call>inspect<arg_key>count</arg_key><arg_value>42</arg_value><arg_key>ratio</arg_key><arg_value>1.25</arg_value>"},
        {"delta_text": "<arg_key>enabled</arg_key><arg_value>false</arg_value></tool_call>"},
    ]),
    "qwen3_coder": scalar_case([
        {"delta_text": "<tool_call><function=inspect><parameter=count>42</parameter><parameter=ratio>1.25</parameter>"},
        {"delta_text": "<parameter=enabled>false</parameter></function></tool_call>"},
    ]),
    "minimax_m2": scalar_case([
        {"delta_text": '<minimax:tool_call><invoke name="inspect"><parameter name="count">42</parameter><parameter name="ratio">1.25</parameter>'},
        {"delta_text": '<parameter name="enabled">false</parameter></invoke></minimax:tool_call>'},
    ]),
    "minimax_m3": scalar_case([
        {"delta_text": ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="inspect">]<]minimax[>[<count>42]<]minimax[>[</count>]<]minimax[>[<ratio>1.25]<]minimax[>[</ratio>'},
        {"delta_text": ']<]minimax[>[<enabled>false]<]minimax[>[</enabled>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>'},
    ]),
}


STRING_CASES = {
    "deepseek_v4": string_case([
        {"delta_text": '<｜DSML｜tool_calls><｜DSML｜invoke name="inspect">'},
        {"delta_text": '<｜DSML｜parameter name="spaced" string="true">  café\n</｜DSML｜parameter>'},
        {"delta_text": '<｜DSML｜parameter name="blank" string="true">\t\r\n </｜DSML｜parameter>'},
        {"delta_text": '<｜DSML｜parameter name="empty" string="true"></｜DSML｜parameter>'},
        {"delta_text": '</｜DSML｜invoke></｜DSML｜tool_calls>'},
    ]),
    "glm47": string_case([
        {"delta_text": "<tool_call>inspect<arg_key>spaced</arg_key><arg_value>  café\n</arg_value>"},
        {"delta_text": "<arg_key>blank</arg_key><arg_value>\t\r\n </arg_value><arg_key>empty</arg_key><arg_value></arg_value>"},
        {"delta_text": "</tool_call>"},
    ]),
    "gemma4": string_case([
        {"delta_text": '<|tool_call>call:inspect{spaced:<|"|>  café\n<|"|>,blank:<|"|>\t\r\n <|"|>'},
        {"delta_text": ',empty:<|"|><|"|>}<tool_call|>'},
    ]),
    "kimi_k2": string_case([
        {"delta_text": "<|tool_calls_section_begin|><|tool_call_begin|>functions.inspect:0<|tool_call_argument_begin|>"},
        {"delta_text": '{"spaced":"  café\\n","blank":"\\t\\r\\n ","empty":""}'},
        {"delta_text": "<|tool_call_end|><|tool_calls_section_end|>"},
    ]),
    "kimi_k3": string_case([
        {"delta_text": '<|open|>tools<|sep|><|open|>call tool="inspect" index="1"<|sep|>'},
        {"delta_text": '<|open|>argument key="spaced" type="string"<|sep|>  café\n<|close|>argument<|sep|>'},
        {"delta_text": '<|open|>argument key="blank" type="string"<|sep|>\t\r\n <|close|>argument<|sep|>'},
        {"delta_text": '<|open|>argument key="empty" type="string"<|sep|><|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>'},
    ]),
    "minimax_m3": string_case([
        {"delta_text": ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="inspect">]<]minimax[>[<spaced>  café\n]<]minimax[>[</spaced>'},
        {"delta_text": ']<]minimax[>[<blank>\t\r\n ]<]minimax[>[</blank>]<]minimax[>[<empty>]<]minimax[>[</empty>'},
        {"delta_text": ']<]minimax[>[</invoke>]<]minimax[>[</tool_call>'},
    ]),
    "muse_glimmer": string_case([
        {"delta_text": '<|start|>assistant to=inspect<|message|><atem:function_calls><atem:invoke name="inspect"><atem:parameter name="spaced">"  café\\n"</atem:parameter>'},
        {"delta_text": '<atem:parameter name="blank">"\\t\\r\\n "</atem:parameter><atem:parameter name="empty">""</atem:parameter>'},
        {"delta_text": '</atem:invoke></atem:function_calls><|eom|>'},
    ]),
    "harmony": string_case([
        {
            "delta_text": '<|channel|>commentary to=functions.inspect <|constrain|>json<|message|>{"spaced":"  café\\n",',
            "delta_token_ids": [200005, 12606, 815, 316, 28, 44580, 145021, 220, 200003, 4108, 200008, 10848, 1148, 18308, 7534, 220, 30469, 3392],
        },
        {
            "delta_text": '"blank":"\\t\\r\\n ","empty":""}<|call|>',
            "delta_token_ids": [4294, 20107, 7534, 59, 83, 26543, 3392, 33200, 6857, 1243, 6371, 92, 200012],
        },
    ]),
}


REASONING_CASES = {
    "deepseek_v4": reasoning_case([
        {"delta_text": "before<think>reason</think>"},
        {"delta_text": '<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather">'},
        {"delta_text": '<｜DSML｜parameter name="location" string="true">Paris</｜DSML｜parameter>'},
        {"delta_text": '</｜DSML｜invoke></｜DSML｜tool_calls>after'},
    ]),
    "kimi_k3": reasoning_case([
        {"delta_text": "<|open|>think<|sep|>reason<|close|>think<|sep|>"},
        {"delta_text": '<|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|><|open|>argument key="location" type="string"<|sep|>Paris<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>'},
        {"delta_text": "after"},
    ]),
    "muse_glimmer": reasoning_case([
        {"delta_text": "<|start|>assistant to=self<|message|>reason<|eom|>"},
        {"delta_text": '<|start|>assistant to=get_weather<|message|><atem:function_calls><atem:invoke name="get_weather"><atem:parameter name="location">Paris</atem:parameter></atem:invoke></atem:function_calls><|eom|>after'},
    ]),
}
