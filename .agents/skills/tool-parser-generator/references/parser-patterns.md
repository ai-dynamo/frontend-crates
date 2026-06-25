# Tool Call Parser Patterns Reference

A recognition guide: how to spot each tool-call grammar in an LLM chat template. The authoritative, current family-to-grammar-to-file mapping (which family owns each model and where the parser lives) is the "Parser families" cheat-sheet in [`../../../../parsers_v2/README.md`](../../../../parsers_v2/README.md) — this doc only helps you recognize the shape; it does not track the model list.

## Pattern Categories

### 1. JSON with Special Tokens

#### Bracket Markers (Mistral-style)
```
[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "NYC"}}]
```
- Models: Mistral, Mixtral
- Parser: `base_json_parser` with bracket config
- Keys: `name`, `arguments`

#### XML-Style Tags (Hermes-style)
```xml
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>
```
- Models: Hermes-2, Jamba
- Parser: `base_json_parser` with XML-style markers
- Keys: `name`, `arguments`

#### Single Token Prefix (Llama-style)
```
<|python_tag|>[{"name": "get_weather", "arguments": {"location": "NYC"}}]
```
- Models: Llama 3.1, Llama 3.2
- Parser: `base_json_parser` with single start token
- Keys: `name`, `arguments`

### 2. XML-Based

#### Qwen3 Coder Style
```xml
<tool_call>
<function=get_weather>
<parameter=location>NYC</parameter>
</function>
</tool_call>
```
- Models: Qwen3-Coder, Nemotron-Nano
- Parser: `xml/parser.rs`
- Attribute-based names and parameters

### 3. Nested Special Tokens

#### DeepSeek V3
```
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "NYC"}
```
<｜tool▁call▁end｜>
```
- Models: DeepSeek-V3
- Parser: `deepseek_v3_parser.rs`
- Multiline with markdown code blocks

#### DeepSeek V3.1
```
<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "NYC"}<｜tool▁call▁end｜>
```
- Models: DeepSeek-V3.1
- Parser: `deepseek_v3_1_parser.rs`
- Inline JSON

### 4. DSML (DeepSeek V3.2 / V4)
```xml
<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>
```
- Models: DeepSeek-V3.2, DeepSeek-V4
- Parser: `dsml/parser.rs`
- Explicit parameter types (`string="true|false"`)

### 5. Pythonic
```python
[get_weather(location="NYC"), get_time(timezone="EST")]
```
- Models: some Llama variants
- Parser: `pythonic/pythonic_parser.rs`
- Python function call syntax

### 6. Harmony
```
<|channel|>commentary to=functions.get_weather
<|constrain|>json
<|message|>{"location": "NYC"}
```
- Models: GPT-OSS
- Parser: `harmony/harmony_parser.rs`
- OpenAI Harmony protocol

### 7. GLM (arg_key / arg_value tags)
```xml
<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>
```
- Models: GLM-4.x, GLM 5.1
- Parser: `xml/glm47_parser.rs`
- Function name is bare text right after `<tool_call>`; each argument is an `<arg_key>`/`<arg_value>` pair (not nested JSON)

### 8. Kimi K2 (special-token sections)
```
<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "NYC"}<|tool_call_end|><|tool_calls_section_end|>
```
- Models: Kimi K2
- Parser: `xml/kimi_k2_parser.rs`
- Calls wrapped in `<|tool_calls_section_begin|>`...`<|tool_calls_section_end|>`; each call is `functions.{name}:{index}` then JSON args after `<|tool_call_argument_begin|>`

### 9. Gemma 4 (custom delimited)
```
<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|>
```
- Models: Google Gemma 4 thinking models
- Parser: `gemma4/parser.rs`
- `<|tool_call>`...`<tool_call|>` wrapper, `call:name`, bare keys, and a custom `<|"|>` string delimiter instead of normal JSON quoting

## Quick Identification Guide

1. **Look for `tojson` filter** → JSON format
2. **Look for `<function=` or `<parameter=`** → XML format
3. **Look for `<｜DSML｜`** → DSML format
4. **Look for `function(arg=val)`** → Pythonic format
5. **Look for `<|channel|>commentary`** → Harmony format
6. **Look for `<arg_key>` / `<arg_value>`** → GLM format
7. **Look for `<|tool_calls_section_begin|>` / `<|tool_call_begin|>`** → Kimi K2 format
8. **Look for `<|tool_call>call:` with `<|"|>` string delimiters** → Gemma 4 format
9. **Check start/end markers** → Match to config preset

## Configuration Keys

For JSON formats, check these keys in the template:
- Function name: Usually `name` or `function`
- Arguments: Usually `arguments` or `parameters`
- Structure: Array `[{...}]` or single object `{...}`

## Matching Logic

1. **Exact match** → Use existing config preset
2. **Similar markers** → Create new config with same parser
3. **New format** → Generate new parser implementation
