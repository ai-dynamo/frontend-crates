---
name: tool-parser-generator
description: Generate optimized tool call parsers for the dynamo-parsers crate from HuggingFace model chat templates. Use this when you need to add support for a new model's tool calling format. Takes a HuggingFace model name, analyzes its chat template, compares with existing parsers, and either maps to an existing parser or generates new Rust code with tests for the tool_calling library. The same workflow applies to reasoning parsers in parsers/src/reasoning/.
license: "Apache-2.0"
---

# Tool Parser Generator Skill

Add support for new models' tool calling formats by analyzing their chat templates and generating appropriate parser implementations for the `dynamo-parsers` crate (`parsers/`).

## Parser goals (read first)

These bind every parser you add here and are the tie-breakers when vLLM, SGLang, and Dynamo disagree. The canonical list is in [`../../../parsers_v2/README.md`](../../../parsers_v2/README.md) ("Parser goals"); in brief:

- **Follow the model's own spec** (its chat template / tool-calling guide defines the grammar), and **record the spec source in the fixture YAML** (a `spec:` URL), not just in code comments.
- **Error recovery is under-specified**, so a divergence from vLLM/SGLang on a recovery / edge case is **expected** — document it with a `reason:`; do not "fix" it by matching a peer.
- **Never leak tool-call or reasoning markup into user-visible `content`/`normal_text`** (the `↯` marker catches tool leaks).
- **Make a reasonable, bounded attempt to recover** an incomplete or stream-truncated call/reasoning span (e.g. close the JSON when only the end marker is missing) — never invent content the model didn't emit, and `tracing::warn!` with a stable `why=`.
- **Preserve as much of the original output as possible**: `normal_text` is the model output minus only the recognized markup spans (prefix, inter-call, and trailing text kept verbatim).
- **Parsing is separate from validation** — emit a call even for a tool not in the request's list; the serving layer validates.
- **v1 (batch) and v2 (streaming) must always agree** — same calls, same `normal_text`. v1 is the simple, inefficient reference (it jails/buffers the whole output, then parses); v2 parses token-incrementally (jailing only the ambiguous suffix) for lower latency. Intentional stream-vs-batch differences go in the `known_divergences` allowlist.

## When to Use This Skill

- User asks to add tool calling support for a specific HuggingFace model
- User wants to understand how a model structures tool calls
- User needs to extend the `dynamo-parsers` tool_calling (or reasoning) library with new formats

## Workflow

Follow this systematic workflow when the user provides a HuggingFace model name.

### Phase 1: Fetch and Extract Chat Template

1. **Fetch tokenizer config from HuggingFace Hub**:
   ```
   URL: https://huggingface.co/{model_id}/resolve/main/tokenizer_config.json
   ```

2. **Extract chat template**:
   - Parse the JSON response
   - Look for `chat_template` field
   - Handle two formats:
     - String: Single template
     - Array: List of templates with `name` and `template` fields
       - Prefer `tool_use` template if available
       - Fall back to `default` template

3. **Extract special tokens** (if relevant):
   - `bos_token`, `eos_token`, `unk_token`
   - `additional_special_tokens`
   - Any tool-specific tokens in the config

### Phase 2: Analyze Chat Template

The chat template is a Jinja template. Analyze it to identify tool call patterns:

1. **Find tool-related sections**:
   - Look for conditional blocks with keywords: `tools`, `tool_call`, `function`, `available_tools`
   - Extract content within `{% if tools %}...{% endif %}` blocks
   - Find `{% for tool in tools %}` loops

2. **Identify markers and format**:
   - **Start markers**: Tokens/strings before tool calls
     - Examples: `<tool_call>`, `[TOOL_CALLS]`, `<|python_tag|>`, `<｜tool▁call▁begin｜>`
   - **End markers**: Tokens/strings after tool calls
     - Examples: `</tool_call>`, `[/TOOL_CALLS]`, `<｜tool▁call▁end｜>`
   - **Special tokens**: Unicode or encoded tokens (DeepSeek, Harmony)
   - **Format type**:
     - JSON: Look for `tojson` filter, `{` `}` brackets
     - XML: Look for `<function=`, `<parameter=` patterns
     - Pythonic: Look for `function(arg=val)` patterns
     - DSML: Look for `<｜DSML｜` tokens

3. **Identify JSON structure** (if JSON format):
   - Name key: Usually `name` or `function`
   - Arguments key: Usually `arguments` or `parameters`
   - Array vs single object
   - Multiple calls handling

### Phase 3: Compare with Existing Parsers

**Identify the family from the authoritative cheat-sheet.** The full, current family-to-grammar-to-file mapping (every tool-call and reasoning family) is the "Parser families" section of [`../../../parsers_v2/README.md`](../../../parsers_v2/README.md) — do not maintain a model list here, it drifts. Use it to pick the closest family, then:

1. **Read that family's parser module** under `parsers/src/tool_calling/<family>/` (the batch impl that owns the grammar) to confirm the markers and structure match the model you analyzed.
2. **Review its config preset** in `tool_calling/config.rs` (`ToolCallConfig::<family>()` — start/end tokens, key names, parser type) and its registry entry in `tool_calling/parsers.rs` (`get_tool_parser_map()` → `ParserType`).
3. **Check whether a streaming (v2) parser already exists** under `parsers_v2/src/tool_calling/<family>.rs`. New work targets v2 (pure streaming); v1 (jail-and-buffer batch) is still in use but will be removed once v2 is done.

**Match the analyzed format**:
- If start/end tokens and format match existing parser → Use existing parser with config
- If similar but different tokens → Adapt existing parser config
- If completely different format → Generate new parser

### Phase 4: Generate or Configure Parser

#### Option A: Use Existing Parser (Preferred)

If a match is found, create a configuration preset:

1. Add a new preset function to `parsers/src/tool_calling/config.rs`:
   ```rust
   impl ToolCallConfig {
       pub fn new_model_name() -> Self {
           Self {
               config: ParserConfig::Json(JsonParserConfig {
                   start_token: Some("<marker>".to_string()),
                   end_token: Some("</marker>".to_string()),
                   function_name_key: Some("name".to_string()),
                   function_arguments_key: Some("arguments".to_string()),
                   parser_type: JsonParserType::Basic,
               }),
           }
       }
   }
   ```

2. Register in parser map in `parsers/src/tool_calling/parsers.rs`

3. **Create tests** to verify the configuration works

#### Option B: Generate New Parser (If Needed)

If no existing parser fits, generate new parser code:

1. **Choose parser template** based on format:
   - JSON format → Use `base_json_parser.rs` as template
   - XML format → Use `xml/parser.rs` as template
   - Custom format → Implement three core functions

2. **Implement required functions**:
   ```rust
   // Detection
   pub fn detect_tool_call_start_<name>(chunk: &str, config: &Config) -> bool

   // Parsing
   pub fn try_tool_call_parse_<name>(
       message: &str,
       config: &Config,
       tools: Option<&[ToolDefinition]>,
   ) -> Result<(Vec<ToolCallResponse>, Option<String>)>

   // End detection (for streaming)
   pub fn find_tool_call_end_position_<name>(chunk: &str, config: &Config) -> usize
   ```

3. **Use regex for token matching**:
   - Use `OnceLock<Regex>` for compiled regexes
   - Escape special characters properly
   - Handle partial tokens for streaming

4. **Parse JSON/XML content**:
   - Use `serde_json` for JSON parsing
   - Use regex for XML extraction (or XML parser if complex)
   - Build `ToolCallResponse` structs

5. **Add to appropriate directory**:
   - JSON variants → `json/` directory
   - XML variants → `xml/` directory
   - New format → Create new subdirectory

### Phase 5: Generate Tests

For any new parser or configuration, generate comprehensive tests:

1. **Basic tests**:
   - Detection of start markers
   - Parsing single tool call
   - Parsing multiple tool calls
   - Normal text extraction

2. **Edge cases**:
   - Empty arguments
   - Missing fields
   - Malformed JSON/XML
   - Partial tokens (streaming)

3. **Integration tests**:
   - End-to-end with real model outputs (if available)
   - Tool validation (if tools list provided)

4. **Add tests** to appropriate location:
   - Inline in parser file (in `#[cfg(test)]` module)
   - Or in `parsers/src/tool_calling/tests.rs`

### Phase 6: Integration

1. **Update module exports**:
   - Add `mod` declaration in parent `mod.rs`
   - Export functions as needed

2. **Register parser** in `parsers.rs` if new parser:
   - Add to `get_tool_parser_map()` function
   - **CRITICAL**: Update `test_get_available_tool_parsers()` test
   - Add your new parser name to the `available_parsers` array in the test

3. **Document the parser**:
   - Add doc comments explaining format
   - Include example input/output
   - Reference model family

4. **Run tests** (from the repo root):
   ```bash
   cargo test -p dynamo-parsers tool_calling
   ```

5. **Verify**:
   - Test with actual model output if possible
   - Verify streaming behavior
   - Check error handling

## Key Reference Files

**This repo (`dynamo-parsers`)**:
- `parsers/src/tool_calling/` - All tool call parsers
- `parsers/src/tool_calling/config.rs` - Configuration presets
- `parsers/src/tool_calling/parsers.rs` - Parser registry
- `parsers/src/reasoning/` - Reasoning parsers (same workflow if you're adding
  a `<think>`-style reasoning parser instead of a tool parser)

Note: chat-template loading/structures (`tokcfg.rs`, `template.rs`) live
upstream in `ai-dynamo/dynamo`'s `lib/llm`, not in this crate. Here you fetch
the model's `tokenizer_config.json` directly from HuggingFace (Phase 1) — you
don't need the dynamo template code to build a parser.

**Reference Implementations**:
- **sglang**: https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/function_call
  - Look at detector pattern (base_format_detector.py)
  - Model-specific detectors (qwen25_detector.py, deepseekv3_detector.py, etc.)
- **vLLM**: https://github.com/vllm-project/vllm/tree/main/vllm/tool_parsers
  - Look at abstract_tool_parser.py
  - Model-specific parsers (llama_tool_parser.py, qwen3xml_tool_parser.py, etc.)
- **HuggingFace**: https://huggingface.co/docs/transformers/chat_templating

## Example: Adding Support for a New Model

User: "Add tool calling support for Qwen/Qwen2.5-72B-Instruct"

**Step 1**: Fetch tokenizer config
- Use WebFetch to get `https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/resolve/main/tokenizer_config.json`

**Step 2**: Analyze chat template
- Extract `chat_template` field
- Identify `{% if tools %}` block
- Find markers: Likely `<tool_call>` and `</tool_call>`
- Identify format: Check for JSON with `tojson` filter

**Step 3**: Compare with existing parsers
- Read `parsers/src/tool_calling/config.rs`
- Check `ToolCallConfig::hermes()` - uses `<tool_call>` markers
- Check if Qwen format matches hermes format

**Step 4**: Use or adapt existing parser
- If matches hermes: Create `qwen2_5()` config preset
- If different: Generate new parser or adapt base_json_parser

**Step 5**: Generate tests
- Create test cases with example Qwen tool calls
- Test detection, parsing, and edge cases

**Step 6**: Integrate
- Add config preset to `config.rs`
- Register in parser map (`get_tool_parser_map()`)
- Update `test_get_available_tool_parsers()` test
- Run tests
- Document

## Tips

- **Always prefer existing parsers**: Most models can use existing parsers with different configs
- **Read reference implementations**: sglang and vLLM often have parsers for popular models
- **Use WebFetch for HF models**: Don't assume - always fetch actual tokenizer config
- **Test with real outputs**: If possible, get actual model outputs to test against
- **Keep it simple**: Prefer straightforward regex over complex parsing when possible
- **Document well**: Future you (or others) will thank you

## Common Patterns

The grammar-recognition guide — how to spot each pattern in a chat template — is in [`references/parser-patterns.md`](references/parser-patterns.md). The authoritative family-to-grammar-to-file mapping is the "Parser families" cheat-sheet in [`../../../parsers_v2/README.md`](../../../parsers_v2/README.md). Use those instead of re-listing patterns here.

## Minimal Changes Philosophy

1. **First**: Try existing parser with new config
2. **Second**: Adapt existing parser with minor tweaks
3. **Last resort**: Create entirely new parser

Most models (>80%) can use existing parsers with appropriate configuration.
