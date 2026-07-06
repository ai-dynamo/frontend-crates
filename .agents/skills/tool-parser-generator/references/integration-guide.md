# Parser Integration Guide

Step-by-step guide for wiring a parser into the v1 batch crate (`parsers/`): the `ToolCallConfig` preset + `get_tool_parser_map()` registry mechanics.

> This is the **v1 (batch / jail-and-buffer)** integration mechanism. v1 is still in use, but the pure-streaming v2 path will fully replace it — v1 is not being merged in, it will be removed once v2 is done — so new parser work targets v2: follow "Adding A Day-0 Tool-Calling Parser" in [`../../../../parsers/v2/README.md`](../../../../parsers/v2/README.md), and pick the family from its "Parser families" cheat-sheet. The steps below still apply when a new model reuses an existing v1 family via config.

## Option 1: Add Configuration Preset (Most Common)

When an existing parser can handle the new model with different configuration.

### Step 1: Add Config Preset

Edit `parsers/v1/src/tool_calling/config.rs`:

```rust
impl ToolCallConfig {
    /// Configuration for ModelName
    pub fn model_name() -> Self {
        Self {
            config: ParserConfig::Json(JsonParserConfig {
                start_token: Some("<start>".to_string()),
                end_token: Some("</end>".to_string()),
                function_name_key: Some("name".to_string()),
                function_arguments_key: Some("arguments".to_string()),
                parser_type: JsonParserType::Basic,
            }),
        }
    }
}
```

### Step 2: Register in Parser Map

Edit `parsers/v1/src/tool_calling/parsers.rs`:

In the `get_tool_parser_map()` function:

```rust
map.insert("model_name".to_string(), ParserType::Json);
```

### Step 3: Add Tests

In the same file or in `tests.rs`:

```rust
#[test]
fn test_model_name_parser() {
    let config = ToolCallConfig::model_name();
    let message = r#"<start>{"name": "test", "arguments": {}}</start>"#;

    let (calls, _) = detect_and_parse_tool_call(message, Some("model_name"), None).unwrap();

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "test");
}
```

### Step 4: Run Tests

```bash
# from the repo root
cargo test -p dynamo-parsers model_name
```

## Option 2: Create New Parser

When the format is truly unique and doesn't fit existing parsers.

### Step 1: Choose Directory

- JSON variant → `json/`
- XML variant → `xml/`
- New format → Create new subdirectory

### Step 2: Create Parser File

Example: `json/model_name_parser.rs`

```rust
// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use regex::Regex;
use std::sync::OnceLock;

use crate::tool_calling::{
    config::{JsonParserConfig, ToolDefinition},
    response::{CalledFunction, ToolCallResponse, ToolCallType},
};

static DETECT_REGEX: OnceLock<Regex> = OnceLock::new();

pub fn detect_tool_call_start_model_name(
    chunk: &str,
    config: &JsonParserConfig,
) -> bool {
    // Implementation
    todo!()
}

pub fn try_tool_call_parse_model_name(
    message: &str,
    config: &JsonParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> Result<(Vec<ToolCallResponse>, Option<String>)> {
    // Implementation
    todo!()
}

pub fn find_tool_call_end_position_model_name(
    chunk: &str,
    config: &JsonParserConfig,
) -> usize {
    // Implementation
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection() {
        // Tests
    }
}
```

### Step 3: Update mod.rs

In the parent directory's `mod.rs`:

```rust
pub mod model_name_parser;

pub use model_name_parser::*;
```

### Step 4: Add Config Enum Variant

Edit `config.rs` to add your parser type if needed:

```rust
pub enum ParserConfig {
    Json(JsonParserConfig),
    Xml(XmlParserConfig),
    // ... existing variants
    ModelName(ModelNameConfig), // If you need custom config
}
```

### Step 5: Register in parsers.rs

Add routing logic in `try_tool_call_parse()`:

```rust
match config {
    // ... existing matches
    ParserConfig::ModelName(cfg) => {
        model_name_parser::try_tool_call_parse_model_name(message, cfg, tools)
    }
}
```

### Step 6: Run All Tests

```bash
# from the repo root
cargo test -p dynamo-parsers tool_calling
cargo clippy
cargo fmt
```

## File Structure

The v1 tool-call crate root is `parsers/v1/src/tool_calling/`:

- `config.rs` — `ToolCallConfig` presets and the `ParserConfig` enum
- `parsers.rs` — `get_tool_parser_map()` registry and dispatch
- `response.rs` — `ToolCallResponse` types; `tools.rs` — high-level APIs; `tests.rs` — integration tests
- per-family grammar modules in subdirectories: `json/`, `xml/`, `dsml/`, `gemma4/`, `harmony/`, `pythonic/`

For which family owns which model and the exact module per family, use the "Parser families" cheat-sheet in [`../../../../parsers/v2/README.md`](../../../../parsers/v2/README.md) rather than a tree here (a tree drifts as families are added). Reasoning parsers live in `parsers/v1/src/reasoning/` (see its README).

## Testing Strategy

### Unit Tests
Test individual functions in the parser file:
- `detect_tool_call_start_*`
- `try_tool_call_parse_*`
- `find_tool_call_end_position_*`

### Integration Tests
Test through the main API in `tests.rs`:
- `detect_and_parse_tool_call()` with parser name
- Streaming behavior
- Tool validation

### Example Output Tests
Use real model outputs when possible:
- Get actual tool call from model
- Verify parsing produces correct structure
- Check normal text extraction

## Common Issues

### Issue: Parser not found
**Solution**: Check registration in `get_tool_parser_map()`

### Issue: Detection works but parsing fails
**Solution**: Check regex patterns and JSON structure keys

### Issue: Streaming produces wrong results
**Solution**: Verify `find_tool_call_end_position_*` implementation

### Issue: Tests fail with "unknown tool"
**Solution**: Either provide tools list or remove validation

## Documentation

Add doc comments to your parser:

```rust
//! ModelName tool call parser
//!
//! Format: <start>JSON</start>
//!
//! Example:
//! ```text
//! <start>{"name": "func", "arguments": {}}</start>
//! ```
//!
//! Models: ModelName family
```

## Checklist

- [ ] Parser implements three required functions
- [ ] Config preset added (if using existing parser)
- [ ] Parser registered in map
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation added
- [ ] Clippy warnings resolved
- [ ] Code formatted with rustfmt
