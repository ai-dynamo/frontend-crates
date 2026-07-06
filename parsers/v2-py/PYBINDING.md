# PyO3 binding guide — `dynamo-parsers-v2`

This document explains how to add a Python binding for a new parser family and make it compatible with vLLM's and SGLang's streaming interfaces. It uses the existing Harmony binding in `parsers/v2-py/src/lib.rs` as the reference implementation.

---

## Why a separate crate

The Rust parser (`parsers/v2`) is a pure `rlib` used by the conformance test suite and other Rust consumers. PyO3 requires a `cdylib` that links against the Python runtime — mixing `rlib` and `cdylib` in one crate causes symbol conflicts and breaks both. `parsers/v2-py` is the thin `cdylib` wrapper that only exists to produce the Python extension module.

---

## vLLM's streaming interface (Python)

```python
# vllm/tool_parsers/abstract_tool_parser.py
class ToolParser:
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,          # full accumulated text: previous + delta
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],  # full accumulated ids
        delta_token_ids: Sequence[int],
        request: "ChatCompletionRequest",
    ) -> DeltaMessage | None:
        ...

# vllm/entrypoints/openai/protocol.py
@dataclass
class DeltaFunctionCall:
    name: str | None = None
    arguments: str | None = None   # JSON fragment, concatenate across chunks

@dataclass
class DeltaToolCall:
    index: int
    id: str | None = None           # present only on FIRST delta for this index
    type: str | None = None         # "function" | None
    function: DeltaFunctionCall | None = None

@dataclass
class DeltaMessage:
    content: str | None = None
    tool_calls: list[DeltaToolCall] | None = None
```

**Key contract:**
- Called once per engine chunk with `delta_token_ids` (and text variants).
- Returns `DeltaMessage | None` — `None` means nothing to emit yet.
- First delta for each `index` carries `id` + `function.name` (no arguments).
- All subsequent deltas for the same `index` carry `function.arguments` only.
- The client concatenates `arguments` fragments; name is committed once.

## SGLang's streaming interface (Python)

```python
# sglang/srt/function_call/base_format_detector.py
class BaseFormatDetector:
    def parse_streaming_increment(
        self,
        new_text: str,              # delta text only — no token ids
        tools: list[Tool],
    ) -> StreamingParseResult:
        ...

@dataclass
class ToolCallItem:
    tool_index: int
    name: str | None
    parameters: str              # JSON string, may be partial

@dataclass
class StreamingParseResult:
    normal_text: str             # text not part of any tool call
    calls: list[ToolCallItem]
```

**Key difference from vLLM:** SGLang is text-only (no token ids). The result type is also different — `normal_text` + `calls[]` rather than vLLM's `DeltaMessage`.

---

## Our Rust types

```rust
// dynamo-parsers: the core chunk type
pub struct ToolCallResponseChunk {
    pub index: u32,
    pub id: Option<String>,          // Some only on first delta per index
    pub tp: Option<ToolCallType>,    // Some(Function) only on first delta
    pub function: Option<CalledFunctionStream>,
}
pub struct CalledFunctionStream {
    pub name: Option<String>,        // Some only on first delta
    pub arguments: Option<String>,   // Some on subsequent deltas; None on first
}

// dynamo-parsers-v2: what the parser emits per chunk
pub struct ToolStreamResult {
    pub tool_call_chunks: Vec<ToolCallResponseChunk>,
}
```

---

## How PyO3 bridges Rust → Python

### 1. The Python-visible chunk type

```rust
#[pyclass(name = "ToolCallChunk", get_all, frozen)]
pub struct PyToolCallChunk {
    pub index: u32,
    pub id: Option<String>,
    pub call_type: Option<String>,    // renamed: "type" is a Python builtin
    pub function_name: Option<String>,
    pub function_arguments: Option<String>,
}
```

`get_all` exposes all fields as Python attributes. `frozen` makes instances immutable (no `__setattr__`). `call_type` is renamed from `type` because `type` is a Python builtin — document this for callers.

### 2. The Python-visible parser class

```rust
#[pyclass(name = "HarmonyToolStreamParser", unsendable)]
pub struct PyHarmonyToolStreamParser { inner: HarmonyToolStreamParser }
```

`unsendable` is required when the inner type is not `Send` (e.g., it holds a `StreamableParser` that contains non-Send internals). This prevents Python threads from moving the object across threads, matching the "one parser per request" contract.

### 3. Type mapping table

| Python (vLLM/SGLang) | Rust | PyO3 bridge |
|---|---|---|
| `Sequence[int]` | `Vec<u32>` | automatic via `FromPyObject` |
| `str` | `&str` | automatic via `FromPyObject` |
| `str \| None` | `Option<String>` | automatic; `None` ↔ `Option::None` |
| `DeltaToolCall` | `ToolCallResponseChunk` | manual `chunk_to_py()` |
| `DeltaMessage \| None` | `Vec<PyToolCallChunk>` | empty list = `None` equivalent |
| `list[ToolCallItem]` (SGLang) | `Vec<PyToolCallChunk>` | same chunks, different consumer |

---

## Adding a new family

1. **Implement the Rust streaming parser** in `parsers/v2/src/tool_calling/`:
   - Add `parse_<family>_streaming_incremental(&mut self, delta_token_ids: &[u32]) -> ToolStreamResult`
   - Add `parse_<family>_streaming_text(&mut self, delta_text: &str) -> ToolStreamResult`
   - Add `finish_<family>_stream(&mut self) -> ToolStreamResult`

2. **Add a Python class** in `parsers/v2-py/src/lib.rs`:
   ```rust
   #[pyclass(name = "FamilyToolStreamParser", unsendable)]
   pub struct PyFamilyParser { inner: FamilyToolStreamParser }

   #[pymethods]
   impl PyFamilyParser {
       #[new]
       fn new() -> PyResult<Self> { ... }
       fn parse(&mut self, delta_token_ids: Vec<u32>) -> Vec<PyToolCallChunk> { ... }
       fn parse_text(&mut self, delta_text: &str) -> Vec<PyToolCallChunk> { ... }
       fn finish(&mut self) -> Vec<PyToolCallChunk> { ... }
   }
   ```

3. **Register in the module**:
   ```rust
   #[pymodule]
   fn dynamo_parsers_v2(m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add_class::<PyHarmonyToolStreamParser>()?;
       m.add_class::<PyFamilyParser>()?;      // add here
       m.add_class::<PyToolCallChunk>()?;
       Ok(())
   }
   ```

---

## vLLM adapter (what vLLM writes)

```python
from dynamo_parsers_v2 import HarmonyToolStreamParser, ToolCallChunk
from vllm.entrypoints.openai.protocol import (
    DeltaFunctionCall, DeltaMessage, DeltaToolCall
)

class DynamoHarmonyParser(ToolParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._parser = HarmonyToolStreamParser()

    def extract_tool_calls_streaming(
        self, previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids, request
    ) -> DeltaMessage | None:
        # Only delta_token_ids is used — the parser maintains its own state.
        chunks: list[ToolCallChunk] = self._parser.parse(list(delta_token_ids))
        if not chunks:
            return None
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=c.index,
                    id=c.id,
                    type=c.call_type,   # note: call_type, not type
                    function=DeltaFunctionCall(
                        name=c.function_name,
                        arguments=c.function_arguments,
                    ) if c.function_name or c.function_arguments else None,
                )
                for c in chunks
            ]
        )

    def extract_tool_calls(self, model_output, request):
        raise NotImplementedError  # use dynamo-parsers batch API instead
```

**Notes:**
- `previous_text`, `current_text`, `delta_text`, `previous_token_ids`, `current_token_ids`, `request` are all ignored — harmony's `StreamableParser` is stateful and only needs `delta_token_ids`.
- Text-only families (if/when added) would use `parse_text(delta_text)` instead.
- Call `self._parser.finish()` when the stream ends (on `finish_reason` set).

## SGLang adapter (what SGLang writes)

```python
from dynamo_parsers_v2 import HarmonyToolStreamParser, ToolCallChunk
from sglang.srt.function_call.base_format_detector import StreamingParseResult, ToolCallItem

class DynamoHarmonyDetector(BaseFormatDetector):
    def __init__(self):
        self._parser = HarmonyToolStreamParser()

    def parse_streaming_increment(self, new_text, tools) -> StreamingParseResult:
        # SGLang is text-only. For harmony, encode text → tokens internally.
        chunks: list[ToolCallChunk] = self._parser.parse_text(new_text)
        calls = [
            ToolCallItem(
                tool_index=c.index,
                name=c.function_name,
                parameters=c.function_arguments or "",
            )
            for c in chunks
            if c.function_name or c.function_arguments
        ]
        return StreamingParseResult(normal_text="", calls=calls)
```

---

## Build

```bash
cd parsers/v2-py
maturin develop          # install into active venv (dev build)
maturin build --release  # produce a wheel for distribution

# Verify:
python3 -c "from dynamo_parsers_v2 import HarmonyToolStreamParser; p = HarmonyToolStreamParser(); print(p.parse([200005]))"
```

## Migration note

`parsers/v2-py` is temporary for the bridge period. It exists so Dynamo parser v2 can expose a Python module without mixing `rlib` and `cdylib` outputs in the same Rust crate. After Dynamo consumes the released frontend-crates parser crate directly and parser-source rsync stops, move this binding surface into the parser crate's normal Python binding package and remove the temporary `dynamo_parsers_v2` module/package name.
