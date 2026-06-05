// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PyO3 0.22 macro expansion still emits `unsafe_op_in_unsafe_fn` under Rust 2024.
#![allow(unsafe_op_in_unsafe_fn)]

//! PyO3 Python bindings for `dynamo-parsers-v2`.
//!
//! Build with maturin:
//!   cd parsers_v2-py
//!   maturin develop          # installs into the active venv
//!   maturin build --release  # produces a wheel
//!
//! ## vLLM integration
//!
//! vLLM's concrete harmony subclass would look like:
//!
//! ```python
//! from dynamo_parsers_v2 import HarmonyToolStreamParser
//!
//! class DynamoHarmonyParser(ToolParser):
//!     def __init__(self, tokenizer):
//!         super().__init__(tokenizer)
//!         self._parser = HarmonyToolStreamParser()
//!
//!     def extract_tool_calls_streaming(
//!         self, previous_text, current_text, delta_text,
//!         previous_token_ids, current_token_ids, delta_token_ids, request
//!     ):
//!         # Only delta_token_ids is needed — the parser maintains its own state.
//!         chunks = self._parser.parse(list(delta_token_ids))
//!         if not chunks:
//!             return None  # nothing to emit yet
//!         return _to_delta_message(chunks)
//!
//!     def extract_tool_calls(self, model_output, request):
//!         # Batch path — not handled here; use dynamo-parsers directly.
//!         raise NotImplementedError
//!
//! def _to_delta_message(chunks):
//!     from vllm.entrypoints.openai.protocol import DeltaMessage, DeltaToolCall, DeltaFunctionCall
//!     tool_calls = []
//!     for c in chunks:
//!         tool_calls.append(DeltaToolCall(
//!             index=c.index,
//!             id=c.id,
//!             type=c.call_type,
//!             function=DeltaFunctionCall(
//!                 name=c.function_name,
//!                 arguments=c.function_arguments,
//!             ) if c.function_name or c.function_arguments else None,
//!         ))
//!     return DeltaMessage(tool_calls=tool_calls)
//! ```

use dynamo_parsers::tool_calling::ToolCallResponseChunk;
use pyo3::prelude::*;

// Import the Rust crate under an alias to avoid the name collision with the
// #[pymodule] entry-point function (both would be `dynamo_parsers_v2`).
use dynamo_parsers_v2::HarmonyToolStreamParser;
use dynamo_parsers_v2::ToolStreamResult;

// ── Python-exposed chunk type ─────────────────────────────────────────────────

/// One streaming delta — maps to one element of vLLM's `DeltaMessage.tool_calls[]`
/// and the OpenAI streaming wire shape.
///
/// The first delta for a given `index` carries `id`, `call_type`, and
/// `function_name`. All subsequent deltas for the same `index` carry only
/// `function_arguments`. This is the OpenAI streaming contract:
/// name is committed once, arguments are streamed as fragments.
#[pyclass(name = "ToolCallChunk", get_all, frozen)]
#[derive(Clone, Debug)]
pub struct PyToolCallChunk {
    /// Zero-based call index. Parallel tool calls use different indices.
    pub index: u32,
    /// Unique call id (e.g. `"call_00000000"`). Present only on the first delta
    /// for this index; `None` on all subsequent argument-fragment deltas.
    pub id: Option<String>,
    /// Always `"function"` when present; `None` on argument-fragment deltas.
    /// Named `call_type` to avoid shadowing Python's built-in `type`.
    pub call_type: Option<String>,
    /// Function name. Present only on the first delta for this index.
    pub function_name: Option<String>,
    /// Argument fragment. `None` on the name delta; a JSON substring on all
    /// subsequent deltas. The consumer concatenates all fragments per index.
    pub function_arguments: Option<String>,
}

#[pymethods]
impl PyToolCallChunk {
    fn __repr__(&self) -> String {
        format!(
            "ToolCallChunk(index={}, id={:?}, call_type={:?}, function_name={:?}, function_arguments={:?})",
            self.index, self.id, self.call_type, self.function_name, self.function_arguments
        )
    }
}

// ── Python-exposed parser ─────────────────────────────────────────────────────

/// Token-incremental Harmony (gpt-oss) tool-call streaming parser.
///
/// Instantiate once per request. Call `parse(delta_token_ids)` for each chunk
/// from the model, then `finish()` at stream end. The parser is stateful —
/// do not share across requests.
///
/// Designed to plug directly into vLLM's `ToolParser.extract_tool_calls_streaming`
/// contract. Only `delta_token_ids` is used; `previous_text`, `current_text`,
/// `delta_text`, etc. are not needed because the inner `StreamableParser`
/// maintains its own incremental state.
#[pyclass(name = "HarmonyToolStreamParser", unsendable)]
pub struct PyHarmonyToolStreamParser {
    inner: HarmonyToolStreamParser,
}

#[pymethods]
impl PyHarmonyToolStreamParser {
    /// Create a new parser. Loads the gpt-oss harmony encoding on first call
    /// (cached globally — subsequent instantiations are fast).
    #[new]
    fn new() -> PyResult<Self> {
        let inner = HarmonyToolStreamParser::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Feed one chunk of token ids and return any deltas emitted.
    ///
    /// Pass the `delta_token_ids` argument from vLLM's
    /// `extract_tool_calls_streaming` directly. Returns an empty list when
    /// there is nothing to emit yet (equivalent to vLLM returning `None`).
    fn parse(&mut self, delta_token_ids: Vec<u32>) -> Vec<PyToolCallChunk> {
        to_py(
            self.inner
                .parse_tool_call_streaming_incremental(&delta_token_ids),
        )
    }

    /// Feed one chunk of text (encodes to token ids internally).
    ///
    /// Convenience wrapper for callers that have `delta_text` but not token ids.
    /// For harmony the token path (`parse`) is preferred.
    fn parse_text(&mut self, delta_text: &str) -> Vec<PyToolCallChunk> {
        to_py(self.inner.parse_tool_call_streaming_text(delta_text))
    }

    /// Signal end of stream. Call once after all chunks have been fed.
    ///
    /// Drives the inner parser to its terminal state. Typically returns an
    /// empty list for harmony since `<|call|>` closes each call inline.
    fn finish(&mut self) -> Vec<PyToolCallChunk> {
        to_py(self.inner.finish_tool_call_stream())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn to_py(result: ToolStreamResult) -> Vec<PyToolCallChunk> {
    result
        .tool_call_chunks
        .into_iter()
        .map(chunk_to_py)
        .collect()
}

fn chunk_to_py(c: ToolCallResponseChunk) -> PyToolCallChunk {
    PyToolCallChunk {
        index: c.index,
        id: c.id,
        call_type: c.tp.map(|_| "function".to_string()),
        function_name: c.function.as_ref().and_then(|f| f.name.clone()),
        function_arguments: c.function.as_ref().and_then(|f| f.arguments.clone()),
    }
}

// ── Module entry point ────────────────────────────────────────────────────────

/// PyO3 module entry point. Named `init_module` to avoid colliding with the
/// `dynamo_parsers_v2` crate import. `#[pymodule(name = ...)]` tells
/// PyO3 to export the `PyInit_dynamo_parsers_v2` symbol regardless.
#[pymodule(name = "dynamo_parsers_v2")]
fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHarmonyToolStreamParser>()?;
    m.add_class::<PyToolCallChunk>()?;
    Ok(())
}
