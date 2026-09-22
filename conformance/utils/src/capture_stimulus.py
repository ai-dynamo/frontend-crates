# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reused case ID does not prove that a historical capture ran today's input."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from fixture_disposition import (
    CAPTURE_SNAPSHOT, capture_layer_sort_key, capture_snapshot_members,
    canonical_unified_record_key, canonicalize_unified_inputs, inactive_fixture_dirs,
    is_source_capture,
)
from unified_tools import unified_tools


def capture_input(record: dict) -> dict:
    return {
        "input": record.get("input", ""),
        "init": record.get("init") or {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        "finish_reason": record.get("finish_reason") or "stop",
        "tools": record.get("tools"),
        "chunks": [{key: value for key, value in row.items() if key in {"delta_text", "token_ids", "finish_reason"}}
                   for row in record.get("chunks", [])],
    }


def capture_peer_results(cases: list[dict], families, capture, *, tools, supports_finish=False) -> dict:
    """Bind only native/default peer executions; unsupported requests never run."""
    ready, results, bindings = [], {}, {}
    for case in cases:
        if case["family"] not in families:
            continue
        key = case["id"]
        if key in bindings:
            raise ValueError(f"duplicate peer capture case: {key}")
        chunks = case.get("chunks") or []
        if not isinstance(chunks, list) or any(not isinstance(chunk, str) for chunk in chunks):
            raise ValueError(f"peer capture needs literal string chunks: {key}")
        actual = capture_input({"input": case["input"], "tools": tools, "chunks": [{"delta_text": chunk} for chunk in chunks]})
        terminal_step = bool(chunks and chunks[-1] == "‹finish›" and "".join(chunks[:-1]) == case["input"])
        requested = capture_input({**case, "chunks": actual["chunks"]})
        bindings[key] = actual
        unsupported = [field for field in ("init", "finish_reason") if requested[field] != actual[field]]
        if "tools" in case and case["tools"] != tools:
            unsupported.append("tools")
        if unsupported:
            results[key] = {"unavailable": "Peer harness supports only native/default initialization and stop termination; unsupported request: " + ", ".join(unsupported)}
        elif terminal_step and not supports_finish:
            actual["chunks"] = actual["chunks"][:-1]
            results[key] = {"unavailable": "Peer detector harness has no explicit finish operation; authored terminal-step schedules cannot be captured by this harness."}
        elif not terminal_step and "".join(chunks) != case["input"]:
            # A display-only finish row is not text delivered to the parser. Do not
            # invent a finish call or silently drop that row to manufacture parity.
            results[key] = {"unavailable": "Peer chunk text differs from input; synthetic finish rows require an explicit engine finish operation and are not literal input."}
        else:
            ready.append({**case, "chunks": chunks[:-1] if terminal_step else chunks, "terminal_step": terminal_step})
    if ready:
        captured = capture(ready)
        expected = {case["id"] for case in ready}
        if captured.keys() != expected:
            raise ValueError(f"peer capture results differ from executed request: missing={sorted(expected - captured.keys())}, extra={sorted(captured.keys() - expected)}")
        results.update(captured)
    return {key: {**result, "capture_input": bindings[key]} for key, result in results.items()}


def read_bindings(directory: Path) -> dict:
    path = directory / "capture-inputs.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != 1 or not isinstance(doc.get("records"), dict):
        raise ValueError(f"invalid capture input bindings: {path}")
    return doc["records"]


def original_capture_input(record: dict, raw: bytes, relative: str, bindings: dict) -> dict | None:
    original = record.get("capture_input")
    if original is None and relative in bindings:
        binding = bindings[relative]
        if binding["capture_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"capture input binding does not match capture bytes: {relative}")
        original = binding["capture_input"]
    return original


def comparison_failure(record: dict, current: dict, raw: bytes, relative: str, bindings: dict) -> str | None:
    original = original_capture_input(record, raw, relative, bindings)
    if original is None:
        return "Capture stimulus unavailable: original input, initialization, and chunk schedule were not retained; this output cannot be compared to the displayed request."
    expected = capture_input(current)
    if isinstance(original, dict) and ("tools" not in original or original["tools"] is None):
        return "Capture stimulus unavailable: original tool schema was not retained; this output cannot be compared to the displayed request."
    if expected["tools"] is None:
        return "Capture stimulus unavailable: displayed request tool schema was not retained; request equality cannot be verified."
    if not isinstance(original, dict) or original.keys() != expected.keys():
        raise ValueError(f"incomplete capture input binding: {relative}")
    changed = [key for key in expected if original[key] != expected[key]]
    if changed:
        return f"Capture stimulus mismatch ({', '.join(changed)}): historical output belongs to a different request and is not scored against this input."
    return None


def current_source_snapshot(directory: Path) -> Path:
    """Deprecated Rust harness compatibility; canonical YAML has no source patches."""
    if not is_source_capture(directory.name):
        return directory
    candidates = [directory]
    candidates.extend(path for path in directory.parent.glob(directory.name + ".patch*")
                      if path.is_dir() and capture_layer_sort_key(path.name)[0] == directory.name)
    selected = max(candidates, key=lambda path: capture_layer_sort_key(path.name))
    marker = selected / CAPTURE_SNAPSHOT
    if selected != directory or marker.is_file():
        if not marker.is_file():
            raise ValueError(f"source overlay has no complete snapshot index: {selected}")
        capture_snapshot_members(marker.read_bytes(),
                                 [str(path.relative_to(selected)) for path in selected.glob("*/*.yaml")])
    return selected


def _effective_capture_records(directory: Path, input_aliases: dict) -> dict:
    base_name, _patch = capture_layer_sort_key(directory.name)
    base = directory.with_name(base_name)
    inactive = inactive_fixture_dirs(directory.parent)
    if "+source." in base_name:
        layers = [current_source_snapshot(base)]
    else:
        layers = [base, *(path for path in base.parent.glob(base.name + ".patch*")
                          if path.is_dir() and capture_layer_sort_key(path.name)[0] == base.name)]
        layers.sort(key=lambda path: capture_layer_sort_key(path.name))
    captures = {}
    for layer in layers:
        if layer.name in inactive:
            continue
        bindings = read_bindings(layer)
        records = {}
        for path in sorted(layer.glob("*/*.yaml")):
            raw = path.read_bytes()
            doc = yaml.safe_load(raw)
            if doc["family"] != path.parent.name:
                raise ValueError(f"capture family differs from its directory: {path}")
            for key, record in doc["cases"].items():
                ident = canonical_unified_record_key(doc["family"], key, input_aliases)
                if ident in records and records[ident][0] != record:
                    raise ValueError(f"conflicting current capture aliases: {ident}")
                records[ident] = (record, raw, str(path.relative_to(layer)), bindings)
        captures.update(records)
    return captures


def validated_current_capture_docs(directory: Path, input_dirs: list[Path]) -> list[dict]:
    """Return the effective records checked here so Rust cannot reread a different layer."""
    raw_inputs = {}
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*/*.yaml")):
            doc = yaml.safe_load(path.read_bytes())
            for key, record in doc["cases"].items():
                raw_inputs[(doc["family"], key)] = record
    inputs, input_aliases = canonicalize_unified_inputs(raw_inputs)
    input_keys = {}
    for (family, key), record in raw_inputs.items():
        ident = canonical_unified_record_key(family, key, input_aliases)
        if inputs.get(ident) == record:
            input_keys[ident] = key
    captures = _effective_capture_records(directory, input_aliases)
    if not inputs or inputs.keys() != captures.keys():
        raise ValueError(f"current capture/input sets differ: missing={sorted(inputs.keys() - captures.keys())}, extra={sorted(captures.keys() - inputs.keys())}")
    tools = unified_tools()
    for ident, current in inputs.items():
        if current.get("tools") != tools:
            raise ValueError(f"current input tools differ from executable shared schema: {ident}")
        record, raw, relative, bindings = captures[ident]
        failure = comparison_failure(record, current, raw, relative, bindings)
        if failure:
            raise ValueError(f"{ident}: {failure}")
        if "unavailable" in record or "error" in record:
            raise ValueError(f"current capture did not succeed: {ident}")
    families = {}
    for (family, key), (record, _raw, _relative, _bindings) in sorted(captures.items()):
        families.setdefault(family, {})[input_keys[(family, key)]] = record
    return [{"family": family, "cases": records} for family, records in families.items()]


def validate_current_capture(directory: Path, input_dirs: list[Path]) -> int:
    return sum(len(doc["cases"]) for doc in validated_current_capture_docs(directory, input_dirs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select-source-snapshot", type=Path, help="deprecated Rust harness compatibility")
    mode.add_argument("--validate-current", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--format", choices=("count", "json"), default="count")
    args = parser.parse_args()
    if args.select_source_snapshot is not None:
        if args.format != "count":
            parser.error("--format json requires --validate-current")
        print(current_source_snapshot(args.select_source_snapshot))
    else:
        if not args.inputs:
            parser.error("--validate-current requires --inputs")
        if args.format == "json":
            print(json.dumps(validated_current_capture_docs(args.validate_current, args.inputs)))
        else:
            print(validate_current_capture(args.validate_current, args.inputs))


if __name__ == "__main__":
    main()
