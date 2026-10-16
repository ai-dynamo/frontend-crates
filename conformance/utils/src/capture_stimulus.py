# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reused case ID does not prove that a historical capture ran today's input."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from fixture_disposition import canonical_unified_record_key, canonicalize_unified_inputs
from impls import UNIFIED_BATCH_IMPLS


_UNAVAILABLE_CATEGORIES = {
    **dict.fromkeys(("peer_request_unsupported", "peer_finish_unsupported", "peer_termination_unsupported",
                     "peer_chunk_schedule_unsupported", "vllm_chunk_schedule_unsupported", "vllm_tool_schema_unsupported",
                     "vllm_rust_guided_json_unsupported", "vllm_rust_named_tool_unsupported", "vllm_rust_starting_state_unsupported"),
                    "unsupported_request_mode"),
    **dict.fromkeys(("peer_detector_unavailable", "vllm_parser_not_registered", "vllm_rust_parser_not_registered"),
                    "unsupported_family"),
}


def capture_failure_detail(payload: dict) -> object:
    return payload.get("detail", payload.get("message", payload.get("code")))


def capture_failure_text(payload: dict) -> str:
    detail = capture_failure_detail(payload)
    text = detail if isinstance(detail, str) else payload.get("code")
    if not isinstance(text, str):
        raise ValueError("capture failure needs string detail, message, or code")
    return text


def normalize_unavailable(record: dict) -> dict:
    """Project typed limitations without changing stored observations or error kinds."""
    if "unavailable" not in record:
        return record
    result = dict(record)
    typed = ((record.get("capture_observation") or {}).get("unavailable")
             or (record.get("capture_stimulus") or {}).get("unavailable") or {})
    if isinstance(record["unavailable"], dict):
        if not typed:
            typed = record["unavailable"]
            result["capture_observation"] = {**(record.get("capture_observation") or {}), "unavailable": dict(typed)}
        result["unavailable"] = capture_failure_text(typed)
    code = typed.get("code") or record.get("unavailable_code")
    if code is not None:
        result["unavailable_code"] = _UNAVAILABLE_CATEGORIES.get(code, code)
    return result


def validate_family_scoped_cases(cases: list[dict]) -> None:
    """Reject jobs whose normalized IDs could collide across families."""
    families = {case.get("family") for case in cases}
    if len(families) > 1:
        raise ValueError(
            "Unified peer capture jobs must contain exactly one family; "
            f"received {sorted(families)!r}"
        )
    ids = [case.get("id") for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate peer capture case IDs: {duplicates!r}")


def capture_input(record: dict) -> dict:
    return {
        "input": record.get("input", ""),
        "init": record.get("init") or {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        "finish_reason": record.get("finish_reason") or "stop",
        "tools": record.get("tools"),
        "chunks": [{key: value for key, value in row.items() if key in {"delta_text", "token_ids", "finish_reason"}}
                   for row in record.get("chunks", [])],
    }


def unavailable_result(
    code: str,
    detail: str,
    *,
    capture: dict | None = None,
    observation: bool = False,
) -> dict:
    """Describe a peer limitation and retain the request that was not executed."""
    field = "capture_observation" if observation else "capture_stimulus"
    result = {
        "unavailable": detail,
        field: {"unavailable": {"code": code, "detail": detail}},
    }
    if capture is not None:
        result["capture_input"] = capture
    return result


def capture_peer_results(
    cases: list[dict],
    families,
    capture,
    *,
    tools,
    supports_finish=False,
    supports_init=False,
    supports_tools=False,
    unsupported_reason=None,
) -> dict:
    """Bind supported peer executions; unsupported requests never run."""
    validate_family_scoped_cases(cases)
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

        def unavailable(code: str, detail: str) -> dict:
            return unavailable_result(code, detail, capture=requested)

        reason = unsupported_reason(case) if unsupported_reason is not None else None
        if reason is not None:
            if key in results:
                raise ValueError(f"duplicate peer capture case: {key}")
            code, detail = reason
            results[key] = unavailable(code, detail)
            continue
        unsupported = []
        if not supports_init and requested["init"] != actual["init"]:
            unsupported.append("init")
        if not supports_finish and requested["finish_reason"] != actual["finish_reason"]:
            unsupported.append("finish_reason")
        if not supports_tools and "tools" in case and case["tools"] != tools:
            unsupported.append("tools")
        if unsupported:
            if key in results:
                raise ValueError(f"duplicate peer capture case: {key}")
            detail = "Peer harness supports only native/default initialization and stop termination; unsupported request: " + ", ".join(unsupported)
            results[key] = unavailable("peer_request_unsupported", detail)
        elif terminal_step and not supports_finish:
            if key in results:
                raise ValueError(f"duplicate peer capture case: {key}")
            actual["chunks"] = actual["chunks"][:-1]
            results[key] = unavailable(
                "peer_finish_unsupported",
                "Peer detector harness has no explicit finish operation; authored terminal-step schedules cannot be captured by this harness.",
            )
        elif not terminal_step and "".join(chunks) != case["input"]:
            # A display-only finish row is not text delivered to the parser. Do not
            # invent a finish call or silently drop that row to manufacture parity.
            if key in results:
                raise ValueError(f"duplicate peer capture case: {key}")
            results[key] = unavailable(
                "peer_chunk_schedule_unsupported",
                "Peer chunk text differs from input; synthetic finish rows require an explicit engine finish operation and are not literal input.",
            )
        else:
            bindings[key] = requested if supports_init or supports_finish or supports_tools else actual
            ready.append({**case, "chunks": chunks[:-1] if terminal_step else chunks, "terminal_step": terminal_step})
    if ready:
        captured = capture(ready)
        expected = {case["id"] for case in ready}
        if captured.keys() != expected:
            raise ValueError(f"peer capture results differ from executed request: missing={sorted(expected - captured.keys())}, extra={sorted(captured.keys() - expected)}")
        for result in captured.values():
            if "error" in result and "capture_observation" not in result:
                result["capture_observation"] = {
                    "error": {
                        "code": "peer_parser_error",
                        "detail": result["error"],
                    }
                }
        results.update(captured)
    return {
        key: ({**result, "capture_input": bindings[key]} if key in bindings else result)
        for key, result in results.items()
    }


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


def select_peer_observation(record: dict, current: dict, *, implementation: str, version: str) -> dict:
    """Use a bound recapture while retaining the original historical observation."""
    recapture = record.get("verified_recapture")
    if recapture is None:
        return record
    batch = implementation in UNIFIED_BATCH_IMPLS
    if implementation not in {"sglang_python", "vllm_python", "vllm_rust"} and not batch:
        raise ValueError("verified recapture requires a supported peer implementation")
    if not isinstance(recapture, dict) or set(recapture) != {"result", "provenance", "resolution"}:
        raise ValueError("invalid verified recapture")
    provenance = recapture["provenance"]
    if implementation != "sglang_python" and provenance.get("implementation") != implementation:
        raise ValueError("verified recapture implementation differs from candidate")
    if provenance.get("version") != version:
        raise ValueError("verified recapture version differs from candidate")
    for key in ("source_sha256", "adapter_sha256", "job_sha256"):
        digest = provenance.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"verified recapture has invalid {key}")
    resolution = recapture["resolution"]
    if resolution == "historical_request_not_retained":
        if record.get("capture_stimulus") != {"unavailable": {"code": "original_request_not_retained"}}:
            raise ValueError("verified recapture requires an unbound historical peer observation")
    elif resolution == "historical_adapter_unsupported" and implementation != "sglang_python":
        if "unavailable" not in record or record.get("capture_input") != capture_input(current):
            raise ValueError("adapter recapture requires a bound historical unavailable result")
    else:
        raise ValueError("unresolved historical recapture disagreement")
    result = recapture["result"]
    if not isinstance(result, dict) or result.get("capture_input") != capture_input(current):
        raise ValueError("verified recapture request differs from displayed request")
    states = sum(key in result for key in ("error", "unavailable", "assembled" if batch else "chunks"))
    if states != 1:
        raise ValueError("verified recapture must carry exactly one result state")
    if batch:
        if "chunks" in result or ("assembled" in result and result["assembled"] != batch_final_events(result.get("batch_final", {}))):
            raise ValueError("invalid verified batch final fields")
        result = {**result, "producer_origin": {"crate_version": version, **provenance}}
    if resolution == "historical_adapter_unsupported" and "unavailable" in result:
        return {**result, "historical_capture_note": (
            "This comparison uses the recapture adapter's verified request compatibility "
            "classification; the unsupported request was not executed. The original "
            "unavailable result from the earlier capture adapter is preserved in YAML."
        )}
    if resolution == "historical_adapter_unsupported":
        return {**result, "historical_capture_note": (
            "This comparison uses a verified execution of the displayed request. The original "
            "unavailable result from the earlier capture adapter is preserved in YAML."
        )}
    return {**result, "historical_capture_note": (
        "This comparison uses a verified recapture of the displayed request. The original "
        "observation is preserved in YAML; its request was not retained, so output differences "
        "cannot establish a parser change."
    )}


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


def _effective_capture_records(directory: Path, input_aliases: dict) -> dict:
    captures = {}
    bindings = read_bindings(directory)
    for path in sorted(directory.glob("*/*.yaml")):
        raw = path.read_bytes()
        doc = yaml.safe_load(raw)
        if doc["family"] != path.parent.name:
            raise ValueError(f"capture family differs from its directory: {path}")
        for key, record in doc["cases"].items():
            ident = canonical_unified_record_key(doc["family"], key, input_aliases)
            if ident in captures and captures[ident][0] != record:
                raise ValueError(f"conflicting current capture aliases: {ident}")
            captures[ident] = (record, raw, str(path.relative_to(directory)), bindings)
    return captures


def validated_current_capture_docs(
    directory: Path, input_dirs: list[Path], families: set[str] | None = None,
) -> list[dict]:
    """Return the effective records checked here so Rust cannot reread a different layer."""
    raw_inputs = {}
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*/*.yaml")):
            doc = yaml.safe_load(path.read_bytes())
            if families is not None and doc["family"] not in families:
                continue
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
    for ident, current in inputs.items():
        if not isinstance(current.get("tools"), list):
            raise ValueError(f"current input has no executable tool schema: {ident}")
        record, raw, relative, bindings = captures[ident]
        failure = comparison_failure(record, current, raw, relative, bindings)
        if failure:
            raise ValueError(f"{ident}: {failure}")
        if "unavailable" in record:
            raise ValueError(f"current capture did not succeed: {ident}")
        if "error" in record:
            known = yaml.safe_load((Path(__file__).resolve().parents[2] / "unified-known-divergences.yaml").read_text())
            expected_error = known.get(ident[0], {}).get(ident[1], {}).get("golden", {}).get("actual")
            if expected_error != f"ERROR: {record['error']}" or "assembled" in record or "chunks" in record:
                raise ValueError(f"current capture did not succeed (unexpected error): {ident}")
    families = {}
    for (family, key), (record, _raw, _relative, _bindings) in sorted(captures.items()):
        families.setdefault(family, {})[input_keys[(family, key)]] = record
    return [{"family": family, "cases": records} for family, records in families.items()]


def validate_current_capture(
    directory: Path, input_dirs: list[Path], families: set[str] | None = None,
) -> int:
    return sum(len(doc["cases"]) for doc in validated_current_capture_docs(directory, input_dirs, families))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-current", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--families", nargs="+", help="restrict validation to these explicitly applicable families")
    parser.add_argument("--format", choices=("count", "json"), default="count")
    args = parser.parse_args()
    if not args.inputs:
        parser.error("--validate-current requires --inputs")
    families = set(args.families) if args.families is not None else None
    if args.format == "json":
        print(json.dumps(validated_current_capture_docs(args.validate_current, args.inputs, families)))
    else:
        print(validate_current_capture(args.validate_current, args.inputs, families))


if __name__ == "__main__":
    main()


def batch_final_events(final: dict) -> list[dict]:
    """Project final message fields without inventing their lost interleaving."""
    if set(final) != {"reasoning_text", "normal_text", "calls"}:
        raise ValueError("batch final fields must contain reasoning_text, normal_text, calls")
    if not isinstance(final["reasoning_text"], str) or not isinstance(final["normal_text"], str) or not isinstance(final["calls"], list):
        raise ValueError("invalid batch final field types")
    events = []
    if final["reasoning_text"]:
        events.append({"kind": "reasoning", "text": final["reasoning_text"]})
    if final["normal_text"]:
        events.append({"kind": "text", "text": final["normal_text"]})
    for call in final["calls"]:
        if set(call) != {"name", "arguments"} or not isinstance(call["name"], str):
            raise ValueError("invalid batch tool call")
        events.append({"kind": "tool_call", **call})
    return events
