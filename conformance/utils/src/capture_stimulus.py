# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reused case ID does not prove that a historical capture ran today's input."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from capture_bindings import original_capture_input, read_bindings
from capture_policy import policy_for_snapshot, reader_metadata
from dynamo_version import crate_version, source_fingerprint
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
    for (family, key), (record, raw, relative, bindings) in sorted(captures.items()):
        families.setdefault(family, {})[input_keys[(family, key)]] = {
            **record, "capture_input": original_capture_input(record, raw, relative, bindings)}
    return [{"family": family, "cases": records} for family, records in families.items()]


def validated_family_capture_docs(root: Path, input_dirs: list[Path]) -> list[dict]:
    raw_inputs = {}
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*/*.yaml")):
            doc = yaml.safe_load(path.read_bytes())
            for key, record in doc["cases"].items():
                raw_inputs[(doc["family"], key)] = record
    inputs, input_aliases = canonicalize_unified_inputs(raw_inputs)
    records = {}
    available = [directory.name for directory in root.iterdir() if directory.is_dir()]
    metadata = reader_metadata("unified", "dynamo_v2", available, policy=policy_for_snapshot(root))
    selected = metadata["selected"]
    if selected is None:
        print(f"historical unified/dynamo_v2 capture unavailable: {metadata['unavailable']}", file=sys.stderr)
        return []
    eligible = metadata["eligible"]
    directories = [root / name for name in eligible[:eligible.index(selected) + 1]]
    for directory in directories:
        families = {path.name for path in directory.iterdir() if path.is_dir()}
        records = {ident: value for ident, value in records.items() if ident[0] not in families}
        for ident, value in _effective_capture_records(directory, input_aliases).items():
            records[ident] = (directory, value)
    families = {}
    skipped = {ident: "no recorded measurement" for ident in inputs.keys() - records.keys()}
    for ident, (directory, (record, raw, relative, bindings)) in sorted(records.items()):
        if "unavailable" in record or "error" in record:
            skipped[ident] = str(record.get("unavailable", record.get("error")))
            continue
        original = original_capture_input(record, raw, relative, bindings)
        failure = historical_replay_failure(original)
        if failure:
            skipped[ident] = failure
            continue
        family = families.setdefault(ident[0], {"version": None, "cases": {}})
        version = capture_layer_sort_key(directory.name)[0].removeprefix("dynamo_v2-")
        if family["version"] not in (None, version):
            raise ValueError(f"family spans multiple latest capture versions: {ident[0]}")
        family["version"] = version
        family["cases"][ident[1]] = {**record, "capture_input": original}
    for (family, key), reason in sorted(skipped.items()):
        print(f"historical Unified regression unavailable: {family}/{key}: {reason}", file=sys.stderr)
    print(f"historical Unified regression: {sum(len(item['cases']) for item in families.values())} recorded requests selected; "
          f"{len(skipped)} cases unavailable", file=sys.stderr)
    return [
        {"family": family, **capture}
        for family, capture in sorted(families.items())
    ]


def historical_replay_failure(original: object) -> str | None:
    """A historical request must be executable without borrowing today's input."""
    if original is None:
        return "original request binding was not retained"
    if not isinstance(original, dict) or original.keys() != {"input", "init", "tools", "chunks", "finish_reason"}:
        return "original request binding is incomplete"
    if not isinstance(original["input"], str) or not isinstance(original["init"], dict) or not isinstance(original["tools"], list):
        return "original input, initialization, or tool schema was not retained"
    chunks = original["chunks"]
    if not isinstance(chunks, list) or any(not isinstance(chunk, dict) or not isinstance(chunk.get("delta_text"), str) for chunk in chunks):
        return "original chunk schedule was not retained"
    if not isinstance(original["finish_reason"], str) or not original["finish_reason"]:
        return "original completion reason was not retained"
    # The producing Unified harness treats the completion reason as metadata;
    # its finish() API has no reason argument (including for length termination).
    if any(chunk.get("token_ids") or chunk.get("delta_token_ids") or chunk.get("finish_reason") for chunk in chunks):
        return "recorded token or per-chunk finish operations are unsupported by the Unified text replay"
    if not chunks or chunks[-1]["delta_text"] != "‹finish›":
        return "recorded schedule has no explicit terminal finish operation"
    if any(chunk["delta_text"] == "‹finish›" for chunk in chunks[:-1]) or "".join(chunk["delta_text"] for chunk in chunks[:-1]) != original["input"]:
        return "recorded chunk schedule does not reconstruct the original input"
    return None


def _source_matches(origin: object, version: str, fingerprint: str) -> bool:
    return isinstance(origin, dict) and origin.get("crate_version") == version and origin.get("source_sha256") == fingerprint


def _checkpoint_evidence(directory: Path) -> dict | None:
    path = directory / "capture-checkpoint.json"
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text())
    if (checkpoint.get("schema_version") != 1 or not isinstance(checkpoint.get("families"), dict)
            or not isinstance(checkpoint.get("records"), dict)):
        raise ValueError(f"invalid measured checkpoint evidence: {path}")
    files = {str(file.relative_to(directory)): file for file in directory.glob("*/*.yaml")}
    if files.keys() != checkpoint["records"].keys():
        raise ValueError(f"measured checkpoint record set differs: {path}")
    for relative, file in files.items():
        if hashlib.sha256(file.read_bytes()).hexdigest() != checkpoint["records"][relative]:
            raise ValueError(f"measured checkpoint record hash differs: {path}/{relative}")
        if file.parent.name not in checkpoint["families"]:
            raise ValueError(f"measured checkpoint family lacks provenance: {file.parent.name}")
    return checkpoint


def validated_current_source_docs(root: Path, input_dirs: list[Path], repo_root: Path) -> list[dict]:
    """Require measured source and request equality; historical regression is separate."""
    version = crate_version(repo_root / "parsers/v2/Cargo.toml")
    fingerprint = source_fingerprint(repo_root)
    candidates = {}
    for directory in root.glob("dynamo_v2-*"):
        if not directory.is_dir():
            continue
        documents = [yaml.safe_load(path.read_bytes()) for path in sorted(directory.glob("*/*.yaml"))]
        if not documents:
            continue
        checkpoint = _checkpoint_evidence(directory)
        if checkpoint is not None:
            # An unchanged measured checkpoint owns its measurement provenance;
            # inherited observations still retain their original producer headers.
            proven = bool(checkpoint["families"]) and all(
                isinstance(provenance, dict) and provenance.get("status") == "captured"
                and _source_matches(provenance.get("origin"), version, fingerprint)
                for provenance in checkpoint["families"].values())
        else:
            proven = all(_source_matches(doc.get("capture_origin") or doc.get("capture_provenance"), version, fingerprint)
                         for doc in documents)
        if proven:
            candidates[directory] = checkpoint
    if not candidates:
        raise ValueError(f"current-source capture unavailable: no measured {version} checkpoint matches source sha256:{fingerprint}; historical captures do not verify today's source")
    selected = max(candidates, key=lambda path: capture_layer_sort_key(path.name))
    checkpoint = candidates[selected]
    # A same-version patch must not reuse a checkpoint's proof for other bytes.
    for _record, raw, relative, _bindings in _effective_capture_records(selected, {}).values():
        if checkpoint is not None:
            proven = checkpoint["records"].get(relative) == hashlib.sha256(raw).hexdigest()
        else:
            document = yaml.safe_load(raw)
            proven = _source_matches(document.get("capture_origin") or document.get("capture_provenance"), version, fingerprint)
        if not proven:
            raise ValueError(f"current-source capture unavailable: effective record {relative} has different source evidence")
    return validated_current_capture_docs(selected, input_dirs)


def validate_current_capture(directory: Path, input_dirs: list[Path]) -> int:
    return sum(len(doc["cases"]) for doc in validated_current_capture_docs(directory, input_dirs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select-source-snapshot", type=Path, help="deprecated Rust harness compatibility")
    mode.add_argument("--validate-current", type=Path)
    mode.add_argument("--validate-current-source", type=Path, help="require exact current source and request evidence")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    mode.add_argument("--validate-latest-by-family", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--format", choices=("count", "json"), default="count")
    args = parser.parse_args()
    if args.validate_current_source is not None:
        if not args.inputs:
            parser.error("--validate-current-source requires --inputs")
        docs = validated_current_source_docs(args.validate_current_source, args.inputs, args.repo_root)
        print(json.dumps(docs) if args.format == "json" else sum(len(doc["cases"]) for doc in docs))
    elif args.select_source_snapshot is not None:
        if args.format != "count":
            parser.error("--format json requires --validate-current")
        print(current_source_snapshot(args.select_source_snapshot))
    elif args.validate_current is not None:
        if not args.inputs:
            parser.error("--validate-current requires --inputs")
        if args.format == "json":
            print(json.dumps(validated_current_capture_docs(args.validate_current, args.inputs)))
        else:
            print(validate_current_capture(args.validate_current, args.inputs))
    else:
        if not args.inputs:
            parser.error("--validate-latest-by-family requires --inputs")
        docs = validated_family_capture_docs(args.validate_latest_by_family, args.inputs)
        print(json.dumps(docs) if args.format == "json" else sum(len(doc["cases"]) for doc in docs))


if __name__ == "__main__":
    main()
