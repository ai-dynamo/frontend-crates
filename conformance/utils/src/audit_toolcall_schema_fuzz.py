#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, exact-argument MiniMax M3 audit of the pinned OpenRouter corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import SchemaError, ValidationError

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parents[1] / "data/toolcall-schema-fuzz"
NS = "]<]minimax[>["
SCHEDULES = ("whole", "characters", "markers", "seeded")
VALIDATOR_VERSION = "4.26.0"


def load_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"non-JSON number: {value}")

    return json.loads(
        text, parse_float=Decimal, parse_constant=constant, object_pairs_hook=pairs
    )


def json_text(value: Any) -> str:
    """Serialize decimal numbers without converting them through a binary float."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimal")
        return str(value)
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json_text(key) + ":" + json_text(item) for key, item in value.items()
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(json_text(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def integer_type(_checker: Any, value: Any) -> bool:
    return (
        type(value) is int
        or isinstance(value, Decimal)
        and value == value.to_integral_value()
    )


SchemaValidator = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine("integer", integer_type),
)


def load_dataset(directory: Path = DATA) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if importlib.metadata.version("jsonschema") != VALIDATOR_VERSION:
        raise ValueError(
            f"install jsonschema=={VALIDATOR_VERSION} using schema-fuzz-requirements.in"
        )
    manifest = load_json((directory / "manifest.json").read_text())
    raw = (directory / manifest["dataset_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("dataset SHA-256 mismatch")
    document = load_json(raw.decode("utf-8"))
    entries = document["entries"]
    if (
        document["version"] != manifest["dataset_version"]
        or len(entries) != manifest["expected_entry_count"]
    ):
        raise ValueError("dataset version or entry count mismatch")
    keys: set[str] = set()
    argument_count = 0
    for entry in entries:
        key = entry["key"]
        if not isinstance(key, str) or key in keys:
            raise ValueError(f"duplicate or invalid dataset key: {key!r}")
        keys.add(key)
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise ValueError(f"invalid tool name: {key}")
        if not isinstance(entry["parameters"], dict) or len(entry["args"]) != 2:
            raise ValueError(f"invalid schema or argument pair: {key}")
        SchemaValidator.check_schema(entry["parameters"])
        validator = SchemaValidator(entry["parameters"])
        for arguments in entry["args"]:
            if not isinstance(arguments, dict):
                raise TypeError(f"expected object arguments: {key}")
            validator.validate(arguments)
            argument_count += 1
    if argument_count != manifest["expected_argument_count"]:
        raise ValueError("dataset argument count mismatch")
    return manifest, entries


def tag(key: str, text: str) -> str:
    return f"{NS}<{key}>{text}{NS}</{key}>"


def native_value(value: Any) -> str:
    if isinstance(value, dict):
        return "".join(tag(key, native_value(item)) for key, item in value.items())
    if isinstance(value, list):
        return "".join(tag("item", native_value(item)) for item in value)
    return value if isinstance(value, str) else json_text(value)


def encode_call(name: str, arguments: dict[str, Any], encoding: str) -> str:
    if encoding not in ("nested", "json_values"):
        raise ValueError(f"unknown encoding: {encoding}")
    body = "".join(
        tag(
            key,
            json_text(value)
            if encoding == "json_values" and isinstance(value, (dict, list))
            else native_value(value),
        )
        for key, value in arguments.items()
    )
    return (
        f'{NS}<tool_call>{NS}<invoke name="{name}">{body}{NS}</invoke>{NS}</tool_call>'
    )


def chunk_input(text: str, schedule: str, case_id: str) -> list[str]:
    if schedule == "whole":
        return [text]
    if schedule == "characters":
        return list(text)
    if schedule == "markers":
        return [part for part in re.split(r"(\]<\]minimax\[>\[<[^>]*>)", text) if part]
    if schedule == "seeded":
        rng = random.Random(
            int.from_bytes(hashlib.sha256(case_id.encode()).digest(), "big")
        )
        chunks: list[str] = []
        cursor = 0
        while cursor < len(text):
            end = cursor + rng.randint(1, 17)
            chunks.append(text[cursor:end])
            cursor = end
        return chunks
    raise ValueError(f"unknown schedule: {schedule}")


def pointer(path: str, key: str | int) -> str:
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """JSON equality: numbers are exact, and booleans never equal numbers."""
    number = (int, Decimal)
    if type(expected) in number and type(actual) in number:
        return [] if expected == actual else [path]
    if type(expected) is not type(actual):
        return [path]
    if isinstance(expected, dict):
        result = [pointer(path, key) for key in sorted(expected.keys() ^ actual.keys())]
        for key in sorted(expected.keys() & actual.keys()):
            result.extend(differences(expected[key], actual[key], pointer(path, key)))
        return result
    if isinstance(expected, list):
        result = []
        for index in range(max(len(expected), len(actual))):
            item_path = pointer(path, index)
            result.extend(
                [item_path]
                if index >= min(len(expected), len(actual))
                else differences(expected[index], actual[index], item_path)
            )
        return result
    return [] if expected == actual else [path]


def unrepresentable_names(arguments: Any) -> list[str]:
    paths: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = pointer(path, key)
                if not key or ">" in key or key.startswith("/"):
                    paths.append(child)
                visit(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, pointer(path, index))

    visit(arguments, "")
    return paths


def constructs(schema: Any) -> list[str]:
    found: set[str] = set()

    def visit(node: Any, root: bool = False) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in SchemaValidator.VALIDATORS or key in {"$defs", "definitions"}:
                found.add(key)
                if key == "type" and isinstance(value, str):
                    found.add("type:" + value)
                if root and key in {"$ref", "anyOf", "oneOf", "allOf"}:
                    found.add("root:" + key)
            if key in {
                "properties",
                "$defs",
                "definitions",
                "patternProperties",
            } and isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif key in {"anyOf", "oneOf", "allOf", "prefixItems"} and isinstance(
                value, list
            ):
                for child in value:
                    visit(child)
            elif key in {"items", "additionalProperties", "not", "if", "then", "else"}:
                visit(value)
        if not node.keys() & {
            "type",
            "const",
            "enum",
            "$ref",
            "anyOf",
            "oneOf",
            "allOf",
        }:
            found.add("no_type_hint")

    visit(schema, True)
    return sorted(found)


def cases(entries: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for entry in entries:
        for index, arguments in enumerate(entry["args"]):
            encodings = ["nested"]
            if any(isinstance(value, (dict, list)) for value in arguments.values()):
                encodings.append("json_values")
            for encoding in encodings:
                text = encode_call(entry["name"], arguments, encoding)
                for schedule in SCHEDULES:
                    case_id = f"toolcall_schema_fuzz-realistic:{entry['key']}:args{index}:{encoding}:{schedule}"
                    yield {
                        "id": case_id,
                        "key": entry["key"],
                        "argument_index": index,
                        "encoding": encoding,
                        "schedule": schedule,
                        "name": entry["name"],
                        "expected": arguments,
                        "constructs": constructs(entry["parameters"]),
                        "request": {
                            "input": text,
                            "init": {
                                "starting_state": "Response",
                                "tool_output_mode": "Native",
                                "named_tool": None,
                            },
                            "finish_reason": "tool_calls",
                            "tools": [
                                {
                                    "name": entry["name"],
                                    "parameters": entry["parameters"],
                                }
                            ],
                            "chunks": [
                                {"delta_text": chunk}
                                for chunk in chunk_input(text, schedule, case_id)
                            ],
                        },
                    }


def score(case: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    labels: list[str] = []
    if (
        case["encoding"] == "nested" and unrepresentable_names(case["expected"])
    ) or any(not key or ">" in key or key.startswith("/") for key in case["expected"]):
        labels.append("unrepresentable_parameter_name")
    result: dict[str, Any] = {
        "failures": failures,
        "different_paths": [],
        "labels": labels,
    }
    if capture.get("error") or capture.get("unavailable"):
        failures.append("parser_error" if capture.get("error") else "unavailable")
        result["error"] = capture.get("error", capture.get("unavailable"))
        return result
    events = capture.get("assembled", [])
    calls = [event for event in events if event.get("kind") == "tool_call"]
    if any(event.get("kind") != "tool_call" for event in events):
        failures.append("unexpected_event")
    if len(calls) != 1:
        failures.append("call_count")
    if len(capture.get("chunks", [])) != len(case["request"]["chunks"]) + 1:
        failures.append("missing_finish_row")
    deltas = [
        delta for row in capture.get("chunks", []) for delta in row.get("expected", [])
    ]
    if (
        any(delta.get("kind") != "tool_call" for delta in deltas)
        and "unexpected_event" not in failures
    ):
        failures.append("unexpected_event")
    tool_deltas = [delta for delta in deltas if delta.get("kind") == "tool_call"]
    terminals = [
        delta
        for delta in deltas
        if delta.get("kind") == "tool_call" and delta.get("complete") is True
    ]
    if len(terminals) != 1:
        failures.append("completion_count")
    elif tool_deltas[-1] is not terminals[0]:
        failures.append("completion_before_end")
    if len(calls) != 1:
        return result
    call = calls[0]
    stream_names = [delta["name"] for delta in tool_deltas if delta.get("name")]
    if call.get("name") != case["name"] or any(
        name != case["name"] for name in stream_names
    ):
        failures.append("wrong_tool")
    if len(stream_names) != 1:
        failures.append("stream_call_count")
    try:
        if not isinstance(call.get("arguments"), str) or any(
            not isinstance(delta.get("arguments"), str) for delta in tool_deltas
        ):
            raise TypeError("capture must preserve arguments as raw JSON")
        # The shared capture helper may substitute its Value projection for bad
        # raw JSON. Judge the literal fragments the consumer receives as well.
        raw = "".join(delta["arguments"] for delta in tool_deltas)
        result["actual_arguments_raw"] = raw
        actual = load_json(raw)
        assembled = load_json(call["arguments"])
    except (ValueError, TypeError) as error:
        failures.append("invalid_json")
        result["error"] = str(error)
        return result
    result["actual_arguments"] = actual
    if differences(actual, assembled):
        failures.append("assembled_stream_mismatch")
    paths = differences(case["expected"], actual)
    result["different_paths"] = paths
    if paths:
        failures.append("argument_mismatch")
    schema = case["request"]["tools"][0]["parameters"]
    violations = list(SchemaValidator(schema).iter_errors(actual))
    result["schema_valid"] = not violations
    result["schema_errors"] = [
        {
            "path": "/"
            + "/".join(
                str(part).replace("~", "~0").replace("/", "~1")
                for part in error.absolute_path
            ),
            "message": error.message,
        }
        for error in violations
    ]
    if violations:
        failures.append("schema_violation")
    if (
        paths
        and not violations
        and isinstance(actual, dict)
        and encode_call(case["name"], actual, case["encoding"])
        == case["request"]["input"]
    ):
        labels.append("demonstrated_wire_ambiguity")
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json_text(value) + "\n", encoding="utf-8")


def parser_identity(root: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    changed = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "parsers/v2",
            "parsers/structural-tag",
            "conformance/tests",
            "Cargo.lock",
            "Cargo.toml",
            "rust-toolchain.toml",
        ],
        cwd=root,
        text=True,
    ).strip()
    if changed:
        raise ValueError(
            "audit requires unchanged parser, harness, and build inputs; use a clean topic worktree"
        )
    return {
        "revision": revision,
        "cargo_lock_sha256": hashlib.sha256(
            (root / "Cargo.lock").read_bytes()
        ).hexdigest(),
    }


def build_runner(root: Path, output: Path) -> Path:
    with (
        (output / "build.jsonl").open("w") as stdout,
        (output / "build.log").open("w") as stderr,
    ):
        subprocess.run(
            [
                "cargo",
                "test",
                "--locked",
                "-p",
                "dynamo-conformance-fixtures-v2",
                "--test",
                "capture_cross_version",
                "--no-run",
                "--message-format=json",
            ],
            cwd=root,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    executables = [
        event["executable"]
        for line in (output / "build.jsonl").read_text().splitlines()
        if (event := json.loads(line)).get("reason") == "compiler-artifact"
        and event.get("target", {}).get("name") == "capture_cross_version"
        and event.get("executable")
    ]
    if len(executables) != 1:
        raise ValueError("Cargo did not identify one capture runner")
    return Path(executables[0])


def run_batch(
    runner: Path, root: Path, output: Path, index: int, batch: list[dict[str, Any]]
) -> dict[str, Any]:
    stem = output / f"batch-{index:04d}"
    jobs_path = stem.with_suffix(".jobs.json")
    capture_path = stem.with_suffix(".captures.json")
    write_json(
        jobs_path,
        [
            {
                "family": "minimax_m3",
                "cases": [
                    {"id": case["id"], "request": case["request"]} for case in batch
                ],
            }
        ],
    )
    env = os.environ.copy()
    env.update(XVER_JOBS=str(jobs_path), XVER_JSON_OUTPUT=str(capture_path))
    try:
        with stem.with_suffix(".log").open("w") as log:
            subprocess.run(
                [str(runner), "--exact", "capture_family_jobs", "--nocapture"],
                cwd=root,
                env=env,
                stdout=log,
                stderr=log,
                check=True,
                timeout=600,
            )
        results = load_json(capture_path.read_text())["minimax_m3"]
        if set(results) != {case["id"] for case in batch}:
            raise ValueError("capture result membership differs from requested cases")
        for case in batch:
            if differences(case["request"], results[case["id"]].get("capture_input")):
                raise ValueError(f"capture request binding mismatch: {case['id']}")
        return results
    except (subprocess.SubprocessError, ValueError, KeyError, OSError) as error:
        return {
            case["id"]: {"error": f"audit infrastructure: {error}"} for case in batch
        }


def audit(
    root: Path,
    output: Path,
    entries: list[dict[str, Any]],
    manifest: dict[str, Any],
    batch_size: int,
) -> int:
    identity = parser_identity(root)
    expected_encoding_cases = sum(
        1 + any(isinstance(value, (dict, list)) for value in arguments.values())
        for entry in entries
        for arguments in entry["args"]
    )
    expected_runs = expected_encoding_cases * len(SCHEDULES)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": 1,
        "parser": identity,
        "dataset": manifest,
        "jsonschema_version": importlib.metadata.version("jsonschema"),
        "encoder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected_entry_keys": [entry["key"] for entry in entries],
        "selected_argument_count": sum(len(entry["args"]) for entry in entries),
        "expected_runs": expected_runs,
    }
    write_json(output / "provenance.json", provenance)
    runner = build_runner(root, output)
    summary: dict[str, Any] = {
        "runs": 0,
        "passed": 0,
        "failed": 0,
        "by_encoding": {},
        "by_schedule": {},
        "by_construct": {},
        "failure_kinds": {},
        "labels": {},
    }
    groups: dict[str, dict[str, Counter[str]]] = {
        name: defaultdict(Counter)
        for name in ("by_encoding", "by_schedule", "by_construct")
    }
    failure_kinds: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    expected_ids: set[str] = set()
    failed_arguments: set[tuple[str, int]] = set()
    failed_encodings: set[tuple[str, int, str]] = set()
    outcomes: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    batch: list[dict[str, Any]] = []
    batch_index = 0

    def consume(records: list[dict[str, Any]], report: Any) -> None:
        nonlocal batch_index
        captures = run_batch(runner, root, output, batch_index, records)
        for case in records:
            if case["id"] in expected_ids:
                raise ValueError("duplicate generated case ID")
            expected_ids.add(case["id"])
            capture = captures[case["id"]]
            result = score(case, capture)
            status = "failed" if result["failures"] else "passed"
            summary["runs"] += 1
            summary[status] += 1
            if status == "failed":
                failed_arguments.add((case["key"], case["argument_index"]))
                failed_encodings.add(
                    (case["key"], case["argument_index"], case["encoding"])
                )
            for group, values in (
                ("by_encoding", [case["encoding"]]),
                ("by_schedule", [case["schedule"]]),
                ("by_construct", case["constructs"]),
            ):
                for value in values:
                    groups[group][value][status] += 1
            failure_kinds.update(result["failures"])
            labels.update(result.get("labels", []))
            outcomes[(case["key"], case["argument_index"], case["encoding"])].add(
                json_text(
                    {
                        "failures": result["failures"],
                        "actual": result.get("actual_arguments"),
                        "events": capture.get("assembled"),
                    }
                )
            )
            report.write(
                json_text(
                    {
                        **case,
                        **result,
                        "capture_file": f"batch-{batch_index:04d}.captures.json",
                    }
                )
                + "\n"
            )
        batch_index += 1
        report.flush()
        print(
            f"{summary['runs']} runs: {summary['passed']} passed, {summary['failed']} failed",
            flush=True,
        )

    with (output / "results.jsonl").open("w", encoding="utf-8") as report:
        for case in cases(entries):
            batch.append(case)
            if len(batch) == batch_size:
                consume(batch, report)
                batch = []
        if batch:
            consume(batch, report)
    if (
        summary["runs"] != expected_runs
        or len(outcomes) != expected_encoding_cases
        or len(expected_ids) != expected_runs
    ):
        raise ValueError("incomplete audit result count")
    summary["encoding_cases"] = len(outcomes)
    summary["source_schemas"] = len(entries)
    summary["source_arguments"] = sum(len(entry["args"]) for entry in entries)
    summary["failed_source_arguments"] = len(failed_arguments)
    summary["failed_source_schemas"] = len({key for key, _index in failed_arguments})
    summary["failed_encoding_cases"] = len(failed_encodings)
    summary["chunk_dependent_cases"] = [
        list(key) for key, values in outcomes.items() if len(values) > 1
    ]
    for name, group in groups.items():
        summary[name] = {key: dict(value) for key, value in sorted(group.items())}
    summary["failure_kinds"] = dict(sorted(failure_kinds.items()))
    summary["labels"] = dict(sorted(labels.items()))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory under this checkout's logs/",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="exact dataset key; repeat to select a smoke subset",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    try:
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        output = args.output.resolve()
        if not output.is_relative_to((ROOT / "logs").resolve()):
            raise ValueError("output must be under this checkout's logs/")
        manifest, entries = load_dataset()
        if args.case:
            selected = set(args.case)
            entries = [entry for entry in entries if entry["key"] in selected]
            if {entry["key"] for entry in entries} != selected:
                raise ValueError("unknown --case key")
        print(
            f"Preflight passed: {manifest['expected_entry_count']} schemas, {manifest['expected_argument_count']} expected calls",
            flush=True,
        )
        return audit(ROOT, output, entries, manifest, args.batch_size)
    except (
        ValueError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
        SchemaError,
        ValidationError,
    ) as error:
        print(f"audit setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
