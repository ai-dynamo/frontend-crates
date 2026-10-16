#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay one deployed GPT-OSS configuration; preserve requests, SSE events and assertions."""
import argparse
import datetime
import hashlib
import concurrent.futures
import copy
import html
import itertools
import json
import subprocess
from pathlib import Path
import time

import httpx
import jsonschema
import yaml


def complete(client, url, request, trace):
    started = time.monotonic()
    result = {"content": "", "reasoning_content": "", "tool_calls": []}
    calls = {}
    fragments = {}
    finish = None
    events = []
    trace["events"] = events
    with client.stream("POST", url.rstrip("/") + "/v1/chat/completions", json=request) as response:
        trace["status"] = response.status_code
        if response.status_code != 200:
            trace["error_body"] = response.read().decode(errors="replace")
            return result, fragments, finish
        if not request["stream"]:
            payload = json.loads(response.read())
            trace["response"] = payload
            choice = payload["choices"][0]
            result.update(choice["message"])
            result["content"] = result.get("content") or ""
            result["reasoning_content"] = result.get("reasoning_content") or ""
            result["tool_calls"] = result.get("tool_calls") or []
            return result, fragments, choice.get("finish_reason")
        done = False
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            payload = json.loads(data)
            events.append({"ms": round((time.monotonic() - started) * 1000, 3), "event": payload})
            if not payload.get("choices"):
                continue
            assert len(payload["choices"]) == 1, "expected one choice"
            choice = payload["choices"][0]
            delta = choice.get("delta") or {}
            for channel in ("content", "reasoning_content"):
                result[channel] += delta.get(channel) or ""
            for call in delta.get("tool_calls") or []:
                index = call["index"]
                assert isinstance(index, int) and index >= 0, "invalid tool index"
                target = calls.setdefault(index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    assert target["id"] in (None, call["id"]), "tool ID changed within a stream"
                    target["id"] = call["id"]
                if call.get("type"):
                    assert call["type"] == "function", "unexpected tool type"
                function = call.get("function") or {}
                target["function"]["name"] += function.get("name") or ""
                argument = function.get("arguments") or ""
                target["function"]["arguments"] += argument
                if argument:
                    fragments[index] = fragments.get(index, 0) + 1
            if choice.get("finish_reason"):
                assert finish is None, "duplicate terminal choice"
                finish = choice["finish_reason"]
        trace["events"] = events
        assert done, "stream ended without [DONE]"
    assert finish is not None, "stream ended without a finish reason"
    assert sorted(calls) == list(range(len(calls))), "non-contiguous tool indices"
    result["tool_calls"] = [calls[index] for index in sorted(calls)]
    return result, fragments, finish


def validate(message, expected, tools, reasoning, fragments, stream, require_fragments):
    actual_calls = []
    ids = []
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    for call in message["tool_calls"]:
        assert call.get("id"), "tool call missing ID"
        ids.append(call["id"])
        assert call.get("type") == "function", "tool call has wrong type"
        function = call["function"]
        assert isinstance(function["arguments"], str), "tool arguments must be a JSON string"
        arguments = json.loads(function["arguments"])
        assert function["name"] in schemas, "unoffered tool"
        jsonschema.Draft202012Validator(schemas[function["name"]]).validate(arguments)
        actual_calls.append({"name": function["name"], "arguments": arguments})
    assert len(set(ids)) == len(ids), "tool IDs are not unique"
    assert actual_calls == expected["calls"], f"calls: expected {expected['calls']!r}; got {actual_calls!r}"
    if "contains" in expected:
        assert expected["contains"] in message["content"], f"missing answer {expected['contains']!r}"
    for channel in ("content", "reasoning_content"):
        assert not any(marker in message[channel] for marker in ("<|start|>", "<|channel|>", "<|message|>", "<|call|>", "<|return|>", "<|end|>")), f"control marker leaked into {channel}"
    if reasoning == "disabled":
        assert not message["reasoning_content"], "reasoning emitted while disabled"
    elif reasoning == "enabled":
        assert message["reasoning_content"].strip(), "enabled reasoning missing"
    if stream and require_fragments and actual_calls:
        assert fragments.get(0, 0) >= 2, f"arguments buffered until completion: {fragments}"


def run_one(args, suite, case, reasoning, stream):
    row = {"id": case["id"], "reasoning": reasoning, "stream": stream,
           "structural_tags": args.structural_tags == "on", "stream_interval": args.stream_interval,
           "implementation": args.implementation, "turns": [], "passed": False}
    started = time.monotonic()
    messages = [{"role": "system", "content": "Follow the requested tool names and argument values exactly. Keep reasoning brief."},
                {"role": "user", "content": case["prompt"]}]
    try:
        with httpx.Client(timeout=args.timeout) as client:
            for turn in [case] + ([case["follow_up"]] if case.get("follow_up") else []):
                tools = case.get("tools", [])
                request = {"model": args.model or suite["model"], "messages": copy.deepcopy(messages),
                           "temperature": 0, "max_tokens": args.max_tokens, "stream": stream,
                           "reasoning_effort": "none" if reasoning == "disabled" else "low", "chat_template_kwargs": {"reasoning_effort": "none" if reasoning == "disabled" else "low"}}
                if "tools" in case:
                    request["tools"] = tools
                choice = turn.get("tool_choice", case.get("tool_choice"))
                if choice is not None:
                    request["tool_choice"] = choice
                if "parallel_tool_calls" in case:
                    request["parallel_tool_calls"] = case["parallel_tool_calls"]
                if args.implementation == "sglang" and args.structural_tags == "on" and reasoning != "disabled" and turn["expected"].get("http_status", 200) == 200:
                    built = subprocess.run([str(args.tag_builder)], input=json.dumps(request), text=True, capture_output=True, check=True)
                    tag = json.loads(built.stdout)
                    if tag is not None:
                        request["response_format"] = tag
                trace = {"request": request}
                row["turns"].append(trace)
                message, fragments, finish = complete(client, args.url, request, trace)
                expected_status = 400 if reasoning == "disabled" else turn["expected"].get("http_status", 200)
                assert trace["status"] == expected_status, f"HTTP {trace['status']}: {trace.get('error_body', '')}"
                if expected_status != 200:
                    continue
                trace.update(message=message, fragments=fragments, finish_reason=finish)
                validate(message, turn["expected"], tools, reasoning if turn is case else "optional", fragments, stream,
                         case.get("expect_argument_fragments", False))
                assert finish == ("tool_calls" if message["tool_calls"] else "stop"), f"wrong finish reason: {finish}"
                if turn is case and case.get("follow_up"):
                    messages.append({"role": "assistant", **message})
                    follow_up = case["follow_up"]
                    for call, result in zip(message["tool_calls"], follow_up["tool_results"], strict=True):
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
                    messages.append({"role": "user", "content": follow_up["prompt"]})
        row["passed"] = True
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["seconds"] = round(time.monotonic() - started, 3)
    return row


def report(directory):
    runs = []
    for path in sorted(directory.glob("*.json")):
        run = json.loads(path.read_text())
        if run.get("results") and run.get("configuration", {}).get("started_at"):
            run["capture_file"] = path.name
            runs.append(run)
    latest = {}
    for run in runs:
        results = run.get("results", [])
        if not results:
            continue
        first = results[0]
        key = (first["implementation"], first["structural_tags"], first["stream_interval"])
        if key not in latest or run["configuration"]["started_at"] > latest[key]["configuration"]["started_at"]:
            latest[key] = run
    rows = [row for run in latest.values() for row in run["results"]]
    summaries = []
    suite = yaml.safe_load(Path(__file__).with_name("cases.yaml").read_text())
    expected = len(suite["cases"]) * 16
    invalid = {case["id"] for case in suite["cases"] if case.get("expected", {}).get("http_status") == 400}
    scores = {}
    for implementation in sorted({row["implementation"] for row in rows}, key={"v2": 0, "v1": 1, "sglang": 2}.get):
        subset = [row for row in rows if row["implementation"] == implementation]
        groups = {
            "functional": [row for row in subset if row["reasoning"] == "enabled" and row["id"] not in invalid],
            "unsupported_disable": [row for row in subset if row["reasoning"] == "disabled" and row["id"] not in invalid],
            "invalid_tools": [row for row in subset if row["id"] in invalid],
            "total": subset,
        }
        scores[implementation] = {name: {"passed": sum(row["passed"] for row in group), "executed": len(group)}
                                  for name, group in groups.items()}
        cells = "".join(f"<td>{sum(row['passed'] for row in group)}/{len(group)}</td>" for group in groups.values())
        summaries.append(f"<tr><td>{html.escape(implementation)}</td>{cells}</tr>")
    (directory / "scores.json").write_text(json.dumps({"planned_per_implementation": expected,
                                                "suite_sha256": hashlib.sha256(Path(__file__).with_name("cases.yaml").read_bytes()).hexdigest(),
                                                "selected_captures": [run["capture_file"] for run in latest.values()],
                                                "scores": scores}, indent=2) + "\n")
    environment_path = directory / "environment.json"
    environment = json.loads(environment_path.read_text()) if environment_path.exists() else {}
    configurations = []
    for (implementation, tags, interval), run in sorted(latest.items()):
        subset = run["results"]
        provenance = run["configuration"].get("provenance", {})
        if not provenance and implementation in {"v1", "sglang"}:
            provenance = environment
        build = provenance.get("core_sha256", {}).get("v2" if implementation == "v2" else "worker_and_v1", "not recorded")
        if implementation == "sglang":
            build = provenance.get("engine_packages", {}).get("sglang", "see environment")
        configurations.append(f"<tr><td>{implementation}</td><td>{tags}</td><td>{interval}</td><td>{sum(row['passed'] for row in subset)}/{len(subset)}</td><td>{html.escape(build[:24])}</td></tr>")
    history = []
    for run in runs:
        subset = run["results"]
        first = subset[0]
        configuration = run["configuration"]
        label = f"{first['implementation']} / tags={first['structural_tags']} / interval={first['stream_interval']}"
        history.append(f"<tr><td>{html.escape(label)}</td><td>{sum(row['passed'] for row in subset)}/{len(subset)}</td>"
                       f"<td>{html.escape(configuration['started_at'])}</td><td>{configuration.get('concurrency', 'not recorded')}</td>"
                       f"<td>{html.escape(configuration['suite_sha256'][:12])}</td>"
                       f"<td><a href=\"{html.escape(run['capture_file'], quote=True)}\">JSON capture</a></td></tr>")
    progress = []
    for row in rows:
        if row["id"] != "long_string_streaming" or not row["stream"] or row["reasoning"] != "enabled":
            continue
        events = row["turns"][0].get("events", [])
        times = [event["ms"] for event in events if any(
            call.get("function", {}).get("arguments")
            for choice in event["event"].get("choices", [])
            for call in ((choice.get("delta") or {}).get("tool_calls") or []))]
        label = f"{row['implementation']} / tags={row['structural_tags']} / interval={row['stream_interval']} / reasoning={row['reasoning']}"
        first = times[0] if times else None
        lead = round(events[-1]["ms"] - first, 3) if first is not None else None
        progress.append(f"<tr><td>{html.escape(label)}</td><td>{len(times)}</td><td>{first}</td><td>{lead}</td></tr>")
    metadata_path = directory / "metadata.json"
    metadata = ""
    if metadata_path.exists():
        environment = json.loads(metadata_path.read_text())
        checks = environment.get("local_validation", {})
        validation = list(checks.items())
        findings = environment.get("findings", [])
        metadata = "<h2>Serving path</h2><p><code>Request → prompt segments → trusted/data token IDs → worker → Harmony UnifiedParser → reasoning, text and incremental tool deltas</code></p><h2>Findings and limits</h2><ul>" + "".join(
            f"<li>{html.escape(finding)}</li>" for finding in findings) + "</ul>"
        metadata += "<h2>Hermetic parser validation</h2><table>" + "".join(
            f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(value))}</td></tr>"
            for label, value in validation) + "</table>"
        evidence = [path for path in ("cases.yaml", "scores.json", "validation/canonical-status.log",
                                     "diagnostics/token-boundary-probe.json", "diagnostics/guided-format-probe.json")
                    if (directory / path).exists()]
        if evidence:
            metadata += "<p>Evidence: " + ", ".join(f'<a href="{path}">{path}</a>' for path in evidence) + "</p>"
        metadata += "<h2>Environment and scope</h2><pre>" + html.escape(metadata_path.read_text()) + "</pre>"
    details = []
    for row in rows:
        name = f"{row['implementation']} / {row['id']} / reasoning={row['reasoning']} / tags={row['structural_tags']} / interval={row['stream_interval']} / stream={row['stream']}"
        view = {**row, "turns": []}
        for turn in row["turns"]:
            events = turn.get("events", [])
            view["turns"].append({**{key:value for key,value in turn.items() if key != "events"},
                                  "event_count": len(events), "first_events": events[:2], "last_events": events[-2:]})
        details.append(f"<details><summary class={'pass' if row['passed'] else 'fail'}>{'PASS' if row['passed'] else 'FAIL'} {html.escape(name)}</summary><pre>{html.escape(json.dumps(view, indent=2, ensure_ascii=False))}</pre></details>")
    (directory / "report.html").write_text("<!doctype html><meta charset=utf-8><title>GPT-OSS parser validation</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse}th,td{padding:8px 16px;text-align:left;border-bottom:1px solid #ddd;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}.pass{color:#246c27}.fail{color:#a22121}details{padding:6px}</style><h1>GPT-OSS parser validation</h1><p>Scores use exact tool arguments, JSON Schema types, reasoning controls, tool IDs, finish reasons and incremental delivery. Request and SSE evidence is retained below. The unchanged 23-case suite covers 368 checks per implementation on one B200.</p><table><tr><th>Implementation</th><th>Functional requests</th><th>Unsupported-disable rejection</th><th>Invalid-tool validation</th><th>Total passed / executed</th></tr>" + "".join(summaries) + "</table><h2>Deployed configurations</h2><table><tr><th>Implementation</th><th>Structural tags</th><th>Stream interval</th><th>Passed / executed</th><th>Build</th></tr>" + "".join(configurations) + "</table><p>Baseline builds missing inline provenance use the frozen deployment identity in environment.json; original capture files are unchanged.</p><h2>Observed SSE argument delivery</h2><table><tr><th>Long-string configuration</th><th>Argument frames</th><th>First argument (ms)</th><th>Time before terminal frame (ms)</th></tr>" + "".join(progress) + "</table><h2>All recorded iterations</h2><p>Earlier failures remain in immutable captures. Input hashes distinguish suite versions.</p><table><tr><th>Configuration</th><th>Passed</th><th>Started UTC</th><th>Concurrency</th><th>Suite SHA</th><th>Evidence</th></tr>" + "".join(history) + "</table>" + metadata + "<h2>Case evidence</h2><p>Full SSE events and timing are preserved in the adjacent JSON captures. Each disclosure includes the request, assembled response and first/last events.</p>" + "".join(details))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.yaml"))
    parser.add_argument("--implementation", choices=["v1", "v2", "sglang"])
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--run-id", help="Optional immutable capture label; defaults to UTC timestamp")
    parser.add_argument("--structural-tags", choices=["on", "off"], default="off")
    parser.add_argument("--stream-interval", type=int, choices=[1, 20], default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--reasoning", choices=["enabled", "disabled"], help="Select one reasoning axis")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--tag-builder", type=Path, help="Production Rust constraint exporter; required for native SGLang with structural tags")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args.output)
        return
    if not args.implementation or not args.url:
        parser.error("--implementation and --url are required to execute requests")
    if args.implementation == "sglang" and args.structural_tags == "on" and not args.tag_builder:
        parser.error("--tag-builder is required for SGLang structural-tag runs")
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    provenance = json.loads(args.provenance.read_text()) if args.provenance else {}
    suite_bytes = args.cases.read_bytes()
    suite = yaml.safe_load(suite_bytes)
    jobs = [(case, reasoning, stream) for case, reasoning, stream in itertools.product(suite["cases"], suite["matrix"]["reasoning"], suite["matrix"]["stream"]) if (not args.case or case["id"] in args.case) and (not args.reasoning or reasoning == args.reasoning)]
    name = f"{args.implementation}-tags-{args.structural_tags}-interval-{args.stream_interval}-{run_id}"
    results = [None] * len(jobs)
    with (args.output / f"{name}.jsonl").open("x") as journal, concurrent.futures.ThreadPoolExecutor(args.concurrency) as executor:
        futures = {executor.submit(run_one, args, suite, *job): index for index, job in enumerate(jobs)}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            results[futures[future]] = row
            journal.write(json.dumps(row, ensure_ascii=False) + "\n")
            journal.flush()
            print(f"{done}/{len(jobs)} {row['id']} reasoning={row['reasoning']} stream={row['stream']}: "
                  + ("PASS" if row['passed'] else row['error']), flush=True)
    path = args.output / f"{name}.json"
    with path.open("x") as output:
        json.dump({"configuration": {"url":args.url,"model":args.model or suite['model'],"started_at":started_at,
                  "concurrency":args.concurrency,"max_tokens":args.max_tokens,"timeout_seconds":args.timeout,
                  "suite_sha256":hashlib.sha256(suite_bytes).hexdigest(),"runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"provenance":provenance},
                  "results": results}, output, indent=2, ensure_ascii=False)
    report(args.output)
    passed = sum(row["passed"] for row in results)
    print(f"{name}: {passed}/{len(results)} passed")
    for row in results:
        if not row["passed"]:
            print(row["id"], row["reasoning"], row["stream"], row["error"])
    raise SystemExit(passed != len(results))


if __name__ == "__main__":
    main()
