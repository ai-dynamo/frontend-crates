# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound, explicitly based checkpoints for the legacy fixture reader views."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import unified_history
from fixture_corpus import split_sel, version_key

SCHEMA = "legacy-checkpoint-v1"
LEDGER = "observations.yaml"
OUTPUT_FIELDS = {"expected", "chunks", "normal_text", "unavailable", "exception", "assembled", "calls", "tool_calls", "error"}


def is_compact(store: Path) -> bool:
    return next(store.glob(f"*/families/*/{LEDGER}"), None) is not None


def _key(filename: str, case_id: str) -> str:
    return json.dumps([filename, case_id], separators=(",", ":"))


def bound_request(document: dict, case: dict) -> dict:
    fields = ("model_text", "tools", "init", "finish_reason", "chat_template_kwargs", "force_reasoning")
    request = {key: case.get(key) for key in fields}
    request["document"] = {key: document.get(key) for key in fields}
    request.update(family=document["family"], mode=document.get("mode"))
    request["chunks"] = [{key: value for key, value in chunk.items()
                          if key in {"delta_text", "delta_token_ids", "token_ids", "finish_reason"}}
                         if isinstance(chunk, dict) else chunk for chunk in case.get("chunks", [])]
    return request


def _rust_stream_observation(previous: dict | None, incoming: dict) -> dict:
    """Match the historical native reader's partial per-chunk updates."""
    observation = copy.deepcopy(incoming)
    prior_value = {} if previous is None else previous["payload"]["value"]
    value = copy.deepcopy(prior_value)
    raw = incoming["payload"]["value"]
    inherits = False
    if raw.get("unavailable") is not None:
        value["unavailable"] = raw["unavailable"]
        inherits = bool(value.get("chunks"))
    else:
        value.pop("unavailable", None)
        chunks = value.setdefault("chunks", [])
        incoming_chunks = raw.get("chunks", [])
        inherits = len(chunks) > len(incoming_chunks)
        for index, chunk in enumerate(incoming_chunks):
            resolved = {"expected": copy.deepcopy(chunk.get("expected", []))}
            if chunk.get("normal_text") is not None:
                resolved["normal_text"] = chunk["normal_text"]
            if index < len(chunks):
                chunks[index] = resolved
            else:
                chunks.append(resolved)
    observation["payload"]["value"] = value
    if previous is not None and previous["payload"] == observation["payload"]:
        observation["producer"] = copy.deepcopy(previous["producer"])
    elif inherits and previous is not None:
        producer = observation["producer"]
        old_producer = previous["producer"]
        contributors = copy.deepcopy(old_producer.get("contributors", {}))
        contributors[old_producer["capture"]] = {key: copy.deepcopy(value) for key, value in old_producer.items() if key != "contributors"}
        producer["contributors"] = contributors
    return observation


def from_records(records: dict[Path, dict], prior_inventory: dict | None = None) -> dict[str, dict]:
    """Freeze both historical readers, retaining observation ownership separately."""
    inventory = {}
    groups = defaultdict(list)
    shared = {}
    for path, record in records.items():
        corpus, family, capture = (record[key] for key in ("corpus", "family", "capture"))
        if capture == "inputs":
            shared[(corpus, family)] = record
            inventory[f"{corpus}/{family}/{capture}"] = {"record": copy.deepcopy(record)}
        else:
            groups[(corpus, family, split_sel(capture)[0])].append(record)
    for (corpus, family, implementation), captures in sorted(groups.items()):
        captures.sort(key=lambda record: (version_key(split_sel(record["capture"])[1]),
                                          "+" in record["capture"] or ".patch" in record["capture"], record["capture"]))
        python, rust = {}, {}
        has_prior = False
        if prior_inventory is not None and corpus != "batch_on_stream":
            prior = [(identity.split("/")[2], snapshot) for identity, snapshot in prior_inventory.items()
                     if identity.startswith(f"{corpus}/{family}/") and "views" in snapshot
                     and split_sel(identity.split("/")[2])[0] == implementation]
            prior.sort(key=lambda item: (version_key(split_sel(item[0])[1]), item[0]))
            if prior:
                has_prior = True
                if (version_key(split_sel(captures[0]["capture"])[1]), captures[0]["capture"]) <= (version_key(split_sel(prior[-1][0])[1]), prior[-1][0]):
                    raise ValueError(f"cannot add out-of-order legacy capture: {captures[0]['capture']}")
                python = copy.deepcopy(prior[-1][1]["views"]["python"])
                rust_prior = [snapshot for _capture, snapshot in prior if "rust" in snapshot["views"]]
                if rust_prior:
                    rust = copy.deepcopy(rust_prior[-1]["views"]["rust"])
        input_documents = shared.get(("batch" if corpus == "batch_on_stream" else corpus, family), {}).get("documents", {})
        if corpus in {"reasoning", "stream"} and not has_prior:
            for filename, document in input_documents.items():
                for case_id, case in document.get("cases", {}).items():
                    expected = case.get("expected") or {}
                    unavailable = case.get("unavailable") or {}
                    if corpus == "reasoning" and implementation in expected:
                        value = {"expected": {implementation: copy.deepcopy(expected[implementation])}}
                    elif corpus == "stream" and isinstance(unavailable, dict) and implementation in unavailable:
                        value = {"unavailable": unavailable[implementation]}
                    else:
                        continue
                    python[_key(filename, case_id)] = {
                        "payload": {"request": bound_request(document, case), "value": value},
                        "annotations": {},
                        "producer": {"capture": "inputs", "document": {key: copy.deepcopy(value) for key, value in document.items() if key != "cases"}, "provenance": None},
                    }
            if corpus == "stream":
                rust = copy.deepcopy(python)
        for record in captures:
            capture = record["capture"]
            release = "+" not in capture and ".patch" not in capture
            if corpus == "batch_on_stream":
                python = {}
            for filename, document in record["documents"].items():
                input_document = input_documents.get(filename, {"family": family, "cases": {}})
                for case_id, case in document.get("cases", {}).items():
                    identity = _key(filename, case_id)
                    source_case = input_document.get("cases", {}).get(case_id, {})
                    payload = {"request": bound_request(input_document, source_case),
                               "value": {key: copy.deepcopy(value) for key, value in case.items()
                                         if key in OUTPUT_FIELDS}}
                    annotations = {key: copy.deepcopy(value) for key, value in case.items() if key not in OUTPUT_FIELDS}
                    producer = {"capture": capture,
                                "document": {key: copy.deepcopy(value) for key, value in document.items() if key != "cases"},
                                "provenance": copy.deepcopy(record.get("provenance"))}
                    observation = {"payload": payload, "annotations": annotations, "producer": producer}
                    # A repeated full extraction does not claim a new producer for inherited output.
                    if identity in python and python[identity]["payload"] == payload:
                        observation["producer"] = copy.deepcopy(python[identity]["producer"])
                    if corpus in {"batch", "reasoning"} and "expected" not in case and identity in python:
                        observation["payload"] = copy.deepcopy(python[identity]["payload"])
                        observation["producer"] = copy.deepcopy(python[identity]["producer"])
                    python[identity] = observation
                    if corpus == "stream" and release:
                        rust[identity] = _rust_stream_observation(rust.get(identity), {
                            "payload": payload, "annotations": annotations, "producer": producer,
                        })
            header = {key: copy.deepcopy(value) for key, value in record.items() if key != "documents"}
            headers = {filename: {key: copy.deepcopy(value) for key, value in document.items() if key != "cases"}
                       for filename, document in record["documents"].items()}
            inventory[f"{corpus}/{family}/{capture}"] = {
                "header": header, "documents": headers,
                "views": {"python": copy.deepcopy(python), **({"rust": copy.deepcopy(rust)} if corpus == "stream" and release else {})},
            }
        if corpus == "stream":
            # Python's numeric selector folds every qualified layer tied with the selected release.
            for record in captures:
                target = version_key(split_sel(record["capture"])[1])
                last = [item for item in captures if version_key(split_sel(item["capture"])[1]) <= target][-1]
                inventory[f"{corpus}/{family}/{record['capture']}"]["views"]["python"] = copy.deepcopy(
                    inventory[f"{corpus}/{family}/{last['capture']}"]["views"]["python"])
    return inventory


def load(store: Path) -> dict[str, dict]:
    inventory = {}
    for directory in sorted(store.glob("*/families/*")):
        if not directory.is_dir() or not list(directory.glob("*.yaml")):
            continue
        ledger_path = directory / LEDGER
        if not ledger_path.exists():
            raise ValueError(f"mixed compact and legacy family: {directory}")
        ledger = unified_history.load_yaml(ledger_path)
        unified_history._validate_ledger(ledger, ledger_path)
        for producer_id, producer in ledger["producers"].items():
            if producer_id != unified_history._request_digest(producer):
                raise ValueError(f"producer fingerprint mismatch: {ledger_path}/{producer_id}")
        captures = {}
        corpus, family = directory.parent.parent.name, directory.name
        for path in sorted(directory.glob("*.yaml")):
            if path.name == LEDGER:
                continue
            record = unified_history.load_yaml(path)
            if record.get("capture") == "inputs":
                inventory[f"{corpus}/{family}/inputs"] = {"record": record}
                continue
            if record.get("schema") != SCHEMA or record.get("corpus") != corpus or record.get("family") != family or record.get("capture") != path.stem:
                raise ValueError(f"legacy checkpoint identity mismatch: {path}")
            captures[path.stem] = record
        resolved = {}

        def resolve(capture: str, view: str, visiting: set[tuple[str, str]]) -> dict:
            key = (capture, view)
            if key in visiting:
                raise ValueError(f"legacy capture inheritance cycle: {directory}/{capture}")
            if key in resolved:
                return copy.deepcopy(resolved[key])
            if capture not in captures or view not in captures[capture]["views"]:
                raise ValueError(f"missing legacy capture base/view: {directory}/{capture}/{view}")
            visiting.add(key)
            checkpoint = captures[capture]["views"][view]
            base = checkpoint["base"]
            if base is not None and split_sel(base)[0] != split_sel(capture)[0]:
                raise ValueError(f"foreign implementation base: {directory}/{capture}")
            state = {} if base is None else resolve(base, view, visiting)
            for identity, change in checkpoint["changes"].items():
                parts = json.loads(identity)
                if not isinstance(parts, list) or len(parts) != 2 or not all(isinstance(part, str) for part in parts):
                    raise ValueError(f"invalid observation case identity: {identity}")
                if change == {"absent": True}:
                    state.pop(identity, None)
                    continue
                ref, producer = change["observation"], change["producer"]
                if ref not in ledger["observations"] or producer not in ledger["producers"]:
                    raise ValueError(f"dangling legacy observation or producer: {directory}/{capture}")
                owner = ledger["producers"][producer]["capture"]
                if owner != "inputs" and split_sel(owner)[0] != split_sel(capture)[0]:
                    raise ValueError(f"foreign legacy observation producer: {directory}/{capture}")
                state[identity] = {"payload": copy.deepcopy(ledger["observations"][ref]),
                                   "producer": copy.deepcopy(ledger["producers"][producer]),
                                   "annotations": {}}
            for identity, annotations in checkpoint["annotations"].items():
                if identity not in state:
                    raise ValueError(f"legacy annotation has no observation: {identity}")
                state[identity]["annotations"] = copy.deepcopy(annotations)
            visiting.remove(key)
            resolved[key] = copy.deepcopy(state)
            return state

        for capture, record in captures.items():
            inventory[f"{corpus}/{family}/{capture}"] = {
                "header": {key: copy.deepcopy(value) for key, value in record.items() if key not in {"schema", "documents", "views"}},
                "documents": copy.deepcopy(record["documents"]),
                "views": {view: resolve(capture, view, set()) for view in record["views"]},
            }
    for identity, snapshot in inventory.items():
        if "views" not in snapshot:
            continue
        corpus, family, _capture = identity.split("/")
        input_corpus = "batch" if corpus == "batch_on_stream" else corpus
        shared = inventory.get(f"{input_corpus}/{family}/inputs", {}).get("record", {}).get("documents", {})
        for state in snapshot["views"].values():
            for key, observation in state.items():
                filename, case_id = json.loads(key)
                document = shared.get(filename)
                if document is None:
                    continue
                case = document.get("cases", {}).get(case_id)
                if case is not None and bound_request(document, case) != observation["payload"]["request"]:
                    raise ValueError(f"captured legacy input fingerprint mismatch; add a new case: {identity}/{case_id}")
    return inventory


def documents(store: Path, inventory: dict[str, dict]) -> dict[Path, dict]:
    output, ledgers, previous = {}, {}, {}
    def order(item):
        corpus, family, capture = item[0].split("/")
        return corpus, family, split_sel(capture)[0] if capture != "inputs" else "", version_key(split_sel(capture)[1]) if capture != "inputs" else (), capture
    for identity, snapshot in sorted(inventory.items(), key=order):
        corpus, family, capture = identity.split("/")
        directory = store / corpus / "families" / family
        ledger = ledgers.setdefault(directory, {"schema": "bound-observations-v1", "observations": {}, "producers": {}})
        if "record" in snapshot:
            output[directory / "inputs_and_golden.yaml"] = snapshot["record"]
            continue
        implementation = split_sel(capture)[0]
        record = {"schema": SCHEMA, **snapshot["header"], "documents": snapshot["documents"], "views": {}}
        for view, state in snapshot["views"].items():
            group = (corpus, family, implementation, view)
            base, prior = previous.get(group, (None, {})) if corpus != "batch_on_stream" else (None, {})
            current, annotations = {}, {}
            for case_id, observation in sorted(state.items()):
                ref = unified_history._request_digest(observation["payload"])
                producer = unified_history._request_digest(observation["producer"])
                ledger["observations"][ref] = observation["payload"]
                ledger["producers"][producer] = observation["producer"]
                current[case_id] = {"observation": ref, "producer": producer}
                if prior.get(case_id, {}).get("annotations", {}) != observation["annotations"]:
                    annotations[case_id] = observation["annotations"]
            prior_refs = {case_id: {"observation": unified_history._request_digest(observation["payload"]),
                                    "producer": unified_history._request_digest(observation["producer"])}
                          for case_id, observation in prior.items()}
            changes = {case_id: current.get(case_id, {"absent": True})
                       for case_id in sorted(current.keys() | prior_refs.keys()) if current.get(case_id) != prior_refs.get(case_id)}
            # Replacing an observation resets its annotations, so repeat only that small metadata.
            for case_id in changes.keys() & state.keys():
                annotations[case_id] = state[case_id]["annotations"]
            record["views"][view] = {"base": base, "changes": changes, "annotations": annotations}
            previous[group] = (capture, state)
        output[directory / f"{capture}.yaml"] = record
    for directory, ledger in ledgers.items():
        output[directory / LEDGER] = ledger
    return output


def records(store: Path, inventory: dict[str, dict], view: str = "python") -> dict[Path, dict]:
    output = {}
    for identity, snapshot in inventory.items():
        corpus, family, capture = identity.split("/")
        directory = store / corpus / "families" / family
        if "record" in snapshot:
            output[directory / "inputs_and_golden.yaml"] = copy.deepcopy(snapshot["record"])
            continue
        if view not in snapshot["views"]:
            continue
        docs = {filename: {**copy.deepcopy(header), "cases": {}} for filename, header in snapshot["documents"].items()}
        for key, observation in snapshot["views"][view].items():
            filename, case_id = json.loads(key)
            doc = docs.setdefault(filename, {**copy.deepcopy(observation["producer"]["document"]), "cases": {}})
            doc["cases"][case_id] = {**copy.deepcopy(observation["payload"]["value"]), **copy.deepcopy(observation["annotations"])}
        output[directory / f"{capture}.yaml"] = {**copy.deepcopy(snapshot["header"]), "documents": docs}
    return output
