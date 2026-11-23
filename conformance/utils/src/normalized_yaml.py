# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalized source storage; callers validate the expanded conformance model."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

from viewer_codec import canonical_typed, encode_typed, typed_equal

SCHEMA = 3
PRODUCER_FIELDS = {"captured_with", "capture_origin"}


def _key(value):
    return json.dumps(canonical_typed(value), ensure_ascii=False, separators=(",", ":"))


def request_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _portable_payload(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is list:
        return all(_portable_payload(item) for item in value)
    return type(value) is dict and all(type(key) is str and _portable_payload(item) for key, item in value.items())


def _payload_digest(value):
    # Existing request digests agree across readers for this exact JSON subset.
    if type(value) is not dict or not _portable_payload(value):
        raise ValueError("result reference requires an integer-only JSON payload")
    return request_digest(value)


def _expand_result_refs(context):
    metadata = context.get("record_metadata")
    if not isinstance(metadata, dict):
        return
    active = metadata.get("verified_recapture")
    if isinstance(active, dict) and "result_ref" in active:
        raise ValueError("active result must be inline")
    archives = metadata.get("verified_recaptures", [])
    if not isinstance(archives, list):
        raise ValueError("invalid archived recaptures")
    references = [receipt for receipt in archives if isinstance(receipt, dict) and "result_ref" in receipt]
    if not references:
        return
    if not isinstance(active, dict) or "result" not in active:
        raise ValueError("result reference has no active payload")
    digest = _payload_digest(active["result"])
    for receipt in references:
        if "result" in receipt or type(receipt["result_ref"]) is not str or receipt["result_ref"] != digest:
            raise ValueError("invalid result reference or digest mismatch")
        expanded = {"result" if key == "result_ref" else key:
                    copy.deepcopy(active["result"]) if key == "result_ref" else value
                    for key, value in receipt.items()}
        receipt.clear()
        receipt.update(expanded)


def _compact_result_refs(context):
    metadata = context.get("record_metadata")
    if not isinstance(metadata, dict):
        return
    active = metadata.get("verified_recapture")
    if not isinstance(active, dict) or type(active.get("result")) is not dict or not _portable_payload(active["result"]):
        return
    payload = active["result"]
    digest = _payload_digest(payload)
    ordered = encode_typed(payload)
    for receipt in metadata.get("verified_recaptures", []):
        if isinstance(receipt, dict) and "result" in receipt and encode_typed(receipt["result"]) == ordered:
            compact = {"result_ref" if key == "result" else key: digest if key == "result" else value
                       for key, value in receipt.items()}
            receipt.clear()
            receipt.update(compact)


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError("invalid normalized YAML fields")
    return value


def _reference(table, key):
    if not isinstance(key, str) or key not in table:
        raise ValueError(f"unknown normalized reference: {key!r}")
    return copy.deepcopy(table[key])


def _intern(table, value, prefix):
    identity = prefix + hashlib.sha256(_key(value).encode()).hexdigest()[:16]
    if identity in table and not typed_equal(table[identity], value):
        raise ValueError("normalized definition identity collision")
    table[identity] = copy.deepcopy(value)
    return identity


def expand_requests(document):
    _fields(document, {"schema_version", "family", "request_defaults", "tools", "requests", "cases"})
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA:
        raise ValueError("invalid normalized authored schema")
    defaults = _fields(document["request_defaults"], {"init", "finish_reason"})
    result = {}
    for identity, stored in document["requests"].items():
        _fields(stored, {"input", "tools"}, {"init", "finish_reason", "chunks", "chunk_sizes", "finish_step"})
        request = {**copy.deepcopy(defaults), **copy.deepcopy(stored)}
        request["tools"] = _reference(document["tools"], stored["tools"])
        if "chunk_sizes" in stored:
            if "chunks" in stored or not isinstance(stored["chunk_sizes"], list) or type(stored.get("finish_step", False)) is not bool:
                raise ValueError("ambiguous normalized chunk schedule")
            chunks = []
            offset = 0
            for size in stored["chunk_sizes"]:
                if type(size) is not int or size < 0 or offset + size > len(stored["input"]):
                    raise ValueError("invalid normalized chunk size")
                chunks.append(stored["input"][offset:offset + size])
                offset += size
            if offset != len(stored["input"]):
                raise ValueError("normalized chunk sizes do not cover input")
            if stored.get("finish_step", False):
                chunks.append("‹finish›")
        else:
            if "finish_step" in stored:
                raise ValueError("finish step requires a chunk schedule")
            chunks = stored.get("chunks", [stored["input"]])
        if not isinstance(chunks, list):
            raise ValueError("invalid normalized chunk schedule")
        request.pop("chunk_sizes", None)
        request.pop("finish_step", None)
        request["chunks"] = [{"delta_text": chunk} if isinstance(chunk, str) else copy.deepcopy(chunk) for chunk in chunks]
        if request_digest(request) != identity:
            raise ValueError("normalized request digest mismatch")
        result[identity] = request
    return result


def expand_family(document):
    if document.get("schema_version") != SCHEMA:
        return document
    requests = expand_requests(document)
    family = document["family"]
    cases = {}
    for identity, stored in document["cases"].items():
        _fields(stored, {"description", "display_id", "request", "golden"}, {"lifecycle", "scenario", "policy", "historical_ids"})
        case = {"lifecycle": "active", "scenario": identity, "policy": [], "historical_ids": [], **copy.deepcopy(stored)}
        case["request"] = None if stored["request"] is None else _reference(requests, stored["request"])
        case["golden"] = None if stored["golden"] is None else {"assembled": copy.deepcopy(stored["golden"])}
        cases[identity] = case
    return {"input_document": {"family": family}, "golden_document": {"family": family}, "cases": cases}


def encode_family(document, frozen_requests=()):
    family = document["input_document"]["family"]
    for header, allowed in ((document["input_document"], {"family": family, "mode": "unified", "model_label": family}),
                            (document["golden_document"], {"family": family, "mode": "unified", "captured_with": {"golden": "v1"}})):
        if any(key not in allowed or not typed_equal(value, allowed[key]) for key, value in header.items()):
            raise ValueError("cannot discard custom authored headers")
    requests = {request_digest(value): copy.deepcopy(value) for value in frozen_requests}
    for case in document["cases"].values():
        if case["request"] is not None:
            requests[request_digest(case["request"])] = copy.deepcopy(case["request"])
    if not requests:
        raise ValueError("normalized authored file needs requests")
    def common(field):
        keys = Counter(_key(value[field]) for value in requests.values())
        chosen = keys.most_common(1)[0][0]
        return copy.deepcopy(next(value[field] for value in requests.values() if _key(value[field]) == chosen))
    defaults = {field: common(field) for field in ("init", "finish_reason")}
    tools = {}; encoded = {}
    for identity, request in sorted(requests.items()):
        stored = {key: copy.deepcopy(value) for key, value in request.items() if key not in defaults or not typed_equal(value, defaults[key])}
        stored["tools"] = _intern(tools, request["tools"], "tools_")
        chunks = request["chunks"]
        if typed_equal(chunks, [{"delta_text": request["input"]}]):
            stored.pop("chunks")
        else:
            texts = [chunk["delta_text"] for chunk in chunks if set(chunk) == {"delta_text"} and isinstance(chunk["delta_text"], str)]
            finish = bool(texts and texts[-1] == "‹finish›" and "".join(texts[:-1]) == request["input"])
            wire = texts[:-1] if finish else texts
            if len(texts) == len(chunks) and "".join(wire) == request["input"]:
                stored.pop("chunks")
                stored["chunk_sizes"] = [len(text) for text in wire]
                if finish:
                    stored["finish_step"] = True
            else:
                stored["chunks"] = [chunk["delta_text"] if set(chunk) == {"delta_text"} and isinstance(chunk["delta_text"], str) else copy.deepcopy(chunk) for chunk in chunks]
        encoded[identity] = stored
    cases = {}
    for identity, case in document["cases"].items():
        stored = copy.deepcopy(case)
        for key, default in (("lifecycle", "active"), ("scenario", identity), ("policy", []), ("historical_ids", [])):
            if typed_equal(stored[key], default):
                del stored[key]
        stored["request"] = None if case["request"] is None else request_digest(case["request"])
        stored["golden"] = None if case["golden"] is None else copy.deepcopy(case["golden"]["assembled"])
        cases[identity] = stored
    return {"schema_version": SCHEMA, "family": family, "request_defaults": defaults,
            "tools": tools, "requests": encoded, "cases": cases}


def expand_snapshot(document, authored):
    if document.get("schema_version") != SCHEMA:
        return document
    _fields(document, {"schema_version", "provenance", "document", "origins", "producers", "contexts", "record_defaults", "records"},
            {"capture_selection", "execution_proof", "historical_evidence"})
    if type(document["schema_version"]) is not int:
        raise ValueError("invalid normalized capture schema")
    requests = expand_requests(authored)
    origins = document["origins"]
    provenance = copy.deepcopy(document["provenance"])
    if "origin" in provenance:
        provenance["origin"] = _reference(origins, provenance["origin"])
    defaults = _fields(document["record_defaults"], {"member", "producer", "context"})
    contexts = copy.deepcopy(document["contexts"])
    for context in contexts.values():
        if isinstance(context, dict):
            _expand_result_refs(context)
    records = {}
    for identity, stored in document["records"].items():
        _fields(stored, {"case_key"}, {"request", "stimulus", "output", "error", "unavailable", "producer", "context", "member", "suppress_inherited_from"})
        if ("request" in stored) == ("stimulus" in stored):
            raise ValueError("normalized record needs exactly one stimulus")
        states = set(stored) & {"output", "error", "unavailable"}
        if len(states) != 1:
            raise ValueError("normalized record needs exactly one observation")
        producer = _reference(document["producers"], stored.get("producer", defaults["producer"]))
        if set(producer) - PRODUCER_FIELDS:
            raise ValueError("unknown normalized producer field")
        if "capture_origin" in producer and producer["capture_origin"] is not None:
            producer["capture_origin"] = _reference(origins, producer["capture_origin"])
        context = _reference(contexts, stored.get("context", defaults["context"]))
        if not isinstance(context, dict):
            raise ValueError("normalized context must be a mapping")
        if set(context) & PRODUCER_FIELDS:
            raise ValueError("normalized context repeats producer fields")
        stimulus = ({"inline": _reference(requests, stored["request"]), "semantic_sha256": stored["request"]}
                    if "request" in stored else copy.deepcopy(stored["stimulus"]))
        state = next(iter(states))
        value = {"case_key": stored["case_key"], "document": {**context, **producer}, "stimulus": stimulus,
                 "observation": {"value" if state == "output" else state: copy.deepcopy(stored[state])}}
        wrapped = {"member": stored.get("member", defaults["member"]), "value": value}
        if "suppress_inherited_from" in stored:
            wrapped["suppress_inherited_from"] = stored["suppress_inherited_from"]
        records[identity] = wrapped
    return {**{key: copy.deepcopy(value) for key, value in document.items() if key not in {"origins", "producers", "contexts", "record_defaults", "records"}},
            "schema_version": 2, "provenance": provenance, "records": records}


def encode_snapshot(document, authored, *, result_refs=True):
    requests = expand_requests(authored)
    origins = {}; producers = {}; contexts = {}; records = {}
    provenance = copy.deepcopy(document["provenance"])
    if "origin" in provenance:
        provenance["origin"] = _intern(origins, provenance["origin"], "origin_")
    for identity, wrapped in document["records"].items():
        value = wrapped["value"]; metadata = value["document"]
        producer = {key: copy.deepcopy(value) for key, value in metadata.items() if key in PRODUCER_FIELDS}
        if "capture_origin" in producer and producer["capture_origin"] is not None:
            producer["capture_origin"] = _intern(origins, producer["capture_origin"], "origin_")
        context = {key: copy.deepcopy(value) for key, value in metadata.items() if key not in PRODUCER_FIELDS}
        stored = {"case_key": value["case_key"], "member": wrapped["member"],
                  "producer": _intern(producers, producer, "producer_"), "context": _intern(contexts, context, "context_")}
        stimulus = value["stimulus"]
        if set(stimulus) == {"inline", "semantic_sha256"} and stimulus["semantic_sha256"] == request_digest(stimulus["inline"]):
            digest = stimulus["semantic_sha256"]
            if digest not in requests or not typed_equal(requests[digest], stimulus["inline"]):
                raise ValueError("capture request is absent from frozen request library")
            stored["request"] = digest
        else:
            stored["stimulus"] = copy.deepcopy(stimulus)
        for key, value in value["observation"].items():
            stored["output" if key == "value" else key] = copy.deepcopy(value)
        if "suppress_inherited_from" in wrapped:
            stored["suppress_inherited_from"] = wrapped["suppress_inherited_from"]
        records[identity] = stored
    if result_refs:
        for context in contexts.values():
            _compact_result_refs(context)
    defaults = {field: Counter(record[field] for record in records.values()).most_common(1)[0][0]
                for field in ("member", "producer", "context")}
    for record in records.values():
        for field, value in defaults.items():
            if record[field] == value:
                del record[field]
    return {**{key: copy.deepcopy(value) for key, value in document.items() if key not in {"schema_version", "provenance", "records"}},
            "schema_version": SCHEMA, "provenance": provenance, "origins": origins, "producers": producers,
            "contexts": contexts, "record_defaults": defaults, "records": records}
