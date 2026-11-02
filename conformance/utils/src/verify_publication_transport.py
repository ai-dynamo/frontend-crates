# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently compare a frozen publication with its chunked-v2 successor."""

from __future__ import annotations
import argparse
import sys
import base64
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from email.utils import parsedate_to_datetime
from functools import lru_cache
import gzip
import hashlib
import json
import os
import multiprocessing
from pathlib import Path
import random
import re
import ssl
import tempfile
import threading
import traceback
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib


MAX_PACK = 1024 * 1024
MAX_RESOURCE = 1024 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(data, expected_hash, expected_size):
    require(len(data) == expected_size and digest(data) == expected_hash, "byte identity mismatch")
    return data


def read(root, relative):
    root = Path(root).resolve(strict=True)
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe resource path")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root), "resource escapes publication")
    if resolved.exists():
        return resolved.read_bytes()
    resources = embedded_publication(root)["resources"]
    require(relative in resources, f"resource is not embedded: {relative}")
    return resources[relative]


HTTP_BUDGET = 180
HTTP_TIMEOUT = 30
HTTP_ATTEMPTS = 4
RETRY_STATUSES = {429, 502, 503, 504}


def retry_delay(value, attempt):
    delay = 2 ** (attempt - 1) + random.uniform(0, 0.25)
    if value is not None:
        try:
            seconds = int(value) if re.fullmatch(r"[0-9]+", value) else parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return delay
        delay = max(delay, seconds)
    return delay


def transient_http_error(error):
    if isinstance(error, HTTPError):
        return error.code in RETRY_STATUSES
    reason = error.reason if isinstance(error, URLError) else error
    return isinstance(reason, (TimeoutError, ConnectionResetError)) and not isinstance(reason, ssl.SSLError)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_records(base_url, paths):
    return {path: {"url": f"{base_url.rstrip('/')}/{quote(path)}", "attempts": [], "outcome": "not-started"} for path in paths}


def _http_request_ids(headers):
    return {key: value for key, value in headers.items()
            if key.lower() in {"x-github-request-id", "x-request-id", "x-served-by", "x-cache"}}


def _scan_http_resources(root, base_url, paths, update, deadline=None, timeout=HTTP_TIMEOUT):
    started = time.monotonic()
    deadline = started + HTTP_BUDGET if deadline is None else deadline
    records = _http_records(base_url, paths)
    opener = build_opener(_NoRedirect())
    lock = threading.Lock()
    failure = None
    stopped = threading.Event()

    def emit(relative):
        with lock:
            update(relative, records[relative])

    def check(relative):
        record = records[relative]
        expected = read(root, relative)
        request = Request(record["url"], headers={"Accept-Encoding": "identity", "Cache-Control": "no-cache"})
        for attempt in range(1, HTTP_ATTEMPTS + 1):
            require(not stopped.is_set(), f"verification stopped: {request.full_url}; attempt {attempt}")
            remaining = deadline - time.monotonic()
            require(remaining > 0, f"verification budget exhausted: {request.full_url}; attempt {attempt}")
            entry = {"attempt": attempt, "status": None, "elapsed_seconds": time.monotonic() - started, "request_ids": {}}
            record["attempts"].append(entry)
            record["outcome"] = "active"
            emit(relative)
            error = None
            retry_after = None
            try:
                with opener.open(request, timeout=min(timeout, remaining)) as response:
                    entry["status"] = response.status
                    entry["request_ids"] = _http_request_ids(response.headers)
                    require(response.status == 200 and response.url == request.full_url,
                            f"unexpected publication HTTP response: {request.full_url}: {response.status}")
                    emit(relative)
                    actual = response.read(len(expected) + 1)
                require(actual == expected, f"published resource differs from qualified artifact: {request.full_url}")
            except (HTTPError, URLError, OSError, ValueError) as caught:
                error = caught
                entry["error"] = repr(caught)
                if isinstance(caught, HTTPError):
                    entry["status"] = caught.code
                    entry["request_ids"] = _http_request_ids(caught.headers)
                    retry_after = caught.headers.get("Retry-After")
                    caught.close()
            finally:
                entry["elapsed_seconds"] = time.monotonic() - started
                with lock:
                    print(json.dumps({"url": request.full_url, **entry}), flush=True)
                    update(relative, record)
            if error is None:
                record["outcome"] = "recovered" if attempt > 1 else "passed"
                record["bytes"] = len(expected)
                emit(relative)
                return len(expected)
            if not transient_http_error(error) or attempt == HTTP_ATTEMPTS:
                raise error
            delay = retry_delay(retry_after, attempt)
            entry["retry_delay_seconds"] = delay
            require(delay < deadline - time.monotonic(),
                    f"verification budget exhausted before retry: {request.full_url}; attempt {attempt}")
            time.sleep(delay)
        raise AssertionError("unreachable HTTP attempt")

    workers = ThreadPoolExecutor(max_workers=8)
    active = {}
    pending = iter(paths)
    total = 0
    try:
        while True:
            while len(active) < 8 and not stopped.is_set() and time.monotonic() < deadline:
                relative = next(pending, None)
                if relative is None:
                    break
                active[workers.submit(check, relative)] = relative
            if not active:
                break
            done, _ = wait(active, timeout=max(0, deadline + timeout - time.monotonic()), return_when=FIRST_COMPLETED)
            if not done:
                failure = failure or TimeoutError("verification budget and active-request grace exhausted")
                break
            for future in done:
                relative = active.pop(future)
                try:
                    total += future.result()
                except Exception as error:
                    # Resource supervisor: retain every failure and finish accounting
                    # for independent files before propagating the first cause.
                    records[relative]["outcome"] = "failed"
                    records[relative]["error"] = repr(error)
                    failure = failure or error
                    stopped.set()
                    emit(relative)
        for relative, record in records.items():
            if record["outcome"] in {"not-started", "active"}:
                record["outcome"] = "not-checked" if stopped.is_set() else "budget-exhausted"
                emit(relative)
                failure = failure or TimeoutError(f"verification budget exhausted: {record['url']}")
        if failure:
            raise failure
        return {"resources": len(paths), "bytes": total}
    finally:
        workers.shutdown(wait=False, cancel_futures=True)


def _http_scan_process(connection, root, base_url, paths, deadline, timeout):
    try:
        result = _scan_http_resources(root, base_url, paths, lambda path, record: connection.send(("resource", path, record)), deadline, timeout)
        connection.send(("result", result))
    except Exception as error:
        # Process supervisor preserves the original failure and traceback across IPC.
        detail = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if isinstance(error, HTTPError):
            detail.update(url=error.url, code=error.code, headers=dict(error.headers))
        connection.send(("error", detail))
    finally:
        connection.close()


def _run_http_scan(root, base_url, paths, update, deadline):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_http_scan_process, args=(sender, root, base_url, paths, deadline, HTTP_TIMEOUT))
    expires = deadline + HTTP_TIMEOUT
    worker.start()
    sender.close()
    try:
        while True:
            remaining = expires - time.monotonic()
            if remaining <= 0 or not receiver.poll(remaining):
                raise TimeoutError("verification budget and active-request grace exhausted")
            message = receiver.recv()
            if message[0] == "resource":
                update(message[1], message[2])
            elif message[0] == "result":
                return message[1]
            else:
                detail = message[1]
                if detail["type"] == "HTTPError":
                    raise HTTPError(detail["url"], detail["code"], detail["message"], detail["headers"], None)
                if detail["type"] == "TimeoutError":
                    raise TimeoutError(detail["message"])
                raise ValueError(detail["message"] + "\n" + detail["traceback"])
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=1)
        if worker.is_alive():
            worker.kill()
            worker.join()
        receiver.close()


def update_delivery_report(path, *, check=None, browser_journey=None):
    failure = sys.exc_info()[1]
    try:
        evidence = json.loads(path.read_bytes()) if path.is_file() and path.stat().st_size else {"checks": []}
        if check is not None:
            evidence["checks"].append(check)
        if browser_journey is not None:
            evidence["browser_journey"] = browser_journey
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    except Exception as error:
        # Reporting must fail CI without replacing the failure it was recording.
        if failure is not None:
            raise failure from error
        raise
    print(f"delivery report: {path}", flush=True)


def _verify_http_resources(root, base_url, paths):
    started = time.monotonic()
    records = _http_records(base_url, paths)
    report = {"base_url": base_url, "resources": records, "success": False}
    report_path = os.environ.get("CONFORMANCE_DELIVERY_REPORT")
    if report_path is None:
        fd, report_path = tempfile.mkstemp(prefix="conformance-delivery-", suffix=".json")
        os.close(fd)
    report_path = Path(report_path).resolve()
    require(not report_path.is_relative_to(root), "delivery report must remain outside publication")
    try:
        deadline = float(os.environ.get("CONFORMANCE_DELIVERY_DEADLINE", started + HTTP_BUDGET))
        require(deadline > started, "verification budget exhausted before scan")
        result = _run_http_scan(root, base_url, paths, lambda path, record: records.__setitem__(path, record), deadline)
        report["success"] = True
        return result
    except Exception as error:
        # Report supervisor records failures before propagating them to CI.
        report["error"] = repr(error)
        raise
    finally:
        for record in records.values():
            if record["outcome"] in {"active", "not-started"}:
                record["outcome"] = "not-checked" if record["outcome"] == "not-started" else "interrupted"
        report["elapsed_seconds"] = time.monotonic() - started
        update_delivery_report(report_path, check=report)


def verify_http_publication(root, base_url):
    parsed = urlsplit(base_url)
    require(parsed.scheme in {"http", "https"} and parsed.netloc and not parsed.username
            and not parsed.query and not parsed.fragment, "publication URL must be an HTTP directory")
    root = Path(root).resolve(strict=True)
    require((root / "index.html").is_file(), "qualified publication lacks index.html")
    paths = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    return _verify_http_resources(root, base_url, paths)


def verify_http_publication_identity(root, base_url):
    root = Path(root).resolve(strict=True)
    relative = "conformance/viewer-publication.json"
    require((root / relative).is_file(), "qualified history lacks publication identity")
    return _verify_http_resources(root, base_url, [relative])


def decompress(data, limit):
    require(0 <= limit <= MAX_RESOURCE, "invalid decompression bound")
    decoder = zlib.decompressobj(31)
    result = decoder.decompress(data, limit + 1)
    require(len(result) <= limit and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, "invalid or oversized gzip resource")
    return result


def compressed_json(root, path):
    return json.loads(decompress(read(root, path), MAX_RESOURCE))


def inline_json(html, identifier):
    match = re.search(r'<script type="application/json" id="' + re.escape(identifier) + r'">(.*?)</script>', html, re.S)
    require(match is not None, f"missing embedded publication block: {identifier}")
    return json.loads(match[1])


@lru_cache(maxsize=4)
def _embedded_publication(path, file_identity):
    html = path.read_text()
    payload = inline_json(html, "conformance-embedded")
    require(payload.get("schema") == 1 and isinstance(payload.get("resources"), list), "invalid embedded publication")
    descriptor = payload["catalog"]
    encoded = identity(base64.b64decode(descriptor["bytes"], validate=True), descriptor["sha256"], descriptor["size"])
    require(0 <= descriptor["decoded_size"] <= 64 * MAX_PACK, "embedded catalog exceeds limit")
    catalog_bytes = decompress(encoded, descriptor["decoded_size"])
    trailer = re.search(rb',"runtime_sha256":"([a-f0-9]{64})"}\n$', catalog_bytes)
    require(trailer is not None and digest(catalog_bytes[:trailer.start()] + b"}\n") == trailer[1].decode(), "embedded runtime catalog integrity mismatch")
    catalog = json.loads(catalog_bytes)
    resources = {}
    inventory = []
    total = 0
    for resource in payload["resources"]:
        descriptor = resource["descriptor"]
        name = descriptor["path"]
        require(not Path(name).is_absolute() and ":" not in name and "\\" not in name
                and all(part not in {"", ".", ".."} for part in name.split("/")), "unsafe embedded path")
        require(name not in resources, "duplicate embedded resource")
        total += descriptor["size"]
        require(total <= 256 * MAX_PACK, "embedded resources exceed limit")
        resources[name] = identity(base64.b64decode(resource["bytes"], validate=True), descriptor["sha256"], descriptor["size"])
        inventory.append([name, descriptor["size"], descriptor["sha256"]])
    require(digest((json.dumps(inventory, separators=(",", ":")) + "\n").encode()) == payload["resources_sha256"], "embedded membership mismatch")
    return {"catalog": catalog, "catalog_bytes": catalog_bytes, "publication": inline_json(html, "conformance-publication"), "resources": resources}


def embedded_publication(root):
    root = Path(root).resolve(strict=True)
    path = (root / "index.html").resolve(strict=True)
    require(path.is_relative_to(root), "embedded publication escapes root")
    stat = path.stat()
    # Restamping, replacement, and corruption must invalidate the cached HTML.
    return _embedded_publication(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def runtime_catalog(root):
    if (Path(root) / "conformance/viewer-resolved-catalog.json.gz").is_file():
        return compressed_json(root, "conformance/viewer-resolved-catalog.json.gz")
    return json.loads(embedded_publication(root)["catalog_bytes"])


def publication_metadata(root):
    path = Path(root) / "conformance/viewer-publication.json"
    return json.loads(path.read_bytes()) if path.is_file() else inline_json((Path(root) / "index.html").read_text(), "conformance-publication")


def materialize_embedded(root, destination):
    """Expose verified HTML transport in private qualification scratch only."""
    publication = embedded_publication(root)
    destination.mkdir(parents=True)
    (destination / "index.html").write_bytes((Path(root) / "index.html").read_bytes())
    for name, data in publication["resources"].items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    conformance = destination / "conformance"
    conformance.mkdir(exist_ok=True)
    (conformance / "viewer-resolved-catalog.json.gz").write_bytes(gzip.compress(publication["catalog_bytes"], mtime=0))
    (conformance / "viewer-publication.json").write_text(json.dumps(publication["publication"]))
    return destination


def read_index(root, descriptor):
    encoded = identity(read(root, descriptor["path"]), descriptor["sha256"], descriptor["size"])
    decoded = decompress(encoded, descriptor["decoded_size"])
    identity(decoded, descriptor["decoded_sha256"], descriptor["decoded_size"])
    value = json.loads(decoded)
    require(value["schema"] == 2, "incorrect index schema")
    return value["resources"]


class Decoder:
    """Bound shared decoded-pack caching to 64 MiB; never trust a cache hit's metadata."""

    def __init__(self, root):
        self.root = root
        self.cache = OrderedDict()
        self.cached_bytes = 0
        self.pack_reads = 0

    def pack(self, descriptor):
        path = descriptor["path"]
        require(path == f"conformance/viewer-data/packs/{descriptor['sha256']}.bin.gz", "invalid pack path")
        size = descriptor["decoded_size"]
        require(type(size) is int and 0 < size <= MAX_PACK, "invalid decoded pack size")
        if path in self.cache:
            original_descriptor, decoded = self.cache.pop(path)
            require(original_descriptor == descriptor, "conflicting cached pack identity")
        else:
            encoded = identity(read(self.root, path), descriptor["sha256"], descriptor["size"])
            decoded = decompress(encoded, size)
            identity(decoded, descriptor["decoded_sha256"], size)
            self.cached_bytes += len(decoded)
            self.pack_reads += 1
        self.cache[path] = (descriptor, decoded)
        while self.cached_bytes > 64 * MAX_PACK:
            _, (_, evicted) = self.cache.popitem(last=False)
            self.cached_bytes -= len(evicted)
        return decoded

    def reconstruct(self, recipe):
        require(recipe["schema"] == 2, "incorrect recipe schema")
        require(type(recipe["size"]) is int and 0 <= recipe["size"] < MAX_RESOURCE, "invalid logical size")
        result = bytearray()
        for chunk in recipe["chunks"]:
            pack = self.pack(chunk["pack"])
            offset, length = chunk["offset"], chunk["size"]
            require(type(offset) is int and type(length) is int and offset >= 0 and 0 < length <= 128 * 1024 and offset + length <= len(pack), "invalid chunk range")
            data = identity(pack[offset:offset + length], chunk["sha256"], length)
            require(len(result) + length <= recipe["size"], "logical resource exceeds declared size")
            result.extend(data)
        return identity(bytes(result), recipe["sha256"], recipe["size"])


def case_count(envelope):
    value = envelope["value"]
    require(value[0] == "object", "authored family must be an object")
    cases = next(value for key, value in value[1] if key == ["str", "cases"])
    require(cases[0] == "object", "authored cases must be an object")
    active = 0
    for _, case in cases[1]:
        lifecycle = next(value for key, value in case[1] if key == ["str", "lifecycle"])
        active += lifecycle != ["str", "retired"]
    return len(cases[1]), active


def compare(baseline, candidate):
    started = time.monotonic()
    before = runtime_catalog(baseline)
    after = runtime_catalog(candidate)
    require(after["transport"]["schema"] == 2, "candidate is not transport v2")
    before_resources = {r["path"]: r for r in before["sources"]["unified"] if not (r["kind"] == "inputs-and-golden" and isinstance(r.get("families"), list))}
    after_resources = {r["path"]: r for r in after["sources"]["unified"]}
    require(before_resources.keys() == after_resources.keys(), "logical capture/input selections differ")
    decoder = Decoder(candidate)
    baseline_decoder = Decoder(baseline)
    packed_baseline = before.get("transport", {}).get("schema") == 2
    counts = {"capture_selections": 0, "family_inputs": 0, "authored_unified_cases_total": 0, "authored_unified_cases_active": 0, "original_resources": 0, "legacy_raw_aliases": 0, "logical_bytes": 0, "original_bytes": 0}
    shared_inputs = None
    for path, original in before_resources.items():
        current = after_resources[path]
        for field in ("kind", "family", "families", "implementation", "version", "format", "record_contract", "capture_identity", "source_fingerprint", "resolved_metadata_sha256"):
            require(original.get(field) == current.get(field), f"descriptor metadata differs: {field}: {path}")
        if packed_baseline:
            old_bytes = baseline_decoder.reconstruct(read_index(baseline, original["recipe_index"])[path])
            identity(old_bytes, original["sha256"], original["size"])
        else:
            encoded = identity(read(baseline, path), original["sha256"], original["size"])
            old_bytes = decompress(encoded, MAX_RESOURCE)
        if original["kind"] == "inputs-and-golden":
            if shared_inputs is None:
                shared_inputs = read_index(candidate, current["recipe_index"])
            recipe = shared_inputs[path]
            counts["family_inputs"] += 1
            total, active = case_count(json.loads(old_bytes))
            counts["authored_unified_cases_total"] += total
            counts["authored_unified_cases_active"] += active
        else:
            recipe = read_index(candidate, current["recipe_index"])[path]
            counts["capture_selections"] += 1
        new_bytes = decoder.reconstruct(recipe)
        identity(new_bytes, current["sha256"], current["size"])
        require(old_bytes == new_bytes, f"decoded payload differs: {path}")
        counts["logical_bytes"] += len(new_bytes)
    if packed_baseline:
        baseline_raw = {}
        for descriptor in before["transport"]["raw_buckets"].values():
            baseline_raw.update(read_index(baseline, descriptor))
        original_index = {path: entry["resource"] for path, entry in baseline_raw.items() if "resource" in entry}
    else:
        original_index = compressed_json(baseline, "conformance/viewer-original-resources.json.gz")["resources"]
    seen = set()
    for bucket, descriptor in after["transport"]["raw_buckets"].items():
        for path, entry in read_index(candidate, descriptor).items():
            require(digest(path.encode())[:2] == bucket, "raw path in incorrect bucket")
            if "resource" not in entry:
                require(entry["path"] in original_index and isinstance(entry["member"], str), "unresolved legacy alias")
                if packed_baseline:
                    require(entry == baseline_raw.get(path), f"legacy alias differs: {path}")
                counts["legacy_raw_aliases"] += 1
                continue
            require(path in original_index and path not in seen, f"unexpected/duplicate original: {path}")
            seen.add(path)
            old = original_index[path]
            resource = entry["resource"]
            if packed_baseline:
                old_bytes = baseline_decoder.reconstruct(old["recipe"])
                require(all(resource[key] == old[key] for key in ("path", "sha256", "size")), "original metadata differs")
            else:
                # Historical envelopes remain supported only by this independent verifier.
                envelope_path = f"conformance/viewer-data/originals/{old['source_sha256']}.json.gz"
                envelope_bytes = identity(read(baseline, envelope_path), old["sha256"], old["size"])
                envelope = json.loads(decompress(envelope_bytes, MAX_RESOURCE))
                require(envelope["schema"] == 1 and envelope["encoding"] == "base64", "invalid baseline envelope")
                old_bytes = identity(base64.b64decode(envelope["bytes"], validate=True), old["source_sha256"], old["source_size"])
                require(resource["path"] == path and resource["sha256"] == old["source_sha256"] and resource["size"] == old["source_size"], "original metadata differs")
            new_bytes = decoder.reconstruct(resource["recipe"])
            require(old_bytes == new_bytes, f"original bytes differ: {path}")
            counts["original_resources"] += 1
            counts["original_bytes"] += len(new_bytes)
    require(seen == original_index.keys(), "candidate omits originals")
    if packed_baseline:
        require(counts["legacy_raw_aliases"] == len(baseline_raw) - len(original_index), "candidate omits Raw aliases")
        require(before["metadata"] == after["metadata"], "report metadata differs")
        require(before["sources"]["candidates"] == after["sources"]["candidates"], "selection identities differ")
        require(before["sources"]["original_unified"] == after["sources"]["original_unified"], "provenance differs")
    return {"schema": 1, "baseline": str(baseline), "candidate": str(candidate), "equal": True, **counts, "pack_reads": decoder.pack_reads, "pack_cache_limit_bytes": 64 * MAX_PACK, "wall_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence outside either publication")
    parser.add_argument("--baseline-counts", type=Path, help="Frozen cross-suite count evidence; preserved by exact equality of every authored resource")
    args = parser.parse_args()
    baseline, candidate, output = args.baseline.resolve(strict=True), args.candidate.resolve(strict=True), args.output.resolve()
    require(not output.is_relative_to(baseline) and not output.is_relative_to(candidate), "evidence must remain outside publication")
    result = compare(baseline, candidate)
    if args.baseline_counts:
        count_bytes = args.baseline_counts.read_bytes()
        count_evidence = json.loads(count_bytes)
        unified_count = next(tab["parser_case_pairs"] for tab in count_evidence["tabs"] if tab["tab"] == "tab-unified")
        require(unified_count == result["authored_unified_cases_active"], "frozen Unified count differs")
        result["preserved_authored_parser_case_pairs"] = count_evidence["parser_case_pairs"]
        result["count_evidence"] = {"path": str(args.baseline_counts.resolve()), "sha256": digest(count_bytes), "size": len(count_bytes), "basis": "Exact equality of all family inputs and all original archives; active Unified cases recounted independently"}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def snapshot(root: Path, *, include_history: bool = False) -> dict:
    conformance = root / "conformance"
    evidence = {}
    directories = [path for path in conformance.glob("fixtures*") if path.is_dir()]
    directories.extend(conformance / name for name in ("captures", "inputs") if (conformance / name).is_dir())
    if include_history:
        directories.append(conformance / "historic")
    for directory in sorted(directories):
        if directory.is_symlink():
            raise ValueError(f"evidence must not contain links: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"evidence must not contain links: {path}")
            if path.is_file():
                with path.open("rb") as source:
                    evidence[path.relative_to(root).as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    yaml_paths = sorted(path.relative_to(root).as_posix() for directory in directories for path in directory.rglob("*.yaml") if path.is_file())
    return {"evidence": evidence, "yaml_paths": yaml_paths}


def evidence_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("snapshot", "check"))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--include-history", action="store_true")
    args = parser.parse_args()
    current = snapshot(args.root.resolve(), include_history=args.include_history)
    if args.operation == "snapshot":
        args.inventory.write_text(json.dumps(current, sort_keys=True) + "\n")
        return 0
    expected = json.loads(args.inventory.read_bytes())
    if expected != current:
        old = expected["evidence"]
        new = current["evidence"]
        changed = sorted(
            path for path in old.keys() | new.keys() if old.get(path) != new.get(path)
        )
        added_yaml = sorted(set(current["yaml_paths"]) - set(expected["yaml_paths"]))
        removed_yaml = sorted(set(expected["yaml_paths"]) - set(current["yaml_paths"]))
        raise ValueError(
            f"report generation changed evidence={changed}, created YAML={added_yaml}, removed YAML={removed_yaml}"
        )
    print(
        f"permanent evidence unchanged: {len(current['evidence'])} files; no per-case YAML created"
    )
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] in (["snapshot"], ["check"]):
        raise SystemExit(evidence_main())
    main()
