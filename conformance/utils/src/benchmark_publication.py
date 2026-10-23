#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare frozen report publications in paired, cold Chromium navigations."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import statistics
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(root: Path) -> dict:
    files = [(path.relative_to(root).as_posix(), path.stat().st_size, digest(path.read_bytes()))
             for path in sorted(root.rglob("*")) if path.is_file()]
    return {"root": str(root), "files": len(files), "bytes": sum(row[1] for row in files),
            "inventory_sha256": digest(json.dumps(files, separators=(",", ":")).encode())}


def default_closure(root: Path) -> set[str] | None:
    catalog = json.loads(gzip.decompress((root / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    if catalog.get("transport", {}).get("schema") != 2:
        return None
    implementation = catalog["metadata"]["selection_policy"]["unified"]["reference"]
    candidates = [item for item in catalog["sources"]["candidates"]
                  if item["implementation"] == implementation and item["mode"] == "unified"]
    selected = max(candidates, key=lambda item: tuple(int(part) for part in item["version"].split(".")))
    families = set(selected["native_unified_families"] if implementation == "dynamo_v2"
                   else selected["capture_families"])
    resources = [item for item in catalog["sources"]["unified"]
                 if (item["kind"] == "inputs-and-golden" and item["family"] in families)
                 or (item["kind"] == "resolved-capture" and item["implementation"] == implementation
                     and item["version"] == selected["version"])]
    allowed = set()
    pack_intervals = {}
    pack_sizes = {}
    for resource in resources:
        index = resource["recipe_index"]
        allowed.add(index["path"])
        recipes = json.loads(gzip.decompress((root / index["path"]).read_bytes()))["resources"]
        for chunk in recipes[resource["path"]]["chunks"]:
            pack = chunk["pack"]
            allowed.add(pack["path"])
            pack_sizes[pack["path"]] = pack["decoded_size"]
            pack_intervals.setdefault(pack["path"], set()).add((chunk["offset"], chunk["offset"] + chunk["size"]))
    for path, intervals in pack_intervals.items():
        end = 0
        for start, stop in sorted(intervals):
            if start > end:
                raise ValueError(f"default pack contains bytes outside its dependency closure: {path}")
            end = max(end, stop)
        if end != pack_sizes[path]:
            raise ValueError(f"default pack has unrelated trailing bytes: {path}")
    return allowed


class PublicationServer:
    def __init__(self, root: Path):
        self.requests = []
        self.lock = threading.Lock()
        owner = self

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)

            def log_message(self, *_args):
                pass

            def do_GET(self):
                start = time.monotonic_ns()
                path = Path(self.translate_path(urlsplit(self.path).path))
                if path.is_dir():
                    path /= "index.html"
                if not path.is_file():
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.send_header("Cache-Control", "no-store")
                    header_size = sum(map(len, self._headers_buffer)) + 2
                    self.end_headers()
                    row = {"url": self.path, "status": 404, "encoded_bytes": 0,
                           "transferred_bytes": header_size, "start_ns": start}
                else:
                    data = path.read_bytes()
                    compress = path.suffix in {".html", ".js", ".json", ".css"}
                    if compress:
                        data = gzip.compress(data, mtime=0)
                    self.send_response(200)
                    self.send_header("Content-Type", self.guess_type(str(path)))
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    if compress:
                        self.send_header("Content-Encoding", "gzip")
                    header_size = sum(map(len, self._headers_buffer)) + 2
                    self.end_headers()
                    self.wfile.write(data)
                    row = {"url": self.path, "status": 200, "encoded_bytes": len(data),
                           "transferred_bytes": len(data) + header_size, "start_ns": start}
                row["end_ns"] = time.monotonic_ns()
                with owner.lock:
                    owner.requests.append(row)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def navigation(driver, server: PublicationServer, label: str, trial: int,
               throughput: int, allowed: set[str] | None) -> dict:
    driver.get("about:blank")
    driver.execute_cdp_cmd("Network.clearBrowserCache", {})
    driver.execute_cdp_cmd("Network.setCacheDisabled", {"cacheDisabled": True})
    driver.execute_cdp_cmd("Storage.clearDataForOrigin", {"origin": server.origin, "storageTypes": "all"})
    driver.execute_cdp_cmd("Network.emulateNetworkConditions", {
        "offline": False, "latency": 0, "downloadThroughput": throughput,
        "uploadThroughput": -1, "connectionType": "ethernet"})
    with server.lock:
        server.requests.clear()
    started = time.monotonic_ns()
    driver.get(server.origin + "/?tab=tab-unified")
    WebDriverWait(driver, 180, poll_frequency=0.05).until(
        lambda page: page.execute_script("return window.__conformanceStartup?.painted || window.__conformanceError"))
    result = driver.execute_async_script("""
        const done = arguments[0];
        if (window.__conformanceError) { done({error: window.__conformanceError}); return; }
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const entries = [...performance.getEntriesByType('navigation'), ...performance.getEntriesByType('resource')];
          done({painted_ready_ms: window.__conformanceStartup.paintedReady,
            observed_ready_ms: performance.now(), phases: window.__conformanceStartup.phases,
            resource_metrics: window.__conformanceStartup.resources,
            document_timing_encoded_bytes: entries.reduce((n, item) => n + item.encodedBodySize, 0),
            document_timing_transferred_bytes: entries.reduce((n, item) => n + item.transferSize, 0),
            viewport: [innerWidth, innerHeight],
            families: window.__conformanceModel.rows.filter(row => !row.section).map(row => row.family)});
        }));
    """)
    if "error" in result:
        raise RuntimeError(f"{label}: {result['error']}")
    finished = time.monotonic_ns()
    with server.lock:
        requests = [row.copy() for row in server.requests if row["start_ns"] >= started]
    paths = {urlsplit(row["url"]).path.lstrip("/") for row in requests}
    errors = []
    if result["viewport"] != [1440, 1000]:
        errors.append(f"unexpected measured viewport: {result['viewport']}")
    if any(row["status"] != 200 and urlsplit(row["url"]).path != "/favicon.ico" for row in requests):
        errors.append("startup requested a missing resource")
    if allowed is not None:
        forbidden = [path for path in paths if "raw-" in path or "/originals/" in path
                     or path.endswith((".yaml", ".tar.gz"))
                     or ("/viewer-data/" in path and path not in allowed)]
        if forbidden:
            errors.append(f"startup requested unrelated/original resources: {sorted(forbidden)}")
    return {"publication": label, "trial": trial, "download_bytes_per_second": throughput,
            "requests": requests, "request_count": len(requests),
            "encoded_bytes": sum(row["encoded_bytes"] for row in requests),
            "transferred_bytes": sum(row["transferred_bytes"] for row in requests),
            "wall_ms": (finished - started) / 1e6, "errors": errors, **result}


def benchmark(baseline: Path, candidate: Path, trials: int) -> dict:
    roots = {"baseline": baseline, "candidate": candidate}
    initial = {name: snapshot(root) for name, root in roots.items()}
    closures = {name: default_closure(root) for name, root in roots.items()}
    servers = {name: PublicationServer(root) for name, root in roots.items()}
    options = Options()
    for option in ["--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--window-size=1440,1000"]:
        options.add_argument(option)
    records = []
    try:
        with webdriver.Chrome(options=options) as driver:
            driver.set_script_timeout(30)
            driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
            browser = driver.capabilities
            for throughput in [-1, 20_000_000]:
                for trial in range(trials):
                    order = ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
                    for name in order:
                        row = navigation(driver, servers[name], name, trial, throughput, closures[name])
                        records.append(row)
                        print(json.dumps({key: row[key] for key in ["publication", "trial", "download_bytes_per_second", "painted_ready_ms", "request_count", "encoded_bytes", "errors"]}), flush=True)
    finally:
        for server in servers.values():
            server.close()
    final = {name: snapshot(root) for name, root in roots.items()}
    if initial != final:
        raise RuntimeError("publication files changed during benchmark")
    summaries = {}
    failures = [error for row in records for error in row["errors"]]
    for throughput in [-1, 20_000_000]:
        medians = {name: {key: statistics.median(row[key] for row in records
                                               if row["publication"] == name and row["download_bytes_per_second"] == throughput)
                          for key in ["painted_ready_ms", "encoded_bytes", "transferred_bytes", "request_count"]}
                   for name in roots}
        for key in medians["baseline"]:
            limit = 1.10 if key == "painted_ready_ms" else 1.0
            if medians["candidate"][key] > medians["baseline"][key] * limit:
                failures.append(f"{throughput} bytes/s: candidate {key} exceeds baseline by allowed factor {limit}")
        summaries[str(throughput)] = {"medians": medians,
            "paired_readiness_ratios": [next(row["painted_ready_ms"] for row in records if row["publication"] == "candidate" and row["trial"] == trial and row["download_bytes_per_second"] == throughput)
                                        / next(row["painted_ready_ms"] for row in records if row["publication"] == "baseline" and row["trial"] == trial and row["download_bytes_per_second"] == throughput)
                                        for trial in range(trials)]}
    return {"schema": 1, "trials": trials, "publications": initial,
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "host_cpus": os.cpu_count(),
            "browser": browser, "viewport": [1440, 1000], "latency_ms": 0,
            "cache_policy": "cold HTTP and IndexedDB; cache disabled; all responses no-store",
            "transfer_scope": "server HTTP response bodies plus HTTP headers across document and workers; excludes TCP framing",
            "timing_scope": "paintedReady application metric; observation waits two additional animation frames",
            "default_dependency_closures": {name: sorted(paths) if paths is not None else None for name, paths in closures.items()},
            "records": records, "summaries": summaries,
            "acceptance": {"passed": not failures, "failures": failures}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    affinity = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, affinity)
    result = benchmark(args.baseline.resolve(), args.candidate.resolve(), args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["acceptance"]))
    return 0 if result["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
