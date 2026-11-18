# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise selection cancellation against held Chromium HTTP response bodies."""
import gzip
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip('selenium')
from selenium.webdriver.support.ui import WebDriverWait

from test_browser_viewer import ROOT, _driver, _server


@pytest.mark.parametrize("transport", ["source", "chunked", "indexed"])
def test_browser_obsolete_downloads_release_slots_and_preserve_shared_raw_reads(transport):
    release = threading.Event()
    requested = set()
    payloads = {f"old-{index}": f"old-{index}".encode() for index in range(8)}
    payloads["current"] = b"current"
    index_decoded_size = 0
    if transport != "source":
        payloads = {key: gzip.compress(value, mtime=0) for key, value in payloads.items()}
        digest = hashlib.sha256(b'current').hexdigest()
        pack = {'path': 'current', 'sha256': hashlib.sha256(payloads['current']).hexdigest(),
                'size': len(payloads['current']), 'decoded_sha256': digest, 'decoded_size': 7}
        index_bytes = json.dumps({'schema': 2, 'resources': {'logical/current': {
            'schema': 2, 'sha256': digest, 'size': 7,
            'chunks': [{'sha256': digest, 'size': 7, 'offset': 0, 'pack': pack}],
        }}}).encode()
        index_decoded_size = len(index_bytes)
        payloads['current-index'] = gzip.compress(index_bytes, mtime=0)
    lock = threading.Lock()
    servers = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            key = self.path.split('?')[0].strip('/')
            with lock:
                requested.add(key)
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(payloads[key])))
            self.end_headers()
            self.wfile.flush()
            if key.startswith('old-'):
                release.wait(30)
            try:
                self.wfile.write(payloads[key])
            except (BrokenPipeError, ConnectionResetError):
                pass

    module_server, _ = _server(root=ROOT)
    driver = _driver()
    try:
        # Six old reads occupy the new selection's HTTP origin; two more
        # occupy separate origins so every scheduler slot owns a real read.
        for _ in range(3):
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        driver.get(f'http://127.0.0.1:{module_server.server_port}/conformance/viewer/viewer.css')
        driver.execute_async_script("""
            const done = arguments[arguments.length - 1], ports = arguments[0], transport = arguments[1], encoded = arguments[2], indexDecodedSize = arguments[3];
            (async () => {
                const {HttpResourceStore, sha256} = await import('/conformance/viewer/data.js');
                const descriptor = async path => {
                    const bytes = new TextEncoder().encode(path), hash = await sha256(bytes);
                    if (transport === 'source') return {path, sha256: hash, size: bytes.length};
                    const compressed = Uint8Array.from(encoded[path]);
                    const pack = {path, sha256: await sha256(compressed), size: compressed.length,
                        decoded_sha256: hash, decoded_size: bytes.length};
                    if (transport === 'indexed' && path === 'current') {
                        const index = Uint8Array.from(encoded['current-index']);
                        return {path: 'logical/current', sha256: hash, size: bytes.length, transport: 'chunked-v2',
                            recipe_index: {path: 'current-index', sha256: await sha256(index), size: index.length,
                                decoded_size: indexDecodedSize}};
                    }
                    return {path: `logical/${path}`, sha256: hash, size: bytes.length, transport: 'chunked-v2',
                        recipe: {schema: 2, sha256: hash, size: bytes.length,
                            chunks: [{sha256: hash, size: bytes.length, offset: 0, pack}]}};
                };
                const resources = await Promise.all([...Array(8)].map((_, i) => descriptor(`old-${i}`)));
                const store = new HttpResourceStore({allowSourceFiles: true, baseUrl: new URL('/', location.href),
                    fetchImpl: (url, options) => {
                        const path = url.pathname.slice(1), index = path.startsWith('old-') && +path.slice(4) >= 6 ? +path.slice(4) - 5 : 0;
                        options.signal?.addEventListener('abort', () => probe.fetchAborts.push(path), {once: true});
                        return fetch(`http://127.0.0.1:${ports[index]}/${path}`, options);
                    }});
                window.probe = {store, resources, descriptor, settled: [], errors: [], fetchAborts: []};
                await store.demand(resources);
                resources.forEach((resource, i) => store.read(resource, {selection: true})
                    .then(() => probe.settled.push(i), error => probe.errors.push(String(error))));
                done();
            })().catch(error => done(String(error)));
        """, [server.server_port for server in servers], transport, {key: list(value) for key, value in payloads.items()}, index_decoded_size)
        WebDriverWait(driver, 10).until(lambda _: len(requested) == 8)
        assert driver.execute_script('return probe.store.scheduler.active') == 8
        assert driver.execute_script('return probe.errors') == []
        driver.execute_async_script("""
            const done = arguments[arguments.length - 1];
            (async () => {
                const {sha256} = await import('/conformance/viewer/data.js');
                // One selected read and one Raw consumer survive the switch.
                probe.raw = probe.store.read(probe.resources[1]).then(() => {probe.rawReady = true;});
                const current = await probe.descriptor('current');
                await probe.store.demand([probe.resources[0], current]);
                probe.store.read(current, {selection: true}).then(value => {
                    probe.current = new TextDecoder().decode(value);
                }, error => probe.errors.push(String(error)));
                done();
            })().catch(error => done(String(error)));
        """)
        WebDriverWait(driver, 5).until(lambda page: page.execute_script('return probe.current === "current"'))
        assert not release.is_set(), 'new selection must finish while obsolete HTTP bodies remain held'
        assert driver.execute_script('return probe.store.scheduler.active') == 2
        assert driver.execute_script('return probe.rawReady || false') is False
        assert driver.execute_script('return probe.fetchAborts.sort()') == [f'old-{i}' for i in range(2, 8)]
        release.set()
        WebDriverWait(driver, 10).until(lambda page: page.execute_script('return probe.rawReady && probe.settled.includes(0)'))
        assert driver.execute_script('return probe.settled.sort()') == [0, 1]
        assert driver.execute_script('return probe.errors.length') == 6
    finally:
        release.set()
        driver.quit()
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda server: (server.shutdown(), server.server_close()), [module_server, *servers]))
