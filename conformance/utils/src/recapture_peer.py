#!/usr/bin/env python3
"""Re-capture ONE peer impl at a PINNED version against the repaired inputs.

Drives capture_driver._container_capture directly instead of going through
capture.sh/fill_streamv2, which rewrite the whole corpus into the pre-#93 flat layout and
mark dynamo_v2 as TODO. This touches exactly one impl's overlay tree.

Usage: recapture_peer.py <container> <impl> <mode> <inputs-dir> <out.json>
   mode: stream | batch
"""
import json
import os
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture_driver as cd  # noqa: E402

container, impl, mode, inputs, out_path = sys.argv[1:6]
PARSERS = {"vllm_python": cd.VLLM, "sglang_python": cd.SGLANG}[impl]
glob = "TOOLCALLING.streamv2*.yaml" if mode == "stream" else "TOOLCALLING.batch*.yaml"

import pathlib  # noqa: E402
jobs = []
for family, parser in sorted(PARSERS.items()):
    d = pathlib.Path(inputs) / family
    if not d.exists():
        continue
    for f in sorted(d.glob(glob)):
        jobs.append({"src": str(f), "container_path": cd._cpath(str(f), mode), "parser": parser})

print(f"{impl} @ {container}: {len(jobs)} fixtures ({mode})", file=sys.stderr)
work = tempfile.mkdtemp(prefix=f"recap_{impl}_")
cd._copy_worker((container,))
# The in-container worker takes the LEGACY impl key (`vllm` / `sglang`).
legacy = {"vllm_python": "vllm", "sglang_python": "sglang"}[impl]
version, by_src = cd._container_capture(container, legacy, mode, jobs, work)
print(f"engine version reported: {version}", file=sys.stderr)

# Reshape to the same envelope capture_vllm_rust emits, so one applier handles both.
fixtures = {src: entry for src, entry in by_src.items()}
json.dump({"version": version, "fixtures": fixtures}, open(out_path, "w"), ensure_ascii=False)
ncase = sum(len(e.get("cases") or {}) for e in fixtures.values())
nerr = sum(1 for e in fixtures.values() for r in (e.get("cases") or {}).values()
           if isinstance(r, dict) and "error" in r)
print(f"captured {ncase} cases ({nerr} parser errors) -> {out_path}", file=sys.stderr)
