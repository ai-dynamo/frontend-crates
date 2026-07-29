#!/usr/bin/env python3
"""Re-align stream-v2 inputs to their batch source after batch text was normalized.

A `streamv2` case is the chunked form of a `batch` case (`ref: derived from ...`), so the
joined `delta_text` must equal the batch `model_text`. Normalizing the batch payload text
without touching the stream corpus broke that for 21 cases.

Chunk boundaries are preserved wherever the text is unchanged: the old joined text and
the new target share a common prefix and suffix, and only the differing middle moves. The
length delta is absorbed by the single chunk that contains the start of the difference, so
every other boundary — and therefore every other parser flush point — stays put.

Safe because the whole corpus is re-captured afterwards; the recorded per-chunk
expectations for these cases are regenerated, not reused.
"""
import json
import pathlib
import sys

import yaml

STREAM = pathlib.Path(sys.argv[1])
BATCH = pathlib.Path(sys.argv[2])
APPLY = "--apply" in sys.argv

batch = {}
for f in BATCH.glob("*/TOOLCALLING.batch*.yaml"):
    doc = yaml.safe_load(f.read_text()) or {}
    for cid, case in (doc.get("cases") or {}).items():
        if isinstance(case, dict) and case.get("model_text") is not None:
            batch[(f.parent.name, cid)] = case["model_text"]


def resplit(deltas, target):
    """Re-split `target` across `deltas`' boundaries, preserving the chunk COUNT.

    Chunks outside the changed region keep their exact text. Chunks that overlap it share
    the new middle in proportion to how much of the old middle each covered, so a payload
    that straddled a boundary still straddles one.
    """
    old = "".join(deltas)
    if old == target:
        return None
    p = 0
    while p < len(old) and p < len(target) and old[p] == target[p]:
        p += 1
    s_ = 0
    while (s_ < len(old) - p and s_ < len(target) - p
           and old[len(old) - 1 - s_] == target[len(target) - 1 - s_]):
        s_ += 1
    dstart, dend = p, len(old) - s_          # changed region in OLD coordinates
    middle = target[dstart:len(target) - s_]  # its replacement
    delta = len(target) - len(old)

    spans = []
    pos = 0
    for i, d in enumerate(deltas):
        a, b = pos, pos + len(d)
        pos = b
        if b > dstart and a < dend:
            spans.append((i, max(a, dstart), min(b, dend)))
    total_old = sum(e - s0 for _, s0, e in spans) or 1
    shares, used = {}, 0
    for n, (i, s0, e) in enumerate(spans):
        if n == len(spans) - 1:
            shares[i] = middle[used:]
        else:
            take = round(len(middle) * (e - s0) / total_old)
            shares[i] = middle[used:used + take]
            used += take

    out, pos = [], 0
    for i, d in enumerate(deltas):
        a, b = pos, pos + len(d)
        pos = b
        if b <= dstart:
            out.append(target[a:b])
        elif a >= dend:
            out.append(target[a + delta:b + delta])
        else:
            head = target[a:dstart] if a < dstart else ""
            tail = target[dend + delta:b + delta] if b > dend else ""
            out.append(head + shares[i] + tail)
    return out if "".join(out) == target else False


changed = files = 0
report = []
for f in sorted(STREAM.glob("*/TOOLCALLING.streamv2*.yaml")):
    fam = f.parent.name
    doc = yaml.safe_load(f.read_text()) or {}
    touched = False
    for cid, case in (doc.get("cases") or {}).items():
        chunks = case.get("chunks") or []
        deltas = [(ch.get("delta_text") or "") for ch in chunks]
        bid = (case.get("ref") or "").replace("derived from ", "").strip("` ")
        # `harmony_text` has no batch corpus of its own; its refs point at the
        # `harmony` batch cases (same envelope, text-mode transport).
        target = batch.get((fam, bid)) or batch.get((fam.replace("_text", ""), bid))
        if target is None:
            continue
        new = resplit(deltas, target)
        if new is None:
            continue
        if new is False:
            report.append(f"  SKIP {fam}/{cid}: could not re-split exactly")
            continue
        for ch, val in zip(chunks, new):
            ch["delta_text"] = val
        touched = True
        changed += 1
        report.append(f"  {fam}/{cid}")
    if not touched:
        continue
    # Re-dump the whole document. These files live inside git-lfs tarballs, so they are
    # not line-reviewed, and safe_dump quotes every scalar correctly — which also clears
    # the single-quote folding hazard for the rest of the file as a side effect.
    out = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096)
    check = yaml.safe_load(out)
    for cid, case in (check.get("cases") or {}).items():
        for ch in (case.get("chunks") or []):
            assert isinstance(ch.get("delta_text", ""), str), f"{f}:{cid} non-string delta_text"
    assert check == doc, f"{f}: round-trip changed the document"
    files += 1
    if APPLY:
        f.write_text(out)

print("\n".join(report))
print(f"{'re-aligned' if APPLY else 'would re-align'}: {changed} cases across {files} files")
