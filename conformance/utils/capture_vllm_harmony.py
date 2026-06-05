#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture vLLM's harmony tool-call streaming, per chunk. Runs in the vLLM container.

vLLM has no pluggable ToolParser for harmony — it parses harmony in a dedicated
serving path. This drives that path's core function,
`extract_harmony_streaming_delta`, directly: feed each chunk's delta_token_ids to
an openai_harmony StreamableParser (token-native, like the frontend-crate v2 Harmony parser),
then call the function to get vLLM's DeltaMessage. Mirrors the per-token loop in
vllm/entrypoints/openai/chat_completion/serving.py.

Emits {version, cases: {cid: [{deltas, normal_text}, ...]}} (openai_harmony version).
Only chunks carrying delta_token_ids are processed (harmony is token-native).
"""
import importlib.metadata as meta
import json
import sys

import yaml
from openai_harmony import HarmonyEncodingName, Role, StreamableParser, load_harmony_encoding
from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    TokenState,
    extract_harmony_streaming_delta,
)


# gpt-oss harmony special tokens (stable ids, as stamped in the fixtures):
#   <|start|> = 200006, assistant = 173781.
# The StreamableParser is created in ExpectStart mode (role=None) so it accepts a
# leading <|start|>. For channel-first inputs (no <|start|>) we prepend
# <|start|>assistant — exactly the normalization the frontend-crate v2 Harmony parser does — so
# both parsers process the identical token stream (a fair comparison).
START_TOKEN = 200006
PREAMBLE = [200006, 173781]


def main():
    fixture = sys.argv[1]
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    doc = yaml.safe_load(open(fixture))
    out = {}
    for cid, case in doc.get("cases", {}).items():
        parser = StreamableParser(enc, role=None)
        per_chunk = []
        prepended = False
        broken = False  # parser hit a terminal/unexpected token; stop feeding
        for chunk in case.get("chunks", []):
            ids = list(chunk.get("delta_token_ids", []) or [])
            # Prepend the preamble once, on the first chunk that carries tokens, if
            # the stream doesn't already start with <|start|>.
            if not prepended and ids:
                prepended = True
                if ids[0] != START_TOKEN:
                    ids = PREAMBLE + ids
            # Drive extract_harmony_streaming_delta ONE TOKEN at a time, mirroring
            # vLLM's serving loop (one engine step per token). Calling it per token
            # keeps the message-close token (<|call|>) in its own step, so a closing
            # call's trailing args aren't mis-attributed to a phantom next index
            # (which happens if args + <|call|> are grouped in one extract call).
            deltas, normal = [], ""
            for tid in ids:
                if broken:
                    break
                prev_recipient = parser.current_recipient
                try:
                    parser.process(tid)
                except Exception:
                    # Trailing/stray tokens after a message closes (e.g. the
                    # repeated <|call|> in stream.4.c) make StreamableParser raise.
                    # Mirror the local parser, which breaks and keeps what it emitted.
                    broken = True
                    break
                ts = [
                    TokenState(
                        parser.current_channel,
                        parser.current_recipient,
                        parser.last_content_delta or "",
                    )
                ]
                dm, _ = extract_harmony_streaming_delta(parser, ts, prev_recipient, False)
                if dm is None:
                    continue
                if getattr(dm, "content", None):
                    normal += dm.content
                for tc in (dm.tool_calls or []):
                    d = {"index": tc.index}
                    if tc.id is not None:
                        d["id"] = True
                    fn = tc.function
                    if fn is not None:
                        if fn.name is not None:
                            d["name"] = fn.name
                        if fn.arguments is not None:
                            d["arguments"] = fn.arguments
                    deltas.append(d)
            per_chunk.append({"deltas": deltas, "normal_text": normal})
        out[cid] = per_chunk
    # Emit the bare {cid: [{deltas, normal_text}, ...]} shape that
    # build_stream_fixtures.py consumes for --vllm (same as the sglang probe and
    # the dynamo recorder). The openai_harmony version is recorded separately via
    # the builder's --captured flag (printed to stderr here for reference).
    print(f"openai_harmony {meta.version('openai_harmony')}", file=sys.stderr)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
