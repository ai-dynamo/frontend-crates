# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import re
from functools import cache
from pathlib import Path

from impls import vllm_tokenizer_markers
from peer_modules import AdapterContractError

# The checkpoint template is retained in the vLLM v0.34.0 Beam parser test.
BEAM_TEMPLATE = Path(__file__).with_name("beam_response_template.json")


class StubTokenizer:
    """Deterministic marker tokenizer for parser-only capture.

    The parser APIs inspect token IDs when deciding whether a prompt ended inside
    reasoning. An empty vocabulary makes that state impossible to exercise, so
    this tokenizer exposes the exact marker strings used by each parser family
    with stable synthetic IDs. It still returns no IDs for ordinary model text.
    """
    all_special_tokens = []
    is_fast = True

    def __init__(self, family):
        self.source_text = None
        self.marker_at_start = False
        markers = vllm_tokenizer_markers(family)
        self._vocab = {marker: 256 + index for index, marker in enumerate(markers)}
        if family == "beam":
            self.response_template = json.loads(BEAM_TEMPLATE.read_bytes())
        self._encoded = {
            marker: [token_id] for marker, token_id in self._vocab.items()
        }
        if family == "kimi_k3":
            self._vocab = {token: 256 + index for index, token in enumerate(
                ("<|open|>", "<|close|>", "<|sep|>", "think", "response", "message"))}
            self._encoded.update({token: [token_id] for token, token_id in self._vocab.items()})
            self._encoded["<|open|>think<|sep|>"] = [256, 259, 258]
            self._encoded["<|close|>think<|sep|>"] = [257, 259, 258]
            self._encoded["<|open|>response<|sep|>"] = [256, 260, 258]
            self._encoded["<|close|>response<|sep|>"] = [257, 260, 258]
            self._encoded["<|close|>message<|sep|>"] = [257, 261, 258]
        self._patterns = []
        for marker, ids in self._encoded.items():
            pattern = re.escape(marker)
            if family == "kimi_k3":
                # vLLM's XTML decoder may insert spaces between marker tokens.
                pattern = pattern.replace("think", r"\s*think\s*")
                pattern = pattern.replace("response", r"\s*response\s*")
                pattern = pattern.replace("message", r"\s*message\s*")
            self._patterns.append((re.compile(pattern), ids))

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, t):
        return self._vocab.get(t)

    def get_added_vocab(self):
        return self._vocab

    def encode(self, t, add_special_tokens=False):
        return [token for _start, _end, tokens in self.token_spans(t) for token in tokens]

    def token_spans(self, text):
        matches = sorted((match.start(), -len(match.group()), match.end(), ids)
                         for pattern, ids in self._patterns for match in pattern.finditer(text))
        spans, cursor = [], 0
        for start, _length, end, ids in matches:
            if start < cursor:
                continue
            spans.extend((index, index + 1, [1000 + ord(text[index])]) for index in range(cursor, start))
            spans.append((start, end, ids))
            cursor = end
        spans.extend((index, index + 1, [1000 + ord(text[index])]) for index in range(cursor, len(text)))
        return spans

    def bind_input(self, text, *, marker_at_start):
        self.source_text, self.marker_at_start = text, marker_at_start

    def generated_ids(self, previous_text, current_text):
        """Expose completed control markers at their authored chunk positions.

        DelegatingParser uses IDs to leave reasoning even when its detector
        extracts text correctly. A split marker must not transition early.
        Ordinary text gets an opaque ID: its vocabulary is irrelevant here.
        """
        if self.marker_at_start:
            if self.source_text is None or not self.source_text.startswith(current_text):
                raise AdapterContractError("Token engine input differs from the bound authored text")
            # A token can precede its fully decoded spelling; the published
            # scanner explicitly defers it until the corresponding text arrives.
            return [token for start, _end, ids in self.token_spans(self.source_text)
                    if len(previous_text) <= start < len(current_text) for token in ids]
        return [token for _start, end, ids in self.token_spans(current_text)
                if end > len(previous_text) for token in ids]

    def decode(self, ids, **k):
        markers = {token_id: marker for marker, token_id in self._vocab.items()}
        return "".join(chr(token_id - 1000) if token_id >= 1000 else markers[token_id] for token_id in ids)

    @property
    def vocab_size(self):
        return 0


@cache
def _harmony_encoding():
    harmony = importlib.import_module("openai_harmony")
    return harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)


class HarmonyCaptureTokenizer(StubTokenizer):
    def __init__(self):
        super().__init__("gpt_oss")
        self.encoding = _harmony_encoding()

    def encode(self, text, add_special_tokens=False):
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, ids, **kwargs):
        return self.encoding.decode(ids)

    def bind_input(self, text, *, marker_at_start):
        super().bind_input(text, marker_at_start=marker_at_start)
        self.token_ends = []
        end = 0
        for token in self.encode(text):
            spelling = self.encoding.decode([token], errors="surrogateescape")
            end += len(spelling.encode("utf-8", errors="surrogateescape"))
            self.token_ends.append((end, token))
        if end != len(text.encode()):
            raise AdapterContractError("Harmony tokens do not span the authored input")

    def generated_ids(self, previous_text, current_text):
        start, end = len(previous_text.encode()), len(current_text.encode())
        return [token for offset, token in self.token_ends if start < offset <= end]


def capture_tokenizer(family):
    return HarmonyCaptureTokenizer() if family == "gpt_oss" else StubTokenizer(family)
