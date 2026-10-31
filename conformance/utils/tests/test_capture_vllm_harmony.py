# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify the prompt-consumed role without rewriting native requests."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import peer_modules
from capture_stimulus import _without_synthetic_finish_step
from capture_unified_batch import capture_cases
from capture_unified_job import build_job


@pytest.fixture
def original():
    root = Path(__file__).resolve().parents[3]
    job = build_job(root / 'conformance/fixtures-unified-v2/families', 'gpt_oss', mode='batch')
    return next(case for case in job['cases'] if case['scenario'] == 'gpt_oss_implicit_recipient_first_header')


@pytest.fixture
def public_api(tmp_path, monkeypatch, original):
    calls = []
    request = original['request']
    package = tmp_path / 'vllm'
    (package / 'parser').mkdir(parents=True)
    (package / 'parser/parser_manager.py').write_bytes(b'# public factory')

    class Encoding:
        special_tokens_set = set()

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

        def decode(self, tokens):
            return ''.join(chr(token) for token in tokens)

    encoding = Encoding()

    class ResponsesRequest:
        def __init__(self, *, model, input, tools):
            assert input == [] and tools == []
            self.tools = tools

    class Parser:
        def __init__(self, tokenizer, tools):
            assert tools == []
            self.tokenizer = tokenizer

        def parse(self, text, schema, *, enable_auto_tools, model_output_token_ids):
            assert text == request['input']
            assert self.tokenizer.decode(model_output_token_ids) == text
            assert schema.tools == [] and enable_auto_tools
            calls.append(('batch', text))
            return None, None, [SimpleNamespace(name='lookup', arguments='{"q":"x"}')]

        def parse_delta(self, text, tokens, schema, *, finished):
            assert schema.tools == []
            assert self.tokenizer.decode(tokens) == text
            calls.append(('stream', text, finished))
            return [{'kind': 'tool_call', 'name': 'lookup', 'arguments': '{"q":"x"}'}] if finished else []

    class Manager:
        @staticmethod
        def get_parser(*, tool_parser_name, reasoning_parser_name, enable_auto_tools, is_harmony):
            assert (tool_parser_name, reasoning_parser_name, enable_auto_tools, is_harmony) == ('openai', 'openai_gptoss', True, True)
            return Parser

    modules = {'vllm': SimpleNamespace(__file__=str(package / '__init__.py')),
               'vllm.entrypoints.openai.parser.harmony_utils': SimpleNamespace(get_encoding=lambda: encoding),
               'vllm.entrypoints.openai.responses.protocol': SimpleNamespace(ResponsesRequest=ResponsesRequest),
               'openai_harmony': SimpleNamespace(HarmonyError=ValueError)}
    monkeypatch.setattr(peer_modules.importlib, 'import_module', modules.__getitem__)
    monkeypatch.setattr(peer_modules, 'resolve_peer_module', lambda *args: SimpleNamespace(ParserManager=Manager))
    return calls, encoding, modules


def test_public_harmony_batch_preserves_original_empty_tools_and_request(original, public_api):
    before = deepcopy(original)
    result = capture_cases({'cases': [original]}, {}, lambda names: False, lambda *args: pytest.fail('generic parser used'),
                           case_override=lambda case: peer_modules.capture_vllm_harmony(case['family'], case['request']))
    observation = result[original['id']]
    assert observation['assembled'] == original['golden']['assembled']
    assert observation['capture_input'] == original['request']
    assert original == before


def test_public_harmony_stream_preserves_every_original_chunk(original, public_api):
    calls, _encoding, _modules = public_api
    before = deepcopy(original)
    result = peer_modules.capture_vllm_harmony('gpt_oss', original['request'], streaming=True, delta_events=lambda value: value)
    expected = [chunk['delta_text'] for chunk in _without_synthetic_finish_step(original['request'])['chunks']]
    assert calls == [('stream', text, False) for text in expected] + [('stream', '', True)]
    assert len(result['chunks']) == len(original['request']['chunks'])
    assert result['capture_input'] == original['request']
    assert original == before


@pytest.mark.parametrize('mutation', ['full_header', 'response', 'offered_tools', 'guided', 'named'])
def test_harmony_scope_rejects_unqualified_contract_without_rewriting(original, public_api, mutation):
    request = deepcopy(original['request'])
    if mutation == 'full_header':
        request['input'] = '<|start|>assistant' + request['input']
    elif mutation == 'response':
        request['init']['starting_state'] = 'Response'
    elif mutation == 'offered_tools':
        request['tools'] = [{'name': 'lookup', 'parameters': {'type': 'object'}}]
    elif mutation == 'guided':
        request['init']['tool_output_mode'] = 'GuidedJson'
    else:
        request['init']['named_tool'] = 'lookup'
    before = deepcopy(request)
    result = peer_modules.capture_vllm_harmony('gpt_oss', request, streaming=True, delta_events=lambda value: value)
    assert 'unavailable' in result and 'error' not in result
    assert request == before and public_api[0] == []


def test_harmony_stream_does_not_rechunk_inside_native_token(original, public_api, monkeypatch):
    calls, encoding, _modules = public_api
    text = original['request']['input']
    monkeypatch.setattr(encoding, 'encode', lambda value, **kwargs: [1])
    monkeypatch.setattr(encoding, 'decode', lambda tokens: text if tokens else '')
    result = peer_modules.capture_vllm_harmony('gpt_oss', original['request'], streaming=True, delta_events=lambda value: value)
    assert 'unavailable' in result and not calls
    assert result['capture_input'] == original['request']


def test_harmony_encoding_dependency_failure_is_not_a_parser_error(original, public_api):
    def absent():
        raise FileNotFoundError('official-vocab-artifact')
    public_api[2]['vllm.entrypoints.openai.parser.harmony_utils'].get_encoding = absent
    with pytest.raises(FileNotFoundError, match='official-vocab-artifact'):
        peer_modules.capture_vllm_harmony('gpt_oss', original['request'])
