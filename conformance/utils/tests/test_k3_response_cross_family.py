# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native message ownership has shared cases without inventing absent XTML forms."""
from copy import deepcopy
import hashlib
import json

import pytest

import gen_unified_golden as golden
from explode_unified_fixtures import _split_only_cases
import unified_taxonomy as taxonomy
from fixture_test_support import FRAMED_FAMILIES, assert_native_response_counterparts


PUBLISHED_K3_CONTRACTS = {
    'kimi_k3_json_marker_then_long_field': '2f185d9c57ce0c9f77832cf577e5dd90bc8e73985a94a58a83688151ed9c0ede',
    'kimi_k3_json_missing_close_literal_response': 'e0136ea0b7fd55c251973da185d4895bdada7769711760aa6ebbd26a910cfa08',
    'kimi_k3_typed_response_marker_payload': '6bb82706f5249d1bef22280e808cb7996fcc4e2e555ce214f2985203b99a8e4c',
    'kimi_k3_typed_response_then_call': '61f620cc6fcf0084a4af6a62c1a633a3af510e834a9fcafb3e570584c67f989b',
    'kimi_k3_typed_response_argument_literal': '994769825e02573a199a4aabd9d3a88ebe93a78f0d14e915e6445d2b140cdc1d',
    'kimi_k3_direct_typed_response_recovery': '141ee357780f79e41fa6cced91e20eb4730cb8b40f558f452a046ccc18ab1ca4',
    'kimi_k3_direct_json_response_recovery': '41fc5f94df97b275ebc2eb79dafd350a4c54a643b733a3c3cb259a432567cae5',
}


@pytest.fixture(scope='module')
def corpus():
    return {family: golden.build_cases(family) for family in FRAMED_FAMILIES}


def test_native_response_cases_have_independently_justified_cross_family_coverage(corpus):
    assert_native_response_counterparts(corpus, golden, taxonomy)


@pytest.mark.parametrize('scenario', PUBLISHED_K3_CONTRACTS)
def test_published_k3_requests_goldens_and_ids_remain_exact(scenario):
    cases = golden.build_cases('kimi_k3')
    key = f'UNIFIED.{scenario}.kimi_k3'
    rows = _split_only_cases('kimi_k3', {'family': 'kimi_k3', 'cases': {key: cases[key]}})
    row = next(iter(rows.values()))
    contract = {field: row[field] for field in ('request', 'golden')}
    actual = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert actual == PUBLISHED_K3_CONTRACTS[scenario]
    published_order = ('typed_response_marker_payload', 'json_missing_close_literal_response',
                       'json_marker_then_long_field', 'typed_response_then_call',
                       'typed_response_argument_literal', 'direct_typed_response_recovery',
                       'direct_json_response_recovery')
    assert taxonomy.tax(scenario) == ('kimi', str(21 + published_order.index(scenario.removeprefix('kimi_k3_'))))
    assert taxonomy.numbered_id(scenario) == 'UNIFIED.kimi-' + scenario.removeprefix('kimi_k3_')


@pytest.mark.parametrize('mutation', ['missing_family', 'unjustified_exclusion', 'payload', 'coordinated_marker', 'discarded_payload', 'policy', 'alias_target', 'alias_k3', 'folded_child'])
def test_cross_family_response_guards_reject_contract_mutations(corpus, monkeypatch, mutation):
    changed = deepcopy(corpus)
    scenario = 'kimi_k3_json_marker_then_long_field'
    family = 'beam'
    if mutation == 'missing_family':
        del changed[family][f'UNIFIED.{scenario}.{family}']
    elif mutation == 'unjustified_exclusion':
        original = golden.scenario_families
        monkeypatch.setattr(golden, 'scenario_families', lambda name: original(name) - {family} if name == scenario else original(name))
    elif mutation == 'payload':
        changed[family][f'UNIFIED.{scenario}.{family}']['golden'][0]['arguments']['q'] += ' incidental'
    elif mutation == 'coordinated_marker':
        case = changed[family][f'UNIFIED.{scenario}.{family}']
        old = case['golden'][0]['arguments']['q']
        wrong = '<|start|>assistant to=user<|message|>literal<|eom|>'
        case['input'] = case['input'].replace(old, wrong)
        case['input_chunks'] = [case['input'][at:at + 64] for at in range(0, len(case['input']), 64)]
        case['golden'][0]['arguments']['q'] = wrong
    elif mutation == 'discarded_payload':
        scenario = 'kimi_k3_typed_response_then_call'
        case = changed[family][f'UNIFIED.{scenario}.{family}']
        case['input'] = case['input'].replace('first', 'other')
        case['input_chunks'] = [case['input'][at:at + 64] for at in range(0, len(case['input']), 64)]
    elif mutation == 'policy':
        scenario = 'kimi_k3_typed_response_then_call'
        changed[family][f'UNIFIED.{scenario}.{family}']['golden'].insert(0, {'kind': 'tool_call', 'name': 'lookup', 'arguments': {'q': 'first'}})
    elif mutation in {'alias_target', 'alias_k3'}:
        original = taxonomy.coverage_case
        raw = 'kimi_k3_direct_json_response_recovery'
        monkeypatch.setattr(taxonomy, 'coverage_case', lambda name, fam=None: 'tool_only' if name == raw and (fam == family if mutation == 'alias_target' else fam == 'kimi_k3') else original(name, fam))
    elif mutation == 'folded_child':
        original = taxonomy.case_fold
        monkeypatch.setattr(taxonomy, 'case_fold', lambda name: ('kimi_k3_typed_response_marker_payload', 'Wrong parent') if name == scenario else original(name))
    with pytest.raises(AssertionError):
        assert_native_response_counterparts(changed, golden, taxonomy)
