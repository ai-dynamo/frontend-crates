# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import gzip
import hashlib
import json

import pytest
import audit_unified_consistency as audit_owner


def write_rows(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def review_fields(classification):
    return dict(classification=classification, resolved=classification != 'insufficient_evidence', reason='Independent exact request review', source_evidence={'source':'published producer source not retained'}, reproducer={'request':'selected_request','procedure':'rebuild exact retained producer and adapter'}, required_correction='Retain measured failure and qualify original producer', missing_verification=['source and adapter identity not independently rebuilt'] if classification == 'insufficient_evidence' else [])


def candidate(tmp_path):
    request={'input':'original','tools':[], 'chunks':['original'], 'init':{'starting_state':'None'}, 'finish_reason':'stop'}
    golden={'assembled':[{'kind':'text','text':'expected'}]}
    output={'assembled':[{'kind':'text','text':'actual'}]}
    original={'inline':request}
    raw={'value':output}
    document={'captured_with':{'peer':'1'}}
    meta=dict(family='beam',case_id='child',status='differs_from_golden',producer_implementation='peer',producer_version='1',parser_path='native',mode='stream',producer_origin={'source_sha256':'unverified'},selected_request_sha256=audit_owner.digest(request),selected_observation_sha256=audit_owner.digest(output),stimulus_sha256=audit_owner.digest(original),original_observation_sha256=audit_owner.digest(raw),document_sha256=audit_owner.digest(document))
    eid=audit_owner.observation_evidence_id(meta)
    identity=dict(family='beam',source_family='beam',scenario='child',case_id='child',implementation='peer')
    history=[dict(family='beam',scenario='child',implementation='peer',capture='peer-1',version='1',mode='stream',status='inventoried',observations=[meta,dict(family='beam',case_id='parent',status='missing_execution')]),dict(family='gemma',scenario='child',implementation='peer',capture='peer-1',version='1',mode='stream',status='pending_applicability')]
    exclusion=dict(classification='genuine_grammar_limitation',resolved=True,reason='Independent grammar proof')
    authored=[dict(family='beam',scenario='child',status='authored',requests=[dict(family='beam',case_id=name,request_sha256=audit_owner.digest(request),golden_sha256=audit_owner.digest(golden)) for name in ('child','parent')]),dict(family='gemma',scenario='child',status='pending_applicability',requests=[],disposition=exclusion)]
    grid=[dict(identity={k:r[k] for k in ('family','scenario','implementation','capture','version','mode')},inventory_row_sha256=audit_owner.digest(r),classification='observation_dispositions',records=[dict(family='beam',case_id='child',evidence_id=eid),dict(family='beam',case_id='parent',execution_claimed=False,**review_fields('missing_execution'))]) for r in history[:1]]
    grid.append(dict(identity={k:history[1][k] for k in ('family','scenario','implementation','capture','version','mode')},inventory_row_sha256=audit_owner.digest(history[1]),classification=exclusion['classification'],authored_disposition_sha256=audit_owner.digest(exclusion),execution_claimed=False))
    selected=[dict(evidence_id=eid,identity=identity,observation_metadata=meta,selected_request=request,active_request=request,golden=golden,selected_observation=output,original_stimulus=original,original_observation=raw,source_document=document)]
    observations=[dict(evidence_id=eid,identity=identity,producer_identity={k:meta[k] for k in ('producer_implementation','producer_version','parser_path','mode','producer_origin')},selected_status=meta['status'],captured_request_sha256=audit_owner.digest(request),active_request_sha256=audit_owner.digest(request),golden_sha256=audit_owner.digest(golden),selected_observation_sha256=audit_owner.digest(output),**review_fields('insufficient_evidence'))]
    retained=[dict(surface='history',family='beam',source_family='beam',case_id='old',implementation='peer',capture='peer-1',selection_version='1',mode='stream',scenario='old',lifecycle='retired',request_retained=False,source_member='original.yaml',stimulus_sha256='request-not-retained',observation_sha256='original-output',document_sha256='original-document',producer_origin={})]
    retired=[dict(inventory_row_sha256=audit_owner.digest(retained[0]),identity={k:retained[0][k] for k in ('surface','family','source_family','case_id','implementation','capture','selection_version','mode','scenario','lifecycle')},**review_fields('historical_request_not_retained'))]
    retired[0]['source_evidence']={k:retained[0][k] for k in ('source_member','stimulus_sha256','observation_sha256','document_sha256','producer_origin')}
    audit=dict(complete=False,reviewed_candidate={},history={'path':'history.gz','combinations':2},retained_evidence={'path':'retained.gz'},family_scenario_inventory=authored,totals={'pending_applicability':0,'pending_differences':0})
    return deepcopy(audit),deepcopy(dict(grid=grid,selected_evidence=selected,observations=observations,retired=retired)),deepcopy(history),deepcopy(retained)


def publish(tmp_path,sidecars,history,retained):
    manifest={'sidecars':{}}
    write_rows(tmp_path/'history.gz',history);write_rows(tmp_path/'retained.gz',retained)
    for name,rows in sidecars.items():
        path=tmp_path/(name+'.gz');write_rows(path,rows)
        manifest['sidecars'][name]={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest));return path


def test_coverage_completion_retains_unresolved_failure_and_missing_execution(tmp_path):
    audit,sidecars,history,retained=candidate(tmp_path)
    audit_owner.apply_producer_review(audit,publish(tmp_path,sidecars,history,retained),tmp_path)
    assert audit['complete'] is True
    assert audit['producer_review']['unresolved_classifications']=={'insufficient_evidence':1}
    assert audit['producer_review']['missing_executions']==1
    assert not audit['producer_review']['source_and_semantic_verification_complete']
    assert not audit['producer_review']['all_observations_match_golden']


@pytest.mark.parametrize('mutation', ['omit_family','omit_supported_family','duplicate_grid','foreign_grid','stale_binding','omit_child','unjustified_exclusion','contradict_inline_disposition','mutate_actual','mutate_metadata','foreign_selected_source_member','resolved_class_with_explicit_proof_gap','mutate_original','mutate_active_request','mutate_golden','omit_selected','duplicate_selected','foreign_selected','omit_disposition','duplicate_disposition','foreign_disposition','omit_proof_gap','resolved_proof_gap','omit_reproducer','changed_classification','omit_retired','mutate_retired','mutate_retired_identity','foreign_retired'])
def test_independent_producer_review_rejects_bound_mutations(tmp_path,mutation):
    audit,rows,history,retained=candidate(tmp_path)
    if mutation=='omit_family':rows['grid'].pop()
    elif mutation=='omit_supported_family':rows['grid'].pop(0)
    elif mutation=='duplicate_grid':rows['grid'].append(deepcopy(rows['grid'][0]))
    elif mutation=='foreign_grid':rows['grid'][0]['identity']['family']='foreign'
    elif mutation=='stale_binding':rows['grid'][0]['inventory_row_sha256']='old'
    elif mutation=='omit_child':rows['grid'][0]['records'].pop()
    elif mutation=='unjustified_exclusion':rows['grid'][0]['classification']='genuine_grammar_limitation'
    elif mutation=='contradict_inline_disposition':rows['grid'][0]['records'][0]['classification']='matches_authored_typed_events'
    elif mutation=='mutate_actual':rows['selected_evidence'][0]['selected_observation']['assembled'][0]['text']='changed'
    elif mutation=='mutate_metadata':rows['selected_evidence'][0]['observation_metadata']['status']='matches_golden'
    elif mutation=='foreign_selected_source_member':rows['selected_evidence'][0]['observation_metadata']['source_member']='foreign/nonexistent.yaml'
    elif mutation=='resolved_class_with_explicit_proof_gap':rows['observations'][0].update(classification='parser_defect',resolved=True)
    elif mutation=='mutate_original':rows['selected_evidence'][0]['source_document']['added']='changed'
    elif mutation=='mutate_active_request':rows['selected_evidence'][0]['active_request']={'input':'changed'}
    elif mutation=='mutate_golden':rows['selected_evidence'][0]['golden']={'assembled':[]}
    elif mutation=='omit_selected':rows['selected_evidence'].clear()
    elif mutation=='duplicate_selected':rows['selected_evidence']*=2
    elif mutation=='foreign_selected':rows['selected_evidence'][0]['evidence_id']='foreign'
    elif mutation=='omit_disposition':rows['observations'].clear()
    elif mutation=='duplicate_disposition':rows['observations']*=2
    elif mutation=='foreign_disposition':rows['observations'][0]['evidence_id']='foreign'
    elif mutation=='omit_proof_gap':rows['observations'][0]['missing_verification']=[]
    elif mutation=='resolved_proof_gap':rows['observations'][0]['resolved']=True
    elif mutation=='omit_reproducer':rows['observations'][0].pop('reproducer')
    elif mutation=='changed_classification':rows['observations'][0]['classification']='genuine_grammar_limitation'
    elif mutation=='omit_retired':rows['retired'].clear()
    elif mutation=='mutate_retired':rows['retired'][0]['inventory_row_sha256']='old'
    elif mutation=='mutate_retired_identity':rows['retired'][0]['identity']['case_id']='foreign'
    elif mutation=='foreign_retired':rows['retired']*=2
    with pytest.raises(ValueError):
        audit_owner.apply_producer_review(audit,publish(tmp_path,rows,history,retained),tmp_path)


@pytest.mark.parametrize("classification", ["adapter_limitation", "genuine_grammar_limitation"])
def test_pre_parser_adapter_failure_keeps_measured_error_without_inventing_parser_defect(tmp_path, classification):
    audit, rows, history, retained = candidate(tmp_path)
    producer_started = False
    marker_profile = None

    def producer():
        nonlocal producer_started
        producer_started = True

    def source_qualified_adapter():
        # This preflight branch fails before the producer is constructed or called.
        if marker_profile is None:
            raise ValueError("missing reasoning marker profile: gemma4")
        producer()

    with pytest.raises(ValueError, match="missing reasoning marker profile") as failure:
        source_qualified_adapter()
    assert producer_started is False
    outcome = {"error": str(failure.value)}
    selected = rows["selected_evidence"][0]
    selected["selected_observation"] = outcome
    selected["original_observation"] = {"value": outcome}
    selected["source_document"]["adapter_phase"] = "preflight before producer construction"
    metadata = history[0]["observations"][0]
    metadata.update(status="parser_error", selected_observation_sha256=audit_owner.digest(outcome),
                    original_observation_sha256=audit_owner.digest(selected["original_observation"]),
                    document_sha256=audit_owner.digest(selected["source_document"]))
    selected["observation_metadata"] = deepcopy(metadata)
    evidence_id = audit_owner.observation_evidence_id(metadata)
    selected["evidence_id"] = evidence_id
    rows["grid"][0]["inventory_row_sha256"] = audit_owner.digest(history[0])
    rows["grid"][0]["records"][0]["evidence_id"] = evidence_id
    disposition = rows["observations"][0]
    disposition.update(evidence_id=evidence_id, selected_status="parser_error", classification=classification,
                       resolved=True, missing_verification=[], selected_observation_sha256=audit_owner.digest(outcome),
                       reason="The independently controlled adapter preflight raises before producer construction.",
                       source_evidence={"control": "source_qualified_adapter", "producer_started": producer_started},
                       required_correction="Keep the original adapter error; supplement through a corrected public adapter. Do not assign a parser defect or grammar exclusion.")
    manifest = publish(tmp_path, rows, history, retained)
    if classification == "genuine_grammar_limitation":
        with pytest.raises(ValueError, match="contradicts retained observation status"):
            audit_owner.apply_producer_review(audit, manifest, tmp_path)
        return
    audit_owner.apply_producer_review(audit, manifest, tmp_path)
    assert audit["producer_review"]["observation_classifications"] == {"adapter_limitation": 1}
    assert disposition["selected_status"] == "parser_error"
    assert audit["producer_review"]["all_observations_match_golden"] is False


@pytest.mark.parametrize("use_matching_member", [True, False])
def test_inherited_source_member_binds_to_its_actual_metadata_and_original_occurrence(tmp_path, use_matching_member):
    audit, rows, history, retained = candidate(tmp_path)
    first = history[0]
    first["observations"][0]["source_member"] = "retained/first.yaml"
    rows["grid"][0]["inventory_row_sha256"] = audit_owner.digest(first)
    second = deepcopy(first)
    second["capture"] = "peer-2"
    second["observations"][0]["source_member"] = "retained/second.yaml"
    document = deepcopy(rows["selected_evidence"][0]["source_document"])
    document["inherited_from"] = "peer-1"
    second["observations"][0]["document_sha256"] = audit_owner.digest(document)
    history.append(second)
    projection = deepcopy(rows["grid"][0])
    projection["identity"]["capture"] = "peer-2"
    projection["inventory_row_sha256"] = audit_owner.digest(second)
    rows["grid"].append(projection)
    audit["history"]["combinations"] += 1
    selected = rows["selected_evidence"][0]
    selected["source_document"] = document
    selected["observation_metadata"] = deepcopy(second["observations"][0])
    if not use_matching_member:
        # Both paths exist, but the first path belongs to another original document.
        selected["observation_metadata"]["source_member"] = "retained/first.yaml"
    manifest = publish(tmp_path, rows, history, retained)
    if not use_matching_member:
        with pytest.raises(ValueError, match="original retained evidence differs"):
            audit_owner.apply_producer_review(audit, manifest, tmp_path)
        return
    audit_owner.apply_producer_review(audit, manifest, tmp_path)
    assert audit["producer_review"]["unique_observations"] == 1
    assert audit["producer_review"]["grid_rows"] == 3
    assert audit["producer_review"]["unresolved_classifications"] == {"insufficient_evidence": 1}


def test_parser_defect_with_explicit_verification_gaps_remains_in_unresolved_summary(tmp_path):
    audit, rows, history, retained = candidate(tmp_path)
    rows["observations"][0].update(classification="parser_defect", resolved=False)
    audit_owner.apply_producer_review(audit, publish(tmp_path, rows, history, retained), tmp_path)
    assert audit["producer_review"]["unresolved_classifications"] == {"parser_defect": 1}
