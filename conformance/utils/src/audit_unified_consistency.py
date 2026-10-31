#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inventory authored contracts and resolved observations without inferring applicability.

An authored scope is evidence of coverage, never proof of a grammar exclusion.
Unreviewed differences and exclusions remain pending in the published audit.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import gen_unified_golden as generator
from capture_stimulus import select_peer_observation, unified_record_contract
from impls import UNIFIED_BATCH_IMPLS
from numeric_cases import canonical_events
import unified_history
import unified_taxonomy as taxonomy
from viewer_codec import canonical_typed, typed_equal


def digest(value):
    return hashlib.sha256(canonical_typed(value).encode()).hexdigest()


def report_family(family):
    # These selectors route to the same Qwen contract. Nemotron retains its
    # distinct default reasoning state despite sharing the Qwen scanner.
    return "qwen3" if family == "qwen3_coder" else family


def authored_inventory(families_root):
    families = defaultdict(dict)
    for path in sorted(families_root.glob("*/inputs_and_golden.yaml")):
        family = path.parent.name
        if family not in generator.UNIFIED_FAMILIES:
            continue
        document = unified_history.load_yaml(path)
        for case_id, case in document["cases"].items():
            if case["lifecycle"] != "active":
                continue
            families[report_family(family)].setdefault(case["scenario"], []).append({
                "family": family, "case_id": case_id, "source": str(path), "case": case,
            })
    expected = {report_family(family) for family in generator.UNIFIED_FAMILIES}
    if families.keys() != expected:
        raise ValueError("registered native family is absent from the authored inventory")
    scenarios = sorted({scenario for cases in families.values() for scenario in cases})
    rows = []
    for family, cases in sorted(families.items()):
        for scenario in scenarios:
            entries = cases.get(scenario, [])
            rows.append({
                "family": family, "scenario": scenario,
                "status": "authored" if entries else "pending_applicability",
                "requests": [{"family": entry["family"], "case_id": entry["case_id"],
                              "source": entry["source"], "request_sha256": digest(entry["case"]["request"]),
                              "golden_sha256": digest(entry["case"]["golden"])} for entry in entries],
            })
    return families, scenarios, rows


def differing_contracts(families, scenarios):
    groups = []
    for scenario in scenarios:
        contracts = defaultdict(lambda: defaultdict(set))
        for family, cases in families.items():
            for entry in cases.get(scenario, []):
                case = entry["case"]
                contract = {key: case["request"][key] for key in ("init", "tools", "finish_reason")}
                contracts[canonical_typed(contract)][family].add(canonical_typed(canonical_events(case["golden"]["assembled"])))
        for contract, family_outcomes in contracts.items():
            outcomes = defaultdict(set)
            conflicts = {}
            for family, values in family_outcomes.items():
                if len(values) != 1:
                    conflicts[family] = [json.loads(value) for value in sorted(values)]
                    continue
                outcomes[next(iter(values))].add(family)
            if len(outcomes) > 1 or conflicts:
                groups.append({"identity": f"scenario:{scenario}:{digest(json.loads(contract))}", "status": "pending_review",
                               "typed_request_contract": json.loads(contract), "conflicting_aliases": conflicts,
                               "outcomes": [{"typed_golden": json.loads(value), "families": sorted(voters),
                                             "family_votes": len(voters)} for value, voters in sorted(outcomes.items())]})
        fold = taxonomy.case_fold(scenario)
        if fold is None:
            continue
        for family, cases in families.items():
            for child in cases.get(scenario, []):
                for parent in cases.get(fold[0], []):
                    if child["family"] != parent["family"]:
                        continue
                    if not typed_equal(canonical_events(child["case"]["golden"]["assembled"]),
                                       canonical_events(parent["case"]["golden"]["assembled"])):
                        groups.append({"identity": f"fold:{child['family']}:{scenario}:{fold[0]}",
                                       "status": "pending_review", "parent": fold[0],
                                       "child_golden_sha256": digest(child["case"]["golden"]),
                                       "parent_golden_sha256": digest(parent["case"]["golden"])})
    return groups


def effective_parser_path(metadata, document):
    return metadata.get("parser_path", metadata.get("parser", document.get("parser_path")))


def observation_row(case, record, *, implementation, version):
    if record is None:
        return {"status": "missing_execution"}
    materialized, document = unified_history.materialized_record(case, record)
    materialized.update(document.get("record_metadata", {}))
    primary_version = document.get("captured_with", {}).get(implementation, document.get("capture_origin", {}).get("crate_version"))
    recapture = materialized.get("verified_recapture")
    measured_version = recapture["provenance"]["version"] if recapture is not None else primary_version
    if recapture is not None and measured_version not in {version, primary_version}:
        raise ValueError("recapture producer differs from both checkpoint and inherited observation")
    selected = select_peer_observation(materialized, case["request"], implementation=implementation,
                                       version=measured_version or version)
    observation = record["observation"]
    origin = selected.get("producer_origin", document.get("capture_origin", {}))
    producer_version = origin.get("version", origin.get("crate_version", measured_version))
    producer_document = ({**document, "capture_origin": origin, "record_metadata": selected}
                         if recapture is not None else document)
    selected_observation = {key: selected[key] for key in (
        "assembled", "error", "unavailable", "unavailable_code", "chunks", "partial_chunks", "batch_final",
    ) if key in selected}
    if "capture_input" not in selected:
        status = "request_binding_unavailable"
    elif not typed_equal(selected["capture_input"], case["request"]):
        status = "request_mismatch"
    elif "assembled" in selected:
        status = ("matches_golden" if typed_equal(canonical_events(selected["assembled"]), canonical_events(case["golden"]["assembled"]))
                  else "differs_from_golden")
    elif "error" in selected:
        status = "parser_error"
    else:
        status = "pending_unavailability_review"
    return {"status": status, "version": version, "selection_version": version,
            "selection_mode": "batch" if implementation in UNIFIED_BATCH_IMPLS else "stream",
            "mode": unified_record_contract(implementation, producer_document),
            "producer_version": producer_version, "primary_producer_version": primary_version,
            "producer_implementation": origin.get("implementation", implementation),
            "producer_origin": origin, "inherited_from": document.get("inherited_from"),
            "parser_path": effective_parser_path(selected, document),
            "recapture_resolution": recapture["resolution"] if recapture is not None else None,
            "stimulus_sha256": digest(record["stimulus"]), "observation_sha256": digest(observation),
            "original_observation_sha256": digest(observation),
            "selected_observation_sha256": digest(selected_observation),
            "selected_request_sha256": digest(selected["capture_input"]) if "capture_input" in selected else None,
            "document_sha256": digest(document), "verified_recapture": selected is not materialized,
             "selected_outcome": {key: selected[key] for key in ("error", "unavailable", "unavailable_code") if key in selected},
            **({"unavailable": observation["unavailable"]} if "unavailable" in observation else {})}


def history_inventory(store, families, scenarios, destination, *, current_store=None):
    counts = Counter()
    observation_counts = Counter()
    evidence_ids = set()
    executed_ids = set()
    stores = [store, current_store] if current_store is not None else [store]
    selections = sorted({(implementation, identity)
                         for source in stores
                         for (family, implementation), history in source.histories.items()
                         if report_family(family) in families
                         for identity in history.captures})
    # Stream complete checkpoints through the existing resolver. Original member,
    # request, recapture and execution metadata remain owned by the history reader.
    with gzip.open(destination, "wt", encoding="utf-8") as output:
        for family, cases in sorted(families.items()):
            aliases = sorted({entry["family"] for entries in cases.values() for entry in entries})
            for implementation, identity in selections:
                snapshots = {}
                members = {}
                histories = {}
                for alias in aliases:
                    for source in stores:
                        history = source.histories.get((alias, implementation))
                        if history is not None and identity in history.captures:
                            snapshots[alias] = next(history.iter_resolved_snapshots({identity}, include_members=True))[1]
                            members[alias] = history.snapshot_members.get(identity, {})
                            histories[alias] = history
                for scenario in scenarios:
                    entries = cases.get(scenario, [])
                    if not entries:
                        row = {"status": "pending_applicability"}
                    else:
                        observations = []
                        for entry in entries:
                            snapshot = snapshots.get(entry["family"], {})
                            observations.append({"family": entry["family"], "case_id": entry["case_id"],
                                                 "source_member": members[entry["family"]].get(entry["case_id"])
                                                 if entry["family"] in histories else None,
                                                 **observation_row(entry["case"], snapshot.get(entry["case_id"]),
                                                                   implementation=implementation,
                                                                   version=histories[entry["family"]].captures[identity]["runtime_version"]
                                                                   if entry["family"] in snapshots else identity)})
                        observation_counts.update(row["status"] for row in observations)
                        for item in observations:
                            if item["status"] == "missing_execution":
                                continue
                            evidence_identity = observation_evidence_id(item)
                            evidence_ids.add(evidence_identity)
                            if item["producer_version"] is not None and item["status"] in {"matches_golden", "differs_from_golden", "parser_error"}:
                                executed_ids.add(evidence_identity)
                        row = {"status": "inventoried", "observations": observations}
                    versions = {history.captures[identity]["runtime_version"] for history in histories.values()}
                    if len(versions) > 1:
                        raise ValueError(f"capture identity has conflicting runtime versions: {identity}")
                    row.update(family=family, scenario=scenario, implementation=implementation, capture=identity,
                               version=next(iter(versions), identity.removeprefix(implementation + "-")),
                               mode="batch" if implementation in UNIFIED_BATCH_IMPLS else "stream")
                    counts[row["status"]] += 1
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    expected = len(families) * len(scenarios) * len(selections)
    if sum(counts.values()) != expected:
        raise ValueError("historical inventory does not reconcile")
    return {"selections": len(selections), "combinations": expected, "counts": dict(counts),
            "observation_counts": dict(observation_counts),
            "unique_observation_evidence": len(evidence_ids), "unique_bound_executed_evidence": len(executed_ids),
            "deduplication": "Family/case, actual producer implementation/version/path/mode, selected request/result, and source/adapter/job identities; inherited checkpoint labels do not create executions.",
            "path": destination.name}


def retained_evidence_inventory(stores, families, destination):
    """Keep retired and orphan evidence outside the active scenario cross product."""
    counts = Counter()
    with gzip.open(destination, "wt", encoding="utf-8") as output:
        for surface, store in stores:
            for (family, implementation), history in sorted(store.histories.items()):
                if report_family(family) not in families:
                    continue
                for identity, records in history.iter_resolved_snapshots(include_members=True):
                    members = history.snapshot_members.get(identity, {})
                    for case_id, record in sorted(records.items()):
                        case = history.family.cases.get(case_id)
                        lifecycle = case.get("lifecycle", "active") if case is not None else "orphan"
                        request_retained = record["stimulus"].get("unavailable", {}).get("code") != "original_request_not_retained"
                        document = record["document"]
                        metadata = document.get("record_metadata", {})
                        row = {"surface": surface, "family": report_family(family), "source_family": family,
                               "implementation": implementation, "capture": identity,
                               "selection_version": history.captures[identity]["runtime_version"],
                               "producer_version": document.get("captured_with", {}).get(implementation),
                               "mode": unified_record_contract(implementation, document),
                               "parser_path": effective_parser_path(metadata, document), "case_id": case_id,
                               "scenario": case.get("scenario") if case is not None else None,
                               "lifecycle": lifecycle, "request_retained": request_retained,
                               "source_member": members.get(case_id),
                               "stimulus_sha256": digest(record["stimulus"]),
                               "observation_sha256": digest(record["observation"]), "document_sha256": digest(document),
                               "producer_origin": document.get("capture_origin", {}),
                               "disposition": ("inventoried_in_active_grid" if lifecycle == "active" else
                                               "retained_without_active_oracle" if request_retained else "historical_request_not_retained")}
                        counts[lifecycle] += 1
                        counts["request_retained" if request_retained else "request_not_retained"] += 1
                        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"records": sum(counts[key] for key in ("active", "retired", "orphan")),
            "counts": dict(counts), "path": destination.name,
            "scope": "Every retained registered-family record, including retired split observations without requests; current and history surfaces remain separate evidence."}


def apply_review_evidence(audit, applicability, reviewed_inventory, differences):
    """Bind independent dispositions to the exact inventoried request identities."""
    if not typed_equal(audit["family_scenario_inventory"], reviewed_inventory["rows"]):
        raise ValueError("applicability evidence does not describe the candidate inventory")
    dispositions = {}
    for row in [*applicability["resolved_aliases"], *applicability["dispositions"],
                *applicability["final_candidate_snapshot"]["new_coverage_aliases"]]:
        identity = row["identity"]
        key = identity["family"], identity["scenario"]
        if key in dispositions and not typed_equal(dispositions[key], row):
            raise ValueError(f"conflicting applicability dispositions: {key}")
        dispositions[key] = row
    counts = Counter()
    for row in audit["family_scenario_inventory"]:
        if row["status"] == "authored":
            continue
        key = row["family"], row["scenario"]
        if key not in dispositions:
            raise ValueError(f"missing applicability disposition: {key}")
        row["disposition"] = dispositions[key]
        counts[row["disposition"]["classification"]] += 1
    reviewed = {row["identity"]: row for row in differences["dispositions"]}
    for row in audit["differences"]:
        disposition = reviewed.get(row["identity"])
        if disposition is None or not typed_equal(row, disposition["original_group"]):
            raise ValueError(f"difference evidence does not describe the candidate group: {row['identity']}")
        row.update(status="reviewed", disposition=disposition)
    audit["totals"]["missing_authored_pairs"] = sum(counts.values())
    audit["totals"]["pending_applicability"] = counts["insufficient_evidence"]
    audit["totals"]["pending_differences"] = sum(not row["disposition"]["resolved"] for row in audit["differences"])
    audit["applicability_classification_counts"] = dict(counts)
    audit["reviewed_candidate"] = reviewed_inventory["snapshot"]
    # Observations and absent executions remain individually inventoried below;
    # resolving authored grammar differences alone cannot complete that review.
    audit["complete"] = False
    audit["limitations"] = [
        "Unresolved applicability pairs remain insufficient evidence, never inferred exclusions.",
        "Historical observation differences and unavailability still require producer-specific review.",
        "Coverage aliases establish semantic coverage, not identical requests or historical execution.",
    ]


def observation_evidence_id(item):
    """Deduplicate executions by their measured producer and exact selected request/result."""
    return digest({key: item.get(key) for key in (
        "family", "case_id", "producer_implementation", "producer_version", "parser_path", "mode", "selected_request_sha256", "selected_observation_sha256",
    )} | {"origin": {key: item["producer_origin"].get(key) for key in (
        "git_commit", "source_sha256", "adapter_sha256", "job_sha256",
    )}})


def json_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def disposition_fields(row):
    if not all(row.get(key) for key in ("reason", "source_evidence", "reproducer", "required_correction")):
        raise ValueError("producer disposition lacks evidence, reason, reproducer or correction")
    if not isinstance(row.get("resolved"), bool):
        raise ValueError("producer disposition lacks explicit resolution")
    if row.get("missing_verification") and row["resolved"]:
        raise ValueError("explicit verification gaps must remain unresolved")
    if row.get("classification") == "insufficient_evidence" and (row["resolved"] or not row.get("missing_verification")):
        raise ValueError("insufficient evidence must retain unresolved proof gaps")


def apply_producer_review(audit, manifest_path, inventory_root):
    """Validate exhaustive dispositions; completion describes coverage, not correctness."""
    if "reviewed_candidate" not in audit or "history" not in audit or "retained_evidence" not in audit:
        raise ValueError("producer review requires authored review and both generated inventories")
    manifest = json.loads(manifest_path.read_text())
    names = ("grid", "observations", "selected_evidence", "retired")
    paths = {}
    for name in names:
        reference = manifest["sidecars"][name]
        path = manifest_path.parent / reference["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"producer sidecar hash differs: {name}")
        paths[name] = path
    authored = {(row["family"], row["scenario"]): row for row in audit["family_scenario_inventory"]}
    requests = {(item["family"], item["case_id"]): item for row in authored.values() for item in row["requests"]}
    expected = {}
    grid_counts = Counter()
    grid_missing = 0
    reviews = iter(json_lines(paths["grid"]))
    for original in json_lines(inventory_root / audit["history"]["path"]):
        review = next(reviews, None)
        identity = {key: original[key] for key in ("family", "scenario", "implementation", "capture", "version", "mode")}
        if review is None or review.get("identity") != identity or review.get("inventory_row_sha256") != digest(original):
            raise ValueError("producer grid missing, duplicated, reordered, foreign or stale")
        grid_counts[review["classification"]] += 1
        if original["status"] == "pending_applicability":
            disposition = authored[(original["family"], original["scenario"])]["disposition"]
            if review["classification"] != disposition["classification"] or review.get("authored_disposition_sha256") != digest(disposition) or review.get("execution_claimed") is not False:
                raise ValueError("producer exclusion differs from independent applicability evidence")
            continue
        if review["classification"] != "observation_dispositions":
            raise ValueError("authored supported request cannot be excluded from producer review")
        records = review.get("records", [])
        if len(records) != len(original["observations"]):
            raise ValueError("producer grid omits an applicable child")
        for observation, record in zip(original["observations"], records):
            if any(record.get(key) != observation[key] for key in ("family", "case_id")):
                raise ValueError("producer grid child identity differs")
            if observation["status"] == "missing_execution":
                disposition_fields(record)
                if record["classification"] != "missing_execution" or record.get("execution_claimed") is not False:
                    raise ValueError("missing execution cannot become an applicability exclusion")
                grid_missing += 1
                continue
            evidence_id = observation_evidence_id(observation)
            if record.get("evidence_id") != evidence_id:
                raise ValueError("producer grid has a broken historical binding")
            original_hashes = tuple(observation[key] for key in ("stimulus_sha256", "original_observation_sha256", "document_sha256"))
            item = expected.setdefault(evidence_id, {"metadata": observation, "variants": set(), "claims": {}, "family": original["family"], "scenario": original["scenario"], "implementation": original["implementation"]})
            metadata_hash = digest({key: value for key, value in observation.items() if key not in {"family", "case_id", "source_member"}})
            item["variants"].add((metadata_hash, original_hashes, observation.get("source_member")))
            if "status" in record and record["status"] != observation["status"]:
                raise ValueError("grid status contradicts retained observation")
            for key in ("classification", "resolved"):
                if key in record:
                    if key in item["claims"] and item["claims"][key] != record[key]:
                        raise ValueError("grid contains conflicting producer dispositions")
                    item["claims"][key] = record[key]
    if next(reviews, None) is not None:
        raise ValueError("producer grid contains foreign or duplicate rows")
    if sum(grid_counts.values()) != audit["history"]["combinations"]:
        raise ValueError("producer grid does not reconcile with historical combinations")
    seen = set()
    for row in json_lines(paths["selected_evidence"]):
        evidence_id = row["evidence_id"]
        if evidence_id not in expected or evidence_id in seen:
            raise ValueError("selected producer evidence is duplicate or foreign")
        seen.add(evidence_id)
        item = expected[evidence_id]
        meta = item["metadata"]
        identity = row["identity"]
        if identity != {"family": item["family"], "source_family": meta["family"], "scenario": item["scenario"], "case_id": meta["case_id"], "implementation": item["implementation"]}:
            raise ValueError("selected producer identity differs from historical inventory")
        if observation_evidence_id(row["observation_metadata"] | {"family": identity["source_family"], "case_id": identity["case_id"]}) != evidence_id:
            raise ValueError("selected producer metadata differs from historical binding")
        request = requests[(meta["family"], meta["case_id"])]
        for field, expected_hash in (("active_request", request["request_sha256"]), ("golden", request["golden_sha256"]), ("selected_observation", meta["selected_observation_sha256"])):
            if digest(row[field]) != expected_hash:
                raise ValueError(f"selected producer payload differs: {field}")
        selected_hash = digest(row["selected_request"]) if row["selected_request"] is not None else None
        if selected_hash != meta["selected_request_sha256"]:
            raise ValueError("selected original request differs")
        raw_original_hashes = tuple(digest(row[key]) for key in ("original_stimulus", "original_observation", "source_document"))
        raw_metadata_hash = digest({key: value for key, value in row["observation_metadata"].items() if key not in {"family", "case_id", "source_member"}})
        if (raw_metadata_hash, raw_original_hashes, row["observation_metadata"].get("source_member")) not in item["variants"]:
            raise ValueError("original retained evidence differs")
    if seen != expected.keys():
        raise ValueError("selected producer evidence is incomplete")
    seen = set()
    counts = Counter()
    unresolved = Counter()
    allowed = {
        "matches_golden": {"matches_authored_typed_events"},
        "differs_from_golden": {"parser_defect", "required_semantic_variation", "insufficient_evidence"},
        "parser_error": {"parser_error", "parser_defect", "insufficient_evidence", "source_build_blocker", "adapter_limitation"},
        "request_mismatch": {"request_mismatch"},
        "request_binding_unavailable": {"request_binding_unavailable", "insufficient_evidence", "adapter_limitation", "source_build_blocker"},
        "pending_unavailability_review": {"request_binding_unavailable", "adapter_limitation", "source_build_blocker", "unsupported_producer_api_mode", "insufficient_evidence"},
    }
    for row in json_lines(paths["observations"]):
        evidence_id = row["evidence_id"]
        if evidence_id not in expected or evidence_id in seen:
            raise ValueError("producer disposition identity is duplicate or foreign")
        seen.add(evidence_id)
        item = expected[evidence_id]
        meta = item["metadata"]
        disposition_fields(row)
        if any(row.get(key) != value for key, value in item["claims"].items()):
            raise ValueError("grid and unique producer disposition disagree")
        expected_identity = {"family": item["family"], "source_family": meta["family"], "scenario": item["scenario"], "case_id": meta["case_id"], "implementation": item["implementation"]}
        if row.get("identity") != expected_identity or row.get("producer_identity") != {key: meta.get(key) for key in ("producer_implementation", "producer_version", "parser_path", "mode", "producer_origin")}:
            raise ValueError("producer disposition identity differs")
        if row["selected_status"] != meta["status"] or row["classification"] not in allowed[meta["status"]]:
            raise ValueError("producer disposition contradicts retained observation status")
        request = requests[(meta["family"], meta["case_id"])]
        if any(row.get(key) != value for key, value in {
            "captured_request_sha256": meta["selected_request_sha256"],
            "active_request_sha256": request["request_sha256"],
            "golden_sha256": request["golden_sha256"],
            "selected_observation_sha256": meta["selected_observation_sha256"],
        }.items()):
            raise ValueError("producer disposition request, GOLDEN or observation binding differs")
        counts[row["classification"]] += 1
        if not row["resolved"]:
            unresolved[row["classification"]] += 1
    if seen != expected.keys():
        raise ValueError("producer dispositions omit selected evidence")
    retired = iter(json_lines(paths["retired"]))
    retired_count = 0
    for original in json_lines(inventory_root / audit["retained_evidence"]["path"]):
        if original["lifecycle"] == "active":
            continue
        row = next(retired, None)
        if row is None or row.get("inventory_row_sha256") != digest(original):
            raise ValueError("retired evidence missing, stale or reordered")
        disposition_fields(row)
        if row.get("identity") != {key: original[key] for key in ("surface", "family", "source_family", "case_id", "implementation", "capture", "selection_version", "mode", "scenario", "lifecycle")}:
            raise ValueError("retired evidence identity differs")
        if row["source_evidence"] != {key: original[key] for key in ("source_member", "stimulus_sha256", "observation_sha256", "document_sha256", "producer_origin")}:
            raise ValueError("retired source evidence differs")
        expected_class = "retained_without_active_oracle" if original["request_retained"] else "historical_request_not_retained"
        if row["classification"] != expected_class:
            raise ValueError("retired evidence cannot acquire an active oracle")
        retired_count += 1
    if next(retired, None) is not None:
        raise ValueError("retired evidence contains foreign or duplicate rows")
    audit["producer_review"] = {
        "disposition_coverage_complete": True,
        "source_and_semantic_verification_complete": False,
        "all_observations_match_golden": not any(key != "matches_authored_typed_events" for key in counts) and not grid_missing,
        "grid_rows": sum(grid_counts.values()), "grid_classifications": dict(grid_counts),
        "unique_observations": len(expected), "observation_classifications": dict(counts),
        "unresolved_classifications": dict(unresolved), "missing_executions": grid_missing,
        "retired_observations": retired_count, "manifest": manifest_path.name, "sidecars": manifest["sidecars"], "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "completion_meaning": "Every inventoried identity has a bound supported disposition; insufficient evidence, parser failures, blockers and missing executions remain explicit. Recorded fingerprints are not independent source/artifact reconstruction.",
    }
    audit["complete"] = True
    audit["limitations"] = [
        "Completion describes exhaustive bound disposition coverage, not producer correctness or independent reconstruction of all retained executables.",
        "Unresolved proof gaps, parser defects, unsupported requests and missing executions remain measured in producer_review.",
        "Coverage aliases establish semantic coverage, not identical requests or historical execution.",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--history-root", type=Path, help="prepared include-history checkout, not a latest-only store")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--applicability-evidence", type=Path)
    parser.add_argument("--reviewed-inventory", type=Path)
    parser.add_argument("--difference-evidence", type=Path)
    parser.add_argument("--producer-review-evidence", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    families, scenarios, rows = authored_inventory(root / "conformance/fixtures-unified-v2/families")
    differences = differing_contracts(families, scenarios)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {"schema_version": 1, "complete": False,
             "comparison": "Lossless typed event equality; family votes guide investigation only. No applicability is inferred from an authored scope.",
             "totals": {"families": len(families), "scenarios": len(scenarios), "pairs": len(rows),
                        "authored": sum(row["status"] == "authored" for row in rows),
                        "pending_applicability": sum(row["status"] == "pending_applicability" for row in rows),
                        "pending_differences": len(differences)},
             "family_scenario_inventory": rows, "differences": differences}
    evidence_paths = (args.applicability_evidence, args.reviewed_inventory, args.difference_evidence)
    if any(evidence_paths):
        if not all(evidence_paths):
            parser.error("all three independent review evidence paths are required together")
        apply_review_evidence(audit, *(json.loads(path.read_text()) for path in evidence_paths))
        audit["evidence"] = [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for path in evidence_paths]
    if args.history_root:
        store = unified_history.load_store(args.history_root.resolve() / "conformance/fixtures-unified-v2")
        current = unified_history.load_store(root / "conformance/fixtures-unified-v2")
        audit["history"] = history_inventory(store, families, scenarios, args.output / "audit-history.jsonl.gz", current_store=current)
        audit["retained_evidence"] = retained_evidence_inventory(
            [("history", store), ("current", current)], families, args.output / "audit-retained-evidence.jsonl.gz")
    if args.producer_review_evidence:
        apply_producer_review(audit, args.producer_review_evidence, args.output)
    (args.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit["totals"], sort_keys=True))


if __name__ == "__main__":
    main()
