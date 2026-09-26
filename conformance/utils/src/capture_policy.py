# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared capture visibility, saved links, and historical reader selection.

Policy describes exceptions and selections, never the available capture inventory.
Historical selection does not certify a capture against the current parser source.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from capture_bindings import original_capture_input, read_bindings
from fixture_corpus import checkpoint_families, complete_family_snapshots, split_sel, version_key
from fixture_disposition import LEGACY_CORPORA, parse_legacy_capture_label

CORPORA = {**LEGACY_CORPORA, "unified": "unified"}


def load_policy(path: Path | None = None) -> dict:
    path = path or Path(os.environ.get("CONFORMANCE_CAPTURE_POLICY", Path(__file__).resolve().parents[2] / "capture-policy.yaml"))
    policy = yaml.safe_load(Path(path).read_text())
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError(f"unsupported capture policy schema: {path}")
    if not isinstance(policy.get("corpora"), dict) or set(policy["corpora"]) != set(CORPORA):
        raise ValueError("capture policy must declare each corpus exactly once")
    for corpus, rules in policy["corpora"].items():
        if not isinstance(rules, dict):
            raise ValueError(f"invalid capture policy for {corpus}")
        for name in ("selectors", "saved_links", "references", "historical_readers"):
            if not isinstance(rules.get(name, {}), dict):
                raise ValueError(f"invalid {corpus}.{name}")
    return policy


def policy_for_snapshot(root: Path, *, path: Path | None = None) -> dict:
    """Use the policy packaged with a snapshot; explicit CLI policy overrides it."""
    if path is not None:
        return load_policy(path)
    resolved = Path(root).resolve()
    for parent in (resolved, *resolved.parents):
        candidate = parent / "capture-policy.yaml"
        if candidate.is_file():
            return load_policy(candidate)
    return load_policy()


def capture_sort_key(identity: str) -> tuple:
    impl, version = split_sel(identity)
    return impl, version_key(version), "+" in version or ".patch" in version, identity


def discover_captures(root: Path) -> list[str]:
    return sorted((p.name for p in root.iterdir() if p.is_dir() and "-" in p.name
                   and p.name not in {"inputs", "golden"}), key=capture_sort_key) if root.is_dir() else []


def reader_metadata(corpus: str, reader: str, available, *, policy: dict | None = None) -> dict:
    rules = (policy or load_policy())["corpora"][corpus].get("historical_readers", {})
    if reader not in rules:
        raise ValueError(f"undeclared historical reader: {corpus}/{reader}")
    rule = rules[reader]
    include_qualified = rule.get("include_qualified", False)
    selection = rule.get("selection", "latest")
    eligible = sorted((name for name in available if split_sel(name)[0] == reader
                       and (include_qualified or ("+" not in name and ".patch" not in name))), key=capture_sort_key)
    selected = None
    reason = rule.get("reason")
    anchor = None
    if selection == "inputs":
        if corpus != "reasoning" or not reason:
            raise ValueError("input-anchor selection is only valid for reasoning with a reason")
        anchor = "inputs"
    elif selection == "latest":
        selected = eligible[-1] if eligible else None
        if selected is None:
            reason = f"no eligible historical capture for {corpus}/{reader}"
    elif selection == "unavailable":
        if not reason:
            raise ValueError(f"unavailable reader needs a reason: {corpus}/{reader}")
    elif selection in eligible:
        selected = selection
    else:
        raise ValueError(f"selected historical capture is missing or ineligible: {corpus}/{reader}: {selection}")
    return {"corpus": corpus, "reader": reader, "include_qualified": include_qualified,
            "selection": selection, "eligible": eligible, "selected": selected, "anchor": anchor,
            "unavailable": reason if selected is None and anchor is None else None}


def observation_digest(state: dict) -> str:
    records = sorted((list(key) if isinstance(key, tuple) else key, value) for key, value in state.items())
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def validate_policy(policy: dict, inventory: dict, observations: dict | None = None) -> None:
    """Validate references, proving declared equivalents from complete resolved states."""
    for corpus, rules in policy["corpora"].items():
        available = set(inventory.get(corpus, []))
        selectors = rules.get("selectors", {})
        for source, rule in selectors.items():
            target = rule.get("equivalent_to")
            if not isinstance(rule.get("visible", True), bool):
                raise ValueError(f"invalid visibility: {corpus}/{source}")
            if target:
                if target not in available:
                    raise ValueError(f"missing equivalent target: {corpus}/{target}")
                if not selectors.get(target, {}).get("visible", True):
                    raise ValueError(f"equivalent target is hidden: {corpus}/{target}")
                states = (observations or {}).get(corpus, {})
                certificate = rule.get("equivalence_sha256")
                matches = (source in states and bool(states[source]) and states[source] == states.get(target))
                certified_removal = (source not in available and source not in states and certificate
                                     and target in states and certificate == observation_digest(states[target]))
                if not (matches or certified_removal):
                    raise ValueError(f"equivalence lacks matching complete observations: {corpus}/{source} -> {target}")
            elif source not in available and not rule.get("unavailable"):
                raise ValueError(f"removed capture needs explicit unavailable state: {corpus}/{source}")
        for key, target in rules.get("saved_links", {}).items():
            if target not in available and not (selectors.get(target, {}).get("unavailable") or selectors.get(target, {}).get("equivalent_to")):
                raise ValueError(f"saved link target is missing: {corpus}/{key}: {target}")
        for reader, rule in rules.get("historical_readers", {}).items():
            reader_metadata(corpus, reader, available, policy=policy)
        if len(rules.get("references", {})) > 1:
            raise ValueError(f"a report corpus has only one default reference: {corpus}")
        for impl, rule in rules.get("references", {}).items():
            selection = rule.get("selection")
            if selection == "inputs":
                if corpus != "reasoning" or not rule.get("reason"):
                    raise ValueError("input-anchor reference is only valid for reasoning with a reason")
            elif selection == "unavailable":
                if not rule.get("reason"):
                    raise ValueError(f"unavailable reference needs a reason: {corpus}/{impl}")
            elif selection != "latest" and (selection not in available or split_sel(selection)[0] != impl):
                raise ValueError(f"selected reference is missing: {corpus}/{impl}: {selection}")
            elif selection != "latest" and not selectors.get(selection, {}).get("visible", True):
                raise ValueError(f"selected reference is hidden: {corpus}/{selection}")


def _resolved_materialized_captures(root: Path, corpus: str, names: list[str], complete: bool) -> dict:
    states, resolved = {}, {}
    if corpus == "reasoning":
        for path in sorted((root / "inputs").glob("*/*.yaml")):
            doc = yaml.safe_load(path.read_text())
            for case, record in doc.get("cases", {}).items():
                for impl, value in record.get("expected", {}).items():
                    states.setdefault(impl, {})[(path.parent.name, path.name, case)] = {"expected": {impl: copy.deepcopy(value)}}
    for name in names:
        impl, _ = split_sel(name)
        state = {} if corpus in {"unified", "batch_on_stream"} else states.setdefault(impl, {})
        if corpus not in {"unified", "batch_on_stream"}:
            for family in sorted(checkpoint_families(root / name)):
                if complete:
                    for key in [key for key in state if key[0] == family]:
                        del state[key]
                state[(family, "__family__")] = {"present": True}
        bindings = read_bindings(root / name) if corpus == "unified" else {}
        for path in sorted((root / name).glob("*/*.yaml")):
            doc = yaml.safe_load(path.read_text())
            if corpus == "unified":
                record = {k: v for k, v in doc.items() if k not in {"captured_with", "capture_origin", "inherited_from", "capture_input"}}
                record["capture_input"] = original_capture_input(doc, path.read_bytes(), str(path.relative_to(root / name)), bindings)
                state[(path.parent.name, path.name)] = record
            else:
                state[(path.parent.name, path.name, "__document__")] = {
                    key: copy.deepcopy(value) for key, value in doc.items()
                    if key not in {"cases", "captured_with", "capture_origin", "inherited_from"}}
                for case, record in doc.get("cases", {}).items():
                    state[(path.parent.name, path.name, case)] = copy.deepcopy(record)
        resolved[name] = copy.deepcopy(state)
    return resolved


def materialized_evidence(snapshot_root: Path) -> tuple[dict, dict]:
    """Compare every case/annotation in each independently resolved implementation.

    Producer headers identify the measurement and intentionally differ between
    equivalent versions. Requests are shared by these legacy loose readers; full
    case payloads retain request overrides and all observation annotations.
    """
    inventory, observations = {}, {}
    complete = complete_family_snapshots(snapshot_root)
    for corpus, relative in CORPORA.items():
        root = snapshot_root / relative
        history_root = snapshot_root / ".reader-views/batch_on_stream"
        if corpus == "batch_on_stream" and history_root.is_dir():
            root = history_root
        names = discover_captures(root)
        inventory[corpus] = names
        resolved = _resolved_materialized_captures(root, corpus, names, complete)
        if corpus == "stream":
            rust_root = snapshot_root / ".reader-views/rust" / relative if complete else root
            rust_names = [name for name in discover_captures(rust_root) if "+" not in name and ".patch" not in name]
            rust_states = _resolved_materialized_captures(rust_root, corpus, rust_names, complete)
            for identity, state in resolved.items():
                if identity not in rust_states:
                    if "+" in identity or ".patch" in identity:
                        state[("__rust_view__",)] = {"eligible": False}
                        continue
                    raise ValueError(f"missing historical Rust stream view: {identity}")
                state[("__rust_view__",)] = sorted((list(key), value) for key, value in rust_states[identity].items())
        if corpus == "batch_on_stream" and not names:
            for path in sorted(root.glob("*/*.yaml")):
                doc = yaml.safe_load(path.read_text())
                for impl, version in doc.get("captured_with", {}).items():
                    name, _ = parse_legacy_capture_label(impl, version)
                    resolved.setdefault(name, {})[(path.parent.name, path.name)] = {
                        case: value[impl] for case, value in doc.get("cases", {}).items() if impl in value}
            inventory[corpus] = sorted(resolved, key=capture_sort_key)
        observations[corpus] = resolved
    return inventory, observations


def validate_materialized_policy(policy: dict, snapshot_root: Path, *, before_root: Path | None = None) -> None:
    inventory, observations = materialized_evidence(snapshot_root)
    if before_root is not None:
        _, before = materialized_evidence(before_root)
        for corpus, states in before.items():
            for identity, state in states.items():
                observations[corpus].setdefault(identity, state)
    validate_policy(policy, inventory, observations)


def selected_removal_references(policy: dict, corpus: str, capture: str, before_root: Path,
                                *, family: str | None = None) -> list[tuple[str, str]]:
    """Resolve symbolic latest choices before removal, including per-family coverage."""
    rules = policy["corpora"][corpus]
    root = before_root / CORPORA[corpus]
    if corpus == "batch_on_stream" and (before_root / ".reader-views/batch_on_stream").is_dir():
        root = before_root / ".reader-views/batch_on_stream"
    available = discover_captures(root)
    if capture not in available:
        return []
    removed_impl, _ = split_sel(capture)
    families = {family} if family is not None else checkpoint_families(root / capture)
    impacted = []
    for section in ("references", "historical_readers"):
        for impl, rule in rules.get(section, {}).items():
            if impl != removed_impl:
                continue
            selection = rule["selection"]
            if selection == capture:
                impacted.append((section, impl))
                continue
            if selection != "latest":
                continue
            eligible = [identity for identity in available if split_sel(identity)[0] == impl]
            if section == "historical_readers" and not rule.get("include_qualified", False):
                eligible = [identity for identity in eligible if "+" not in identity and ".patch" not in identity]
            if section == "references":
                eligible = [identity for identity in eligible if rules.get("selectors", {}).get(identity, {}).get("visible", True)]
            if any((matches := [identity for identity in eligible if (root / identity / name).is_dir()])
                   and matches[-1] == capture for name in families):
                impacted.append((section, impl))
    return impacted


def validate_partial_removal_policy(policy: dict, corpus: str, capture: str, family: str,
                                    before_root: Path) -> None:
    impacted = selected_removal_references(policy, corpus, capture, before_root, family=family)
    if impacted:
        raise ValueError(f"removing selected family capture {corpus}/{family}/{capture} requires a candidate policy "
                         "with an explicit replacement or unavailable reference first")


def retire_capture(policy: dict, corpus: str, capture: str, *, replacement: str | None = None,
                   unavailable: str | None = None, before_root: Path | None = None) -> dict:
    """Return removal policy; equivalence is proved against the pre-removal state."""
    if replacement and unavailable:
        raise ValueError("choose a replacement or unavailable reason")
    if replacement == capture:
        raise ValueError("a removed capture cannot replace itself")
    result = copy.deepcopy(policy)
    rules = result["corpora"][corpus]
    selectors = rules.setdefault("selectors", {})
    dependents = [source for source, rule in selectors.items() if rule.get("equivalent_to") == capture]
    selected_keys = (selected_removal_references(policy, corpus, capture, before_root) if before_root is not None
                     else [(section, impl) for section in ("references", "historical_readers")
                           for impl, rule in rules.get(section, {}).items() if rule.get("selection") == capture])
    if before_root is None and any(rule.get("selection") == "latest"
                                   for section in ("references", "historical_readers")
                                   for impl, rule in rules.get(section, {}).items() if impl == split_sel(capture)[0]):
        raise ValueError("latest removal selection requires the original snapshot evidence")
    selected = [rules[section][impl] for section, impl in selected_keys]
    links = [key for key, value in rules.get("saved_links", {}).items() if value == capture]
    if (dependents or selected or links) and not (replacement or unavailable):
        raise ValueError(f"removing selected capture {corpus}/{capture} requires replacement or unavailable")
    reason = unavailable or "This historical capture was removed."
    for source in [capture, *dependents]:
        selectors[source] = ({"visible": False, "equivalent_to": replacement} if replacement
                             else {"visible": False, "unavailable": reason})
    if replacement:
        if before_root is None:
            raise ValueError("equivalent removal requires the original snapshot evidence")
        _, evidence = materialized_evidence(before_root)
        states = evidence[corpus]
        for source in [capture, *dependents]:
            previous = policy["corpora"][corpus].get("selectors", {}).get(source, {})
            target_state = states.get(replacement)
            matches = bool(target_state) and states.get(source) == target_state
            certified = (source not in states and bool(target_state) and previous.get("equivalent_to") == capture
                         and previous.get("equivalence_sha256") == observation_digest(target_state))
            if not (matches or certified):
                raise ValueError(f"replacement is not equivalent: {source} -> {replacement}")
            selectors[source]["equivalence_sha256"] = observation_digest(target_state)
    for rule in selected:
        rule["selection"] = replacement or "unavailable"
        if unavailable:
            rule["reason"] = unavailable
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--reader-metadata", choices=CORPORA, required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--captures-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reader_metadata(args.reader_metadata, args.reader,
                                    discover_captures(args.captures_root), policy=policy_for_snapshot(args.captures_root, path=args.policy))))


if __name__ == "__main__":
    main()
