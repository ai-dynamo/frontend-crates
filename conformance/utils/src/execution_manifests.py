# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lossless factoring of repeated execution metadata, preserving absent keys."""

import copy
import re

from viewer_codec import canonical_typed, typed_equal as _equal, typed_key_identity

FIELDS = {"schema_version", "family", "implementation", "version", "checkpoint", "cases", "defaults", "exceptions"}
CHECKPOINT_FIELDS = {"provenance", "document", "document_overrides"}


def source_reference(value, directory):
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return f"{directory}/{value}.yaml"
    if not isinstance(value, str) or re.fullmatch(rf"{directory}/[A-Za-z0-9][A-Za-z0-9_.+-]*\.yaml", value) is None:
        raise ValueError(f"invalid {directory} filename reference: {value!r}")
    return value


def _key_items(record):
    return {typed_key_identity(key): (key, value) for key, value in record.items()}


def common_mapping(records):
    if not records:
        return {}
    common = {}
    indexed = [_key_items(record) for record in records]
    for identity, (key, first) in indexed[0].items():
        if not all(identity in record for record in indexed):
            continue
        values = [record[identity][1] for record in indexed]
        if all(_equal(value, first) for value in values):
            common[key] = copy.deepcopy(first)
        elif isinstance(first, dict) and all(isinstance(value, dict) for value in values):
            nested = common_mapping(values)
            if nested:
                common[key] = nested
    return common


def remainder(record, defaults):
    result = {}
    indexed = _key_items(defaults)
    for identity, (key, value) in _key_items(record).items():
        if identity not in indexed:
            result[key] = copy.deepcopy(value)
            continue
        default = indexed[identity][1]
        if _equal(value, default):
            continue
        if isinstance(default, dict) and isinstance(value, dict):
            result[key] = remainder(value, default)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge(defaults, exception):
    result = copy.deepcopy(defaults)
    for key, value in exception.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _receipt_slots(metadata):
    for key in metadata:
        if isinstance(key, str) and (key == "capture_provenance" or key.endswith("_capture_provenance")):
            yield metadata, key
    owners = [metadata.get("verified_recapture")]
    archived = metadata.get("verified_recaptures", [])
    if isinstance(archived, list):
        owners.extend(archived)
    for owner in owners:
        if isinstance(owner, dict) and "provenance" in owner:
            yield owner, "provenance"


def _canonical_receipt(value):
    if isinstance(value, dict):
        return {key: _canonical_receipt(item) for key, item in sorted(value.items(), key=lambda pair: canonical_typed(pair[0]))}
    if isinstance(value, list):
        return [_canonical_receipt(item) for item in value]
    return value


def compact(document, *, family, implementation, version, receipt_documents=None, receipt_names=None):
    metadata = copy.deepcopy(document["metadata_changes"])
    proof = copy.deepcopy(document.get("execution_proof"))
    if not metadata and proof is None:
        return None
    memberships = {}
    slots = [(owner, key, [case]) for case, value in metadata.items() for owner, key in _receipt_slots(value)]
    if proof is not None:
        slots.append((proof, "receipt", list(proof["requests"])))
    for owner, key, cases in slots:
        receipt = owner[key]
        if isinstance(receipt, str) and receipt.startswith("receipts/"):
            raise ValueError("unexpanded receipt reference in execution metadata")
        if not isinstance(receipt, dict) or not {"version", "job_sha256", "adapter_sha256"} <= receipt.keys():
            continue
        if receipt_documents is None or receipt_names is None:
            raise ValueError("execution receipt pooling requires a shared document owner")
        identity = canonical_typed(receipt)
        reference = receipt_names.get(identity)
        if reference is None:
            prefix = f"receipts/{family}--{implementation}-{version}--receipt-"
            index = 1
            reference = f"{prefix}{index:03}.yaml"
            while reference in receipt_names.values():
                index += 1
                reference = f"{prefix}{index:03}.yaml"
            receipt_names[identity] = reference
        receipt_documents[reference] = _canonical_receipt(receipt)
        memberships.setdefault(reference, set()).update(cases)
        owner[key] = reference
    defaults = common_mapping(list(metadata.values()))
    exceptions = {case: remainder(value, defaults) for case, value in metadata.items()}
    return {"schema_version": 1, "family": family, "implementation": implementation, "version": version,
            "checkpoint": {key: copy.deepcopy(document[key]) for key in sorted(CHECKPOINT_FIELDS)},
            "cases": sorted(metadata), "defaults": defaults,
            "exceptions": {case: value for case, value in exceptions.items() if value},
            **({"proof": proof} if proof is not None else {}),
            **({"receipts": {digest: sorted(cases) for digest, cases in sorted(memberships.items())}} if memberships else {})}


def expand(manifest, *, family, implementation, version, receipt_documents=None):
    if not isinstance(manifest, dict) or set(manifest) - {"receipts", "proof"} != FIELDS or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("invalid execution manifest fields or schema")
    if (manifest["family"], manifest["implementation"], manifest["version"]) != (family, implementation, version):
        raise ValueError("execution manifest producer identity mismatch")
    checkpoint = manifest["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != CHECKPOINT_FIELDS:
        raise ValueError("invalid execution checkpoint metadata")
    cases, defaults, exceptions = manifest["cases"], manifest["defaults"], manifest["exceptions"]
    if not isinstance(cases, list) or (not cases and "proof" not in manifest) or any(not isinstance(case, str) for case in cases) or len(cases) != len(set(cases)):
        raise ValueError("invalid execution case membership")
    if not isinstance(defaults, dict) or not isinstance(exceptions, dict) or set(exceptions) - set(cases):
        raise ValueError("execution metadata differs from case membership")
    if not cases and defaults:
        raise ValueError("execution defaults require metadata case membership")
    if any(not isinstance(value, dict) for value in exceptions.values()):
        raise ValueError("execution metadata exception must be a mapping")
    metadata = {case: merge(defaults, exceptions.get(case, {})) for case in cases}
    memberships = manifest.get("receipts", {})
    if not isinstance(memberships, dict):
        raise ValueError("execution receipt membership must be a mapping")
    observed = {}
    proof = copy.deepcopy(manifest.get("proof"))
    if "proof" in manifest:
        if not isinstance(proof, dict) or set(proof) != {"receipt", "requests"} or not isinstance(proof["requests"], dict):
            raise ValueError("invalid execution proof fields")
        reference = proof["receipt"]
        if isinstance(reference, str) and re.fullmatch(r"receipt:[0-9a-f]{64}", reference):
            reference = reference.removeprefix("receipt:")
        source_reference(reference, "receipts")
        digest = reference
        if digest not in memberships or receipt_documents is None or digest not in receipt_documents:
            raise ValueError("execution proof receipt body or membership is missing")
        observed[digest] = set(proof["requests"])
        proof["receipt"] = copy.deepcopy(receipt_documents[digest])
    for case, value in metadata.items():
        for owner, key in _receipt_slots(value):
            reference = owner[key]
            legacy_receipt = isinstance(reference, str) and re.fullmatch(r"receipt:[0-9a-f]{64}", reference)
            if not isinstance(reference, str) or not (reference.startswith("receipts/") or legacy_receipt):
                continue
            if legacy_receipt:
                reference = reference.removeprefix("receipt:")
            source_reference(reference, "receipts")
            digest = reference
            if digest not in memberships or receipt_documents is None or digest not in receipt_documents:
                raise ValueError("execution receipt body or membership is missing")
            observed.setdefault(digest, set()).add(case)
            owner[key] = copy.deepcopy(receipt_documents[digest])
    for digest, members in memberships.items():
        source_reference(digest, "receipts")
        if not isinstance(members, list) or not members or any(not isinstance(case, str) for case in members) or len(members) != len(set(members)) or set(members) != observed.get(digest, set()):
            raise ValueError("execution receipt case membership mismatch")
    return {**copy.deepcopy(checkpoint), "metadata_changes": metadata,
            **({"execution_proof": proof} if proof is not None else {})}
