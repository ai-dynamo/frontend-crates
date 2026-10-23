# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from capture_execution import validate_adapter_bundle, validate_execution


@pytest.mark.parametrize("expected,actual", [(True, 1), (1, True), (0.0, -0.0), (-0.0, 0.0)])
def test_rehashed_capture_cannot_change_request_value_types(expected, actual):
    job = {"cases": [{"id": "case", "request": {"tools": [{"value": expected}]}}]}
    job_bytes = json.dumps(job).encode()
    results = {"case": {"capture_input": {"tools": [{"value": actual}]}, "chunks": []}}
    document = {"vllm_version": "1.0.0", "results": results, "provenance": {
        "implementation": "vllm_python", "version": "1.0.0",
        "source_sha256": "a" * 64, "adapter_sha256": "b" * 64,
        "job_sha256": hashlib.sha256(job_bytes).hexdigest(),
        "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }}
    with pytest.raises(ValueError, match="request mismatch"):
        validate_execution(document, job, job_bytes, "vllm_python", "1.0.0")


@pytest.mark.parametrize("corruption", [None, "empty", "traversal", "symlink", "bytes", "support"])
def test_adapter_bundle_validates_ordered_bytes_and_support(tmp_path, corruption):
    (tmp_path / "probe").write_bytes(b"probe")
    (tmp_path / "helper").write_bytes(b"helper")
    manifest = {"files": [{"path": "probe", "sha256": hashlib.sha256(b"probe").hexdigest()}],
                "support_files": [{"path": "helper", "sha256": hashlib.sha256(b"helper").hexdigest()}]}
    if corruption == "empty":
        manifest["files"] = []
    elif corruption == "traversal":
        manifest["files"][0]["path"] = "../probe"
    elif corruption == "symlink":
        (tmp_path / "probe").unlink()
        (tmp_path / "probe").symlink_to(tmp_path / "helper")
    elif corruption == "bytes":
        (tmp_path / "probe").write_bytes(b"altered")
    elif corruption == "support":
        (tmp_path / "helper").write_bytes(b"altered")
    if corruption:
        with pytest.raises(ValueError, match="adapter"):
            validate_adapter_bundle(tmp_path, manifest)
    else:
        assert validate_adapter_bundle(tmp_path, manifest) == hashlib.sha256(b"probe").hexdigest()
