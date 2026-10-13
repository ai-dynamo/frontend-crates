# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The maintained JavaScript engine owns these reader contracts."""
from conformance.utils.tests.viewer_test_support import run_node_contract


def test_status_reader_contracts():
    output = run_node_contract("retirement_contract.test.js", "--suite", "status")
    assert "status contracts passed" in output
