# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from conformance.utils.tests.viewer_test_support import run_node_contract


def test_shared_node_status_contract():
    assert "status schemas" in run_node_contract("status.test.js")
