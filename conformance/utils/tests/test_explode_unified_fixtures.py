# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from explode_unified_fixtures import _peer_cell  # noqa: E402


def test_peer_error_wins_over_partial_output() -> None:
    result = {
        "error": "UnifiedParserError::ParsingFailed",
        "assembled": [{"kind": "reasoning", "text": "partial"}],
        "chunks": [[{"kind": "reasoning", "text": "partial"}]],
    }

    assert _peer_cell(result) == {"error": "UnifiedParserError::ParsingFailed"}
