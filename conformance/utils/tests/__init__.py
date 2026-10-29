# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

MODEL_CONTRACTS = json.loads((Path(__file__).parent / "fixtures/model_contracts.json").read_text())
GPT_OSS_STANDALONE_HEADER_SCENARIOS = frozenset(MODEL_CONTRACTS["gptOssStandaloneHeaderScenarios"])
