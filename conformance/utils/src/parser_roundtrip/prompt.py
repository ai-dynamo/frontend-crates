# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generation prompt the model continued: the request rendered by the
model's chat template with the generation prompt added. Its tail decides where
the output starts, so the parser's starting state and the model turn both
depend on it."""

from __future__ import annotations

from .bundle import Bundle
from .trace import Case


def generation_prompt(bundle: Bundle, case: Case) -> str:
    """The request's conversation rendered up to the open assistant turn."""
    return bundle.render_request(case.request, True)
