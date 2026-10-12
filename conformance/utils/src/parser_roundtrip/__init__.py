# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parser round trip for the v2 Unified parser: decode recorded output tokens
into x, parse x into ordered events with the `parser_roundtrip` example, fold
them into one assistant message, render it with the model's chat template, and
require the rendered turn to reproduce x.

Unlike the rest of conformance/, the CLI reads a recorded trace and a model's
tokenizer and chat template from outside the repository. Its unit tests build
a toy model in a temporary directory, so they stay hermetic."""
