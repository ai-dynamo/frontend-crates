<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Former archive store

The active conformance fixtures now live in two reviewable YAML stores: [`fixtures-v1/`](../fixtures-v1/) for batch, legacy streaming, batch-on-stream, and reasoning, and [`fixtures-unified-v2/`](../fixtures-unified-v2/) for Unified. `conformance/fixtures-manifest.json` pins both stores by digest. This directory contains no active fixture archives.

`extract_fixtures.py` materializes the YAML into the same loose paths consumed by the Rust tests and HTML/JSON renderer. Each legacy family has an `inputs_and_golden.yaml` file and separate implementation/version capture files. The lowest recorded version remains the full anchor for batch and streaming; later files remain changed-only overlays. Batch-on-stream headers select each document's captures and preserve the original implementation order. Capture provenance retains the exact `captured_with` label and any recorded commit SHA; a label without a version uses an `unversioned` filename and a null `runtime_version`.

New captures go through `conformance/utils/src/package_fixtures.py`, which updates the affected YAML files and manifest pin. See the [fixture workflow](../README.md#fixture-workflows) for capture and render commands.
