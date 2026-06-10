#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Thin wrapper over capture_cli.py (audit B3): one argparse CLI for capturing parser
# behavior to refresh v2 fixtures. Subcommands: stream | batch-on-stream |
# dynamo-stream | dynamo-batch-on-stream | token-ids. Run `capture.sh --help` for
# options. `CARGO` env overrides the cargo binary; `--dry-run` prints commands.
exec python3 "$(dirname "$0")/src/capture_cli.py" "$@"
