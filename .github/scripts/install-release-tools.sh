#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the pinned release-plz used by the release and preview jobs.
set -euo pipefail

RELEASE_PLZ_VERSION=0.3.158

curl -sSLf "https://github.com/release-plz/release-plz/releases/download/release-plz-v$RELEASE_PLZ_VERSION/release-plz-x86_64-unknown-linux-gnu.tar.gz" \
  | tar xz -C /usr/local/bin release-plz
release-plz --version
