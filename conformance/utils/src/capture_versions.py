# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared capture version grammar and ordering; safe before CI dependency setup."""
import re


UNIFIED_VERSION_PATTERN = r"\d+(?:\.\d+){2,}(?:\.post\d+|-[0-9A-Za-z.-]+)?"
UNIFIED_VERSION_RE = re.compile(UNIFIED_VERSION_PATTERN)


def capture_release_sort_key(runtime_version: str) -> tuple:
    if not isinstance(runtime_version, str) or not UNIFIED_VERSION_RE.fullmatch(runtime_version):
        raise ValueError(f"invalid Unified capture version: {runtime_version}")
    release_version = runtime_version
    release, separator, prerelease = release_version.partition("-")
    if ".post" in release:
        release, post = release.rsplit(".post", 1)
        return tuple(int(part) for part in release.split(".")), 2, int(post)
    numeric = tuple(int(part) for part in release.split("."))
    return numeric, 0 if separator else 1, prerelease


def provenance_release_version(version: str) -> str:
    """Retain a frozen source identity while comparing its underlying release."""
    if not isinstance(version, str):
        raise ValueError(f"invalid provenance version: {version}")
    match = re.fullmatch(rf"({UNIFIED_VERSION_PATTERN})(?:\+source\.[0-9a-f]{{64}})?", version)
    if match is None:
        raise ValueError(f"invalid provenance version: {version}")
    return match[1]
