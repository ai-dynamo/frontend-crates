# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen producer identities never become new capture filenames."""
import pytest
from capture_versions import UNIFIED_VERSION_RE, provenance_release_version


def test_frozen_source_identity_compares_as_release_without_becoming_capture_version():
    qualified = '0.7.25+source.' + 'a' * 64
    assert provenance_release_version(qualified) == '0.7.25'
    assert not UNIFIED_VERSION_RE.fullmatch(qualified)


@pytest.mark.parametrize('version', ['0.7.25+source.a', '0.7.25+other.abc', '0.7.25+source./bad'])
def test_malformed_frozen_source_identity_is_rejected(version):
    with pytest.raises(ValueError):
        provenance_release_version(version)
