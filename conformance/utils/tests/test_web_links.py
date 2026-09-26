# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Link-target guards for local reports and GitHub Pages publication."""

import sys
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import pytest

UTILS = Path(__file__).resolve().parents[1]
REPO = UTILS.parents[1]
if str(UTILS / "src") not in sys.path:
    sys.path.insert(0, str(UTILS / "src"))

from tables import common  # noqa: E402

OUTPUT = REPO / "conformance" / "CONFORMANCE_v2.html"
REVISION = "A" * 40


@pytest.fixture(autouse=True)
def restore_local_link_context():
    yield
    common.set_links(OUTPUT, REPO)


def test_local_links_keep_destination_relative_behavior(monkeypatch):
    links = common.set_links(OUTPUT, REPO)
    monkeypatch.setattr(common, "_fixtures_cache_root", lambda: str(REPO.parent / "fixture-cache"))

    assert links["toolcalling_src"] == "../parsers/v1/src/tool_calling/"
    assert links["toolcalling_cases"] == "utils/lib/parsers/TOOLCALLING_CASES.md"
    assert common.repository_href("parsers/v1/src/tool_calling/config.rs") == (
        "../parsers/v1/src/tool_calling/config.rs"
    )
    assert common.fixture_href("deepseek_v4/TOOLCALLING.batch.1.yaml") == (
        "../../fixture-cache/toolcalling/fixtures-batch-v1/inputs/"
        "deepseek_v4/TOOLCALLING.batch.1.yaml"
    )


@pytest.mark.parametrize("filename", ["TOOLCALLING.batch.1.yaml", "TOOLCALLING.batch.1 #2.yaml"])
def test_local_fixture_link_resolves_from_file_and_http_report(tmp_path, monkeypatch, filename):
    repo = tmp_path / "repo"
    output = repo / "conformance/report.html"
    cache = tmp_path / "fixture-cache"
    target = cache / "toolcalling/fixtures-batch-v1/inputs/deepseek_v4" / filename
    target.parent.mkdir(parents=True)
    target.write_text("cases: {}\n")
    common.set_links(output, repo)
    monkeypatch.setattr(common, "_fixtures_cache_root", lambda: str(cache))

    href = common.fixture_href(f"deepseek_v4/{filename}")

    assert urlsplit(href).scheme == ""
    local = urlsplit(urljoin(output.as_uri(), href))
    assert local.scheme == "file"
    assert Path(unquote(local.path)) == target
    served = urlsplit(urljoin("http://example.test/repo/conformance/report.html", href))
    assert served.scheme == "http"
    assert served.netloc == "example.test"
    assert served.fragment == ""
    assert (tmp_path / unquote(served.path).lstrip("/")).read_bytes() == target.read_bytes()


def test_fixture_links_without_render_context_keep_file_fallback(monkeypatch):
    monkeypatch.setattr(common, "_LINK_CONTEXT", None)
    monkeypatch.setattr(common, "_fixtures_cache_root", lambda: "/fixture-cache")
    assert common.fixture_href("deepseek_v4/TOOLCALLING.batch.1.yaml") == (
        "file:///fixture-cache/toolcalling/fixtures-batch-v1/inputs/deepseek_v4/TOOLCALLING.batch.1.yaml"
    )
    assert common.fixture_href("") == ""
    assert common.fixture_href("https://example.test/case.yaml") == "https://example.test/case.yaml"


def test_web_links_pin_sources_and_publish_fixture_inputs():
    links = common.set_links(
        OUTPUT,
        REPO,
        github_repository="ai-dynamo/frontend-crates",
        github_revision=REVISION,
        fixture_base_url="https://ai-dynamo.github.io/frontend-crates/fixtures",
    )
    revision = REVISION.lower()

    assert links["toolcalling_src"] == (
        f"https://github.com/ai-dynamo/frontend-crates/tree/{revision}/"
        "parsers/v1/src/tool_calling/"
    )
    assert common.repository_href("parsers/v1/src/tool_calling/config.rs") == (
        f"https://github.com/ai-dynamo/frontend-crates/blob/{revision}/"
        "parsers/v1/src/tool_calling/config.rs"
    )
    assert links["toolcalling_cases"] == (
        f"https://github.com/ai-dynamo/frontend-crates/blob/{revision}/"
        "conformance/utils/lib/parsers/TOOLCALLING_CASES.md"
    )
    assert links["toolcalling_fixture_store"] == (
        f"https://github.com/ai-dynamo/frontend-crates/tree/{revision}/"
        "conformance/fixtures-v1/batch/"
    )
    assert common.fixture_href("deepseek_v4/TOOLCALLING.streamv1.1.yaml") == (
        "https://ai-dynamo.github.io/frontend-crates/fixtures/toolcalling/"
        "fixtures-stream-v1/inputs/deepseek_v4/TOOLCALLING.streamv1.1.yaml"
    )
    assert common.fixture_href("deepseek_r1/REASONING.batch.1.yaml") == (
        "https://ai-dynamo.github.io/frontend-crates/fixtures/reasoning/"
        "fixtures-v1/inputs/deepseek_r1/REASONING.batch.1.yaml"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"github_repository": "ai-dynamo/frontend-crates"},
        {
            "github_repository": "ai-dynamo/frontend-crates/extra",
            "github_revision": "a" * 40,
            "fixture_base_url": "https://example.test/fixtures/",
        },
        {
            "github_repository": "ai-dynamo/frontend-crates",
            "github_revision": "short",
            "fixture_base_url": "https://example.test/fixtures/",
        },
        {
            "github_repository": "ai-dynamo/frontend-crates",
            "github_revision": "a" * 40,
            "fixture_base_url": "http://example.test/fixtures/",
        },
        {
            "github_repository": "ai-dynamo/frontend-crates",
            "github_revision": "a" * 40,
            "fixture_base_url": "https://example.test/fixtures/?snapshot=1",
        },
    ],
)
def test_web_link_configuration_rejects_partial_or_mutable_targets(kwargs):
    with pytest.raises(ValueError):
        common.set_links(OUTPUT, REPO, **kwargs)
