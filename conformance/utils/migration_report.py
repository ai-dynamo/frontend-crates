#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export every report cell without dropping candidate labels or tooltip fields.

PYTHONPATH selects the report revision's own hydration and status readers.
"""

import argparse
from functools import cache
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import yaml

from model import hydrate_page
from validate_conformance_status import build_status


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def keyed(items, key):
    return unique_object((item[key], item) for item in items)


@cache
def semantic_hash(path):
    value = yaml.safe_load(path.read_text())
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verified_link(href, html, root, *, fixture=False):
    parsed = urlsplit(href)
    if parsed.scheme not in {"", "file"} or parsed.netloc or not parsed.path:
        raise ValueError(f"expected local report link: {href}")
    target = html.parent / unquote(parsed.path)
    if not target.is_file():
        raise ValueError(f"broken report link: {href}")
    record = {"path": str(target.resolve().relative_to(root.resolve())),
              "sha256": semantic_hash(target) if fixture else hashlib.sha256(target.read_bytes()).hexdigest(),
              "fragment": parsed.fragment}
    if parsed.query:
        record["query"] = parsed.query
    return record


def source_links(value, html, source, path=()):
    if isinstance(value, dict):
        return {key: source_links(item, html, source, (*path, key)) for key, item in value.items()}
    if isinstance(value, list):
        return [source_links(item, html, source, (*path, index)) for index, item in enumerate(value)]
    if not isinstance(value, str):
        return value
    # Only these model owners contain presentation links. A parser can emit an
    # identical-looking href as literal text or an argument; those bytes matter.
    presentation = (
        len(path) == 2 and path[1] in {"legend_html", "toolbar_desc_html", "details_note_html"}
        or len(path) == 3 and path[1:] == ("parser", "html")
        or len(path) == 5 and path[1:3] == ("tooltip", "dynamo_notes") and path[4] == 1
    )
    if not presentation:
        return value

    def replace(match):
        href = match[1]
        parsed = urlsplit(href)
        if parsed.scheme not in {"", "file"} or parsed.netloc or not parsed.path:
            return match[0]
        target = html.parent / unquote(parsed.path)
        relative = target.resolve().relative_to(source.resolve())
        canonical = urlunsplit(("", "", quote(str(relative), safe="/"), parsed.query, parsed.fragment))
        return f'href="{canonical}"'

    return re.sub(r'href="([^"]+)"', replace, value)


def report_snapshot(html, status, fixtures, source):
    text = html.read_text()
    matches = re.findall(r'<script type="application/json" id="conformance-model">(.*?)</script>', text, re.S)
    if len(matches) != 1:
        raise ValueError("expected exactly one report model")
    page = hydrate_page(json.loads(matches[0], object_pairs_hook=unique_object))
    supplied = json.loads(status.read_text(), object_pairs_hook=unique_object)
    calculated = build_status(page, page["tabs"], [], html)
    if calculated != supplied:
        raise ValueError("HTML/JSON disagreement")
    links = {}
    records = {}
    # These are the only representation exclusions: interned strings and the
    # tables used to expand candidates/links have already been hydrated above.
    records["page"] = {k: v for k, v in page.items() if k not in {"tabs", "strings", "meta"}}
    records["meta"] = {k: v for k, v in page["meta"].items() if k not in {"stamp", "sha", "short_sha", "output"}}
    tabs = keyed(page["tabs"], "id")
    for tab_id, tab in tabs.items():
        tab = dict(tab)
        rows = tab.pop("rows")
        tab.pop("cand_meta", None)
        tab.pop("fixture_href_base", None)
        if "candidates" in tab:
            tab["candidates"] = keyed(tab["candidates"], "key")
        if "case_docs_href" in tab:
            tab["case_docs_href"] = verified_link(tab["case_docs_href"], html, source)
        records[tab_id] = tab
        seen = set()
        for index, row in enumerate(rows):
            row = dict(row)
            cells = row.pop("cells")
            family = row.get("family")
            identity = f"{family}/{row['model_label']}" if family else f"section-{index}"
            if identity in seen:
                raise ValueError(f"duplicate row: {tab_id}/{identity}")
            seen.add(identity)
            records[f"{tab_id}/{identity}"] = row
            for scenario, cell in cells.items():
                cell = dict(cell)
                href = cell.get("fixture_href")
                if href:
                    cell["fixture_href"] = verified_link(href, html, fixtures, fixture=True)
                    links[cell["fixture_href"]["path"]] = cell["fixture_href"]["sha256"]
                if cell.get("tooltip"):
                    cell["tooltip"]["candidates"] = keyed(cell["tooltip"]["candidates"], "key")
                records[f"{tab_id}/{identity}/{scenario}"] = cell
    records["status"] = supplied["reports"]
    records = source_links(records, html, source)
    return {"inventory": sorted(tabs), "records": records, "links": links,
            "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest(),
            "json_sha256": hashlib.sha256(status.read_bytes()).hexdigest(), "producer_meta": page["meta"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = report_snapshot(args.html, args.json, args.fixtures, args.source)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps({"records": len(result["records"]), "verified_links": len(result["links"])}))


if __name__ == "__main__":
    main()
