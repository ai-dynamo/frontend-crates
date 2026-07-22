#!/usr/bin/env python3
"""Inventory Dynamo-owned protocol types against two async-openai sources."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TYPE_RE = re.compile(
    r"(?m)^[ \t]*pub[ \t]+(struct|enum|type)[ \t]+([A-Za-z_][A-Za-z0-9_]*)"
)
RATIONALE_RE = re.compile(
    r"async[- ]openai|upstream|pinned|workaround|unsupported|missing|"
    r"does not|doesn't|requires|required|optional|relax|override|vendor",
    re.IGNORECASE,
)
UPSTREAM_COMPONENTS = {
    "chat", "completions", "embeddings", "images",
    "realtime", "responses", "shared",
}


@dataclass(frozen=True)
class Definition:
    name: str
    kind: str
    component: str
    file: str
    line: int
    normalized: str
    rationale: str | None


def types_root(path: Path) -> Path:
    path = path.resolve()
    if path.name == "types" and path.is_dir():
        return path
    candidate = path / "src" / "types"
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"cannot find src/types under {path}")


def rust_files(
    root: Path, exclude_anthropic: bool = False, upstream_only: bool = False
) -> Iterable[Path]:
    if upstream_only:
        candidates = []
        for component in UPSTREAM_COMPONENTS:
            path = root / component
            if path.is_dir():
                candidates.extend(path.rglob("*.rs"))
            elif path.with_suffix(".rs").is_file():
                candidates.append(path.with_suffix(".rs"))
        if (root / "stream.rs").is_file():
            candidates.append(root / "stream.rs")
    else:
        candidates = list(root.rglob("*.rs"))

    for path in sorted(set(candidates)):
        if exclude_anthropic and path.name == "anthropic.rs":
            continue
        yield path


def item_end(text: str, match_end: int, kind: str) -> int:
    if kind == "type":
        end = text.find(";", match_end)
        return len(text) if end < 0 else end + 1

    brace = text.find("{", match_end)
    if brace < 0:
        return match_end

    depth = 0
    block_depth = 0
    in_string = False
    escaped = False
    i = brace
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if block_depth:
            if ch == "/" and nxt == "*":
                block_depth += 1
                i += 2
                continue
            if ch == "*" and nxt == "/":
                block_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            newline = text.find("\n", i + 2)
            i = len(text) if newline < 0 else newline + 1
            continue
        if ch == "/" and nxt == "*":
            block_depth = 1
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1

    return len(text)


def normalize(body: str) -> str:
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.DOTALL)
    body = re.sub(r"//.*?$", " ", body, flags=re.MULTILINE)
    return " ".join(body.split())


def rationale_for(lines: list[str], declaration_line: int) -> str | None:
    context = lines[max(0, declaration_line - 18):declaration_line]
    selected = []
    for line in context:
        stripped = line.strip().lstrip("/").strip()
        if stripped and RATIONALE_RE.search(stripped):
            selected.append(stripped)
    if not selected:
        return None
    return " ".join(selected)[:500]


def collect(
    root: Path, display_root: Path, exclude_anthropic: bool = False,
    upstream_only: bool = False,
) -> dict[str, list[Definition]]:
    result: dict[str, list[Definition]] = {}
    for path in rust_files(
        root, exclude_anthropic=exclude_anthropic, upstream_only=upstream_only
    ):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for match in TYPE_RE.finditer(text):
            kind, name = match.groups()
            end = item_end(text, match.end(), kind)
            body = text[match.start():end]
            line = text.count("\n", 0, match.start()) + 1
            try:
                shown_path = str(path.relative_to(display_root))
            except ValueError:
                shown_path = str(path)
            component = Path(path.relative_to(root).parts[0]).stem
            if component == "completion":
                component = "completions"
            definition = Definition(
                name=name,
                kind=kind,
                component=component,
                file=shown_path,
                line=line,
                normalized=normalize(body),
                rationale=rationale_for(lines, line - 1),
            )
            result.setdefault(name, []).append(definition)
    return result


def body_set(items: list[Definition] | None) -> set[str]:
    return {item.normalized for item in items or []}


def locations(items: list[Definition] | None) -> list[str]:
    return [f"{item.file}:{item.line}" for item in items or []]


def build_report(repo: Path, old_source: Path, new_source: Path) -> dict[str, object]:
    local_root = repo / "protocols" / "src" / "types"
    if not local_root.is_dir():
        raise SystemExit(f"cannot find {local_root}")

    local = collect(local_root, repo, exclude_anthropic=True)
    old = collect(types_root(old_source), old_source, upstream_only=True)
    new = collect(types_root(new_source), new_source, upstream_only=True)

    overlaps = []
    for name in sorted(local):
        components = {item.component for item in local[name]}
        allowed_components = components | {"shared"}
        old_items = [
            item for item in old.get(name, []) if item.component in allowed_components
        ]
        new_items = [
            item for item in new.get(name, []) if item.component in allowed_components
        ]
        if not new_items:
            continue
        old_bodies = body_set(old_items)
        new_bodies = body_set(new_items)
        local_bodies = body_set(local.get(name))
        if not old_items:
            status = "new upstream name"
        elif old_bodies != new_bodies:
            status = "upstream definition changed"
        else:
            status = "upstream definition unchanged"
        overlaps.append(
            {
                "name": name,
                "status": status,
                "exact_local_match": bool(local_bodies & new_bodies),
                "local": locations(local.get(name)),
                "old_upstream": locations(old_items),
                "new_upstream": locations(new_items),
            }
        )

    rationale = []
    for name, definitions in sorted(local.items()):
        for definition in definitions:
            if definition.rationale:
                rationale.append(
                    {
                        "name": name,
                        "location": f"{definition.file}:{definition.line}",
                        "text": definition.rationale,
                    }
                )

    return {
        "repo": str(repo),
        "old_source": str(old_source.resolve()),
        "new_source": str(new_source.resolve()),
        "summary": {
            "local_public_definitions": sum(len(v) for v in local.values()),
            "local_upstream_overlaps": len(overlaps),
            "newly_available_local_names": sum(
                item["status"] == "new upstream name" for item in overlaps
            ),
            "changed_upstream_overlaps": sum(
                item["status"] == "upstream definition changed" for item in overlaps
            ),
            "exact_local_matches": sum(
                item["exact_local_match"] for item in overlaps
            ),
        },
        "overlaps": overlaps,
        "rationale_markers": rationale,
    }


def print_markdown(report: dict[str, object]) -> None:
    summary = report["summary"]
    print("# async-openai upgrade inventory")
    print()
    print(f"- Repository: `{report['repo']}`")
    print(f"- Old source: `{report['old_source']}`")
    print(f"- New source: `{report['new_source']}`")
    print(
        "- Summary: "
        f"{summary['local_public_definitions']} local definitions, "
        f"{summary['local_upstream_overlaps']} upstream name overlaps, "
        f"{summary['changed_upstream_overlaps']} changed upstream overlaps, "
        f"{summary['newly_available_local_names']} newly available names, "
        f"{summary['exact_local_matches']} exact declaration matches"
    )
    print()
    print("## Local/upstream overlaps")
    print()
    print("| Name | Upstream status | Exact declaration | Local location |")
    print("|---|---|---:|---|")
    for item in report["overlaps"]:
        exact = "yes" if item["exact_local_match"] else "no"
        local = "<br>".join(f"`{value}`" for value in item["local"])
        print(f"| `{item['name']}` | {item['status']} | {exact} | {local} |")
    print()
    print("## Ownership rationale markers")
    print()
    if not report["rationale_markers"]:
        print("None found.")
    for item in report["rationale_markers"]:
        print(f"- `{item['name']}` at `{item['location']}`: {item['text']}")
    print()
    print(
        "This inventory is not a removability verdict. Compare Serde behavior, "
        "wire-shape tests, public API compatibility, and downstream consumers."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory Dynamo-owned types across an async-openai upgrade."
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(args.repo.resolve(), args.old_source, args.new_source)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_markdown(report)


if __name__ == "__main__":
    main()
