# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python3 conformance/utils/src/parser_roundtrip --trace T --model-dir D
--family F --output-dir O [--parser-bin B]

Writes results.jsonl, summary.json, summary.md and the parser's stderr, each
mode 0600, and prints only the counts. Results hold model text, so the output
directory must be outside any Git work tree or an ignored path under its logs/
directory. Needs transformers and tokenizers, which conformance CI does not
install."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if not __package__:
    # Run as a directory: import the package, so trace.py cannot shadow the
    # standard library's trace module.
    sys.path[0] = str(Path(__file__).resolve().parents[1])

from parser_roundtrip.bundle import Bundle  # noqa: E402
from parser_roundtrip.judge import (  # noqa: E402
    PARSER_COMMAND,
    markdown,
    open_private,
    run,
    summarize,
)
from parser_roundtrip.trace import load  # noqa: E402

RESULTS = "results.jsonl"
PARSER_STDERR = "parser.stderr"
OUTPUTS = (RESULTS, "summary.json", "summary.md", PARSER_STDERR)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 conformance/utils/src/parser_roundtrip"
    )
    parser.add_argument("--trace", type=Path, required=True, help="canonical JSONL")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--family", required=True, help="Unified parser family, e.g. deepseek_v4"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parser-bin",
        type=Path,
        help="prebuilt parser_roundtrip example; default: cargo run",
    )
    return parser.parse_args(argv)


def git(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True
    )


def private_dir(path: Path) -> Path:
    """Create the output directory with mode 0700, refusing any place in a Git
    work tree other than an ignored path under its top-level logs/."""
    path = path.resolve()
    existing = next(parent for parent in (path, *path.parents) if parent.is_dir())
    top = git(existing, "rev-parse", "--show-toplevel")
    if top.returncode == 0:
        logs = Path(top.stdout.strip()).resolve() / "logs"
        ignored = all(
            git(existing, "check-ignore", "-q", "--", str(path / name)).returncode == 0
            for name in OUTPUTS
        )
        if not path.is_relative_to(logs) or not ignored:
            raise SystemExit("--output-dir must be outside Git or ignored under logs/")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)  # mkdir leaves an existing directory's mode alone
    return path


def write_private(path: Path, text: str) -> None:
    with open_private(path) as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = private_dir(args.output_dir)
    command = [str(args.parser_bin.resolve())] if args.parser_bin else PARSER_COMMAND
    header, results = run(
        load(args.trace),
        Bundle.load(args.model_dir),
        args.family,
        output / PARSER_STDERR,
        command,
    )
    summary = summarize(
        header,
        results,
        {
            "trace": str(args.trace),
            "model_dir": str(args.model_dir),
            "family": args.family,
        },
    )
    lines = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results)
    write_private(output / RESULTS, lines)
    write_private(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    write_private(output / "summary.md", markdown(summary))
    counts = {key: summary[key] for key in ("counts", "notes", "incomplete_calls")}
    print(json.dumps({"output_dir": str(output), **counts}))
    return 1 if summary["counts"].get("fail") else 0


if __name__ == "__main__":
    sys.exit(main())
