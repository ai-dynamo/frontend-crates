# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render PARITY.md from the committed `<model>.json` tables.

    python multimodal/tests/fixtures/mm_parity/report.py [--check]

`--check` writes nothing and exits 1 when PARITY.md differs.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRATE = "dynamo-multimodal"
SHORT = {
    "input_ids_sha256": "ids",
    "num_input_ids": "ids",
    "offsets": "offsets",
    "item_offsets": "offsets",
    "image_grid_thw": "grid",
    "image_sizes": "grid",
    "feature_shapes": "shapes",
    "feature_dtypes": "features",
    "feature_sha256": "features",
    "mrope_sha256": "mrope",
    "mrope_delta": "mrope",
    "error": "accepts",
}
LEGEND = """\
# Multimodal parity

`report.py` writes this file from the `<model>.json` tables. Do not edit it.

Each table's baseline is the checkpoint's own Hugging Face processor on the
CPU (`hf`). A cell compares one framework with the baseline on every field
the table records: `=` equal, `≠` differs on the named fields, `omits` the
framework does not give the named fields (it is equal on the others unless
`≠` names them), `rejects` the framework refuses the input, `n/a` the
framework cannot run the case, `gap` a known gap of this crate (README.md).
The crate is gated on the column marked *target*. Fields: `ids` the expanded
prompt, `offsets` the feature spans, `grid` the patch grid, `shapes` and
`features` the pixel values, `mrope` the M-RoPE positions and delta,
`accepts` a case the baseline refuses.
"""


def short(fields):
    return ", ".join(dict.fromkeys(SHORT.get(field, field) for field in fields))


def mark(cell, case, name):
    if name == CRATE and "known_gap" in case:
        return "gap"
    marks = [f"≠ {short(cell['differs'])}"] if "differs" in cell else []
    marks += [f"omits {short(cell['omits'])}"] if "omits" in cell else []
    if marks:
        return "; ".join(marks)
    return "=" if "=" in cell else "rejects" if "rejects" in cell else "n/a"


def matches_target(case, target):
    """The replay's rule: the crate's cell is the target's, or both refuse and
    the crate refuses with `rust_error`."""
    if "known_gap" in case:
        return False
    if target != "hf":
        cell = case["frameworks"][target]
    else:
        cell = {"rejects": case["error"]} if "error" in case else {"=": True}
    crate = case["frameworks"][CRATE]
    if "rejects" in cell:
        return crate.get("rejects") == case.get("rust_error")
    # The crate is gated on the baseline's values of the fields the target omits.
    return crate == ({key: value for key, value in cell.items() if key != "omits"} or {"=": True})


def summary(name, table):
    cells = [case["frameworks"][name] for case in table["cases"]]
    omit = sum("omits" in cell and "differs" not in cell for cell in cells)
    counts = [sum(key in cell for cell in cells) for key in ("=", "differs", "rejects", "n/a")]
    line = f"- `{name}`: {counts[0]} of {len(cells)} cases equal to HF ({counts[1]} differ, "
    line += f"{omit} equal but omit fields, {counts[2]} reject, {counts[3]} n/a)"
    if name == CRATE:
        target = table["target"]
        same = sum(matches_target(case, target) for case in table["cases"])
        gaps = sum("known_gap" in case for case in table["cases"])
        line += f"; {same} equal to the target `{target}`, {gaps} known gap"
    return line + "."


def section(table):
    names = list(table["cases"][0]["frameworks"])
    baseline, target = table["baseline"], table["target"]
    lines = [
        f"## {table['model']}",
        "",
        f"Revision `{table['revision']}`. Baseline: `{baseline['processor']}`, transformers "
        f"{table['frameworks']['hf']['version']}, `{baseline['backend']}` backend, "
        f"`{baseline['decode']}`. Target: `{target}`"
        + (f": {table['target_reason']}" if "target_reason" in table else "."),
        "",
        "| framework | version | libraries | source | omits |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, column in table["frameworks"].items():
        omits = ", ".join(column.get("omits", [])) or "-"
        libraries = ", ".join(f"{k} {v}" for k, v in column.get("versions", {}).items()) or "-"
        lines.append(
            f"| `{name}` | {column['version']} | {libraries} | {column['source']} | {omits} |"
        )
    lines += ["", *(summary(name, table) for name in names), ""]
    heads = [f"{name} (target)" if name == target else name for name in names]
    lines += ["| case | " + " | ".join(heads) + " |", "| --- |" + " --- |" * len(names)]
    for case in table["cases"]:
        marks = [mark(case["frameworks"][name], case, name) for name in names]
        lines.append(f"| {case['name']} | " + " | ".join(marks) + " |")
    return "\n".join(lines) + "\n"


def render():
    tables = sorted(path for path in HERE.glob("*.json") if path.name != "decode.json")
    return LEGEND + "".join("\n" + section(json.loads(path.read_text())) for path in tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    args = parser.parse_args()
    out, text = HERE / "PARITY.md", render()
    if not args.check:
        out.write_text(text)
        print(f"wrote {out}")
    elif not out.exists() or out.read_text() != text:
        sys.exit("PARITY.md differs from the tables: run report.py")
    else:
        print("PARITY.md matches the tables")


if __name__ == "__main__":
    main()
