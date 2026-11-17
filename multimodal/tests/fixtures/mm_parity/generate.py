# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang `rust/sglang-processor/tests/scripts/generate_mm_parity.py`
# (Apache-2.0).
"""Record one family's parity table, `<model>.json`: the HF baseline and one cell per framework.

Usage, from the reference environment that README.md names:

    python multimodal/tests/fixtures/mm_parity/generate.py --family glm5_next \
        --model-dir <checkpoint files> --vllm-python <python> --llm-mm-harness <binary> [--check]
    python multimodal/tests/fixtures/mm_parity/generate.py --family glm5_next --crate [--check]

A family is a module in families/; a framework is a module in frameworks/.
The baseline, `hf`, writes each case's `expected`; every other framework gets
a cell against it. `--crate` fills only the `dynamo-multimodal` column, from
tests/mm_parity.rs. `--check` writes nothing and exits 1 when the table differs.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from frameworks import sha256, versions

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
# Column name: module in frameworks/.
FRAMEWORKS = {"hf": "hf", "sglang": "sglang", "vllm": "vllm", "llm-multimodal": "llm_multimodal"}
CRATE = "dynamo-multimodal"
FIELDS = ("offsets", "image_grid_thw", "feature_shapes", "feature_sha256")
FIELDS += ("mrope_sha256", "mrope_delta")


def fields(output, id_map, names):
    """A framework's output as table fields: ids mapped back to the case's, then hashed."""
    table = {}
    if "input_ids" in output:
        back = {checkpoint: case for case, checkpoint in id_map.items()}
        ids = [back.get(id_, id_) for id_ in output["input_ids"]]
        table = {"input_ids_sha256": sha256(ids, "<i4"), "num_input_ids": len(ids)}
    return {**table, **{field: output[field] for field in names if field in output}}


def cell(result, row, id_map, names):
    """`=`, or the fields that differ from `expected` and the fields the framework
    omits; or the framework's `rejects`/`n/a`."""
    if "output" not in result:
        return result
    actual = fields(result["output"], id_map, names)
    if "error" in row:
        return {"differs": ["error"], **actual}
    expected = row["expected"]
    omits = sorted(field for field in expected if field not in actual)
    differs = sorted(
        field for field in expected if actual.get(field, expected[field]) != expected[field]
    )
    out = {"differs": differs, **{field: actual[field] for field in differs}} if differs else {}
    if omits:
        out["omits"] = omits
    return out or {"=": True}


def header(model_dir):
    """`model` and `revision` of a checkpoint that `hf download --local-dir`
    (`<org>__<repo>`) or the hub cache (`models--<org>--<repo>/snapshots/<sha>`) holds."""
    metadata = model_dir / ".cache/huggingface/download/config.json.metadata"
    if metadata.exists():
        return {
            "model": model_dir.name.replace("__", "/"),
            "revision": metadata.read_text().split()[0],
        }
    repo = model_dir.parent.parent.name.removeprefix("models--").replace("--", "/")
    return {"model": repo, "revision": model_dir.name}


def record(args, family, committed):
    model_dir = args.model_dir.resolve()
    names = getattr(family, "FIELDS", FIELDS)
    cases = list(family.cases(model_dir))
    request = {
        "family": args.family,
        "model_dir": str(model_dir),
        "images_dir": str(IMAGES),
        "cases": [
            {
                "name": case["name"],
                "images": case["images"],
                "input_ids": [family.ID_MAP.get(id_, id_) for id_ in case["input_ids"]],
                **family.request(case),
            }
            for case in cases
        ],
    }
    # A family gives its own `run` for a column whose module cannot run its processor.
    adapters = getattr(family, "ADAPTERS", {})
    responses = {
        name: adapters.get(name, importlib.import_module(f"frameworks.{module}").run)(request, args)
        for name, module in FRAMEWORKS.items()
        if name in args.frameworks
    }
    baseline = responses["hf"]
    committed = committed or {"frameworks": {}, "cases": []}
    environment = versions()
    columns = {}
    for name in [*FRAMEWORKS, CRATE]:
        if name in responses:
            response = responses[name]
            columns[name] = {key: response[key] for key in ("version", "source", "omits")}
            # A worker gives the versions of its own environment.
            columns[name]["versions"] = response.get("versions", environment)
        elif name in committed["frameworks"]:
            columns[name] = committed["frameworks"][name]

    # A column that does not run keeps its committed cell of a case while the
    # baseline's result for the case stays the same.
    old_rows = {case["name"]: case for case in committed["cases"]}
    rows, stale = [], []
    for case in cases:
        result = baseline["cases"][case["name"]]
        row = dict(case)
        if "output" in result:
            row["expected"] = fields(result["output"], family.ID_MAP, names)
            if hasattr(family, "stages"):
                images = [(IMAGES / name).read_bytes() for name in case["images"]]
                row["stages"] = family.stages(case, images, row["expected"])
        else:
            row["error"] = result["rejects"]
        for key in (case["name"], *case["images"]):
            row.update(family.CRATE.get(key, {}))
        old = old_rows.get(case["name"], {})
        same = all(old.get(key) == row.get(key) for key in ("expected", "error"))
        row["frameworks"] = {}
        for name in [*FRAMEWORKS, CRATE][1:]:
            kept = old.get("frameworks", {}).get(name)
            if name in responses:
                row["frameworks"][name] = cell(
                    responses[name]["cases"][case["name"]], row, family.ID_MAP, names
                )
            elif name == CRATE:
                # tests/mm_parity.rs fails on a stale or missing crate cell.
                if kept:
                    row["frameworks"][name] = kept
            elif name in columns:
                if kept and same:
                    row["frameworks"][name] = kept
                else:
                    stale.append(f"{case['name']} ({name})")
        target = row if family.TARGET == "hf" else row["frameworks"].get(family.TARGET, {})
        if "error" in target or "rejects" in target:
            assert (
                "rust_error" in row
            ), f"{case['name']}: the target refuses it; declare its rust_error"
        else:
            assert (
                "rust_error" not in row or "known_gap" in row
            ), f"{case['name']}: the target accepts it; a rust_error needs a known_gap"
        rows.append(row)
    if stale:
        sys.exit(
            "The baseline changed or the case is new, so these committed cells are not valid: "
            f"{', '.join(stale)}. Run their columns again with --frameworks."
        )

    return {
        **header(model_dir),
        # The PIL backend's bicubic output changes across Pillow releases.
        "pillow": environment["pillow"],
        "versions": {
            key: environment[key] for key in ("torch", "torchvision", "transformers", "numpy")
        },
        # Spec fields every case has, such as a token table.
        **(
            {"spec_shared": family.spec_shared(model_dir)} if hasattr(family, "spec_shared") else {}
        ),
        "baseline": {
            "processor": baseline["processor"],
            "backend": baseline["backend"],
            "decode": baseline.get("decode", "transformers.image_utils.load_image"),
        },
        "target": family.TARGET,
        **({"target_reason": family.TARGET_REASON} if hasattr(family, "TARGET_REASON") else {}),
        "frameworks": columns,
        "cases": rows,
    }


def record_crate(table, file):
    """The crate's cells, as tests/mm_parity.rs computes them against `expected`."""
    with tempfile.TemporaryDirectory() as out:
        command = ["cargo", "test", "-p", "dynamo-multimodal", "--test", "mm_parity", "--locked"]
        done = subprocess.run(command, cwd=HERE, env={**os.environ, "MM_PARITY_RECORD_CRATE": out})
        if done.returncode:
            print(
                "tests/mm_parity.rs fails; the column records the crate as it is", file=sys.stderr
            )
        crate = json.loads((Path(out) / file).read_text())
    table["frameworks"][CRATE] = {
        "version": crate["version"],
        "source": "tests/mm_parity.rs through the public API",
    }
    for case in table["cases"]:
        case["frameworks"][CRATE] = crate["cases"][case["name"]]
    return table


def dump(table):
    """The table's fields one per line, and each case on one line."""
    lines = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in table.items() if k != "cases"]
    cases = ",\n".join(f"    {json.dumps(case)}" for case in table["cases"])
    return "{\n" + ",\n".join(lines) + ',\n  "cases": [\n' + cases + "\n  ]\n}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--family", required=True, help="a module in families/")
    parser.add_argument(
        "--model-dir", type=Path, help="the checkpoint's config, processor and tokenizer"
    )
    parser.add_argument(
        "--frameworks",
        type=lambda text: set(text.split(",")) | {"hf"},
        default=set(FRAMEWORKS),
        help="the columns to run; the others keep their committed cells",
    )
    parser.add_argument("--vllm-python", default=os.environ.get("VLLM_PYTHON"))
    parser.add_argument("--llm-mm-harness", default=os.environ.get("LLM_MM_HARNESS"))
    parser.add_argument("--crate", action="store_true", help="fill only the crate's column")
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    args = parser.parse_args()

    family = importlib.import_module(f"families.{args.family}")
    out = HERE / family.FILE
    committed = json.loads(out.read_text()) if out.exists() else None
    if args.crate:
        table = record_crate(committed, family.FILE)
    else:
        assert args.model_dir, "--model-dir: the checkpoint's config, processor and tokenizer"
        assert args.frameworks <= set(FRAMEWORKS), f"--frameworks: a subset of {list(FRAMEWORKS)}"
        for name, tool in (("vllm", args.vllm_python), ("llm-multimodal", args.llm_mm_harness)):
            assert (
                name not in args.frameworks or tool
            ), f"{name}: pass its tool, or leave it out of --frameworks"
        table = record(args, family, committed)
    names = [case["name"] for case in table["cases"]]
    assert len(set(names)) == len(names), "case names are unique"

    files = {out: dump(table)}
    # The data files of the family's own tests, beside the table.
    if hasattr(family, "files") and not args.crate:
        files.update(
            {HERE / name: text for name, text in family.files(args.model_dir.resolve()).items()}
        )
    for path, text in files.items():
        if not args.check:
            path.write_text(text)
            print(f"wrote {path.name}")
        elif not path.exists() or path.read_text() != text:
            sys.exit(f"{path.name} differs from what the frameworks give")
    print(f"{out.name}: {len(names)} cases" + (" match" if args.check else ""))


if __name__ == "__main__":
    main()
