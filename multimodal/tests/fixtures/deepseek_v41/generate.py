# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate geometry goldens from pinned SGLang, with Python stdlib only."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from urllib.request import urlopen

SOURCE = (
    "https://raw.githubusercontent.com/sgl-project/sglang/"
    "ffac53d779c08dcdab2d07e5e2a41dba83f0e65c/"
    "python/sglang/srt/multimodal/deepseek_v41_image_processing.py"
)
FUNCTIONS = {
    "num_image_tokens", "llm_grid", "solve_resize_ratio", "safe_resize", "plan_image_grid"
}


def generate(source):
    # Execute only the pure geometry functions, avoiding torch/numpy/PIL imports.
    tree = ast.parse(source)
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    assert {node.name for node in tree.body} == FUNCTIONS
    namespace = {"math": math}
    exec(compile(tree, SOURCE, "exec"), namespace)
    specs = [
        dict(vision_patch_size=p, vision_downsample_ratio=d,
             vision_min_pixels=minimum, vision_max_n_token=budget,
             vision_max_wh_ratio=cap)
        for p, d, minimum, budget, cap in [
            (14, 3, 0, 1024, None),
            (14, 3, 3136, 256, 4.0),
            (16, 2, 1024, 4, None),
            (14, 3, 0, 5, None),
            (14, 3, 0, 257, 2.5),
            (14, 3, 0, 1024, 0.5),
        ]
    ]
    dimensions = [
        (1, 1), (13, 15), (14, 14), (41, 41), (42, 42), (43, 43),
        (511, 512), (512, 511), (4096, 4096), (1, 1_000_000),
        (1_000_000, 1), (1000, 250), (1001, 250), (1000, 251),
        (2**32 - 1, 2**32 - 1), (1, 2**32 - 1), (2**32 - 1, 1),
    ]
    rng = random.Random(41)
    dimensions += [(rng.randrange(1, 10000), rng.randrange(1, 10000)) for _ in range(20)]
    cases = []
    for spec in specs:
        for width, height in dimensions:
            lh, lw, h, w = namespace["plan_image_grid"](width, height, SimpleNamespace(**spec))
            p = spec["vision_patch_size"]
            cases.append(dict(width=width, height=height, spec=spec, plan=dict(
                resized_height=h, resized_width=w, vit_height=h // p, vit_width=w // p,
                llm_height=lh, llm_width=lw,
                num_image_tokens=namespace["num_image_tokens"](lh, lw),
            )))
    return (
        '{\n  "source": ' + json.dumps(SOURCE) + ',\n  "source_sha256": '
        + json.dumps(hashlib.sha256(source).hexdigest()) + ',\n  "cases": [\n'
        + ',\n'.join('    ' + json.dumps(case, separators=(',', ':')) for case in cases)
        + '\n  ]\n}\n'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source", type=Path, help="Use a local copy of the pinned source")
    args = parser.parse_args()
    if args.source:
        source = args.source.read_bytes()
    else:
        with urlopen(SOURCE, timeout=30) as response:
            source = response.read()
    rendered = generate(source)
    destination = Path(__file__).with_name("geometry.json")
    if args.check:
        assert destination.read_text() == rendered, "Geometry fixtures differ; regenerate them"
    else:
        destination.write_text(rendered)
