# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate GLM-5.3-Flash image geometry goldens from pinned transformers, stdlib only.

The pinned source is `image_processing_glm5_next.py` at the transformers
v5.19.0 tag (unchanged since v5.16.1). Only the pure `smart_resize` function
is executed; the thin wrapper below repeats the geometry-relevant lines of
`Glm5NextImageProcessor.resize` / `_preprocess` (canvas → patch grid →
merged token count), which otherwise need torch.

`--verify-hf` additionally runs the installed `Glm5NextImageProcessor` on
synthetic images and asserts its `image_grid_thw` matches the planned grid;
it needs transformers, torch, torchvision and Pillow and is not required to
regenerate the fixture.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
from urllib.request import urlopen

SOURCE = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    "c4c4605f961877852b89b97478536bb3a26b6fd1/"
    "src/transformers/models/glm5_next/image_processing_glm5_next.py"
)
FUNCTIONS = {"smart_resize"}

# (patch_size, merge_size, temporal_patch_size, patch_expand_factor,
#  min_image_tokens, max_image_tokens). The first row is zai-org/GLM-5.3-Flash;
# the rest are mathematical test inputs, not model defaults.
SPECS = [
    (14, 2, 2, 1, 16, 8000),
    (14, 2, 2, 1, 16, 64),
    (14, 2, 2, 1, 1024, 8000),
    (14, 2, 2, 1, 16, 1),
    (14, 2, 2, 1, 9000, 8000),
    (16, 2, 2, 1, 16, 8000),
    (14, 2, 2, 2, 16, 8000),
    (14, 3, 1, 1, 4, 512),
]
DIMENSIONS = [
    (1, 1), (13, 15), (14, 14), (27, 28), (28, 28), (29, 29), (56, 56),
    (100, 76), (448, 448), (640, 480), (1024, 768), (768, 1024),
    (1920, 1080), (1080, 1920), (3840, 2160), (2160, 3840), (4000, 4000),
    (8000, 8000), (1, 10000), (10000, 1), (1000, 250), (1001, 250),
    (1000, 251), (2800, 2800), (3360, 3360), (12345, 6789), (65535, 65535),
    # Largest sides whose pixel product stays below 2**53, where Python's
    # int / int true division and f64 division agree exactly.
    (2**26, 2**26), (2**26 - 1, 2**26 + 1), (1, 2**32 - 1), (2**32 - 1, 1),
]


def load_smart_resize(source):
    # Execute only the pure geometry function, avoiding torch/torchvision imports.
    tree = ast.parse(source)
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    assert {node.name for node in tree.body} == FUNCTIONS
    namespace = {"math": math}
    exec(compile(tree, SOURCE, "exec"), namespace)
    return namespace["smart_resize"]


def spec_dict(row):
    keys = ("patch_size", "merge_size", "temporal_patch_size",
            "patch_expand_factor", "min_image_tokens", "max_image_tokens")
    return dict(zip(keys, row))


def plan(smart_resize, width, height, spec):
    """Geometry lines of `Glm5NextImageProcessor.resize` and `_preprocess`."""
    factor = spec["patch_size"] * spec["merge_size"] * spec["patch_expand_factor"]
    temporal = spec["temporal_patch_size"]
    resized_height, resized_width = smart_resize(
        num_frames=temporal,
        height=height,
        width=width,
        temporal_factor=temporal,
        factor=factor,
        min_pixels=spec["min_image_tokens"],
        max_pixels=spec["max_image_tokens"],
    )
    grid_h, grid_w = resized_height // spec["patch_size"], resized_width // spec["patch_size"]
    return dict(
        resized_height=resized_height, resized_width=resized_width,
        grid_h=grid_h, grid_w=grid_w,
        num_image_tokens=(grid_h * grid_w) // spec["merge_size"] ** 2,
    )


def generate(source):
    smart_resize = load_smart_resize(source)
    rng = random.Random(53)
    dimensions = DIMENSIONS + [
        (rng.randrange(1, 10000), rng.randrange(1, 10000)) for _ in range(24)
    ]
    groups = []
    for row in SPECS:
        spec = spec_dict(row)
        cases = []
        for width, height in dimensions:
            try:
                expected = plan(smart_resize, width, height, spec)
            except ValueError as error:
                expected = {"error": str(error)}
            cases.append([width, height, expected])
        groups.append(
            '    {"spec": ' + json.dumps(spec) + ', "cases": [\n'
            + ',\n'.join('      ' + json.dumps(case) for case in cases)
            + '\n    ]}'
        )
    return (
        '{\n  "source": ' + json.dumps(SOURCE) + ',\n  "source_sha256": '
        + json.dumps(hashlib.sha256(source).hexdigest()) + ',\n  "groups": [\n'
        + ',\n'.join(groups)
        + '\n  ]\n}\n'
    )


def verify_hf(source, max_side=4096):
    """Cross-check the plan against the installed HF processor's image_grid_thw."""
    import torch  # noqa: F401  (torchvision backend)
    import transformers
    from PIL import Image
    from transformers.models.glm5_next.image_processing_glm5_next import (
        Glm5NextImageProcessor,
    )

    smart_resize = load_smart_resize(source)
    checked = 0
    for row in SPECS:
        spec = spec_dict(row)
        processor = Glm5NextImageProcessor(**spec)
        for width, height in DIMENSIONS:
            if max(width, height) > max_side:
                continue
            try:
                expected = plan(smart_resize, width, height, spec)
            except ValueError:
                continue
            image = Image.new("RGB", (width, height), (37, 141, 203))
            grid = processor(images=[image], return_tensors="pt")["image_grid_thw"][0].tolist()
            assert grid == [1, expected["grid_h"], expected["grid_w"]], (spec, width, height, grid, expected)
            checked += 1
    print(f"verified {checked} cases against transformers {transformers.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-hf", action="store_true")
    parser.add_argument("--source", type=Path, help="Use a local copy of the pinned source")
    args = parser.parse_args()
    if args.source:
        source = args.source.read_bytes()
    else:
        with urlopen(SOURCE, timeout=30) as response:
            source = response.read()
    if args.verify_hf:
        verify_hf(source)
    rendered = generate(source)
    destination = Path(__file__).with_name("geometry.json")
    if args.check:
        assert destination.read_text() == rendered, "Geometry fixtures differ; regenerate them"
    else:
        destination.write_text(rendered)
