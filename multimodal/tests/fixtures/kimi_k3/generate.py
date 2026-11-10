# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the Kimi-K3 fixtures that the parity table cannot hold: family.json and counts.json.

Usage:
    CUDA_VISIBLE_DEVICES="" python generate.py [--check] <moonshotai/Kimi-K3 dir> <moonshotai/Kimi-K3.1-Pro dir>

The image cases are rows of `../mm_parity/kimi-k3.json`. This script runs the
same two references, SGLang's Python CPU path and the checkpoint's own
`KimiK3Processor`, and stops unless they agree:

- `family.json` `cases`: the prompt expansion from sizes alone (`sizes`), and
  two drawn images (7300x64 and 9000x4000) at the checkpoint's limits that
  `draw` builds from integer noise, which the Rust test repeats, so they need
  no file.
- `family.json` `checkpoints`: the files `spec_from_hf_configs` reads, and
  what `save_pretrained` writes, for both checkpoints.
- `counts.json`: the checkpoint's `navit_resize_image` and the prompt span
  with the real tokenizer's header, over a size sweep.

Kimi-K3.1-Pro must give the same token table, the same layouts and drawn
images, and the same cases as the committed `kimi-k3.json`, so run the driver
first. The driver's `files` hook takes one checkpoint, and this script needs
two, so it is a second script. `--check` writes nothing and fails when a
committed file differs.
"""

import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import transformers
from sglang.srt.multimodal.processors.kimi_k3 import (
    KimiK3ImageProcessor,
    _expand_k3_image_prompt_token_ids,
)
from sglang.srt.multimodal.processors.kimi_k25 import navit_resize_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mm_parity"))
import generate  # noqa: E402
from corpus import draw, encode  # noqa: E402
from families import kimi_k3  # noqa: E402
from frameworks import sglang, sha256  # noqa: E402

ONE = "What is in {}?"


def revision(model_dir):
    metadata = model_dir / ".cache/huggingface/download/config.json.metadata"
    return metadata.read_text().split()[0]


class References:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.hf = kimi_k3.HfPath(model_dir)
        self.cfg = self.hf.processor.image_processor.media_proc_cfg
        self.spec = {
            "family": "kimi_k3",
            "image_token_id": self.hf.pad,
            **{key: self.cfg[key] for key in kimi_k3.SPEC_KEYS},
        }

    def navit(self, size):
        keys = ["patch_size", "merge_kernel_size", "in_patch_limit", "patch_limit_on_one_side"]
        keys += ["fixed_output_tokens"]
        return navit_resize_config(*size, *[self.cfg[key] for key in keys])

    def spans(self, input_ids, offsets):
        """Each image's whole expansion, `<|media_begin|>` to `<|media_end|>`."""
        (begin,) = self.hf.encode("<|media_begin|>")
        (end,) = self.hf.encode("<|media_end|>")
        spans = []
        for first, last in offsets:
            start = max(i for i in range(first) if input_ids[i] == begin)
            assert input_ids[last + 1] == end
            spans.append(last + 1 - start + 1)
        return spans

    def expected(self, output):
        """The fields the Rust test checks, from a reference's output."""
        ids = output["input_ids"]
        return {
            "input_ids_sha256": sha256(ids, "<i4"),
            "num_input_ids": len(ids),
            "offsets": output["offsets"],
            "spans": self.spans(ids, output["offsets"]),
            **{key: output[key] for key in ("image_grid_thw", "feature_shapes", "feature_sha256")},
        }

    def layout(self, name, prompt, sizes):
        """SGLang's expansion from the sizes, and the checkpoint's `make_image_prompt`."""
        input_ids = self.hf.encode(prompt.format(*["<|media_pad|>"] * len(sizes)))
        counts = [self.navit(size)["num_tokens"] for size in sizes]
        tokenizer = self.hf.processor.tokenizer
        ids = _expand_k3_image_prompt_token_ids(input_ids, self.hf.pad, counts, sizes, tokenizer)
        ids = ids.flatten().tolist()
        offsets = KimiK3ImageProcessor.get_mm_items_offset(torch.tensor(ids), self.hf.pad)
        image_prompts = [self.hf.processor.image_processor.make_image_prompt(*s) for s in sizes]
        text = prompt.format(*image_prompts)
        counts = iter(counts)
        checkpoint = [
            id_
            for model_id in self.hf.encode(text)
            for id_ in ([model_id] * next(counts) if model_id == self.hf.pad else [model_id])
        ]
        assert checkpoint == ids, name
        offsets = [list(offset) for offset in offsets]
        expected = {
            "input_ids_sha256": sha256(ids, "<i4"),
            "num_input_ids": len(ids),
            "offsets": offsets,
            "spans": self.spans(ids, offsets),
        }
        sizes = [list(size) for size in sizes]
        return {"name": name, "spec": self.spec, "input_ids": input_ids, "sizes": sizes} | {
            "expected": expected
        }

    def draws(self, recipes):
        """Each recipe as a PNG through both references, which decode it as Pillow does."""
        with tempfile.TemporaryDirectory() as images:
            cases = []
            for name, recipe in recipes:
                image = draw(**recipe)
                Path(images, f"{len(cases)}.png").write_bytes(
                    encode(image, "PNG", compress_level=1)
                )
                ids = self.hf.encode(ONE.format("<|media_pad|>"))
                case = {"name": name, "spec": self.spec, "input_ids": ids, "draw": [recipe]}
                # Too slow for an unoptimized test build; the Rust test ignores it.
                if image.width * image.height > 1 << 22:
                    case["checkpoint_scale"] = True
                cases.append(case)
            request = {
                "family": "kimi_k3",
                "model_dir": str(self.model_dir),
                "images_dir": images,
                "cases": [
                    {**case, "images": [f"{i}.png"], "backend": "pil", "image_kwargs": {}}
                    for i, case in enumerate(cases)
                ],
            }
            results = [self.hf.run(request), kimi_k3.run_sglang(request, None)]
        for case in cases:
            first, second = [result["cases"][case["name"]]["output"] for result in results]
            case["expected"] = self.expected(first)
            assert self.expected(second) == case["expected"], case["name"]
        return cases

    def layouts(self):
        yield self.layout("header 7 ids", ONE, [(300, 200)])
        yield self.layout("header 9 ids", ONE, [(1024, 7168)])
        yield self.layout("header 13 ids", ONE, [(1234567890, 1000000007)])
        rng = random.Random(1)
        sizes = [
            (int(10 ** rng.uniform(0, 9.6)), int(10 ** rng.uniform(0, 9.6))) for _ in range(96)
        ]
        # The crate plans exactly only while the patch cells stay below 2^53.
        sizes = [(w, h) for w, h in sizes if (w // 14) * (h // 14) < 2**53][:64]
        yield self.layout("header sweep", " ".join(["{}"] * 64), sizes)

    def checkpoint(self):
        """The two files `spec_from_hf_configs` reads, config.json cut to its
        keys, and what `save_pretrained` writes for the image processor and for
        the whole processor."""
        config = json.loads((self.model_dir / "config.json").read_text())
        processor = transformers.AutoProcessor.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        with tempfile.TemporaryDirectory() as saved:
            processor.image_processor.save_pretrained(f"{saved}/image")
            processor.save_pretrained(f"{saved}/whole")
            resaved = {
                name: json.loads(Path(saved, path).read_text())
                for name, path in [
                    ("preprocessor_config", "image/preprocessor_config.json"),
                    ("processor_config", "whole/processor_config.json"),
                ]
            }
        return {
            "revision": revision(self.model_dir),
            "config": {key: config[key] for key in ("model_type", "media_placeholder_token_id")},
            "preprocessor_config": json.loads(
                (self.model_dir / "preprocessor_config.json").read_text()
            ),
            "resaved": resaved,
        }

    def counts(self):
        """`[width, height, num_tokens, new_width, new_height, prompt span]` rows."""
        end = len(self.hf.encode("<|media_end|>"))
        rows = []
        for size in sweep_sizes():
            image_processor = self.hf.processor.image_processor
            cfg = image_processor.get_resize_config(
                {"type": "image", "image": SimpleNamespace(size=size)}
            )
            assert all(cfg[key] == value for key, value in self.navit(size).items())
            text = f"<|media_begin|>image {size[0]}x{size[1]}<|media_content|>"
            span = len(self.hf.encode(text)) + cfg["num_tokens"] + end
            rows.append([*size, cfg["num_tokens"], cfg["new_width"], cfg["new_height"], span])
        return json.dumps({"spec": self.spec, "rows": rows}, separators=(",", ":")) + "\n"


def sweep_sizes():
    edges = [1, 2, 13, 14, 15, 27, 28, 29, 55, 56, 57, 100, 224, 512, 1000, 1023, 1024]
    edges += [3583, 3584, 3585, 7167, 7168, 7169, 7170, 10000, 14335, 14336, 65535]
    edges += [100000, 2**31, 2**32 - 1]
    sizes = [(w, h) for w in edges for h in edges]
    ns = sorted({int(round(1.25**k)) for k in range(100)} | {2**32 - 1})
    for n in [n for n in ns if n < 2**32]:
        sizes += [(1, n), (n, 1), (2, n), (n, 2)]
    for e in range(20, 33):
        n = min(2**e, 2**32 - 1)
        sizes += [(n, n), (n, n - 1), (n - 1, n), (n, max(1, n // 3))]
    rng = random.Random(0)
    sizes += [(int(2 ** rng.uniform(0, 32)), int(2 ** rng.uniform(0, 32))) for _ in range(1200)]
    sizes += [(rng.randint(100, 8000), rng.randint(100, 8000)) for _ in range(600)]
    # Sizes where `width * scale` lands next to an integer, so a one-ulp scale
    # error would change `int()`.
    near = []
    for _ in range(200000):
        w, h = rng.randint(7000, 300000), rng.randint(7000, 300000)
        s1 = math.sqrt(65536 / (max(1.0, w // 14) * max(1.0, h // 14)))
        scale = min(1.0, s1, 7168 / w, 7168 / h)
        near += [(abs(v - round(v)), w, h) for v in (w * scale, h * scale)]
    near.sort()
    sizes += [(w, h) for _, w, h in near[:120]]
    return [(max(1, w), max(1, h)) for w, h in sizes]


def references(table):
    """A table's `spec_shared` and cases with the two references' values only."""
    return table["spec_shared"], [
        {**case, "frameworks": {"sglang": case["frameworks"]["sglang"]}} for case in table["cases"]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kimi_k3", type=Path)
    parser.add_argument("kimi_k31", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    # Kimi-K3.1-Pro's server args read the CUDA device (SGLang's DSA hook).
    kimi_k3.SglangPath.server_model_dir = args.kimi_k3
    k3, k31 = References(args.kimi_k3), References(args.kimi_k31)
    committed = json.loads((HERE.parent / "mm_parity" / kimi_k3.FILE).read_text())
    on_k31 = argparse.Namespace(
        family="kimi_k3", model_dir=args.kimi_k31, frameworks={"hf", "sglang"}
    )
    table = generate.record(on_k31, kimi_k3, None)
    assert references(table) == references(committed), f"Kimi-K3.1-Pro differs: {kimi_k3.FILE}"
    layouts = list(k3.layouts())
    assert list(k31.layouts()) == layouts, "Kimi-K3.1-Pro expands otherwise"
    recipes = [
        ("side cap 7300x64", dict(kind="rgba", width=7300, height=64, seed=50)),
        ("budget cap 9000x4000", dict(kind="rgb", width=9000, height=4000, seed=51)),
    ]
    draws = k3.draws(recipes)
    assert k31.draws(recipes) == draws, "Kimi-K3.1-Pro gives other drawn images"
    cases = layouts + draws
    fixture = {
        "model": "moonshotai/Kimi-K3",
        "versions": {
            "sglang": sglang.version(),
            "pillow": PIL.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "checkpoints": {
            "moonshotai/Kimi-K3": k3.checkpoint(),
            "moonshotai/Kimi-K3.1-Pro": k31.checkpoint(),
        },
        "cases": cases,
    }
    files = {"family.json": generate.dump(fixture), "counts.json": k3.counts()}
    if args.check:
        stale = [name for name, text in files.items() if (HERE / name).read_text() != text]
        assert not stale, f"differs: {stale}"
        print(f"checked {len(cases)} cases and {len(files)} files")
        return
    for name, text in files.items():
        (HERE / name).write_text(text)
    print(f"wrote {len(cases)} cases")


if __name__ == "__main__":
    main()
