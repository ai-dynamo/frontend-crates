# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron 3.5 Super VL (`nemotron_vl`): one case per decode form, an image
already on its tile, a tall image, request budgets, prompts the processor
refuses, and no image. `files` writes the family's own sizes table."""

import asyncio
import functools
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from transformers.image_utils import load_image

from frameworks import sha256, spans
from frameworks import sglang as sglang_framework

FILE = "nemotron-3.5-super-vl.json"
# SGLang sizes every image against the request's token budget (`--context-length`
# less the text ids), resizes with PIL bicubic and sends bfloat16. The HF remote
# code sizes each image alone and resizes with torch's antialiased bicubic in float32.
TARGET = "sglang"
ID_MAP = {}
# The default fields and `feature_dtypes`: SGLang sends bfloat16, HF float32.
FIELDS = ("offsets", "image_grid_thw", "feature_shapes", "feature_dtypes", "feature_sha256")
FIELDS += ("mrope_sha256", "mrope_delta")
# SGLang's processor turns PIL's decompression-bomb check off for the whole process.
PIL_MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS

IMAGE, TEXT = 18, 1000
DESCRIBE = [5847, 13089, 1046]
# SGLang's format sweep, by name, so that an image another family adds to the
# corpus does not become a case here.
FORMATS = [
    "rgb_96x80.png",
    "rgb_333x251.jpg",
    "rgb_444_120x90.jpg",
    "progressive_200x150.jpg",
    "cmyk_80x64.jpg",
    "exif_orientation6_64x48.jpg",
    "gray_64x64.png",
    "rgba_80x60.png",
    "palette_transparency_70x50.png",
    "gray16_40x30.png",
    "animated_48x40.gif",
    "lossy_90x70.webp",
    "lossless_alpha_60x40.webp",
    "transparent_56x56.gif",
    "rgba_raw_exif6_56x48.png",
    "rgb_50x40.bmp",
    "rgba_bitfields_56x56.bmp",
    "tiny_10x10.png",
    "large_800x600.jpg",
    "aspect_201_4020x20.png",
    "la_40x30.png",
    # More pixels than PIL's decompression-bomb error threshold (178,956,970).
    "huge_palette_14000x13000.png",
]
REFUSED = {"rust_error": "InvalidInput"}
CRATE = {
    "gray16_40x30.png": {
        "known_gap": "Dynamo rejects 16-bit samples, which PIL's convert('RGB') clips",
        **REFUSED,
    },
    "huge_palette_14000x13000.png": {
        "known_gap": "the crate's PNG, WebP and BMP decoder refuses a decoded buffer above "
        "512 MiB; SGLang turns PIL's decompression-bomb check off",
        **REFUSED,
    },
    "reject two placeholders one image": REFUSED,
    "reject no placeholder one image": REFUSED,
    "reject one placeholder no image": REFUSED,
    "reject corrupt bytes": REFUSED,
}


def cases(model_dir):
    """Each case runs at a `--context-length`; the spec is what SGLang resolved for it."""
    specs = {}

    def case(name, images, input_ids, context_length=16384):
        if context_length not in specs:
            path = SglangPath(model_dir, context_length)
            specs[context_length] = path.spec()
            path.close()
        return {
            "name": name,
            "images": images,
            "context_length": context_length,
            "spec": specs[context_length],
            "input_ids": input_ids,
        }

    for name in FORMATS:
        yield case(name, [name], [IMAGE, *DESCRIBE])
    # No resize: every entry of the normalize table, as is.
    yield case("on_tile_512x512.png", ["on_tile_512x512.png"], [IMAGE, *DESCRIBE])
    yield case("tall_10x1100.png", ["tall_10x1100.png"], [IMAGE, *DESCRIBE])
    # No text: the request that `num_media_tokens` sizes. The budget downscales the image.
    yield case("lone image ctx 256", ["large_800x600.jpg"], [IMAGE], 256)
    yield case(
        "text between images ctx 1024",
        ["large_800x600.jpg", "rgb_333x251.jpg", "rgb_300x200.png"],
        [1, IMAGE, 5847, IMAGE, 13089, IMAGE] + [TEXT] * 94,
        1024,
    )
    yield case(
        "text longer than context",
        ["large_800x600.jpg", "rgb_333x251.jpg"],
        [IMAGE, IMAGE] + [TEXT] * 300,
        256,
    )
    # Each image at least `min_num_patches`: SGLang serves more tokens than the budget.
    yield case(
        "served over budget",
        [
            "rgb_96x80.png",
            "rgb_333x251.jpg",
            "large_800x600.jpg",
            "rgb_300x200.png",
            "tiny_10x10.png",
        ],
        [1] + [IMAGE] * 5,
        1024,
    )
    yield case("no image", [], DESCRIBE)
    yield case("reject two placeholders one image", ["rgb_96x80.png"], [IMAGE, IMAGE])
    yield case("reject no placeholder one image", ["rgb_96x80.png"], DESCRIBE)
    yield case("reject one placeholder no image", [], [IMAGE, *DESCRIBE])
    yield case("reject corrupt bytes", ["corrupt.png"], [IMAGE, *DESCRIBE])


def request(case):
    """The case as a framework runs it: the engine's context length. The HF
    processor's backend is torchvision; it takes no image kwargs per request."""
    return {"backend": "torchvision", "image_kwargs": {}, "max_model_len": case["context_length"]}


@functools.cache
def sglang_files(model_dir):
    """The config and processor that SGLang loads for the checkpoint."""
    from sglang.srt.hf_transformers.config import get_config
    from sglang.srt.hf_transformers.processor import get_processor

    return (
        get_config(str(model_dir), trust_remote_code=True),
        get_processor(str(model_dir), trust_remote_code=True),
    )


class SglangPath:
    """SGLang's Nemotron VL processor on the CPU for one `--context-length`."""

    def __init__(self, model_dir, context_length):
        from sglang.srt.multimodal.processors.nano_nemotron_vl import (
            NanoNemotronVLImageProcessor,
        )
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(
            model_path=str(model_dir),
            trust_remote_code=True,
            context_length=context_length,
            mm_feature_transport="cpu",
            # The image path never reads these. They keep SGLang's launch checks
            # for this GPU model from needing a CUDA device.
            device="cpu",
            mamba_radix_cache_strategy="no_buffer",
            disable_overlap_schedule=True,
        )
        publish(server_args, role="tokenizer")
        hf_config, processor = sglang_files(model_dir)
        self.processor = NanoNemotronVLImageProcessor(
            hf_config, server_args, processor, None, skip_mm_pool=True
        )

    def spec(self):
        """The crate spec with the values this processor resolved."""
        p = self.processor
        factor = int(1 / p.downsample_ratio)
        assert factor == int(round(1 / p.downsample_ratio)) == 2
        assert p.dynamic_resolution and p.audio_extractor is None
        return {
            "family": "nemotron_vl",
            "sizing": "sglang",
            "image_token_id": p.mm_tokens.image_token_id,
            "image_start_token_id": p.img_start_token_id,
            "image_end_token_id": p.img_end_token_id,
            "patch_size": p.patch_size,
            "downsample_factor": factor,
            "min_num_patches": p.min_num_patches,
            "max_num_patches": p.max_num_patches,
            "max_model_len": p.max_model_len,
            "image_mean": list(p.norm_mean),
            "image_std": list(p.norm_std),
        }

    def process(self, input_ids, images):
        """The framework's own steps: decode, then the processor."""
        return asyncio.run(
            self.processor.process_mm_data_async(
                image_data=images,
                audio_data=None,
                input_text=list(input_ids),
                request_obj=SimpleNamespace(video_data=None, audio_data=None, rid="parity"),
            )
        )

    def record(self, output):
        features = [item.feature for item in output.mm_items]
        assert all(f.dtype == torch.bfloat16 and f.device.type == "cpu" for f in features)
        assert all(item.model_specific_data["is_dynamic"] for item in output.mm_items)
        p = self.processor
        ids = (output.im_token_id, output.im_start_id, output.im_end_id)
        assert ids == (p.mm_tokens.image_token_id, p.img_start_token_id, p.img_end_token_id)
        return {
            "input_ids": [int(id_) for id_ in output.input_ids],
            # The inclusive span of each item's `<image>` run.
            "offsets": [list(span) for (span,) in (item.offsets for item in output.mm_items)],
            "feature_shapes": [list(f.shape) for f in features],
            "feature_dtypes": ["bfloat16"] * len(features),
            "feature_sha256": [
                sha256(f.view(torch.int16).numpy().view(np.uint16), "<u2") for f in features
            ],
            "mrope_sha256": None,
            "mrope_delta": None,
        }

    def close(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()


def run_sglang(request, args):
    paths, cases = {}, {}
    images_dir = Path(request["images_dir"])
    for case in request["cases"]:
        context_length = case["max_model_len"]
        if context_length not in paths:
            paths[context_length] = SglangPath(request["model_dir"], context_length)
        images = [(images_dir / name).read_bytes() for name in case["images"]]
        try:
            output = paths[context_length].process(case["input_ids"], images)
        except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
            cases[case["name"]] = {"rejects": f"{type(error).__name__}: {error}"}
            continue
        cases[case["name"]] = {"output": paths[context_length].record(output)}
    for path in paths.values():
        path.close()
    source = (
        "NanoNemotronVLImageProcessor.process_mm_data_async(image bytes, token ids) on the CPU, "
        "PIL decode, ServerArgs(context_length=the case's max_model_len)"
    )
    return {"version": sglang_framework.version(), "source": source, "omits": [], "cases": cases}


class HfPath:
    """The checkpoint's remote-code processor, loaded by transformers alone."""

    def __init__(self, model_dir):
        self.hf_processor = transformers.AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        self.processor = self.hf_processor.image_processor
        self.default_max_model_len = self.processor.max_model_len
        self.budgets = {}

    def spec(self):
        """The crate spec with the values the HF processor resolved."""
        p, image = self.hf_processor, self.processor
        return {
            "family": "nemotron_vl",
            "sizing": "hf",
            "image_token_id": p.image_token_id,
            "image_start_token_id": p.tokenizer.convert_tokens_to_ids(p.image_start_token),
            "image_end_token_id": p.tokenizer.convert_tokens_to_ids(p.image_end_token),
            "patch_size": image.patch_size,
            "downsample_factor": image._downsample_factor,
            "min_num_patches": image.min_num_patches,
            "max_num_patches": image.max_num_patches,
            "max_model_len": image.max_model_len,
            "image_mean": list(image.norm_mean),
            "image_std": list(image.norm_std),
        }

    def prompt(self, case):
        tokenizer = self.hf_processor.tokenizer
        text = tokenizer.decode(case["input_ids"])
        assert tokenizer.encode(text, add_special_tokens=False) == case["input_ids"], case["name"]
        return text

    def process(self, case, text, data):
        """The framework's own steps: `load_image`, then the processor with the
        image processor's `max_model_len` set to the case's."""
        images = [load_image(Image.open(io.BytesIO(image))) for image in data]
        self.processor.max_model_len = case["max_model_len"] or self.default_max_model_len
        try:
            return self.hf_processor(text=[text], images=images or None)
        finally:
            self.processor.max_model_len = self.default_max_model_len

    def record(self, out):
        ids = list(out["input_ids"][0])
        # One image's slice of the batch: stacked when every tile has one size, else a list.
        features = [pixels[None] for pixels in out.get("pixel_values", [])]
        return {
            "input_ids": ids,
            "offsets": spans(ids, self.hf_processor.image_token_id),
            "feature_shapes": [list(feature.shape) for feature in features],
            "feature_dtypes": [str(feature.dtype).removeprefix("torch.") for feature in features],
            "feature_sha256": [sha256(feature.numpy(), "<f4") for feature in features],
            "mrope_sha256": None,
            "mrope_delta": None,
        }

    def request(self, sizes, max_model_len):
        """`_preprocess` on blank images: `[width, height, tokens]` per image."""
        self.processor.max_model_len = max_model_len or self.default_max_model_len
        try:
            output = self.processor(images=[Image.new("RGB", size) for size in sizes])
        finally:
            self.processor.max_model_len = self.default_max_model_len
        return [
            [width, height, tokens]
            for (height, width), tokens in zip(output["imgs_sizes"], output["num_tokens"])
        ]

    def budget(self, num_images, max_model_len):
        """The per-image budget `_preprocess` hands `_compute_target_patches`."""
        key = (num_images, max_model_len)
        if key not in self.budgets:
            seen = []
            self.processor._compute_target_patches = lambda _image, tokens: (
                seen.append(tokens),
                (2, 2),
            )[1]
            self.processor.max_model_len = max_model_len
            try:
                self.processor(images=[Image.new("RGB", (32, 32))] * num_images)
            finally:
                del self.processor._compute_target_patches
                self.processor.max_model_len = self.default_max_model_len
            assert len(seen) == num_images and len(set(seen)) == 1
            self.budgets[key] = seen[0]
        return self.budgets[key]

    def plan(self, width, height, budget):
        """`[width, height, tokens]` in pixels, or None where the processor raises."""
        try:
            w, h = self.processor._compute_target_patches(
                SimpleNamespace(width=width, height=height), budget
            )
        except ZeroDivisionError:
            return None
        if w == 0 or h == 0:
            return None
        factor = self.processor._downsample_factor
        patch = self.processor.patch_size
        return [w * patch, h * patch, (w * h) // factor**2]


def run_hf(request, args):
    # The baseline decodes under PIL's own limits.
    Image.MAX_IMAGE_PIXELS = PIL_MAX_IMAGE_PIXELS
    path, cases = HfPath(request["model_dir"]), {}
    for case in request["cases"]:
        text = path.prompt(case)
        data = [(Path(request["images_dir"]) / name).read_bytes() for name in case["images"]]
        try:
            out = path.process(case, text, data)
        except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
            cases[case["name"]] = {"rejects": f"{type(error).__name__}: {error}"}
            continue
        cases[case["name"]] = {"output": path.record(out)}
    return {
        "version": transformers.__version__,
        "source": "AutoProcessor(trust_remote_code=True)(text, images=load_image(bytes)), with "
        "the image processor's max_model_len set to the case's; 1-D positions",
        "processor": type(path.hf_processor).__name__,
        "backend": path.processor.backend,
        "omits": [],
        "cases": cases,
    }


# Columns whose module in frameworks/ cannot run this checkpoint's processor.
ADAPTERS = {"hf": run_hf, "sglang": run_sglang}


SIZES = "../nemotron_vl/sizes.json"
# With 2699 tokens left, SGLang's budget loop is still shrinking the third image
# when it stops: 9 rounds give it 341 tokens, 10 give 330 and 11 give 290.
ROUND_CAP_SIZES = [
    (640, 1080),
    (1920, 480),
    (3840, 1440),
    (800, 1080),
    (1920, 240),
    (1920, 480),
    (320, 600),
    (1024, 600),
    (1280, 1080),
    (320, 1080),
    (1280, 240),
]
# (sizes, max_model_len, text_tokens). A thin image ends above `max_num_patches`,
# and over its budget SGLang's loop raises its maximum further.
SGLANG_REQUESTS = [
    ([(2000000, 1)], 8192, 20),
]
HF_REQUESTS = [
    # (sizes, max_model_len); None keeps the checkpoint's 16384.
    ([(224, 224)], None),
    ([(64, 32)], None),
    ([(512, 512)], None),
    ([(1000, 500)], None),
    ([(500, 1000)], None),
    ([(1024, 1024)], None),
    ([(1536, 1536)], None),
    ([(1920, 1080)], None),
    ([(1280, 720)], None),
    ([(4000, 3000)], None),
    ([(1, 1)], None),
    ([(640, 480)], None),
    ([(3000, 200), (200, 3000), (10000, 1)], None),
    ([(1920, 1080)] * 3, None),
    # vLLM's text-aware budget gives 1344 tokens each here with 10 text tokens.
    ([(1920, 1080)] * 3, 4096),
    ([(1920, 1080)] * 3, 1000),
    ([(1920, 1080)] * 5, 1000),
    ([(640, 480), (1920, 1080)], 260),
    ([(1920, 1080)], 4),
    ([(1000000, 1)], None),
    ([(100000, 2)], 260),
]


def size_cases():
    """(name, context_length, sizes, input_ids): SGLang's processor on blank images,
    for what does not depend on the pixels."""
    one = [IMAGE, *DESCRIBE]

    def many(sizes, text_tokens):
        return sizes, [1] + [IMAGE] * len(sizes) + [TEXT] * (text_tokens - 1)

    for group, sizes in [
        ("below min", [(1, 1), (16, 16), (224, 224), (64, 32)]),
        ("identity", [(640, 480), (1024, 1024)]),
        ("tie", [(1000, 500), (1280, 720), (1840, 1840)]),
        ("above max", [(2600, 1900), (4000, 3000)]),
        ("thin", [(3000, 200), (200, 3000), (10000, 1), (1, 10000)]),
    ]:
        for width, height in sizes:
            yield f"{group} {width}x{height}", 16384, [(width, height)], one
    yield "thin 100000x1 and 1x100000", 16384, *many([(100000, 1), (1, 100000)], 1)
    yield "lone image 640x480 ctx 256", 256, [(640, 480)], [IMAGE]
    yield "lone image 4000x3000 ctx 2000", 2000, [(4000, 3000)], [IMAGE]
    for context, text_tokens in [(16384, 1), (16384, 12000), (8192, 4000), (4096, 10)]:
        name = f"budget 3x1920x1080 ctx {context} text {text_tokens}"
        yield name, context, *many([(1920, 1080)] * 3, text_tokens)
    yield (
        "budget mixed sizes with text between images",
        1024,
        [(1280, 720), (640, 480), (800, 600)],
        [1, IMAGE, 5847, IMAGE, 13089, IMAGE] + [TEXT] * 94,
    )
    for name, context, count, text_tokens in [
        ("budget 5x4000x3000 ctx 16384 text 300", 16384, 5, 300),
        ("budget 2x4000x3000 ctx 8192 text 2000", 8192, 2, 2000),
        ("context length unset 3x4000x3000", None, 3, 1),
    ]:
        yield name, context, *many([(4000, 3000)] * count, text_tokens)
    yield "context length unset 2x1280x720 text 7000", None, *many([(1280, 720)] * 2, 7000)
    yield "text longer than context 2x1920x1080", 4096, *many([(1920, 1080)] * 2, 5000)
    yield "served over budget 5x640x480", 1024, *many([(640, 480)] * 5, 1)
    yield "round cap 11 images", 2700, *many(ROUND_CAP_SIZES, 1)
    # 1024 tokens of budget: counting the placeholder as text would shrink the image.
    yield "budget met exactly 1024x1024", 1027, [(1024, 1024)], one
    yield "no image", 16384, [], DESCRIBE
    yield "reject two placeholders one image", 16384, [(64, 64)], [IMAGE] * 2
    yield "reject no placeholder one image", 16384, [(64, 64)], DESCRIBE
    yield "reject one placeholder no image", 16384, [], one


def rle(ids):
    runs = []
    for token in ids:
        if runs and runs[-1][0] == token:
            runs[-1][1] += 1
        else:
            runs.append([token, 1])
    return runs


def blank_png(size):
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format="PNG")
    return buffer.getvalue()


def record_size_cases(model_dir):
    paths, cases = {}, []
    for name, context_length, sizes, input_ids in size_cases():
        if context_length not in paths:
            paths[context_length] = SglangPath(model_dir, context_length)
        path = paths[context_length]
        case = {
            "name": name,
            "context_length": context_length,
            "spec": path.spec(),
            "images": [list(size) for size in sizes],
            "input_ids": rle(input_ids),
        }
        cases.append(case)
        try:
            output = path.process(input_ids, [blank_png(size) for size in sizes])
        except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
            case["error"] = f"{type(error).__name__}: {error}"
            continue
        output = path.record(output)
        case["expected"] = {
            "input_ids_sha256": sha256(output["input_ids"], "<i4"),
            "num_input_ids": len(output["input_ids"]),
            "offsets": output["offsets"],
            "num_tokens": [end - start + 1 for start, end in output["offsets"]],
            "feature_shapes": output["feature_shapes"],
        }
    for path in paths.values():
        path.close()
    return cases


class Lcg:
    """64-bit LCG shared with tests/nemotron_vl_planning.rs."""

    def __init__(self, seed):
        self.state = seed

    def below(self, bound):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.state >> 33) % bound


def line(size, plan):
    return f"{size[0]} {size[1]} " + ("error" if plan is None else "%d %d %d" % tuple(plan))


def digest(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def record_sizes(model_dir, spec):
    from sglang.srt.multimodal.internvl_utils import compute_budgeted_image_sizes

    patch, ratio = spec["patch_size"], 1 / spec["downsample_factor"]
    low, high = spec["min_num_patches"], spec["max_num_patches"]

    def sglang(sizes, max_model_len, text_tokens):
        return [
            list(plan)
            for plan in compute_budgeted_image_sizes(
                sizes, max_model_len - text_tokens, patch, ratio, low, high
            )
        ]

    hf = HfPath(model_dir)
    hf_requests = []
    for sizes, max_model_len in HF_REQUESTS:
        request = {"sizes": sizes, "max_model_len": max_model_len or hf.default_max_model_len}
        try:
            request["plans"] = hf.request(sizes, max_model_len)
        except Exception as error:  # noqa: BLE001 -- the rejection is the recorded result
            request["error"] = f"{type(error).__name__}: {error}"
        hf_requests.append(request)

    # Lone images at the default context: all sizes, then thin ones, where SGLang
    # clamps a side to the pixel-shuffle factor and HF raises.
    default_budget = hf.budget(1, hf.default_max_model_len)
    sweeps = {
        "single_image_sweep": {"count": 20000, "max_long_side": 4096, "max_short_side": 4096},
        "thin_image_sweep": {"count": 4000, "max_long_side": 300000, "max_short_side": 48},
    }
    for seed, sweep in enumerate(sweeps.values(), start=1):
        rng = Lcg(seed)
        sglang_lines, hf_lines = [], []
        for _ in range(sweep["count"]):
            size = (1 + rng.below(sweep["max_long_side"]), 1 + rng.below(sweep["max_short_side"]))
            if rng.below(2):
                size = size[::-1]
            sglang_lines.append(line(size, sglang([size], 16384, 0)[0]))
            hf_lines.append(line(size, hf.plan(*size, default_budget)))
        sweep.update(seed=seed, sglang_sha256=digest(sglang_lines), hf_sha256=digest(hf_lines))

    # Whole requests: SGLang's text-aware loop and the HF per-call budget.
    rng = Lcg(3)
    requests = {"count": 4000, "max_side": 4096, "max_images": 6, "seed": 3}
    contexts = [4, 260, 1024, 2048, 4096, 8192, 16384]
    sglang_lines, hf_lines = [], []
    for _ in range(requests["count"]):
        sizes = [
            (1 + rng.below(requests["max_side"]), 1 + rng.below(requests["max_side"]))
            for _ in range(1 + rng.below(requests["max_images"]))
        ]
        max_model_len = contexts[rng.below(len(contexts))]
        text_tokens = rng.below(max_model_len + 2000)
        head = f"{max_model_len} {text_tokens}"
        plans = sglang(sizes, max_model_len, text_tokens)
        sglang_lines.append(head + " | " + " | ".join(map(line, sizes, plans)))
        budget = hf.budget(len(sizes), max_model_len)
        plans = [hf.plan(*size, budget) for size in sizes]
        hf_lines.append(head + " | " + " | ".join(map(line, sizes, plans)))
    requests.update(
        contexts=contexts,
        sglang_sha256=digest(sglang_lines),
        hf_sha256=digest(hf_lines),
    )

    return {
        "sglang_spec": spec,
        "hf_spec": hf.spec(),
        "sglang_requests": [
            {
                "sizes": sizes,
                "max_model_len": max_model_len,
                "text_tokens": text_tokens,
                "plans": sglang(sizes, max_model_len, text_tokens),
            }
            for sizes, max_model_len, text_tokens in SGLANG_REQUESTS
        ],
        "hf_requests": hf_requests,
        **sweeps,
        "request_sweep": requests,
    }


def dump(fixture):
    """One field per line, and each element of a list field on one line."""
    lines = []
    for key, value in fixture.items():
        if isinstance(value, list):
            body = ",\n".join(f"    {json.dumps(item)}" for item in value)
            lines.append(f"  {json.dumps(key)}: [\n{body}\n  ]")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value)}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def files(model_dir):
    """`sizes.json` for tests/nemotron_vl_planning.rs: SGLang's processor on blank
    images, `_preprocess` of the HF remote code, and digested sweeps of both."""
    size_cases = record_size_cases(model_dir)
    header = {
        "model": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "sglang": sglang_framework.version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        # PIL's bicubic output changes across Pillow releases.
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }
    sizes = record_sizes(model_dir, size_cases[0]["spec"])
    return {SIZES: dump({**header, **sizes, "sglang_cases": size_cases})}
