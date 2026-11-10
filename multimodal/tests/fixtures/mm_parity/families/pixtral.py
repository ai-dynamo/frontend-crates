# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pixtral (`pixtral`, Mistral Large 4): one case per decode form, sizes that
upscale or fit into the 1540-pixel longest edge on both backends, multi-image
prompts, and the inputs the baseline refuses. `files` writes the family's own
sizing sweep."""

import json
import types

import numpy as np
import PIL
import tokenizers
import torch
import torchvision
import transformers
from PIL import Image
from transformers.image_utils import ChannelDimension
from transformers.models.pixtral.image_processing_pixtral import get_resize_output_image_size

from frameworks import hf

FILE = "mistral-large-4.json"
TARGET = "hf"
ID_MAP = {}
# Each item's rows of `[IMG]` are its spans; the batch is padded, so each image
# has its own `image_sizes`.
FIELDS = (
    "item_offsets",
    "image_sizes",
    "feature_shapes",
    "feature_sha256",
    "mrope_sha256",
    "mrope_delta",
)

# One image per decode form, as SGLang's format sweep orders them, and gray with alpha.
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
    "gray_alpha_noise_84x60.png",
]
# Both backends: an upscale to one token, a size with no resize, upscales, a
# fit into the longest edge (wide and large), and an image 3 pixels high.
SIZES = [
    "tiny_10x10.png",
    "rgb_56x56.png",
    "rgb_96x80.png",
    "rgb_333x251.jpg",
    "rgba_80x60.png",
    "large_800x600.jpg",
    "aspect_201_4020x20.png",
    "rgb_2048x1536.png",
    "thin_640x3.png",
]
# Each image of a multi-image prompt is its own crop of HF's padded batch.
MULTI = [
    ("two images", ["rgb_96x80.png", "tall_10x1100.png"], "[IMG]x[IMG]"),
    (
        "three one-row images",
        ["tiny_10x10.png", "thin_640x3.png", "aspect_201_4020x20.png"],
        "[IMG][IMG][IMG]",
    ),
]
# HF expands the one `[IMG]` with the first image and still returns the pixels
# and `image_sizes` of both images.
FEWER_PLACEHOLDERS = ("two images, one [IMG]", ["rgb_96x80.png", "rgb_333x251.jpg"])
# A truncated file, bytes that are no image, and a short side that fits to 0 pixels.
REJECTED = ["truncated_96x80.png", "garbage.bin", "thin_3080x1.png"]
EXIF = (
    "Dynamo's Pixtral decode is decode_rgb, which keeps the stored orientation; "
    "transformers' load_image applies the EXIF orientation"
)
CRATE = {
    "gray16_40x30.png": {
        "known_gap": "Dynamo rejects 16-bit samples, which PIL's convert('RGB') clips"
    },
    "exif_orientation6_64x48.jpg": {"known_gap": EXIF},
    "rgba_raw_exif6_56x48.png": {"known_gap": EXIF},
    FEWER_PLACEHOLDERS[0]: {
        "known_gap": "Dynamo refuses a prompt with fewer [IMG] than images",
        "rust_error": "InvalidInput",
    },
    **{name: {"rust_error": "InvalidInput"} for name in REJECTED},
}


def cases(model_dir):
    processors = {
        "auto": hf.processor(model_dir, "torchvision"),
        "pil": hf.processor(model_dir, "pil"),
    }
    tokenizer = processors["auto"].tokenizer

    def case(name, images, backend="auto", placeholders="[IMG]"):
        """The crate's spec for this backend, and the chat template's prompt
        around the `[IMG]` placeholders."""
        text = f"<s>[INST]{placeholders}Describe.[/INST]"
        return {
            "name": name,
            "images": images,
            "image_processor_backend": backend,
            "spec": spec(processors[backend]),
            "input_ids": tokenizer.encode(text, add_special_tokens=False),
        }

    for name in FORMATS:
        yield case(name, [name])
    for name in SIZES:
        if name not in FORMATS:
            yield case(name, [name])
    for name in SIZES:
        yield case(f"{name} pil", [name], backend="pil")
    for name, images, placeholders in MULTI:
        yield case(name, images, placeholders=placeholders)
        yield case(f"{name} pil", images, backend="pil", placeholders=placeholders)
    yield case(*FEWER_PLACEHOLDERS)
    for name in REJECTED:
        yield case(name, [name])


def request(case):
    """The case as a framework runs it: the transformers backend, no image kwargs."""
    backend = "pil" if case["image_processor_backend"] == "pil" else "torchvision"
    return {"backend": backend, "image_kwargs": {}}


def spec(processor):
    """The spec the crate gets for this processor: ids and knobs read off it."""
    image_processor = processor.image_processor
    return {
        "family": "pixtral",
        "image_token_id": processor.image_token_id,
        "image_break_token_id": processor.image_break_token_id,
        "image_end_token_id": processor.image_end_token_id,
        "patch_size": processor.patch_size,
        "spatial_merge_size": processor.spatial_merge_size,
        "longest_edge": image_processor.size["longest_edge"],
        "image_mean": list(image_processor.image_mean),
        "image_std": list(image_processor.image_std),
        "resample": {"torchvision": "aten_u8", "pil": "pil"}[image_processor.backend],
    }


GEOMETRY = "../pixtral/geometry.json"
MODEL = "mistralai/Mistral-Large-4-1T-A52B"
REVISION = "af26e7869af601441d5cdf75a1272469ddc3d059"
SIDES = [1, 3, 28, 29, 56, 1539, 1540, 1541, 3080, 3081, 65535]
EXTRA_SIZES = [
    (100, 60),
    (640, 480),
    (1920, 1080),
    (4032, 3024),
    (3024, 4032),
    (4020, 20),
    (3080, 2),
    (2, 5000),
    (12345, 6789),
    # Floor and mistral-common's round give different grids here.
    (1600, 408),
    (36, 1912),
    # The f64 quotient of the fit is just below an integer: one row or column
    # less than exact integer arithmetic gives.
    (2520, 414),
    (414, 2520),
    (2044, 1971),
]


def geometry(processor):
    """`[width, height, rows, cols]` from HF's `get_resize_output_image_size`; a
    zero row or column count is a size the processor call rejects. Each size up
    to 1.5 megapixels is also run through the processor call, which must give
    the same size or raise."""
    longest = processor.image_processor.size["longest_edge"]
    patch = processor.patch_size * processor.spatial_merge_size
    sizes = [(w, h) for w in SIDES for h in SIDES] + EXTRA_SIZES
    sizes += [(2**32 - 1, h) for h in (1, 2788939, 2788940, 2**32 - 1)] + [(1, 2**32 - 1)]
    rows_out = []
    for width, height in sizes:
        # The sizing function reads only the shape.
        resized = get_resize_output_image_size(
            types.SimpleNamespace(shape=(height, width, 3)),
            size=(longest, longest),
            patch_size=(patch, patch),
            input_data_format=ChannelDimension.LAST,
        )
        rows, cols = resized[0] // patch, resized[1] // patch
        rows_out.append([width, height, rows, cols])
        if width * height > 1_500_000:
            continue
        # The same answer from the processor call on a real image.
        try:
            out = processor.image_processor(
                images=[Image.new("RGB", (width, height))],
                patch_size=patch,
                return_tensors="pt",
                device="cpu",
            )
            assert out["image_sizes"].tolist() == [[rows * patch, cols * patch]], (width, height)
        except ValueError:
            assert rows == 0 or cols == 0, (width, height)
        else:
            assert rows > 0 and cols > 0, (width, height)
            # HF's count misreads a 1 or 3 pixel high image as channels-first.
            if height not in (1, 3):
                count = processor._get_num_multimodal_tokens(image_sizes=[(height, width)])
                assert count.num_image_tokens == [rows * (cols + 1)], (width, height)
    return rows_out


def checkpoint(model_dir):
    """The config files `spec_from_model_dir` reads, trimmed to what it reads."""

    def read(name):
        return json.loads((model_dir / name).read_text())

    config = read("config.json")
    return {
        "params.json": {"vision_encoder": read("params.json")["vision_encoder"]},
        "processor_config.json": read("processor_config.json"),
        "config.json": {key: config[key] for key in ("model_type", "image_token_index")},
    }


def render(head, cases):
    """One head key and one case per line: small files, readable diffs."""
    lines = [f" {json.dumps(key)}: {json.dumps(value)}" for key, value in head.items()]
    lines.append(' "cases": [\n' + ",\n".join("  " + json.dumps(case) for case in cases) + "\n ]")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def files(model_dir):
    """`geometry.json` for tests/pixtral_planning.rs: the config files that
    `spec_from_model_dir` reads, and the sizing sweep."""
    processor = transformers.AutoProcessor.from_pretrained(model_dir)
    assert processor.image_processor.backend == "torchvision"
    provenance = {
        "generator": "multimodal/tests/fixtures/mm_parity/generate.py --family pixtral",
        "model": MODEL,
        "revision": REVISION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "tokenizers": tokenizers.__version__,
    }
    head = {"provenance": provenance, "checkpoint": checkpoint(model_dir)}
    return {GEOMETRY: render(head, geometry(processor))}
