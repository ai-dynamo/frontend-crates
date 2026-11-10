# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen-VL (`qwen_vl`): both resamplers, both smart_resize branches, two
images, and a tall image that crosses Pillow's pass-order rule."""

import io

import numpy as np
import transformers
from PIL import Image

from frameworks import sha256

FILE = "qwen2-vl.json"
TARGET = "hf"
# The cases' ids, and the Qwen2-VL checkpoint's: <|image_pad|>, <|vision_start|>, <|vision_end|>.
ID_MAP = {900: 151655, 901: 151652, 902: 151653}
CRATE = {}

SPEC = {
    "family": "qwen_vl",
    "image_token_id": 900,
    "resample": "aten_u8",
    "patch_size": 14,
    "merge_size": 2,
    "temporal_patch_size": 2,
    "min_pixels": 3136,
    "max_pixels": 1003520,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}
VISION_START, IMAGE_PAD, VISION_END = 901, 900, 902


def cases(model_dir):
    """The checkpoint's processor runs each case with the case's pixel bounds;
    its other image settings must be the spec's."""
    image_processor = transformers.AutoImageProcessor.from_pretrained(model_dir)
    for knob in ("patch_size", "merge_size", "temporal_patch_size", "image_mean", "image_std"):
        assert list(np.ravel(getattr(image_processor, knob))) == list(np.ravel(SPEC[knob])), knob
    for name, images, resample, max_pixels in [
        ("aten_downscale", ["rgb_300x200.png"], "aten_u8", 12544),
        ("aten_multi", ["rgb_40x40.png", "rgb_56x56.png"], "aten_u8", None),
        ("aten_upscale", ["rgb_40x40.png"], "aten_u8", None),
        ("pil_round", ["rgb_100x76.png"], "pil", None),
        ("pil_tall", ["tall_10x1100.png"], "pil", None),
    ]:
        spec = dict(SPEC, resample=resample)
        spec["max_pixels"] = max_pixels or spec["max_pixels"]
        block = [VISION_START, IMAGE_PAD, VISION_END]
        yield {
            "name": name,
            "images": images,
            "spec": spec,
            "input_ids": [7, *block * len(images), 8],
        }


def request(case):
    spec = case["spec"]
    return {
        "backend": "pil" if spec["resample"] == "pil" else "torchvision",
        "image_kwargs": {"min_pixels": spec["min_pixels"], "max_pixels": spec["max_pixels"]},
    }


def stages(case, images, expected):
    """What `ProcessedItem` does not carry, for tests/qwen_vl_golden.rs: each
    image's content hash and resized buffer."""
    import blake3

    spec, resized = case["spec"], []
    for i, data in enumerate(images):
        rgb = resized_rgb(spec, Image.open(io.BytesIO(data)).convert("RGB"))
        # The resized buffer must reproduce the baseline's features on its own.
        assert sha256(patchify(spec, rgb), "<f4") == expected["feature_sha256"][i], case["name"]
        resized.append(sha256(rgb, "|u1"))
    return {
        # Decimal strings: a JSON number cannot carry a full u64.
        "content_hash": [
            str(int.from_bytes(blake3.blake3(data).digest()[:8], "big")) for data in images
        ],
        "resized_sha256": resized,
    }


def resized_rgb(spec, image):
    """The resampler stage alone, as the HF backend that `resample` mirrors runs it."""
    from torchvision.transforms.v2 import functional as tvF
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    factor = spec["patch_size"] * spec["merge_size"]
    h, w = smart_resize(image.height, image.width, factor, spec["min_pixels"], spec["max_pixels"])
    if spec["resample"] == "pil":
        return np.asarray(image.resize((w, h), Image.BICUBIC))
    tensor = tvF.pil_to_tensor(image)
    out = tvF.resize(tensor, [h, w], tvF.InterpolationMode.BICUBIC, antialias=True)
    return out.permute(1, 2, 0).contiguous().numpy()


def patchify(spec, rgb):
    """Normalize as the HF backend does, then HF's patch order."""
    mean = np.asarray(spec["image_mean"], dtype=np.float32)
    std = np.asarray(spec["image_std"], dtype=np.float32)
    if spec["resample"] == "pil":
        x = (rgb.astype(np.float64) * (1 / 255)).astype(np.float32)
        x = (x - mean) / std
    else:
        x = (rgb.astype(np.float32) - mean * np.float32(255)) / (std * np.float32(255))
    ps, m, tps = spec["patch_size"], spec["merge_size"], spec["temporal_patch_size"]
    h, w, _ = rgb.shape
    gh, gw = h // ps, w // ps
    x = x.transpose(2, 0, 1).reshape(3, gh // m, m, ps, gw // m, m, ps)
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)  # (gh/m, gw/m, m, m, C, ps, ps)
    x = np.repeat(x[..., None, :, :], tps, axis=-3)  # C, tps, ps, ps
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * tps * ps * ps))
