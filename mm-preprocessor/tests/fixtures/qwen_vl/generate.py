# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the qwen_vl golden fixtures from the HF processor.

Each `<case>/case.json` carries the spec and the tokenized prompt; the case's
`input_<i>.png` files are the images. This script writes, per case:

* `resized_<i>.u8`      — HWC u8 RGB after `smart_resize` + the spec's resampler
                          (the stage boundary the Rust kernel must hit bitwise);
* `pixel_values.f32le`  — HF `pixel_values`, little-endian f32, all items;
* `mrope.i64le`         — `get_rope_index` positions, `[3, len]` row-major;
* the derived fields of `case.json` (input_ids, grids, offsets, hashes,
  mrope_delta) plus the library versions that produced them.

The resized buffer is cross-checked here: normalizing and patchifying it must
reproduce HF's `pixel_values` byte for byte, so the two fixtures describe the
same pipeline. Run with a Python environment holding transformers, torch,
torchvision, Pillow, numpy and blake3:

    python mm-preprocessor/tests/fixtures/qwen_vl/generate.py [case ...]
"""

import json
import os
import sys

import blake3
import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from torchvision.transforms.v2 import functional as tvF
from transformers import Qwen2VLConfig
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorPil,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    smart_resize,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

ROOT = os.path.dirname(os.path.abspath(__file__))
# Fixed placeholder ids shared by every case's prompt.
VISION_START, IMAGE_PAD, VISION_END = 901, 900, 902

PROCESSOR_KWARGS = (
    "min_pixels",
    "max_pixels",
    "patch_size",
    "merge_size",
    "temporal_patch_size",
    "image_mean",
    "image_std",
)


def hf_processor(spec):
    cls = Qwen2VLImageProcessorPil if spec["resample"] == "pil" else Qwen2VLImageProcessor
    return cls(**{k: spec[k] for k in PROCESSOR_KWARGS})


def resized_u8(spec, image):
    """The resampler stage in isolation, as the mirrored HF backend runs it."""
    factor = spec["patch_size"] * spec["merge_size"]
    h, w = smart_resize(
        image.height, image.width, factor, spec["min_pixels"], spec["max_pixels"]
    )
    if spec["resample"] == "pil":
        return np.asarray(image.resize((w, h), Image.BICUBIC))
    tensor = tvF.pil_to_tensor(image)
    out = tvF.resize(tensor, [h, w], tvF.InterpolationMode.BICUBIC, antialias=True)
    return out.permute(1, 2, 0).contiguous().numpy()


def normalize_patchify(spec, rgb):
    """Reference normalize + HF flatten order, from the resized HWC u8 buffer."""
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


def rope_model(spec):
    cfg = Qwen2VLConfig(
        text_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=1024,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=1,
            embed_dim=32,
            hidden_size=32,
            num_heads=4,
            mlp_ratio=1,
            patch_size=spec["patch_size"],
            spatial_merge_size=spec["merge_size"],
            temporal_patch_size=spec["temporal_patch_size"],
        ),
        image_token_id=IMAGE_PAD,
        video_token_id=903,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
    )
    with torch.device("meta"):
        return Qwen2VLModel(cfg)


def generate(case_dir):
    path = os.path.join(case_dir, "case.json")
    with open(path) as f:
        case = json.load(f)
    spec = case["spec"]
    assert spec["family"] == "qwen_vl" and spec["image_token_id"] == IMAGE_PAD

    images, hashes = [], []
    i = 0
    while os.path.exists(os.path.join(case_dir, f"input_{i}.png")):
        with open(os.path.join(case_dir, f"input_{i}.png"), "rb") as f:
            data = f.read()
        hashes.append(str(int.from_bytes(blake3.blake3(data).digest()[:8], "big")))
        images.append(Image.open(os.path.join(case_dir, f"input_{i}.png")).convert("RGB"))
        i += 1

    out = hf_processor(spec)(images=images, return_tensors="pt")
    pixel_values = out["pixel_values"].contiguous().numpy().astype(np.float32)
    grids = out["image_grid_thw"].tolist()

    # Stage fixture + cross-check: resized → normalize/patchify == HF pixel_values.
    offset = 0
    for i, image in enumerate(images):
        rgb = resized_u8(spec, image)
        patches = normalize_patchify(spec, rgb)
        n = patches.shape[0]
        assert np.array_equal(patches, pixel_values[offset : offset + n]), (
            f"{case_dir}: resized_{i} does not reproduce pixel_values"
        )
        offset += n
        with open(os.path.join(case_dir, f"resized_{i}.u8"), "wb") as f:
            f.write(rgb.tobytes())
    assert offset == pixel_values.shape[0]

    counts = [t * h * w // spec["merge_size"] ** 2 for t, h, w in grids]
    input_ids, offsets, item = [], [], 0
    for tok in case["prompt_ids"]:
        if tok == IMAGE_PAD:
            offsets.append([len(input_ids), len(input_ids) + counts[item] - 1])
            input_ids.extend([IMAGE_PAD] * counts[item])
            item += 1
        else:
            input_ids.append(tok)
    assert item == len(images)

    ids = torch.tensor([input_ids])
    positions, delta = rope_model(spec).get_rope_index(
        ids,
        (ids == IMAGE_PAD).int(),
        image_grid_thw=torch.tensor(grids),
        attention_mask=torch.ones_like(ids),
    )
    positions = positions[:, 0, :].reshape(-1).to(torch.int64).numpy()

    with open(os.path.join(case_dir, "pixel_values.f32le"), "wb") as f:
        f.write(pixel_values.tobytes())
    with open(os.path.join(case_dir, "mrope.i64le"), "wb") as f:
        f.write(positions.astype("<i8").tobytes())

    case.update(
        input_ids=input_ids,
        grids=grids,
        offsets=offsets,
        hashes=hashes,
        mrope_delta=int(delta.reshape(-1)[0]),
        provenance={
            "generator": "mm-preprocessor/tests/fixtures/qwen_vl/generate.py",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
        },
    )
    with open(path, "w") as f:
        json.dump(case, f, indent=1)
    print(f"{os.path.basename(case_dir)}: grids={grids} tokens={counts} delta={case['mrope_delta']}")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(
        d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))
    )
    for name in names:
        generate(os.path.join(ROOT, name))
