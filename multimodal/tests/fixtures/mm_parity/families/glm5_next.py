# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash (`glm5_next`): one case per decode form, the PIL backend,
two images, and token-count overrides."""

import transformers
from PIL import Image

from frameworks import hf

FILE = "glm-5.3-flash.json"
TARGET = "sglang"
TARGET_REASON = (
    "SGLang serves M-RoPE positions (HF's model counts 1-D positions) and puts alpha over "
    "a gray or white chosen from the edges (HF's load_image drops it)."
)
ID_MAP = {}

# One image per decode form, as SGLang's format sweep orders them.
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
]
# Upscale to the minimum, an odd size, a non-RGB source, and a downscale.
RESIZE = ["tiny_10x10.png", "rgb_333x251.jpg", "rgba_80x60.png", "large_800x600.jpg"]
CRATE = {
    "gray16_40x30.png": {
        "known_gap": "Dynamo rejects 16-bit samples, which PIL's convert('RGB') clips"
    }
}


def cases(model_dir):
    config = transformers.AutoConfig.from_pretrained(model_dir)
    processors = {
        "auto": hf.processor(model_dir, "torchvision"),
        "pil": hf.processor(model_dir, "pil"),
    }
    block = [config.image_start_token_id, config.image_token_id, config.image_end_token_id]
    text = processors["auto"].tokenizer.encode("Describe.", add_special_tokens=False)

    def case(name, images, backend="auto", overrides=None):
        """The crate's spec for this backend, and one `<start> <image> <end>`
        block per image before the text."""
        image_processor = processors[backend].image_processor
        return {
            "name": name,
            "images": images,
            "image_processor_backend": backend,
            "image_overrides": overrides,
            "spec": spec(image_processor, config, overrides or {}),
            "input_ids": block * len(images) + text,
        }

    for name in FORMATS:
        yield case(name, [name])
    for name in RESIZE:
        yield case(f"{name} pil", [name], backend="pil")
    yield case("two images", ["rgb_96x80.png", "rgb_333x251.jpg"])
    tokens = {"min_image_tokens": 4, "max_image_tokens": 64}
    yield case("pixel overrides", ["large_800x600.jpg"], overrides=tokens)


def request(case):
    """The case as a framework runs it: the transformers backend and image kwargs."""
    backend = "pil" if case["image_processor_backend"] == "pil" else "torchvision"
    return {"backend": backend, "image_kwargs": case["image_overrides"] or {}}


def spec(image_processor, hf_config, overrides):
    """SGLang's Rust MM gate for glm5_next (`_resolve_glm5_next`): the spec the
    crate gets for this processor."""
    resample = {"Glm5NextImageProcessor": "aten_u8", "Glm5NextImageProcessorPil": "pil"}
    stages = ("do_resize", "do_rescale", "do_normalize")
    assert all(getattr(image_processor, stage, True) for stage in stages)
    assert image_processor.rescale_factor == 1 / 255
    assert image_processor.resample == Image.Resampling.BICUBIC
    assert set(overrides) <= {"min_image_tokens", "max_image_tokens"}
    knobs = ("patch_size", "merge_size", "temporal_patch_size", "patch_expand_factor")
    spec = {"family": "glm5_next"}
    spec.update({knob: getattr(image_processor, knob) for knob in knobs})
    for knob in ("min_image_tokens", "max_image_tokens"):
        spec[knob] = overrides.get(knob, getattr(image_processor, knob))
    spec.update(
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        image_mean=[float(x) for x in image_processor.image_mean],
        image_std=[float(x) for x in image_processor.image_std],
        resample=resample[type(image_processor).__name__],
    )
    return spec
