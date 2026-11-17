# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 (`kimi_k3`): alpha through the composite, sizes below one patch and
at extreme aspect ratios, every resized decode form at a 4-patch budget, two
images, and each `media_proc_cfg` knob. The decode forms at the shipped limits
are rows of decode.json (`decode_native`)."""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import transformers

from frameworks import reason, sha256, spans
from frameworks import sglang as sglang_framework
from frameworks import vllm as vllm_framework

FILE = "kimi-k3.json"
# HF and SGLang's CPU path give the same values. SGLang's CUDA path resizes on
# the GPU, which is not bit-exact: run the driver with CUDA_VISIBLE_DEVICES="".
TARGET = "sglang"
ID_MAP = {}
NATIVE_MODE = (
    "Dynamo refuses a palette, 1-bit, L with tRNS or CMYK source that needs a "
    "resize; Pillow resizes it in its native mode"
)
CRATE = {
    "gray16_40x30.png": {
        "rust_error": "InvalidInput",
        "known_gap": "Dynamo rejects 16-bit samples, which PIL's convert('RGB') clips",
    },
    "truncated_96x80.png": {"rust_error": "InvalidInput"},
    "odd side limit": {"rust_error": "InvalidInput"},
    "two placeholders one image": {"rust_error": "InvalidInput"},
    **{
        f"{name} small": {"rust_error": "Unsupported", "known_gap": NATIVE_MODE}
        for name in [
            "cmyk_80x64.jpg",
            "palette_transparency_70x50.png",
            "animated_48x40.gif",
            "gray_trns_64x48.png",
            "palette_64x48.png",
            "bilevel_64x48.png",
            "bilevel_trns_64x48.png",
            "transparent_56x56.gif",
        ]
    },
}

# At the shipped limits: the alpha forms, which the composite changes, the
# sizes, and two refusals.
SHIPPED = [
    "rgb_96x80.png",
    "rgba_80x60.png",
    "palette_transparency_70x50.png",
    "gray16_40x30.png",
    "lossless_alpha_60x40.webp",
    "tiny_10x10.png",
    "aspect_201_4020x20.png",
    "alpha_noise_84x60.png",
    "gray_alpha_noise_84x60.png",
    "rgb_trns_64x48.png",
    "gray_trns_64x48.png",
    "bilevel_trns_64x48.png",
    "transparent_56x56.gif",
    "lossy_alpha_60x40.webp",
    "rgba_bitfields_56x56.bmp",
    "wide_300x24.png",
    "wide_alpha_300x24.png",
    "tall_2x1000.png",
    "tall_alpha_3x700.png",
    "one_1x1.png",
    "rgb_3x5.png",
    "truncated_96x80.png",
]
# Every decode form the crate accepts with more than 4 patches: the budget resizes each.
SMALL = [
    "rgb_96x80.png",
    "rgb_333x251.jpg",
    "rgb_444_120x90.jpg",
    "progressive_200x150.jpg",
    "cmyk_80x64.jpg",
    "exif_orientation6_64x48.jpg",
    "gray_64x64.png",
    "rgba_80x60.png",
    "palette_transparency_70x50.png",
    "animated_48x40.gif",
    "lossy_90x70.webp",
    "lossless_alpha_60x40.webp",
    "rgb_50x40.bmp",
    "large_800x600.jpg",
    "aspect_201_4020x20.png",
    "alpha_noise_84x60.png",
    "gray_alpha_noise_84x60.png",
    "rgb_trns_64x48.png",
    "gray_trns_64x48.png",
    "palette_64x48.png",
    "bilevel_64x48.png",
    "bilevel_trns_64x48.png",
    "gray4_64x48.png",
    "transparent_56x56.gif",
    "lossy_alpha_60x40.webp",
    "rgbx_56x40.bmp",
    "rgba_bitfields_56x56.bmp",
    "gray_90x70.jpg",
    "wide_300x24.png",
    "wide_alpha_300x24.png",
    "tall_2x1000.png",
    "tall_alpha_3x700.png",
    "libvpx_filter_above63_96x80.webp",
    "libvpx_filter_in_range_96x80.webp",
]
MEDIUM = {"in_patch_limit": 16}
BACKGROUND = {
    "pattern": "chessboard",
    "chessboard_square_size": 5,
    "chessboard_square_on_top_left": False,
    "chessboard_white_value": 240,
    "chessboard_gray_value": 100,
}
CLIP = {
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}
SPEC_KEYS = [
    "patch_size",
    "merge_kernel_size",
    "in_patch_limit",
    "patch_limit_on_one_side",
    "image_mean",
    "image_std",
    "transparent_bg_config",
    "transparent_bg_fill_stage",
    "fixed_output_tokens",
]
ONE = "What is in <|media_pad|>?"
TWO = "A <|media_pad|> and B <|media_pad|> end"


def tokenizer(model_dir):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return lambda text: list(tokenizer.encode(text, allowed_special="all"))


def cases(model_dir):
    """The prompt has one `<|media_pad|>` per image; the overrides go into `media_proc_cfg`."""
    encode = tokenizer(model_dir)
    shipped = json.loads((model_dir / "preprocessor_config.json").read_text())["media_proc_cfg"]
    (pad,) = encode("<|media_pad|>")

    def case(name, images, overrides=None, prompt=ONE):
        cfg = {**shipped, **(overrides or {})}
        return {
            "name": name,
            "images": images,
            "media_proc_cfg": overrides,
            "spec": {
                "family": "kimi_k3",
                "image_token_id": pad,
                **{key: cfg[key] for key in SPEC_KEYS},
            },
            "input_ids": encode(prompt),
        }

    for name in SHIPPED:
        yield case(name, [name])
    for name in SMALL:
        yield case(f"{name} small", [name], {"in_patch_limit": 4})
    yield case("gray_alpha_noise_84x60.png medium", ["gray_alpha_noise_84x60.png"], MEDIUM)
    yield case("side cap", ["wide_300x24.png"], {"patch_limit_on_one_side": 8})
    yield case("tall", ["tall_2x1000.png"], {"patch_limit_on_one_side": 16})
    yield case("two images", ["rgb_96x80.png", "alpha_noise_84x60.png"], prompt=TWO)
    two = ["rgb_333x251.jpg", "alpha_noise_84x60.png"]
    yield case("two images medium", two, MEDIUM, prompt=TWO)
    background = {**MEDIUM, "transparent_bg_config": BACKGROUND}
    yield case("background", ["alpha_noise_84x60.png"], background)
    yield case("clip normalize", ["rgb_96x80.png"], CLIP)
    # 7 px * 14 = 98 px pads to 112 px = 4 tokens of 2 patches, over the limit.
    yield case("odd side limit", ["wide_300x24.png"], {"patch_limit_on_one_side": 7})
    yield case("two placeholders one image", ["rgb_96x80.png"], prompt=TWO)


def request(case):
    """Kimi-K3 has one image processor; `media_proc_cfg` takes the overrides."""
    return {"backend": "pil", "image_kwargs": case["media_proc_cfg"] or {}}


class HfPath:
    """The checkpoint's remote-code `KimiK3Processor`: the text with one
    `<|kimi_image_placeholder|>` per image, and the image bytes, which its
    `media_utils` decodes. Its model expands each `<|media_pad|>` to the
    image's token count."""

    SOURCE = (
        "KimiK3Processor(medias=[image bytes], text) with media_proc_cfg overrides, "
        "each <|media_pad|> expanded to media_tokens_calculator"
    )
    DECODE = "media_utils.ensure_media_type (the checkpoint's remote code)"

    def __init__(self, model_dir):
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.shipped = copy.deepcopy(self.processor.image_processor.media_proc_cfg)
        (self.pad,) = self.encode("<|media_pad|>")

    def encode(self, text):
        return list(self.processor.tokenizer.encode(text, allowed_special="all"))

    def prompt(self, case):
        text = self.processor.tokenizer.decode(case["input_ids"])
        assert self.encode(text) == case["input_ids"], case["name"]
        return text.replace("<|media_pad|>", self.processor.image_placeholder)

    def process(self, text, data):
        """The framework's own steps: each image's token count, then the processor."""
        medias = [{"type": "image", "image": image} for image in data]
        calculator = self.processor.image_processor.media_tokens_calculator
        counts = [calculator(dict(media)) for media in medias]
        return counts, self.processor(medias=medias, text=text, return_tensors="pt")

    def record(self, counts, out):
        counts = iter(counts)
        ids = [
            id_
            for model_id in out["input_ids"][0].tolist()
            for id_ in ([model_id] * next(counts) if model_id == self.pad else [model_id])
        ]
        grids = out["grid_thws"].tolist()
        features = out["pixel_values"].float().numpy()
        features = np.split(features, np.cumsum([t * h * w for t, h, w in grids])[:-1])
        return {
            "input_ids": ids,
            "offsets": spans(ids, self.pad),
            "image_grid_thw": grids,
            "feature_shapes": [list(feature.shape) for feature in features],
            "feature_sha256": [sha256(feature, "<f4") for feature in features],
            "mrope_sha256": None,
            "mrope_delta": None,
        }

    def run(self, request):
        cases = {}
        cfg = self.processor.image_processor.media_proc_cfg
        for case in request["cases"]:
            text = self.prompt(case)
            cfg.clear()
            cfg.update(copy.deepcopy(self.shipped), **case["image_kwargs"])
            data = [(Path(request["images_dir"]) / name).read_bytes() for name in case["images"]]
            try:
                counts, out = self.process(text, data)
            except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
                cases[case["name"]] = {"rejects": reason(error)}
                continue
            cases[case["name"]] = {"output": self.record(counts, out)}
        return {
            "version": transformers.__version__,
            "source": self.SOURCE,
            "processor": type(self.processor).__name__,
            "backend": type(self.processor.image_processor).__name__,
            "decode": self.DECODE,
            "omits": [],
            "cases": cases,
        }


class SglangPath(sglang_framework.SglangPath):
    """SGLang's Kimi-K3 processor over one `media_proc_cfg`. With CUDA it
    resizes on the GPU, which is not bit-exact with the checkpoint's processor."""

    # The checkpoint whose server args SGLang resolves, when not `model_dir`; the
    # image path reads only their multimodal settings.
    server_model_dir = None

    def __init__(self, model_dir, backend, overrides):
        import torch
        from sglang.srt.hf_transformers.config import get_config
        from sglang.srt.hf_transformers.processor import get_processor
        from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        assert not torch.cuda.is_available(), 'run with CUDA_VISIBLE_DEVICES=""'
        publish(
            ServerArgs(
                model_path=str(self.server_model_dir or model_dir),
                trust_remote_code=True,
                device="cpu",
                mm_feature_transport="cpu",
                mm_preprocess_cache_size_mb=0,
                # Scheduler settings a CPU host needs to resolve this model's args.
                mamba_radix_cache_strategy="no_buffer",
                disable_overlap_schedule=True,
            ),
            role="tokenizer",
        )
        # PIL decode (read by a classmethod), not nvJPEG.
        KimiK3ImageProcessor.gpu_image_decode = False
        hf_config = get_config(str(model_dir), trust_remote_code=True)
        processor = get_processor(str(model_dir), trust_remote_code=True)
        processor.image_processor.media_proc_cfg.update(overrides or {})
        self.processor = KimiK3ImageProcessor(
            hf_config, SimpleNamespace(), processor, None, skip_mm_pool=True
        )
        assert not self.processor.mm_preprocess_cache.enabled

    @staticmethod
    def grid(item):
        return item.model_specific_data["image_grid_thw"].flatten().tolist()


def run_hf(request, args):
    return HfPath(request["model_dir"]).run(request)


def run_sglang(request, args):
    return sglang_framework.run(request, args, SglangPath)


def run_vllm(request, args):
    """vLLM's processor reads `media_proc_cfg` from the checkpoint only, so a
    case with overrides is `n/a`."""
    plain = [case for case in request["cases"] if not case["image_kwargs"]]
    response = vllm_framework.run({**request, "cases": plain}, args)
    why = "vLLM's Kimi-K3 processor reads media_proc_cfg from the checkpoint only"
    for case in request["cases"]:
        if case["image_kwargs"]:
            response["cases"][case["name"]] = {"n/a": why}
    return response


# Columns whose module in frameworks/ cannot run this checkpoint's processor as
# the case asks: the remote-code processor, SGLang's Kimi-K3 class, and the
# `media_proc_cfg` overrides that vLLM does not read.
ADAPTERS = {"hf": run_hf, "sglang": run_sglang, "vllm": run_vllm}


def spec_shared(model_dir):
    """The size header's token table (`KimiK3Spec::size_tokens`), checked
    against the tokenizer over 60000 sizes."""
    encode = tokenizer(model_dir)

    def single(text):
        (token,) = encode(text)
        return token

    tokens = {
        "begin": encode("<|media_begin|>image "),
        "by": encode("x"),
        "content": encode("<|media_content|>"),
        "end": encode("<|media_end|>"),
        "digits": [single(str(n).zfill(width)) for width in (1, 2, 3) for n in range(10**width)],
    }
    rng = random.Random(0)
    sizes = [(side, 7) for side in range(1, 20001)] + [(7, side) for side in range(1, 20001)]
    sizes += [(int(2 ** rng.uniform(0, 32)), int(2 ** rng.uniform(0, 32))) for _ in range(20000)]
    for width, height in sizes:
        text = f"<|media_begin|>image {width}x{height}<|media_content|>"
        assert header(tokens, width, height) == encode(text), (width, height)
    return {"size_tokens": tokens}


def header(tokens, width, height):
    """`SizeTokens::header`: each number in 3-digit groups from the left."""

    def number(value):
        text = str(value)
        groups = [text[i : i + 3] for i in range(0, len(text), 3)]
        return [tokens["digits"][[0, 10, 110][len(group) - 1] + int(group)] for group in groups]

    return tokens["begin"] + number(width) + tokens["by"] + number(height) + tokens["content"]
