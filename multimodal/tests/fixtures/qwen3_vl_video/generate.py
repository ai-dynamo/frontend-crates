# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the Qwen3-VL video golden fixtures from HF's processor.

Each case's media are u8 RGB noise from `noise` below (THWC for a video), which
the Rust test rebuilds. `<case>/case.json` holds what HF produces: the spec,
the media, the prompt ids, `Qwen3VLProcessor.__call__`'s `input_ids` and the
token ids of each timestamp text, the video grids, the M-RoPE delta, and the
blake3 of each stage: a resized video's frames after
`Qwen3VLVideoProcessor.resize`, `pixel_values_videos` (little-endian f32, every
video in order), and `get_rope_index`'s positions (little-endian i64, `[3, len]`).

`input_ids` come from the processor with Qwen3-VL-2B-Instruct's tokenizer, so
the expansion is HF's own; the Rust test encodes timestamps through the
recorded map, as a host encodes them through its tokenizer.

    python multimodal/tests/fixtures/qwen3_vl_video/generate.py
"""

import json
import os

import blake3
import numpy as np
import torch
import torchvision
import transformers
from torchvision.transforms.v2 import functional as tvF
from transformers import AutoTokenizer, Qwen3VLConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
    smart_resize,
)
from transformers.video_utils import VideoMetadata

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD = 151652, 151653, 151655, 151656
V = "<|vision_start|><|video_pad|><|vision_end|>"
I = "<|vision_start|><|image_pad|><|vision_end|>"

# Qwen/Qwen3-VL-2B-Instruct's processor configs, except where a case narrows
# a bound to keep the fixture small or to force a downscale.
IMAGE = dict(patch_size=16, merge_size=2, temporal_patch_size=2,
             min_pixels=65536, max_pixels=16777216,
             image_mean=[0.5] * 3, image_std=[0.5] * 3)
VIDEO = dict(patch_size=16, merge_size=2, temporal_patch_size=2,
             size=dict(shortest_edge=4096, longest_edge=25165824),
             image_mean=[0.5] * 3, image_std=[0.5] * 3)

# media: ("video", (frames, height, width), frame_indices, fps) or ("image", (height, width)).
CASES = {
    # An odd frame count: HF repeats the last frame to fill the temporal patch.
    "odd_3f": dict(prompt=f"Watch {V}now.",
                   media=[("video", (3, 32, 64), [0, 14, 29], 29.97002997002997)]),
    # Above a narrowed clip maximum: every frame is downscaled.
    "downscale_6f": dict(prompt=f"Describe {V}briefly.",
                         media=[("video", (6, 60, 80), [0, 9, 19, 29, 39, 49], 25.0)],
                         longest_edge=12288),
    # Two videos around an image: items in placeholder order, one resized
    # and one unresized clip.
    "video_image_video": dict(prompt=f"{V}then {I} and {V}Describe.",
                              media=[("video", (3, 30, 40), [0, 7, 15], 30.0),
                                     ("image", (48, 80)),
                                     ("video", (3, 32, 32), [0, 3, 5], 2.0)],
                              image_min_pixels=1024),
    # A 3-frame temporal patch for video while images keep 2; the timestamps
    # 0.25 and 0.75 are exact ties.
    "tps3": dict(prompt=f"{V}What?", media=[("video", (4, 32, 32), [0, 1, 2, 3], 4.0)], tps=3),
}


def noise(shape, seed):
    """Byte `i` is `(((i + seed) * 2654435761) >> 20) & 255`, as in the fixtures README."""
    i = np.arange(int(np.prod(shape)), dtype=np.uint64)
    return (((i + np.uint64(seed)) * np.uint64(2654435761)) >> np.uint64(20)).astype(np.uint8).reshape(shape)


def digest(array):
    return blake3.blake3(np.ascontiguousarray(array).tobytes()).hexdigest()


def image_spec(case):
    return dict(IMAGE, min_pixels=case.get("image_min_pixels", IMAGE["min_pixels"]))


def video_processor(case):
    video = dict(VIDEO, temporal_patch_size=case.get("tps", 2))
    if "longest_edge" in case:
        video["size"] = dict(video["size"], longest_edge=case["longest_edge"])
    return Qwen3VLVideoProcessor(**video), video


def resized_frames(vp, frames_tchw):
    """`Qwen3VLVideoProcessor.resize` alone, on the stacked uint8 frames, or
    `None` when the frames keep their size."""
    t, _, h, w = frames_tchw.shape
    th, tw = smart_resize(
        num_frames=t, height=h, width=w, temporal_factor=vp.temporal_patch_size,
        factor=vp.patch_size * vp.merge_size,
        min_pixels=vp.size["shortest_edge"], max_pixels=vp.size["longest_edge"])
    if (th, tw) == (h, w):
        return None
    return tvF.resize(frames_tchw, [th, tw], tvF.InterpolationMode.BICUBIC, antialias=True)


def normalize_patchify(video, frames_thwc):
    """Fused normalize and HF's video flatten order, from resized frames."""
    mean = np.asarray(video["image_mean"], dtype=np.float32)
    std = np.asarray(video["image_std"], dtype=np.float32)
    x = (frames_thwc.astype(np.float32) - mean * np.float32(255)) / (std * np.float32(255))
    ps, m, tps = video["patch_size"], video["merge_size"], video["temporal_patch_size"]
    t, h, w, _ = x.shape
    if t % tps:
        x = np.concatenate([x, np.repeat(x[-1:], tps - t % tps, axis=0)])
        t = x.shape[0]
    gt, gh, gw = t // tps, h // ps, w // ps
    x = x.transpose(0, 3, 1, 2).reshape(gt, tps, 3, gh // m, m, ps, gw // m, m, ps)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)  # gt, gh/m, gw/m, m, m, C, tps, ps, ps
    return np.ascontiguousarray(x.reshape(gt * gh * gw, 3 * tps * ps * ps))


def rope_model():
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=4, head_dim=8,
                         vocab_size=1024,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2],
                                       "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, deepstack_visual_indexes=[]),
        image_token_id=IMAGE_PAD, video_token_id=VIDEO_PAD,
        vision_start_token_id=VISION_START, vision_end_token_id=VISION_END,
    )
    with torch.device("meta"):
        return Qwen3VLModel(cfg)


def generate(name, case, tokenizer):
    case_dir = os.path.join(ROOT, name)
    os.makedirs(case_dir, exist_ok=True)
    vp, video = video_processor(case)
    processor = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(**image_spec(case)),
                                 tokenizer=tokenizer, video_processor=vp)

    images, videos, metadata, media, features, timestamp_ids = [], [], [], [], [], {}
    for k, (kind, shape, *timing) in enumerate(case["media"]):
        seed = 1000 * k + sum(name.encode())
        rgb = noise((*shape, 3), seed)
        entry = dict(kind=kind, shape=list(shape), seed=seed)
        if kind == "image":
            images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
            media.append(entry)
            continue
        indices, fps = timing
        tchw = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
        videos.append(tchw)
        metadata.append(VideoMetadata(total_num_frames=max(indices) + 1, fps=fps,
                                      frames_indices=indices))
        resized = resized_frames(vp, tchw)
        if resized is not None:
            entry["resized_blake3"] = digest(resized.permute(0, 2, 3, 1).contiguous().numpy())
            tchw = resized
        features.append(normalize_patchify(video, tchw.permute(0, 2, 3, 1).numpy()))
        for seconds in Qwen3VLProcessor._calculate_timestamps(None, list(indices), fps,
                                                              vp.temporal_patch_size):
            text = f"<{seconds:.1f} seconds>"
            timestamp_ids[text] = tokenizer.encode(text, add_special_tokens=False)
        media.append(dict(entry, frame_indices=indices, fps=fps))

    kwargs = dict(images=images) if images else {}
    out = processor(text=[case["prompt"]], videos=videos, video_metadata=metadata,
                    do_sample_frames=False, cap_pixels_per_frame=False, return_tensors="pt",
                    **kwargs)
    pixel_values_videos = out["pixel_values_videos"].numpy().astype(np.float32)
    assert np.array_equal(np.concatenate(features), pixel_values_videos), (
        f"{name}: the resized frames do not reproduce pixel_values_videos")

    ids = out["input_ids"]
    positions, delta = rope_model().get_rope_index(
        ids, out["mm_token_type_ids"], image_grid_thw=out.get("image_grid_thw"),
        video_grid_thw=out["video_grid_thw"], attention_mask=torch.ones_like(ids))
    mrope = positions[:, 0, :].reshape(-1).numpy().astype("<i8")

    spec = dict(family="qwen_vl", image_token_id=IMAGE_PAD, **image_spec(case),
                resample="aten_u8",
                video=dict(video_token_id=VIDEO_PAD, vision_start_token_id=VISION_START,
                           vision_end_token_id=VISION_END,
                           processor=dict(video_processor_type="Qwen3VLVideoProcessor",
                                          **{k: video[k] for k in ("patch_size", "merge_size",
                                                                   "temporal_patch_size", "size")}),
                           image_mean=video["image_mean"], image_std=video["image_std"]))
    record = dict(
        spec=spec, media=media,
        prompt_ids=tokenizer.encode(case["prompt"], add_special_tokens=False),
        timestamp_ids=timestamp_ids, input_ids=ids[0].tolist(),
        video_grids=out["video_grid_thw"].tolist(), mrope_delta=int(delta.reshape(-1)[0]),
        pixel_values_videos_blake3=digest(pixel_values_videos.astype("<f4")),
        mrope_blake3=digest(mrope),
        provenance={"generator": "multimodal/tests/fixtures/qwen3_vl_video/generate.py",
                    "tokenizer": f"{MODEL}@{REVISION}",
                    "transformers": transformers.__version__, "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "numpy": np.__version__})
    with open(os.path.join(case_dir, "case.json"), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    assert tokenizer.convert_tokens_to_ids(
        ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    ) == [VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD]
    for name, case in CASES.items():
        generate(name, case, tokenizer)
