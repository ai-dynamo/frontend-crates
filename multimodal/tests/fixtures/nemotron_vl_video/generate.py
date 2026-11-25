# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the Nemotron 3.5 Super VL video golden fixtures.

Each clip is u8 RGB noise from `noise` below (THWC), which the Rust test
rebuilds. SGLang's `NanoNemotronVLImageProcessor` samples the clip itself
(`sample_video_frames` at 2 fps); every case feeds the frames it picked, with
their indices and the clip's frame rate, to one reference:

- `sglang`: `process_mm_data_async` on the CPU with the checkpoint's config,
  which resolves no `video_target_num_patches`. A case may set the processor's
  `video_target_num_patches` or turn EVS off, as another config would.
- `hf`: the checkpoint's remote-code processor, with `video_metadata` or
  without it.

`<case>/case.json` holds the crate spec with the values the reference resolved,
the clip, the prompt ids with one `<image>` placeholder where the reference prompt
has `<video>`, the tokenizer's ids for each frame label, and what the reference
returned: `input_ids`, the feature shape and dtype, and for SGLang the item
offsets and the blake3 of the bfloat16 feature bits.

    python generate.py <NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 dir, non-weight files>
"""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import blake3
import numpy as np
import torch
import transformers
from PIL import Image
from transformers.video_utils import VideoMetadata

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO, REVISION = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe"
IMAGE, IMG_START, IMG_END = 18, 19, 20
BEFORE = "<|im_start|>user\n"
AFTER = "\nDescribe this video.<|im_end|>\n<|im_start|>assistant\n"

# clip: (total frames, height, width, average fps)
CLIPS = {
    "wide": (60, 90, 160, 30.0),
    "portrait_odd": (25, 128, 96, 10.0),
    "single_frame": (7, 64, 64, 24.0),
    "thin": (45, 100, 300, 15.0),
    "wide_b": (40, 90, 160, 25.0),
}
# name: (reference, clip or clips, options)
CASES = {
    "sglang_wide": ("sglang", "wide", {}),
    "sglang_portrait_odd": ("sglang", "portrait_odd", {}),
    "sglang_single_frame": ("sglang", "single_frame", {}),
    "sglang_thin": ("sglang", "thin", {}),
    "sglang_target_1024_wide": ("sglang", "wide", {"target_num_patches": 1024}),
    "sglang_target_1024_portrait_odd": ("sglang", "portrait_odd", {"target_num_patches": 1024}),
    "sglang_no_evs_portrait_odd": ("sglang", "portrait_odd", {"no_evs": True}),
    "sglang_two_videos": ("sglang", ["wide", "portrait_odd"], {}),
    "sglang_target_1024_two_videos": ("sglang", ["wide", "wide_b"], {"target_num_patches": 1024}),
    "hf_wide": ("hf", "wide", {}),
    "hf_portrait_odd": ("hf", "portrait_odd", {}),
    "hf_single_frame": ("hf", "single_frame", {}),
    "hf_thin": ("hf", "thin", {}),
    "hf_no_fps_portrait_odd": ("hf", "portrait_odd", {"no_fps": True}),
}


def noise(shape, seed):
    """Byte `i` is `(((i + seed) * 2654435761) >> 20) & 255`, as in the fixtures README."""
    i = np.arange(int(np.prod(shape)), dtype=np.uint64)
    return (((i + np.uint64(seed)) * np.uint64(2654435761)) >> np.uint64(20)).astype(np.uint8).reshape(shape)


class Clip:
    """The decoded video SGLang's processor reads: its length, rate and frames."""

    def __init__(self, frames, fps):
        self.frames, self.avg_fps = frames, fps

    def __len__(self):
        return len(self.frames)

    def get_frames_at(self, indices):
        return self.frames[indices]


def sglang_processor(model_dir):
    from sglang.srt.hf_transformers.config import get_config
    from sglang.srt.hf_transformers.processor import get_processor
    from sglang.srt.multimodal.processors.nano_nemotron_vl import NanoNemotronVLImageProcessor
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(model_path=model_dir, trust_remote_code=True, context_length=16384,
                             mm_feature_transport="cpu", device="cpu",
                             mamba_radix_cache_strategy="no_buffer", disable_overlap_schedule=True)
    publish(server_args, role="tokenizer")
    return NanoNemotronVLImageProcessor(get_config(model_dir, trust_remote_code=True), server_args,
                                        get_processor(model_dir, trust_remote_code=True), None,
                                        skip_mm_pool=True)


def sglang_spec(p):
    evs = p.evs.evs_config
    return dict(family="nemotron_vl", sizing="sglang", image_token_id=p.mm_tokens.image_token_id,
                image_start_token_id=p.img_start_token_id, image_end_token_id=p.img_end_token_id,
                patch_size=p.patch_size, downsample_factor=int(1 / p.downsample_ratio),
                min_num_patches=p.min_num_patches, max_num_patches=p.max_num_patches,
                max_model_len=p.max_model_len, image_mean=list(p.norm_mean),
                image_std=list(p.norm_std),
                video=dict(temporal_patch_size=p.video_temporal_patch_size,
                           target_num_patches=p.video_target_num_patches,
                           maintain_aspect_ratio=p.video_maintain_aspect_ratio,
                           image_size=p.image_size,
                           pruning_rate=evs.video_pruning_rate if evs else 0.0))


def hf_spec(hf, config):
    ip = hf.image_processor
    tokens = hf.tokenizer.convert_tokens_to_ids
    return dict(family="nemotron_vl", sizing="hf", image_token_id=hf.image_token_id,
                image_start_token_id=tokens(hf.image_start_token),
                image_end_token_id=tokens(hf.image_end_token), patch_size=ip.patch_size,
                downsample_factor=ip._downsample_factor, min_num_patches=ip.min_num_patches,
                max_num_patches=ip.max_num_patches, max_model_len=ip.max_model_len,
                image_mean=list(ip.norm_mean), image_std=list(ip.norm_std),
                video=dict(temporal_patch_size=hf.video_temporal_patch_dim,
                           target_num_patches=ip.video_target_num_patches,
                           maintain_aspect_ratio=ip.video_maintain_aspect_ratio,
                           image_size=config["vision_config"]["image_size"],
                           pruning_rate=config["video_pruning_rate"]))


def labels(tokenizer, ids, start, newline):
    """Each frame label, from `start` or `</img>` to `<img>`, with its ids. After a
    newline id, the rest is recorded too: a later video's first label follows the
    prompt's newline."""
    out = {}
    for k, id_ in enumerate(ids):
        if id_ == IMG_START:
            for begin in (start, start + len(newline)):
                text = tokenizer.decode(ids[begin:k])
                out[text] = tokenizer.encode(text, add_special_tokens=False)
                assert out[text] == ids[begin:k], text
                if ids[start:start + len(newline)] != newline:
                    break
        elif id_ == IMG_END:
            start = k + 1
    return out


def runs(ids):
    spans, start = [], None
    for k, id_ in enumerate(ids + [-1]):
        if id_ == IMAGE and start is None:
            start = k
        elif id_ != IMAGE and start is not None:
            spans.append([start, k - 1])
            start = None
    return spans


def bf16_digest(feature):
    return blake3.blake3(feature.contiguous().view(torch.int16).numpy().astype("<i2").tobytes()).hexdigest()


def main(model_dir):
    from sglang.srt.multimodal.media_io import sample_video_frames

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    hf = transformers.AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = hf.tokenizer
    before = tokenizer.encode(BEFORE, add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    after = tokenizer.encode(AFTER, add_special_tokens=False)
    sglang = sglang_processor(model_dir)
    defaults = dict(target=sglang.video_target_num_patches, evs=sglang.evs.evs_config)
    sglang_commit = subprocess.run(["git", "-C", os.path.dirname(sys.modules["sglang"].__file__),
                                    "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    for name, (reference, clip_names, options) in CASES.items():
        clip_names = [clip_names] if isinstance(clip_names, str) else clip_names
        clips, media = [], []
        for clip_name in clip_names:
            total, height, width, fps = CLIPS[clip_name]
            seed = sum(clip_name.encode())
            clips.append(Clip(noise((total, height, width, 3), seed), fps))
            picked = sample_video_frames(clips[-1], desired_fps=2, max_frames=128)
            media.append(dict(shape=[total, height, width], seed=seed, frame_indices=picked,
                              fps=fps))
        # Videos are joined by a newline in the prompt.
        prompt_ids = before + [IMAGE] + (newline + [IMAGE]) * (len(clips) - 1) + after
        text = BEFORE + "\n".join(["<video>"] * len(clips)) + AFTER
        record = dict(media=media, prompt_ids=prompt_ids)
        if reference == "sglang":
            sglang.video_target_num_patches = options.get("target_num_patches", defaults["target"])
            sglang.evs.evs_config = None if options.get("no_evs") else defaults["evs"]

            async def load(**_):
                return SimpleNamespace(images=[], videos=clips, audios=[])

            sglang.load_mm_data = load
            out = asyncio.run(sglang.process_mm_data_async(
                image_data=None, audio_data=None,
                input_text=tokenizer.encode(text, add_special_tokens=False),
                request_obj=SimpleNamespace(video_data=clips, audio_data=None, rid=name)))
            (item,) = out.mm_items
            assert item.is_video() and item.feature.dtype == torch.bfloat16
            ids = [int(id_) for id_ in out.input_ids]
            assert [list(span) for span in item.offsets] == runs(ids)
            record.update(spec=sglang_spec(sglang), input_ids=ids,
                          offsets=[list(span) for span in item.offsets],
                          feature_shape=list(item.feature.shape), feature_dtype="bfloat16",
                          feature_blake3=bf16_digest(item.feature))
        else:
            ((clip,), (entry,)) = (clips, media)
            meta = {} if options.get("no_fps") else dict(video_metadata=VideoMetadata(
                total_num_frames=len(clip), fps=clip.avg_fps, frames_indices=entry["frame_indices"]))
            if options.get("no_fps"):
                entry["fps"] = 0.0
            out = hf(text=[text], videos=[Image.fromarray(frame)
                                          for frame in clip.frames[entry["frame_indices"]]], **meta)
            ids = [int(id_) for id_ in out["input_ids"][0]]
            pixels = out["pixel_values_videos"]
            record.update(spec=hf_spec(hf, config), input_ids=ids, offsets=runs(ids),
                          feature_shape=list(pixels.shape),
                          feature_dtype=str(pixels.dtype).removeprefix("torch."))
        assert ids[:len(before)] == before
        record["label_ids"] = labels(tokenizer, ids, len(before), newline)
        record["provenance"] = {
            "generator": "multimodal/tests/fixtures/nemotron_vl_video/generate.py",
            "checkpoint": f"{REPO}@{REVISION}",
            "reference": "the checkpoint's remote code" if reference == "hf"
            else f"sgl-project/sglang@{sglang_commit} NanoNemotronVLImageProcessor",
            "transformers": transformers.__version__, "torch": torch.__version__,
            "numpy": np.__version__, "pillow": Image.__version__}
        case_dir = os.path.join(ROOT, name)
        os.makedirs(case_dir, exist_ok=True)
        with open(os.path.join(case_dir, "case.json"), "w") as f:
            json.dump(record, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
