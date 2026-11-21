# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the GLM-5.3-Flash video parity fixtures from two references.

Each case's media are u8 RGB noise from `noise` below (THWC for a source clip),
which the Rust test rebuilds. Both references take the whole source clip and
sample it themselves:

- `hf`: transformers' `Glm5NextProcessor` with `videos=` the clip and its
  `VideoMetadata`.
- `sglang`: SGLang's GLM processor (`process_mm_data_async`), with a decoder
  that serves the same frames, through mm_parity's `frameworks.sglang.SglangPath`.

`<case>/case.json` holds, per reference, the sampled source frames, the
expanded `input_ids`, the video grids, the inclusive feature spans of each
video, and the blake3 of `pixel_values_videos` (little-endian f32), plus SGLang's
M-RoPE (little-endian i64, `[3, len]`) and delta. `sglang_differs` names the fields
where SGLang and HF differ. The Rust test gates on `sglang`.

    python multimodal/tests/fixtures/glm5_next_video/generate.py <GLM-5.3-Flash dir>

The directory needs the checkpoint's json and jinja files, not the weights.
"""

import asyncio
import io
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import blake3
import numpy as np
import torch
import torchvision
import transformers
from PIL import Image
from transformers.video_utils import VideoMetadata

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "mm_parity"))
from frameworks import reason, spans  # noqa: E402
from frameworks.sglang import SglangPath  # noqa: E402
from frameworks.sglang import version as sglang_version  # noqa: E402

MODEL, REVISION = "zai-org/GLM-5.3-Flash", "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
IMAGE_TOKEN, VIDEO_TOKEN = 154854, 154855
VIDEO_START, VIDEO_END = 154832, 154833
V = "<|begin_of_video|><|video|><|end_of_video|>"
I = "<|begin_of_image|><|image|><|end_of_image|>"

# media: ("video", (frames, height, width), fps) or ("image", (height, width)).
# `max_image_tokens` narrows the video processor's budget for the whole clip.
CASES = {
    # 3 s at 30 fps: one frame per 0.5 s; small frames are scaled up and padded.
    "fps30_3s": dict(prompt=f"{V}Describe.", media=[("video", (90, 48, 64), 30.0)]),
    # 2.5 s at 10 fps: the loop stops at 2 s, linspace refills 5 frames, and the
    # odd count repeats the last frame.
    "odd_fill": dict(prompt=f"Watch {V}now.", media=[("video", (25, 40, 40), 10.0)]),
    # 1 fps is below the 2 fps target: linspace repeats frames, which are then deduplicated.
    "slow_source": dict(prompt=f"{V}What?", media=[("video", (4, 56, 28), 1.0)]),
    # NTSC rate, a tall frame, and timestamps that are not multiples of 0.5.
    "ntsc_tall": dict(prompt=f"{V}Go.", media=[("video", (60, 70, 30), 30000 / 1001)]),
    # Under 0.5 s: HF's sampler keeps no frame.
    "short_clip": dict(prompt=f"{V}Hi.", media=[("video", (6, 28, 28), 30.0)]),
    # Above a narrowed budget: the binary search shrinks the shared canvas.
    "budget_cut": dict(prompt=f"{V}Sum up.", media=[("video", (12, 96, 128), 4.0)],
                       max_image_tokens=64),
    # An image between two videos under a budget that two videos split.
    "video_image_video": dict(prompt=f"{V}then {I} and {V}Compare.",
                              media=[("video", (8, 140, 140), 4.0),
                                     ("image", (48, 80)),
                                     ("video", (8, 140, 112), 4.0)],
                              max_image_tokens=80),
}


def noise(shape, seed):
    """Byte `i` is `(((i + seed) * 2654435761) >> 20) & 255`, as in the Qwen3-VL fixtures."""
    i = np.arange(int(np.prod(shape)), dtype=np.uint64)
    return (((i + np.uint64(seed)) * np.uint64(2654435761)) >> np.uint64(20)).astype(np.uint8).reshape(shape)


def digest(array):
    return blake3.blake3(np.ascontiguousarray(array).tobytes()).hexdigest()


def media_of(name, case):
    out = []
    for k, (kind, shape, *rest) in enumerate(case["media"]):
        seed = 1000 * k + sum(name.encode())
        out.append(dict(kind=kind, shape=list(shape), seed=seed, fps=rest[0] if rest else None,
                        rgb=noise((*shape, 3), seed)))
    return out


def model_variant(model_dir, max_image_tokens):
    """A copy of the checkpoint's processor files with a narrowed video budget."""
    if max_image_tokens is None:
        return model_dir
    variant = tempfile.mkdtemp(prefix="glm5_next_video_")
    for entry in os.listdir(model_dir):
        if entry.endswith((".json", ".jinja")):
            shutil.copy(os.path.join(model_dir, entry), variant)
    path = os.path.join(variant, "processor_config.json")
    with open(path) as f:
        config = json.load(f)
    config["video_processor"]["max_image_tokens"] = max_image_tokens
    with open(path, "w") as f:
        json.dump(config, f)
    return variant


def video_spans(ids):
    """The inclusive spans of image tokens inside each video, one list per video."""
    videos = []
    for start, end in spans(ids, IMAGE_TOKEN):
        if ids[:start].count(VIDEO_START) == ids[:start].count(VIDEO_END):
            continue
        if not videos or VIDEO_START in ids[videos[-1][-1][1]:start]:
            videos.append([])
        videos[-1].append([start, end])
    return videos


def summary(ids, sampled, video_grids, features):
    return dict(sampled=sampled, input_ids=ids, video_grids=video_grids,
                video_spans=video_spans(ids),
                pixel_values_videos_blake3=digest(np.concatenate(features).astype("<f4")),
                pixel_values_videos_shape=list(np.concatenate(features).shape))


def run_hf(model_dir, case, media, tokenizer):
    processor = transformers.AutoProcessor.from_pretrained(model_dir)
    images = [Image.fromarray(m["rgb"]) for m in media if m["kind"] == "image"]
    videos, metadata = [], []
    for m in media:
        if m["kind"] == "video":
            videos.append(torch.from_numpy(m["rgb"]).permute(0, 3, 1, 2).contiguous())
            total = m["shape"][0]
            # transformers' own video loaders set duration = total / fps.
            metadata.append(VideoMetadata(total_num_frames=total, fps=m["fps"],
                                          duration=total / m["fps"],
                                          frames_indices=list(range(total))))
    try:
        out = processor(text=[case["prompt"]], images=images or None, videos=videos,
                        video_metadata=metadata, return_tensors="pt")
    except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
        return {"rejects": reason(error)}
    # The processor drops its metadata; its sampler is a pure function of it.
    sampled = [processor.video_processor.sample_frames(meta).tolist() for meta in metadata]
    grids = out["video_grid_thw"].tolist()
    assert [g[0] * 2 for g in grids] == [len(s) for s in sampled], (grids, sampled)
    pixels = out["pixel_values_videos"].float().numpy()
    return summary(out["input_ids"][0].tolist(), sampled, grids, [pixels])


class NoiseVideo:
    """A decoded source clip in SGLang's decoder interface, frames as NHWC uint8."""

    def __init__(self, frames, fps):
        self.frames, self.fps = frames, fps

    def __len__(self):
        return len(self.frames)

    @property
    def avg_fps(self):
        return self.fps

    def get_frames_as_tensor(self, indices):
        return torch.from_numpy(self.frames[list(indices)])

    def close(self):
        pass


def run_sglang(path, case, media, tokenizer):
    import sglang.srt.multimodal.multimodal_processor_mixin as mixin
    import sglang.srt.multimodal.processors.glm4v as glm4v
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    # glm4v routes only VideoDecoderWrapper instances through its sampler.
    source = type("NoiseVideoWrapper", (NoiseVideo, VideoDecoderWrapper), {})
    clips = {f"noise:{k}": source(m["rgb"], m["fps"]) for k, m in enumerate(media) if m["kind"] == "video"}
    sampled = {}
    load_video, sample_and_decode = mixin.load_video, glm4v.glm_sample_and_decode_sync

    def recording(vr, *args, **kwargs):
        frames, meta = sample_and_decode(vr, *args, **kwargs)
        sampled[id(vr)] = [int(i) for i in meta["frames_indices"]]
        return frames, meta

    mixin.load_video = lambda data, use_gpu=True: clips[data] if data in clips else load_video(data, use_gpu)
    glm4v.glm_sample_and_decode_sync = recording
    images = []
    for m in media:
        if m["kind"] == "image":
            png = io.BytesIO()
            Image.fromarray(m["rgb"]).save(png, format="PNG")
            images.append(png.getvalue())
    try:
        output = asyncio.run(path.processor.process_mm_data_async(
            image_data=images, input_text=tokenizer.encode(case["prompt"], add_special_tokens=False),
            request_obj=SimpleNamespace(video_data=list(clips), audio_data=None, rid="parity",
                                        mm_content_hashes=None)))
    except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
        return {"rejects": reason(error)}
    finally:
        mixin.load_video, glm4v.glm_sample_and_decode_sync = load_video, sample_and_decode
    videos = [item for item in output.mm_items if item.modality.name == "VIDEO"]
    grids = [row for item in videos for row in item.video_grid_thw.reshape(-1, 3).tolist()]
    features = [item.feature.detach().cpu().float().numpy() for item in videos]
    record = summary([int(i) for i in output.input_ids], [sampled[id(c)] for c in clips.values()],
                     grids, features)
    item_spans = [list(span) for item in videos for span in item.offsets]
    assert item_spans == [span for video in record["video_spans"] for span in video], item_spans
    mrope = output.mrope_positions.cpu().numpy().astype("<i8")
    record.update(mrope_blake3=digest(mrope), mrope_delta=int(output.mrope_position_delta.item()))
    return record


def sglang_on_cpu(model_dir):
    """SglangPath on a host without an accelerator: server argument resolution
    needs a GPU, and the CPU processor reads none of the resolved values."""
    from sglang.srt.server_args import ServerArgs

    ServerArgs.resolve_once = lambda self: None
    return SglangPath(model_dir, "torchvision", None)


def checkpoint_files(model_dir):
    """config.json, cut to the fields the Rust resolver reads, and processor_config.json."""
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(model_dir, "processor_config.json")) as f:
        processor = json.load(f)
    ids = [k for k in config if k.endswith("_token_id") and k.split("_")[0] in ("image", "video")]
    vision = {k: config["vision_config"][k]
              for k in ("patch_size", "spatial_merge_size", "temporal_patch_size")}
    return dict(config=dict(model_type=config["model_type"], **{k: config[k] for k in ids},
                            vision_config=vision),
                processor_config=processor)


def rust_spec(files):
    """The spec the Rust test expects the resolver to give for these files."""
    config = files["config"]
    image, video = files["processor_config"]["image_processor"], files["processor_config"]["video_processor"]
    knobs = ("patch_size", "merge_size", "temporal_patch_size", "patch_expand_factor",
             "min_image_tokens", "max_image_tokens")
    return dict(family="glm5_next", image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN,
                **{k: image[k] for k in knobs}, image_mean=image["image_mean"],
                image_std=image["image_std"], resample="aten_u8",
                video=dict(image_start_token_id=config["image_start_token_id"],
                           image_end_token_id=config["image_end_token_id"],
                           fps=video["fps"], max_frames=2048, **{k: video[k] for k in knobs},
                           image_mean=video["image_mean"], image_std=video["image_std"]))


def main(model_dir):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    assert tokenizer.convert_tokens_to_ids(
        ["<|image|>", "<|video|>", "<|begin_of_video|>", "<|end_of_video|>"]
    ) == [IMAGE_TOKEN, VIDEO_TOKEN, VIDEO_START, VIDEO_END]
    with open(os.path.join(model_dir, "config.json")) as f:
        mrope_section = '"mrope_section"' in f.read()
    for name, case in CASES.items():
        variant = model_variant(model_dir, case.get("max_image_tokens"))
        media = media_of(name, case)
        hf = run_hf(variant, case, media, tokenizer)
        path = sglang_on_cpu(variant)
        try:
            sglang = run_sglang(path, case, media, tokenizer)
        finally:
            path.close()
        texts = set()
        for ref in (hf, sglang):
            for sampled, m in zip(ref.get("sampled", []), [m for m in media if m["kind"] == "video"]):
                texts |= {f"{i / m['fps']:.1f} seconds" for i in sampled[::2]}
        fields = ("sampled", "video_grids", "input_ids", "video_spans", "pixel_values_videos_blake3")
        differs = (["rejects"] if ("rejects" in hf) != ("rejects" in sglang) else
                   [f for f in fields if hf.get(f) != sglang.get(f)])
        files = checkpoint_files(variant)
        record = dict(
            checkpoint=files, spec=rust_spec(files),
            media=[{k: v for k, v in m.items() if k != "rgb" and v is not None} for m in media],
            prompt_ids=tokenizer.encode(case["prompt"], add_special_tokens=False),
            timestamp_ids={t: tokenizer.encode(t, add_special_tokens=False) for t in sorted(texts)},
            sglang=sglang, hf=hf, sglang_differs=differs,
            provenance={"generator": "multimodal/tests/fixtures/glm5_next_video/generate.py",
                        "checkpoint": f"{MODEL}@{REVISION}", "config_mrope_section": mrope_section,
                        "sglang": sglang_version(), "transformers": transformers.__version__,
                        "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "numpy": np.__version__})
        os.makedirs(os.path.join(ROOT, name), exist_ok=True)
        with open(os.path.join(ROOT, name, "case.json"), "w") as f:
            json.dump(record, f, indent=1)
            f.write("\n")
        print(name, "sglang differs from hf in:", differs or "nothing")


if __name__ == "__main__":
    main(sys.argv[1])
