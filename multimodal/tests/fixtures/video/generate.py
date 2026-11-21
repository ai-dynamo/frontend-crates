# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the video golden cases: `smart_resize.json` and
`glm_frame_indices.txt`.

The expected sizes come from HF's own `smart_resize` in
`Qwen3VLVideoProcessor`, executed verbatim: the function is cut out of the
pinned source file with `ast`, so no transformers/torch install is needed.
The per-frame cap lines are copied from `Qwen3VLVideoProcessor.resize`
(`cap_pixels_per_frame`), with `frame_cap` standing in for
`max_video_tokens * factor * factor`.

    python multimodal/tests/fixtures/video/generate.py [path-to-video_processing_qwen3_vl.py]

With no argument the pinned file is downloaded.

`glm_frame_indices.txt` holds, per line, `frames fps target_fps max_frames:`
and the source frames SGLang's `glm_sample_frame_indices` (25c1c439) keeps,
with `duration = frames / fps` as SGLang's caller passes it. SGLang must be
importable.
"""

import ast
import json
import math
import sys
import urllib.request
from pathlib import Path

SOURCE_SHA = "528c26713c8f3774fb56d409da766bc95452aafb"  # transformers, 2026-09-30
SOURCE_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    f"{SOURCE_SHA}/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py"
)


def load_smart_resize(text: str):
    tree = ast.parse(text)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "smart_resize"
    )
    ns = {"math": math}
    exec(compile(ast.Module([fn], []), "smart_resize", "exec"), ns)  # noqa: S102
    return ns["smart_resize"]


def capped_max_pixels(longest_edge, shortest_edge, num_frames, frame_cap):
    """`Qwen3VLVideoProcessor.resize` when `cap_pixels_per_frame` is set."""
    pixels_per_frame = max(
        min(frame_cap, longest_edge // num_frames), int(shortest_edge * 1.05)
    )
    return pixels_per_frame * num_frames


def write_glm_frame_indices() -> None:
    from sglang.srt.multimodal.processors.glm4v import glm_sample_frame_indices

    cases = []
    # Fixture rates, NTSC, below the 2 fps target; under 0.5 s, odd counts.
    for fps in [0.5, 1.0, 1.5, 2.0, 2.5, 10.0, 23.976, 24000 / 1001, 25.0, 29.97,
                30000 / 1001, 30.0, 59.94]:
        for frames in [1, 2, 3, 6, 7, 15, 16, 25, 31, 61, 91]:
            cases.append((frames, fps, 2.0, 2048))
    # Counts around max_frames: the boundary picks, then the whole-clip linspace.
    for fps, base in [(30.0, 960), (29.97, 959), (2.0, 64), (1.0, 32), (23.976, 768)]:
        for delta in [-31, -1, 0, 1, 31]:
            cases.append((base + delta, fps, 2.0, 64))
    for max_frames in [1, 2, 3, 4, 16]:
        for frames, fps in [(300, 30.0), (25, 10.0), (7, 1.0), (6, 30.0)]:
            cases.append((frames, fps, 2.0, max_frames))
    for target in [1.0, 0.7, 3.0]:
        for frames, fps in [(300, 30.0), (25, 10.0), (91, 29.97)]:
            cases.append((frames, fps, target, 2048))

    lines = []
    for frames, fps, target, max_frames in cases:
        picked = glm_sample_frame_indices(
            frames, fps, frames / fps, target_fps=target, max_frame_count=max_frames
        )
        lines.append(f"{frames} {fps!r} {target!r} {max_frames}: {' '.join(map(str, picked))}")
    out = Path(__file__).with_name("glm_frame_indices.txt")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(cases)} cases)")


def main() -> None:
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text()
    else:
        text = urllib.request.urlopen(SOURCE_URL).read().decode()  # noqa: S310
    smart_resize = load_smart_resize(text)

    # Qwen3-VL video: patch 16, merge 2 -> factor 32, temporal 2.
    factor, temporal = 32, 2
    min_px, max_px = 65536, 16777216

    resize_cases = []
    for frames, h, w, lo, hi in [
        (8, 480, 640, min_px, max_px),  # inside the budget: plain rounding
        (32, 1080, 1920, min_px, max_px),  # 1080p, default budget
        (64, 1080, 1920, min_px, 8_000_000),  # budget binds: downscale
        (128, 2160, 3840, min_px, 16777216),  # 4K, many frames
        (6, 90, 160, 200_000, max_px),  # below min_pixels: upscale
        (4, 20, 200, min_px, max_px),  # frame smaller than factor
        (4, 31, 33, min_px, max_px),  # just under the factor
        (16, 100, 4000, min_px, 2_000_000),  # very wide, budget binds
        (2, 1, 1, 1024, 1024),  # extreme upscale
        # t_bar = round(n / 2) * 2. Ties go to even (n=5: round(2.5) = 2, t_bar 4),
        # and the budget sits between the tie-even and tie-away clip volumes, so
        # rounding ties away from zero (t_bar 6) would downscale instead.
        (5, 480, 640, min_px, 1_300_000),
        (9, 480, 640, min_px, 2_800_000),  # same, round(4.5) = 4, t_bar 8 vs 10
        (4, 20, 200, 1024, 30_000),  # scale-up, then the max branch
        (4, 20, 200, 2_000_000, max_px),  # scale-up, then the min branch
        (4, 10, 3000, min_px, max_px),  # aspect ratio > 200 after scale-up: error
        (1, 480, 640, min_px, max_px),  # fewer frames than temporal_factor: error
    ]:
        try:
            expected = list(smart_resize(frames, h, w, temporal, factor, lo, hi))
        except ValueError:
            expected = None
        resize_cases.append(
            dict(num_frames=frames, height=h, width=w, temporal_factor=temporal,
                 factor=factor, min_pixels=lo, max_pixels=hi, expected=expected)
        )

    # The budget cases' model, as a `video_preprocessor_config.json` would give it.
    video_config = dict(video_processor_type="Qwen3VLVideoProcessor", patch_size=16,
                        merge_size=2, temporal_patch_size=temporal,
                        size=dict(shortest_edge=min_px, longest_edge=max_px))
    budget_cases = []
    for frames, h, w, total, cap in [
        (32, 1080, 1920, None, None),
        (32, 1080, 1920, 4_000_000, None),
        (32, 1080, 1920, None, 100_000),
        (32, 1080, 1920, 8_000_000, 200_000),
        (8, 1080, 1920, 16_777_216, 1_000_000),
        (64, 720, 1280, 6_000_000, 50_000),  # cap below the floor: floor wins
        (16, 720, 1280, 1_000_000, 10_000_000),  # cap above the share; the 1.05*min floor wins
    ]:
        longest = total if total is not None else max_px
        max_pixels = longest
        if cap is not None:
            max_pixels = capped_max_pixels(longest, min_px, frames, cap)
        expected = list(smart_resize(frames, h, w, temporal, factor, min_px, max_pixels))
        budget_cases.append(
            dict(num_frames=frames, height=h, width=w, total_pixels=total,
                 max_pixels_per_frame=cap, expected=expected)
        )

    out = Path(__file__).with_name("smart_resize.json")
    out.write_text(json.dumps(
        dict(source=SOURCE_URL, video_config=video_config, resize=resize_cases,
             budget=budget_cases),
        indent=1) + "\n")
    print(f"wrote {out}")
    write_glm_frame_indices()


if __name__ == "__main__":
    main()
